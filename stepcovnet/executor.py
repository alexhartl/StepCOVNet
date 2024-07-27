import json
import os
from abc import ABC, abstractmethod

import joblib
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from scipy.signal import find_peaks

from stepcovnet import (
    config,
    encoder,
    inputs,
    training,
    model,
    constants,
    tf_config,
    utils,
    data
)


class AbstractExecutor(ABC):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel, *args, **kwargs):
        self.stepcovnet_model = stepcovnet_model
        tf_config.tf_init()

    @abstractmethod
    def execute(self, input_data: inputs.AbstractInput):
        pass


class InferenceExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel, verbose: bool = False):
        super(InferenceExecutor, self).__init__(stepcovnet_model=stepcovnet_model)
        self.verbose = verbose
        self.binary_arrow_encoder = encoder.BinaryArrowEncoder()
        self.label_arrow_encoder = encoder.LabelArrowEncoder()
        self.onehot_arrow_encoder = encoder.OneHotArrowEncoder()

    def execute(self, input_data: inputs.InferenceInput) -> list[str]:
        arrow_input = input_data.arrow_input_init
        arrow_mask = input_data.arrow_mask_init
        pred_arrows = []
        tokenizer = (
            None if input_data.config.tokenizer_name is None else data.Tokenizers[input_data.config.tokenizer_name].value
        )
        lookback = input_data.config.lookback
        inferer = self.stepcovnet_model.model.signatures["serving_default"]

        # whether we feed detection back into the network. unfortunately,
        # this does not seem to give good results currently
        autoregressive_input = False
        
        all_predictions = np.ones((len(input_data.audio_features), constants.NUM_ARROW_COMBS), dtype=np.float32)
        empty_predictions_detrended = np.empty(len(input_data.audio_features), dtype=np.float32)
        audio_features_index = 0
        last_nonempty_frame = -1
        pbar = tqdm(total=len(input_data.audio_features) - lookback//2)
        while audio_features_index < len(input_data.audio_features) - lookback//2:
            pbar.update(audio_features_index - pbar.n)
            audio_features = utils.get_samples_ngram_with_mask(
                samples=input_data.audio_features[
                    max(
                        audio_features_index + 1 - lookback//2, 0
                    ) : audio_features_index + lookback//2
                    + 1
                ],
                lookback=input_data.config.lookback,
                squeeze=False,
            )[0][-1]
            audio_input = utils.apply_scalers(
                features=audio_features, scalers=input_data.config.scalers
            )
            onehot_arrows_probs = inferer(
                arrow_input=tf.convert_to_tensor(arrow_input.astype(np.int32)),
                arrow_mask=tf.convert_to_tensor(arrow_mask.astype(np.int32)),
                audio_input=tf.convert_to_tensor(audio_input[None]),
            )
            onehot_arrows_probs = (
                next(iter(onehot_arrows_probs.values())).numpy().ravel()
            )
            all_predictions[audio_features_index] = onehot_arrows_probs
            arrows = '0000'
            if audio_features_index > 50:
                empty_predictions_detrended[audio_features_index] = \
                    onehot_arrows_probs[0] - np.mean(all_predictions[max(0,audio_features_index-100):audio_features_index,0])
            if audio_features_index > 70:
                # how many frames we need detection to be distant from each other
                min_distance = 5
                peaks,_ = find_peaks(-empty_predictions_detrended[audio_features_index-20:audio_features_index], height=input_data.config.threshold, distance=min_distance)
                peaks += audio_features_index-20
                peaks = peaks[peaks > last_nonempty_frame+min_distance]
                if peaks.size:
                    audio_features_index = last_nonempty_frame = np.min(peaks)
                    pred_arrows = pred_arrows[:audio_features_index]
                    p = all_predictions[audio_features_index,1:].copy()
                    p /= p.sum()
                    arrows = self.onehot_arrow_encoder.decode(1+np.random.choice(len(p), p=p))

            pred_arrows.append(arrows)
            if tokenizer is not None:
                tokenizer_input = pred_arrows[-lookback:]
                if not autoregressive_input:
                    tokenizer_input = ['0000'] * len(tokenizer_input)
                arrow_input = tokenizer(' '.join(tokenizer_input), return_tensors="tf", add_prefix_space=True)["input_ids"].numpy().astype(np.int32)
                arrow_mask = np.ones_like(arrow_input)
            else:
                # Roll and append predicted arrow to input to predict next sample
                arrow_input = np.roll(arrow_input, -1, axis=0)
                arrow_mask = np.roll(arrow_mask, -1, axis=0)
                arrow_input[0][-1] = self.label_arrow_encoder.encode(arrows)
                arrow_mask[0][-1] = 1
            if self.verbose and audio_features_index % 100 == 0:
                print(
                    "[%d/%d] Samples generated"
                    % (audio_features_index, len(input_data.audio_features))
                )
            audio_features_index += 1
        pbar.close()
        return pred_arrows


class TrainingExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel):
        super(TrainingExecutor, self).__init__(stepcovnet_model=stepcovnet_model)

    def execute(self, input_data: inputs.TrainingInput) -> model.StepCOVNetModel:
        hyperparameters = input_data.config.hyperparameters

        weights = (
            self.stepcovnet_model.model.get_weights()
            if hyperparameters.retrain
            else None
        )

        self.stepcovnet_model.model.compile(
            loss=hyperparameters.loss,
            metrics=hyperparameters.metrics,
            optimizer=hyperparameters.optimizer,
        )
        self.stepcovnet_model.model.summary()
        # Saving scalers and metadata in the case of errors during training
        self.save(input_data.config, pretrained=True, retrained=False)
        history = self.train(input_data, self.get_training_callbacks(hyperparameters))
        self.save(input_data.config, training_history=history, retrained=False)

        if hyperparameters.retrain:
            epochs_final = len(history.history["val_loss"])
            retraining_history = self.retrain(
                input_data,
                saved_original_weights=weights,
                epochs=epochs_final,
                callback_list=self.get_retraining_callbacks(hyperparameters),
            )
            self.save(
                input_data.config, training_history=retraining_history, retrained=True
            )
        return self.stepcovnet_model

    def get_training_callbacks(
        self, hyperparameters: training.TrainingHyperparameters
    ) -> list[callbacks.Callback]:
        model_out_path = self.stepcovnet_model.model_root_path
        model_name = self.stepcovnet_model.model_name
        log_path = hyperparameters.log_path
        patience = hyperparameters.patience
        callback_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_out_path, model_name + "_callback"),
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
            )
        ]
        if patience > 0:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, verbose=0
                )
            )

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=os.path.join(log_path, "split_dataset"),
                    histogram_freq=1,
                    profile_batch=100000000,
                )
            )
        return callback_list

    @staticmethod
    def get_retraining_callbacks(
        hyperparameters: training.TrainingHyperparameters,
    ) -> list[callbacks.Callback]:
        log_path = hyperparameters.log_path
        callback_list = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=os.path.join(log_path, "whole_dataset"),
                    histogram_freq=1,
                    profile_batch=100000000,
                )
            )
        return callback_list

    def train(
        self, input_data: inputs.TrainingInput, callback_list: list[callbacks.Callback]
    ) -> callbacks.History:
        print(
            "Training on %d samples (%d songs) and validating on %d samples (%d songs)"
            % (
                input_data.train_feature_generator.num_samples,
                len(input_data.train_feature_generator.train_indexes),
                input_data.val_feature_generator.num_samples,
                len(input_data.val_feature_generator.train_indexes),
            )
        )
        print("\nStarting training...")
        history = self.stepcovnet_model.model.fit(
            x=input_data.train_generator,
            epochs=input_data.config.hyperparameters.epochs,
            steps_per_epoch=len(input_data.train_feature_generator),
            validation_steps=len(input_data.val_feature_generator),
            callbacks=callback_list,
            class_weight=input_data.config.train_class_weights,
            validation_data=input_data.val_generator,
            verbose=1,
        )
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(
        self,
        input_data: inputs.TrainingInput,
        saved_original_weights: np.ndarray,
        epochs: int,
        callback_list: list[callbacks.Callback],
    ) -> callbacks.History:
        print(
            "Training on %d samples (%d songs)"
            % (
                input_data.all_feature_generator.num_samples,
                len(input_data.all_feature_generator.train_indexes),
            )
        )
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history: callbacks.History = self.stepcovnet_model.model.fit(
            x=input_data.all_generator,
            epochs=epochs,
            steps_per_epoch=len(input_data.all_feature_generator),
            callbacks=callback_list,
            class_weight=input_data.config.all_class_weights,
            verbose=1,
        )
        print("\n*******************************")
        print("***** RETRAINING FINISHED *****")
        print("*******************************\n")
        return history

    def save(
        self,
        training_config: config.TrainingConfig,
        retrained: bool,
        training_history: callbacks.History | None = None,
        pretrained: bool = False,
    ):
        model_out_path = self.stepcovnet_model.model_root_path
        model_name = self.stepcovnet_model.model_name
        if pretrained:
            if training_config.all_scalers is not None:
                joblib.dump(
                    training_config.all_scalers,
                    open(
                        os.path.join(model_out_path, model_name + "_scaler.pkl"), "wb"
                    ),
                )
        elif retrained:
            model_name += "_retrained"
        if not pretrained:
            print('Saving model "%s" at %s' % (model_name, model_out_path))
            self.stepcovnet_model.model.save(os.path.join(model_out_path, model_name))
        if self.stepcovnet_model.metadata is None:
            self.stepcovnet_model.build_metadata_from_training_config(training_config)
        if training_history is not None and not pretrained:
            history_name = "retraining_history" if retrained else "training_history"
            self.stepcovnet_model.metadata[history_name] = training_history.history
        print("Saving model metadata at %s" % model_out_path)
        with open(os.path.join(model_out_path, "metadata.json"), "w") as json_file:
            json_file.write(json.dumps(self.stepcovnet_model.metadata))
