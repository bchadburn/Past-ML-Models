import argparse
# import glob
# import matplotlib.pyplot as plt
# import numpy as np
import tarfile
import tensorflow as tf
import pandas as pd
import numpy as np
import PIL
import PIL.Image
import random
from pathlib import Path
from sklearn import metrics
from sklearn import model_selection
from tensorflow.keras import callbacks
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.layers.experimental import preprocessing
import os

IMAGE_SIZE = 224

from utils.gpu_config import *

# # Check GPU is configured and compare GPU vs CPU time
# print_system_info()
# limit_memory(enable=True)
# get_processor_time()


CLASSES = None


class Dataset:
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size

        self.classes = None
        self.total_images = None
        self.class_weight = None

        self.train_dataset = None
        self.validation_dataset = None

        self.validation_metadata = None

    def load(self, dataset_directory, classes=None):
        self.dataset_directory = dataset_directory

        tmp_classes = classes if classes else set([' '.join(i[0].split('-')) for i in os.listdir(dataset_directory)])
        self.classes = list(tmp_classes)
        print(f"Classes: {len(self.classes)}. {self.classes}")

        image_filenames, image_targets = self._load_images()

        (
            image_filenames_train,
            image_filenames_validation,
            image_targets_train,
            image_targets_validation
        ) = model_selection.train_test_split(
            image_filenames,
            image_targets,
            train_size=0.8,
            stratify=image_targets,
            shuffle=True,
            random_state=42
        )

        self.train_dataset = self._dataset(image_filenames_train, image_targets_train,
                                           self.batch_size, repeat=True)
        self.validation_dataset = self._dataset(image_filenames_validation, image_targets_validation,
                                                self.batch_size, repeat=False)
        self.validation_metadata = self._dataset(image_filenames_validation, image_targets_validation, batch_size=0,
                                                 repeat=False, metadata=True)

    def _load_images(self):
        image_filenames = []
        for classname in self.classes:
            for name in os.listdir(self.dataset_directory):
                if name.split('-')[0] == classname:
                    image_filenames.append(os.path.join(self.dataset_directory, name))

        self.total_images = len(image_filenames)
        image_filenames = sorted(image_filenames)

        image_targets = [dataset.classes.index(name.split("\\")[1].split('-')[0]) for name in image_filenames]

        self.class_weight = dict(zip(np.arange(len(image_targets)), 1.0 / np.bincount(image_targets)))
        return image_filenames, image_targets

    def _dataset(self, image_filenames, image_targets, batch_size, repeat=False, metadata=False):
        image_filenames_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

        target_dataset = tf.data.Dataset.from_tensor_slices(image_targets)
        image_dataset = image_filenames_dataset if metadata else image_filenames_dataset.map(self._parse_image,
                                                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, target_dataset))

        if batch_size > 0:
            dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def _parse_image(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image


def create_model(num_classes):
    data_augmentation = Sequential([
        preprocessing.RandomFlip("horizontal", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), seed=42),
        preprocessing.RandomRotation(factor=(-0.2, 0.2), seed=42),
        preprocessing.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=42),
        preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=42),
    ])

    base_model = resnet_v2.ResNet50V2(
        weights='imagenet',
        include_top=False
    )

    base_model.trainable = False

    model_inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = data_augmentation(model_inputs)
    x = resnet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    model_outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(name="classification", inputs=model_inputs, outputs=model_outputs)

    return model


def evaluate_model(session, model, dataset):
    y_true = list(dict(dataset.validation_metadata.as_numpy_iterator()).values())
    validation_predictions = model.predict(dataset.validation_dataset)
    y_pred = np.argmax(validation_predictions, axis=1)

    print(metrics.classification_report(
        y_true,
        y_pred,
        target_names=dataset.classes
    ))

    confusion_matrix = pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred),
        columns=dataset.classes,
        index=dataset.classes
    )
    confusion_matrix.to_csv(os.path.join("/opt/ml/model", f"{session}-confusion-matrix.csv"))

    files = list(
        map(lambda s: s.decode().split("/")[-1], list(dict(dataset.validation_metadata.as_numpy_iterator()).keys())))

    predictions = pd.DataFrame({
        "filename": files,
        "target": np.array(dataset.classes)[y_true],
        "prediction": np.array(dataset.classes)[y_pred],
        "confidence": np.max(validation_predictions, axis=1)
    })

    predictions.to_csv(os.path.join("/opt/ml/model", f"{session}-predictions.csv"))

    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Validation accuracy: {accuracy}")


def fit_model(
        model,
        dataset,
        learning_rate,
        epochs,
        fine_tuning_learning_rate,
        fine_tuning_epochs
):
    checkpoint_filepath = "classification"
    os.makedirs(checkpoint_filepath, exist_ok=True)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(model.summary())

    # Save the weights of the best model
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min"
    )

    history = model.fit(
        dataset.train_dataset,
        validation_data=dataset.validation_dataset,
        epochs=epochs,
        steps_per_epoch=dataset.total_images / dataset.batch_size,
        callbacks=[model_checkpoint],
        class_weight=dataset.class_weight,
        verbose=2
    )

    model.load_weights(checkpoint_filepath)

    evaluate_model("training", model, dataset)

    print("Classification model trained successfully.")

    if fine_tuning_epochs == 0:
        print("Fine tuning classification model skipped.")
        return model

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tuning_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max"
    )

    history = model.fit(
        dataset.train_dataset,
        validation_data=dataset.validation_dataset,
        epochs=fine_tuning_epochs,
        steps_per_epoch=dataset.total_images / dataset.batch_size,
        callbacks=[model_checkpoint],
        class_weight=dataset.class_weight,
        verbose=2
    )

    model.load_weights(checkpoint_filepath)

    evaluate_model("fine-tuning", model, dataset)

    print("Classification model fine-tuned successfully.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='indoor_outdoor_images')

    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="The learning rate that will be used to train the classification model."
                        )

    parser.add_argument('--epochs', type=int, default=30,
                        help=(
                                "The number of epochs that will be used to train the classification model. If zero is specified, the model will be created " +
                                "from a previously set of trained weights."
                        )
                        )

    parser.add_argument("--fine_tuning_learning_rate", type=float, default=1e-4,
                        help="The learning rate that will be used to fine tune the classification model."
                        )

    parser.add_argument('--fine_tuning_epochs', type=int, default=15,
                        help=(
                                "The number of epochs that will be used to fine tune the classification model. If zero is specified, the model will not " +
                                "go through the fine tuning process."
                        )
                        )

    parser.add_argument("--evaluation", type=int, default=1,
                        help=(
                            "If in evaluation mode. If so, we evaluate the validation after fine-tuning"
                            "and don't save a copy of the model. If not, we save a copy after training. "
                        )
                        )

    args, _ = parser.parse_known_args()

    evaluation = args.evaluation == 1

    if evaluation:
        tf.random.set_seed(123)


    dataset = Dataset(batch_size=32, image_size=IMAGE_SIZE)

    dataset.load(Path(dataset_path), classes=CLASSES)

    model = create_model(len(dataset.classes))

    model = fit_model(
        model,
        dataset,
        # args.learning_rate,
        # args.epochs,
        # args.fine_tuning_learning_rate,
        # args.fine_tuning_epochs
        learning_rate,
        epochs,
        fine_tuning_learning_rate,
        fine_tuning_epochs
    )

    model_id = random.randint(0, 1000)
    model_filepath = os.path.join("model", str(model_id))
    model.save(model_filepath)

    print("Model was successfully saved.")

dataset_path = Path('indoor_outdoor_images')
learning_rate = 1e-4
fine_tuning_learning_rate = 1e-5
fine_tuning_epochs = 15
epochs = 20

