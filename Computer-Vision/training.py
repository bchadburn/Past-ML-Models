import pandas as pd
import numpy as np
import random
from sklearn import metrics
from sklearn import model_selection
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from model import model
from utils.utils import *
from utils.process_image import _parse_image
from utils.config import *


class Dataset:
    def __init__(self, dataset_directory, batch_size, image_size):
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.image_size = image_size

        self.classes = None
        self.total_images = None
        self.class_weight = None

        self.train_dataset = None
        self.validation_dataset = None

        self.validation_metadata = None

    def load(self, classes=None):
        """Creates training and validation dataset from directory of images. The prefix of the image name is used to
        identify the class."""

        tmp_classes = classes if classes else set(
            [' '.join(i[0].split('-')) for i in os.listdir(self.dataset_directory)])
        self.classes = list(tmp_classes)

        print(f"Classes: {len(self.classes)}. {self.classes}")

        all_classes_str = [str(i) for i in tmp_classes]
        diff_list = list(set(all_classes_str) - set(tmp_classes))
        if len(diff_list) > 0:
            raise ValueError(f"Classes in image directory don't match classes defined in config.py file. "
                             f"Class differences: {diff_list}")

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
        """Loads image filenames and their respective classes. Calculated total images and class weights in case of
        class imbalance."""
        image_filenames = []
        for classname in self.classes:
            for name in os.listdir(self.dataset_directory):
                if name.split('-')[0] == classname:
                    image_filenames.append(os.path.join(self.dataset_directory, name))

        self.total_images = len(image_filenames)
        image_filenames = sorted(image_filenames)

        image_targets = [self.classes.index(name.split("\\")[1].split('-')[0]) for name in image_filenames]

        self.class_weight = dict(zip(np.arange(len(image_targets)), 1.0 / np.bincount(image_targets)))
        return image_filenames, image_targets

    def _dataset(self, image_filenames, image_targets, batch_size, repeat=False, metadata=False):
        """Creates tf.data tensors of filenames, targets used for training. Sets up parallel processing
        for image loading and parsing. Sets prefecth to load next batch in CPU, while GPU is in training."""
        image_filenames_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

        target_dataset = tf.data.Dataset.from_tensor_slices(image_targets)
        image_dataset = image_filenames_dataset if metadata else image_filenames_dataset.map(
            lambda x: _parse_image(x, IMAGE_SIZE),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, target_dataset))

        # batch_size is 0 when creating dataset of the image filenames, otherwise a batch of images is created for training
        if batch_size > 0:
            dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        # Prefetch data in CPU, while GPU is training
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


def evaluate_model(session, model, dataset):
    """Returns predictions on validation data set along with image names to review specific predictions. Prints
    classification report, saves confusion matrix and predictions for each filename."""
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

    confusion_matrix.to_csv(os.path.join(MODEL_RESULTS_PATH, f"{session}-confusion-matrix.csv"))

    # Create list of image files to match to validation predictions during evaluation
    files = list(
        map(lambda s: s.decode().split("/")[-1], list(dict(dataset.validation_metadata.as_numpy_iterator()).keys())))

    predictions = pd.DataFrame({
        "filename": files,
        "target": np.array(dataset.classes)[y_true],
        "prediction": np.array(dataset.classes)[y_pred],
        "confidence": np.max(validation_predictions, axis=1)
    })

    predictions.to_csv(os.path.join(MODEL_RESULTS_PATH, f"{session}-predictions.csv"))

    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Validation accuracy: {accuracy}")
    return accuracy


@time_it
def fit_model(
        model,
        dataset,
        learning_rate,
        epochs,
        fine_tuning_learning_rate,
        fine_tuning_epochs
):
    """Compiles and trains model with model checkpoint. Returns performance on entire validation set.
     If fine-tuning epochs>0, if runs fine-tuning where all layers are unfrozen.
     Finally, it loads best weights, and returns performance on validation set."""
    checkpoint_filepath = f"{MODEL_CHECKPOINT_PATH}/cp.ckpt"
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

    model.fit(
        dataset.train_dataset,
        validation_data=dataset.validation_dataset,
        epochs=epochs,
        steps_per_epoch=dataset.total_images / dataset.batch_size,
        callbacks=[model_checkpoint],
        class_weight=dataset.class_weight,
        verbose=2
    )

    print("Classification model trained successfully.")

    model.load_weights(checkpoint_filepath)

    accuracy = evaluate_model("training", model, dataset)
    if accuracy == 1 or fine_tuning_epochs == 0:
        print("Fine tuning classification model skipped.")
        if accuracy == 1:
            print(f"Accuracy reached {accuracy:.0%}")
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

    model.fit(
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
    """Builds dataset from image files, creates and fits model and saves model weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=TRAINING_IMAGES_PATH)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="The learning rate that will be used to train the classification model."
                        )

    parser.add_argument('--epochs', type=int, default=15,
                        help=(
                            "The number of epochs that will be used to train the initial classification model."
                        )
                        )

    parser.add_argument("--fine_tuning_learning_rate", type=float, default=1e-5,
                        help="The learning rate that will be used to fine tune the classification model."
                        )

    parser.add_argument('--fine_tuning_epochs', type=int, default=15,
                        help=(
                                "The number of epochs that will be used to fine tune the classification model. If zero is specified, the model will not " +
                                "go through the fine tuning process."
                        )
                        )

    args, _ = parser.parse_known_args()

    # Check GPU is configured and compare GPU vs CPU compute time
    check_cpu_gpu()

    # Create paths if they don't already exist
    verify_create_paths([MODEL_RESULTS_PATH, MODEL_CHECKPOINT_PATH])

    # Create filename, target datasets and filename tensors in prep for model training
    dataset = Dataset(args.image_path, batch_size=32, image_size=IMAGE_SIZE)
    dataset.load(classes=None)

    model = model.create_model(len(dataset.classes), IMAGE_SIZE)

    model = fit_model(
        model,
        dataset,
        args.learning_rate,
        args.epochs,
        args.fine_tuning_learning_rate,
        args.fine_tuning_epochs
    )

    model_id = random.randint(0, 1000)
    model.save(os.path.join(MODEL_RESULTS_PATH, 'fine_tuned_model'))

    print("Model was successfully saved.")
