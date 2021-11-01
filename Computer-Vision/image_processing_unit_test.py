
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import argparse
from utils.process_image import process_single_image
import os

image_directory = 'indoor_outdoor_images/0-z2_jDDLg2wc.jpg'
IMAGE_SIZE = 224
MODEL_PATH = "model_results/fine_tuned_model"
CLASS_LABELS = ('outdoor', 'indoor')
METHODS = ('INTEGER_FAST', 'INTEGER_ACCURATE')  # TF decode methods


def _parse_image(filename, method):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3, dct_method=method)
    image = image.numpy()
    return image


def pil_process_single_image(filename):
    image = Image.open(filename)
    image = np.array(image)
    print(image.dtype)
    print(np.max(image))
    return image


def open_cv_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.dtype)
    print(np.max(image))
    return image


def resize_pil_image_for_prediction(filename, image_size):
    image = pil_process_single_image(filename)
    image = np.resize(image, (image_size, image_size, 3))
    image = np.expand_dims(image, axis=0)
    return image


def dir_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"image_name:{path} is not a valid path")


if __name__ == "__main__":
    """Returns predictions for a single image"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="image_path", required=True,
                        help="Full path to image, must be .jpg", metavar="FILE", type=lambda x: dir_file(x))

    # Load images from Pillow, CV2 and TF
    pil_image = pil_process_single_image(image_directory)
    cv2_image = open_cv_image(image_directory)
    # tf_image_method_0 = _parse_image(image_directory, methods[0])
    tf_image_method_1 = _parse_image(image_directory, METHODS[1])

    print('decode_jpeg method: ', METHODS[1])
    count = np.count_nonzero(tf_image_method_1 != pil_image)
    print('Number of different values between tf and pillow:', count)
    print('Image arrays are equal: ', (tf_image_method_1 == pil_image).all())

    print('decode_jpeg method: ', METHODS[1])
    count = np.count_nonzero(tf_image_method_1 != cv2_image)
    print('Number of different values between tf and cv2:', count)
    print('Image arrays are equal: ', (tf_image_method_1 == cv2_image).all())

    count = np.count_nonzero(pil_image != cv2_image)
    print('Number of different values between pillow and cv2:', count)
    print('Image arrays are equal: ', (pil_image == cv2_image).all())

    # Load model
    print("Loading model...")
    fine_tuned_model = tf.keras.models.load_model(MODEL_PATH)

    # Process images
    tf_processed_image = process_single_image(image_directory, IMAGE_SIZE)
    pil_processed_image = resize_pil_image_for_prediction(image_directory, IMAGE_SIZE)

    # Return predictions
    tf_prediction = fine_tuned_model.predict(tf_processed_image)
    prediction = fine_tuned_model.predict(pil_processed_image)

    # Compare
    pred_index = np.argmax(tf_prediction, axis=1)
    print('image prediction from TF processing:')
    print('class: ', CLASS_LABELS[pred_index[0]])
    print('score: ', tf_prediction[0][pred_index][0])

    pred_index = np.argmax(prediction, axis=1)
    print('image prediction from PIL processing:')
    print('class: ', CLASS_LABELS[pred_index[0]])
    print('score: ', prediction[0][pred_index][0])