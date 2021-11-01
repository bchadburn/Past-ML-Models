from utils.process_image import *
import os
import argparse

IMAGE_SIZE = 224

CLASS_LABELS = ('outdoor', 'indoor')

MODEL_PATH = "model_results/fine_tuned_model"
IMAGE_FOLDER_PATH = "indoor_outdoor_images"


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

    args, _ = parser.parse_known_args()
    image = process_single_image(args.image_path, IMAGE_SIZE)
    fine_tuned_model = tf.keras.models.load_model(MODEL_PATH)

    prediction = fine_tuned_model.predict(image)
    pred_index = np.argmax(prediction, axis=1)
    print('class: ', CLASS_LABELS[pred_index[0]])
    print('score: ', prediction[0][pred_index][0])

