import os
import cv2
import numpy as np
import argparse
import warnings
import time
from mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')
import gc
import logging
import support
from face_crop import crop_face

logger = logging.getLogger(__name__)

SAMPLE_IMAGE_PATH = "./images/sample/"

IMAGE_FOLDER = "image_folder"


def check_image(image):
    height, width, channel = image.shape
    logger.info("height is {}".format(height))
    logger.info("width is {}".format(width))
    if width/height != 3/4:
        logger.info("Image is not appropriate!!! Height Width should be 4/3.")
        return True,"Image is not appropriate!!! Height Width should be 4/3."
    else:
        logger.info("image dimensions are appropriate")
        return True, "NA"


def test(image_name, model_dir, device_id):
    logger.info("inside the presentation attack detection")
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    print("image_name", image_name)
    img = crop_face(image_name)
    if img is None:
        img = image

    width = img.shape[1]
    height = img.shape[0]

    height_width_ratio = height / width
    logger.info('height_width_ratio = {}'.format(height_width_ratio))

    result, warning = check_image(img)

    if height_width_ratio < 0.7 or height_width_ratio > 1.4:
        warning = (
                warning
                + "Image is not appropriate!!! Height Width ratio should be 1.33 or close to that, your image's height_width_ratio is "
                + str(height_width_ratio)
                + ". This may also affect your result. "
        )
    else:
        warning = warning + ""

    # Check if the person is wearing a mask
    if warning == "":
        try:
            mask_model = load_model("mask_detector.h5")
            logger.info("Mask model is loaded successfully")
        except:
            logger.error("Error found while loading Mask model")

        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            test_image = cv2.resize(face_img, (150, 150))
            test_image = np.array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255
            result = mask_model.predict(test_image)

            if result[0][0] > 0.8:
                logger.info(
                    "Mask is detected with {} confidence score".format(result[0][0])
                )
                warning = (
                        warning
                        + "Person is wearing a mask in the image, please remove the mask and try again."
                )
            else:
                warning = warning + ""

    image_bbox = model_test.get_bbox(img)
    prediction = np.zeros((1, 3))
    test_speed = 0
    num_models = 0  # Keep track of the number of models for averaging

    # Iterate through models
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        # Crop the image
        cropped_img = image_cropper.crop(**param)

        # Check if the dimensions match the model's expected input dimensions
        if cropped_img.shape[0] == h_input and cropped_img.shape[1] == w_input:
            cv2.imwrite("cropped.jpg", cropped_img)

            # Make predictions using the current model
            start = time.time()
            model_prediction = model_test.predict(
                cropped_img, os.path.join(model_dir, model_name)
            )
            test_speed += time.time() - start

            # Check confidence threshold before summing up
            confidence_threshold = 0.5
            if np.max(model_prediction) >= confidence_threshold:
                prediction += model_prediction
                num_models += 1

    # Average the predictions
    if num_models > 0:
        prediction /= num_models

    # Draw the result of the final prediction
    label = np.argmax(prediction)
    value = (prediction[0][label] / 2) * 100

    # Log the prediction result
    if label == 1:
        logger.info(
            "Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value)
        )
        result_text = "{:.2f}".format(value)
        result_text_secondary = "{:.4}".format(value_2)
    else:
        logger.info(
            "Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value)
        )
        result_text = "{:.2f}".format(value)
        result_text_secondary = "{:.4f}".format(value_2)

    logger.info("Prediction cost {:.2f} s".format(test_speed))

    # Return the result
    return label, result_text, warning, label_2, result_text_secondary

# Example usage:
# label, value, warning = test("path/to/your/image.jpg", "path/to/your/model_directory", "your_device_id")
