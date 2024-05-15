import os
import cv2
import numpy as np
import argparse
import warnings
import time
import gc
import logging

from face_crop import crop_face
from mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

SAMPLE_IMAGE_PATH = "./images/sample/"

IMAGE_FOLDER = "image_folder"


def check_image(image):
    height, width, channel = image.shape
    logger.info("height is {}".format(height))
    logger.info("width is {}".format(width))
    if width / height != 3 / 4:
        logger.info("Image is not appropriate!!! Height Width should be 4/3.")
        return True, "Image is not appropriate!!! Height Width should be 4/3."
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
    # img = crop_head(image_name)
    if img is None:
        img = image
    # image_cropper = CropImage()
    # if "." in image_name:
    #     if "png" in image_name.split(".")[1]:
    #         im = Image.open(image_name)
    #         rgb_im = im.convert('RGB')
    #         file_mask_name = "test_mask" + str(support.id_generator()) + ".jpg"
    #         save_path = os.path.join(IMAGE_FOLDER, file_mask_name)
    #         rgb_im.save(save_path)
    #         image = cv2.imread(save_path)
    #         image_name = save_path

    width = (image.shape)[1]
    height = (image.shape)[0]

    height_width_ratio = (height / width)

    logger.info('height_width_ratio = {}'.format(height_width_ratio))

    result, warning = check_image(image)

    if (height_width_ratio < 0.7) or (height_width_ratio > 1.4):
        warning = warning + ("Image is not appropriate!!! Height Width ratio should be 1.33 or close to that, "
                             "your image's height_width_ratio is")
        warning = warning + str(height_width_ratio)
        warning = warning + ". This may also affect your result. "
    else:
        warning = warning + ""

    # ------Check if the person is wearing mask or not-----------------------------------------------

    if warning == "":

        try:
            mask_model = load_model("mask_detector.h5")
            logger.info("Mask model is loaded successfully")
        except:
            logger.error("Error found while loading Mask model")

        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            test_image = cv2.resize(face_img, (150, 150))
            test_image = np.array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255
            result = mask_model.predict(test_image)

            if result[0][0] > 0.8:
                logger.info("Mask is detected with {} confidence score".format(result[0][0]))
                warning = warning + "Person is wearing a mask in the image, please remove the mask and try again."
            else:
                warning = warning + ""

    #        del mask_model
    # --------------------------------------------------------------------------------
    image_bbox = model_test.get_bbox(img)
    # cv2.imwrite("/image_folder", image_bbox)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            # "org_img": image,
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)

        logger.info("w_input = {}, h_input = {}".format(w_input, h_input))

        #    cv2.imshow("Test Image", img)
        #    cv2.waitKey(0)

        cv2.imwrite("cropped.jpg", img)

        # print("Image shape: ", img.shape)
        start = time.time()
        # print("Prediction before - ", prediction)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        # print("Prediction after - ", prediction)
        test_speed += time.time() - start

    # draw result of prediction
    logger.info("prediction is {}".format(prediction))
    label = np.argmax(prediction)
    value = (prediction[0][label] / 2) * 100
    # print("label: {}, value: {}".format(label, value))

    # print("NP MAX", np.max(prediction, axis=0))

    label_2 = np.argsort(np.max(prediction, axis=0))[-3]
    value_2 = (prediction[0][label_2] / 2) * 100
    # print("label2: {}, value2: {}".format(label_2, value_2))
    logger.info("value is {}".format(value))
    logger.info("label is {}".format(label))
    if label == 1:
        label_2 = 0  # "secondary class -> Passive Liveness Failed"
        label = 1  # "Passive Liveness Passed"
        logger.info("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "{:.2f}".format(value)
        result_text_secondary = "{:.4}".format(value_2)
        color = (255, 0, 0)
    else:
        label = 0  # "Passive Liveness Failed"
        label_2 = 1  # "secondary class -> Passive Liveness Passed"
        logger.info("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "{:.2f}".format(value)
        result_text_secondary = "{:.4f}".format(value_2)
        color = (0, 0, 255)
    logger.info("Prediction cost {:.2f} s".format(test_speed))

    logger.info("warning")

    out2_list = [label, result_text, warning, label_2, result_text_secondary]
    gc.collect()

    print("Out2_list =====> ", out2_list)

    return out2_list
