from test_pad import test
import logging
import os
import time
from flask import Flask, request, jsonify
import random
import string
from werkzeug.utils import secure_filename
from face_crop import remove_and_add_white_background


FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s]: %(message)s'
# logging.basicConfig(format=FORMAT, level=logging.INFO, filename="logs/passiveliveliness.out")

app = Flask(__name__)
DATE_FORMAT = "%d-%m-%Y"
Temp_Dir = "image_folder"


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@app.route("/")
def welcome():
    logging.critical("Welcome msg dispatched to: {}".format(request.remote_addr))
    return "<h1>Hello passive_liveliness is running here..!!</h1>"


@app.route('/passive_liveliness', methods=['POST'])
def passive_liveliness():
    start = time.time()
    results = {}
    document = request.files['document']
    filename_complete_name = secure_filename(document.filename)
    filename_front_name = filename_complete_name.split(".")[0]
    idx = len(filename_complete_name.split('.')) - 1
    filename_ext = filename_complete_name.split(".")[idx]
    filename_new_name = filename_front_name + str(id_generator()) + "." + filename_ext
    document.save(os.path.join(Temp_Dir, filename_new_name))
    image_path = os.path.join(Temp_Dir, filename_new_name)

    result, confidence, warning, label_2, result_text_secondary = tuple(
        test(image_path, "./resources/anti_spoof_models", 0))

    result_json_pad_attack = {}

    print("CONFIiiiiiiiiiii", confidence)
    print("RESULT", result)

    if (result == 1):
        result_json_pad_attack = {'error_code': 0, 'response': {'Predicted_class': "RealFace",
                                                                'Real_Face_confidence': str(confidence) + "%"},
                                  'result': 'pass', 'warning': str(warning)}
    else:
        if float(confidence) < 90.0:
            print("INSUDE REMOVEEEEEEEEE=======")
            image_path_for_br_image = remove_and_add_white_background(image_path)

            result, confidence, warning, label_2, result_text_secondary = tuple(
                test(image_path_for_br_image, "./resources/anti_spoof_models", 0))
            # print("=====================================")
            # print("RESULT", result, "CONFIDENCE", confidence)
            if (result == 1):

                result_json_pad_attack = {'error_code': 0, 'response': {'Predicted_class': "RealFace",
                                                                        'Real_Face_confidence': str(confidence) + "%"},
                                          'result': 'pass', 'warning': str(warning)}
            else:
                result_json_pad_attack = {'error_code': -1, 'response': {'Predicted_class': "FakeFace",
                                                                         'Fake_Face_confidence': str(confidence) + "%"},
                                          'result': 'fail', 'warning': str(warning)}

        else:
            result_json_pad_attack = {'error_code': -1, 'response': {'Predicted_class': "FakeFace",
                                                                     'Fake_Face_confidence': str(confidence) + "%"},
                                      'result': 'fail', 'warning': str(warning)}

    if "mask" in warning:
        result_json_pad_attack = {'error_code': -4,
                                  'response': {'Predicted_class': "NA", 'Real_Face_confidence': "NA",
                                               'Fake_Face_confidence': "NA"},
                                  'result': 'fail', 'warning': warning}

    results['passive_liveliness_check'] = result_json_pad_attack
    logging.info("passive liveliness check is done")

    os.remove(os.path.join(image_path))
    end = time.time()
    response = {
        "error_code": 0,
        "response": results,
        "response_time": end - start
    }
    print("=============================================================")
    print(jsonify(response))
    print("=============================================================")
    return jsonify(response)
