import os
import base64
import random
import string
import logging
import subprocess
from PIL import Image
from pathlib import Path
from io import BytesIO

BASE_DIR = os.curdir
test_storage = os.path.join(BASE_DIR, 'test')
train_storage = os.path.join(BASE_DIR, 'data/dataset')

logger = logging.getLogger(__name__)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    this will generate random name
    :param size:
    :param chars:
    :return:
    """
    return ''.join(random.choice(chars) for _ in range(size))


def base64_to_image(user_id, base64_string, image_for='test', extension='.png'):
    """
    this will convert the base64 string to image
    :param extension: extension of image
    :param user_id: user_id to enroll
    :param base64_string: base64 encoded image string
    :param image_for: train/test
    :return: filename of the writtened file
    """
    logger.info('extension received: "{}"'.format(extension))
    if not extension.startswith('.'):
            extension = '.' + extension

    if image_for == 'train':
        user_dir = Path(train_storage)/user_id
        if not user_dir.exists():
            user_dir.mkdir()

        for count, image in enumerate(base64_string.split(',')):
            filename = os.path.join(str(user_dir), str(count) + '.png')

            try:
                # img_data = base64.b64decode(image)
                # with open(filename, 'wb') as f:
                #     f.write(img_data)
                im = Image.open(BytesIO(base64.b64decode(image)))
                logger.info("loaded base 64 images")
                dimension_tup = im.size
                logger.info("got dimension: {}".format(dimension_tup))
                if dimension_tup[0] > 1000 or dimension_tup[1] > 1000:
                    im = im.resize((int(dimension_tup[0]/4), int(dimension_tup[1]/4)), Image.ANTIALIAS)
                    logger.info("dimension reduced to 1/4")

                im.save(filename, 'png', quality=85)
                logger.info("image saved with reduced quality")

            except:
                logger.error("Bad base64 string provided")
                return None
        logger.info("train data saved: '{}'".format(str(user_dir)))
        return user_dir

    else:
        if user_id:
            filename = os.path.join(test_storage, user_id + '.png')
        else:
            filename = os.path.join(test_storage, id_generator() + '.png')

        try:
            # img_data = base64.b64decode(base64_string)
            im = Image.open(BytesIO(base64.b64decode(base64_string)))
            logger.info("loaded base 64 images")
            dimension_tup = im.size
            logger.info("got dimension: {}".format(dimension_tup))

            if dimension_tup[0] > 1000 or dimension_tup[1] > 1000:
                im = im.resize((int(dimension_tup[0]/4),int(dimension_tup[1]/4)),Image.ANTIALIAS)
                logger.info("dimension reduced to 1/4")

            im.save(filename, 'png', quality=85)
            logger.info("image saved with reduced quality")
            # with open(filename, 'wb') as f:
            #     print(img_data)
            #     f.write(img_data)
        except:
            logger.exception("Bad base64 string provided")

        logger.info("test data saved: '{}'".format(str(filename)))
        return filename


def validate_this_document(file_name):
    """
    this will validate the document
    :param file_name:
    :return:
    """
    logger.info("[BLOCK:VALIDATE_DOCUMENT] validating document: '{}'".format(file_name))
    try:
        try:
            dpi = Image.open(file_name).info['dpi']
            logger.info('[BLOCK:VALIDATE_DOCUMENT] Image DPI: [{}]'.format(dpi))

        except:
            test = subprocess.Popen(["identify", "-format", "%x,%y", file_name], stdout=subprocess.PIPE, encoding='utf8')
            dpi = tuple(map(int, test.communicate()[0].split(',')))
            logger.info('[BLOCK:VALIDATE_DOCUMENT] Image DPI: [{}]'.format(dpi))

        test = subprocess.Popen(["identify", "-format", "%w,%h", file_name], stdout=subprocess.PIPE, encoding='utf8')
        resolution = tuple(map(int, test.communicate()[0].split(',')))
        logger.info('[BLOCK:VALIDATE_DOCUMENT] Image Resolution: [{}]'.format(resolution))

    except Exception as vEx:
        logger.exception('[BLOCK:VALIDATE_DOCUMENT] exception occurred while validating image: \n{}'.format(vEx))

