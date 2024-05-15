import cv2
from rembg import remove
from PIL import Image

# from google.colab.patches import cv2_imshow


def remove_background(input_path, output_format='PNG'):
    output_path = 'image_folder/background_removed.jpeg'
    # Load the input image
    input_image = Image.open(input_path)

    # Remove the background
    output_image = remove(input_image)

    # Convert RGBA to RGB for JPEG (if needed)
    if output_format == 'JPEG':
        output_image = output_image.convert('RGB')

    # Save the output image
    output_image.save(output_path, format=output_format)
    return output_path


def crop_face(image_path):
    output_path = 'image_folder/cropped_face.jpg'

    # Load the pre-trained Haarcascades face classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    original_image = cv2.imread(image_path)

    # Check if the image reading is successful
    if original_image is None:
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Check if any faces are detected
    if len(faces) == 0:
        return None

    # Assume the first face is the desired one (you can modify this logic based on your needs)
    x, y, w, h = faces[0]

    # Crop the face from the original image
    cropped_face = original_image[y:y + h, x:x + w]

    # Save or display the cropped face
    cv2.imwrite(output_path, cropped_face)

    return cropped_face


def remove_and_add_white_background(input_path, output_format='PNG'):
    output_path2 = 'image_folder/cropped_face.jpg'
    # Load the input image
    input_image = Image.open(input_path)

    # Remove the background
    transparent_image = remove(input_image)

    # Create a new image with a white background
    white_background = Image.new("RGB", transparent_image.size, "white")

    # Paste the transparent image onto the white background
    white_background.paste(transparent_image, (0, 0), transparent_image)

    # Save the result
    white_background.save(output_path2, format=output_format)
    return output_path2

