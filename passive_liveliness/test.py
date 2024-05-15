import os
from main import passive_liveliness_copy
#
#
#
# Directory containing images
directory = 'SELF/'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # Filter only image files
        image_path = os.path.join(directory, filename)

        print("===================================================================")
        print("FILENAME_", filename)
        result = passive_liveliness_copy(image_path)
        print("RESPONSE ---->", result)
        print("===================================================================")

    else:
        print("Skipping non-image file:", filename)


