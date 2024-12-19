import os
from backend.model_inference import classify_image

# Paths
model_path = "backend/model/trained_model.h5"  # Path to the trained model
test_folder = "test_images/"                  # Path to the test images folder

# Iterate through test folders
categories = ["cats", "dogs"]  # Class folders
for category in categories:
    folder_path = os.path.join(test_folder, category)
    print(f"\nTesting images in category: {category.upper()}")

    # Iterate through images in each folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        result = classify_image(model_path, image_path)
        print(f"{image_file} is classified as: {result}")
