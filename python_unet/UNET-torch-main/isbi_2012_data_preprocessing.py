# Preprocessing ISBI 2012 dataset from tiff to png
import tifffile as tiff
import skimage.io as io
import os
import numpy as np

# Define folder paths for preprocessed data
preprocessed_train_img_folder_path = os.path.join('isbi_2012', 'preprocessed', 'train_imgs')
preprocessed_train_label_folder_path = os.path.join('isbi_2012', 'preprocessed', 'train_labels')
preprocessed_test_img_folder_path = os.path.join('isbi_2012', 'preprocessed', 'test_imgs')

# Load the ISBI 2012 dataset
train_images = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'train-volume.tif'))
train_labels = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'train-labels.tif'))
test_images = tiff.imread(os.path.join('isbi_2012', 'raw_data', 'test-volume.tif'))

# Print dataset dimensions
print('Train images shape:', train_images.shape)
print('Train labels shape:', train_labels.shape)
print('Test images shape:', test_images.shape)

# Check if the preprocessing folder path exists; if not, create them
os.makedirs(preprocessed_train_img_folder_path, exist_ok=True)
os.makedirs(preprocessed_train_label_folder_path, exist_ok=True)
os.makedirs(preprocessed_test_img_folder_path, exist_ok=True)

# Ensure proper data format (in case it is loaded in a different type like float32)
# Convert images to uint8 format before saving
train_images = np.uint8(train_images)
train_labels = np.uint8(train_labels)
test_images = np.uint8(test_images)

# Iterate over the dataset and save images and labels as PNG
for image_index, (each_train_image, each_train_label, each_test_image) in enumerate(zip(train_images, train_labels, test_images)):
    # Save train image and label
    try:
        io.imsave(os.path.join(preprocessed_train_img_folder_path, f"{image_index}.png"), each_train_image)
        io.imsave(os.path.join(preprocessed_train_label_folder_path, f"{image_index}.png"), each_train_label)
        io.imsave(os.path.join(preprocessed_test_img_folder_path, f"{image_index}.png"), each_test_image)
    except Exception as e:
        print(f"Error saving images for index {image_index}: {e}")

print('ISBI 2012 Preprocessing finished!')
