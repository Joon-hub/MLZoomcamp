import os
import numpy as np

print('Running')
# defining variables
data = []
labels = []
classes = 10
cur_path = os.getcwd()
print(cur_path)

# Retrieving the images and their labels 
print(f"Contents of Data folder: {os.listdir('Data')}")
train_dir = os.path.join('Data','raw','Train')
print(f"Looking for directory: {train_dir}")

# Check if the directory exists
if not os.path.exists(train_dir):
    print(f"Error: Directory {train_dir} does not exist.")
else:
    # Listing all subdirectories 
    all_classes = os.listdir(train_dir)
    print(f"All items in train_dir: {all_classes}")

    # Filter out non-numeric directory names and sort
    numeric_classes = [x for x in all_classes if x.isdigit()]
    sorted_classes = sorted(numeric_classes, key=int)

    # Selecting the first 10 classes
    selected_classes = sorted_classes[:10]

    print(f"Selected classes: {selected_classes}")

# Numebr of images per class
total_images = 0
for class_name in selected_classes:
    class_dir = os.path.join(train_dir, class_name)
    num_images = len(os.listdir(class_dir))
    total_images += num_images

# percentage distribution per class
for class_name in selected_classes:
    class_dir = os.path.join(train_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class: {class_name}, Number of Images: {num_images} , Percentage distribution:{np.round(num_images/total_images*100,2)}%")
print(f"Total Images: {total_images}")
