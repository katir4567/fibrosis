import os
import shutil
import random
import matplotlib.pyplot as plt

# Folder paths
main_folder = "fibrosis/Fibrosis"    # Main folder containing all images
train_folder = "fibrosis/Train"
validation_folder = "fibrosis/Validation"
test_folder = "fibrosis/Test"

# Define split ratios
train_ratio = 0.68         # 80% of the data will go to training
validation_ratio = 0.12   # 10% will go to validation
test_ratio = 0.2          # 20% will go to testing

# Get all images from the main folder
all_images = os.listdir(main_folder)
random.shuffle(all_images)  # Shuffle to randomize the order

# Calculate the split sizes
train_size = int(train_ratio * len(all_images))
validation_size = int(validation_ratio * len(all_images))

# Split the data
train_images = all_images[:train_size]
validation_images = all_images[train_size:train_size + validation_size]
test_images = all_images[train_size + validation_size:]

# Function to copy images to the specified folder
def copy_images(image_list, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for image in image_list:
        src = os.path.join(main_folder, image)
        dest = os.path.join(destination_folder, image)
        shutil.copy(src, dest)

# Copy images into Train, Validation, and Test folders
copy_images(train_images, train_folder)
copy_images(validation_images, validation_folder)
copy_images(test_images, test_folder)

# Visualize the number of images in each set
def visualize_data_distribution():
    set_folders = [train_folder, validation_folder, test_folder]
    set_names = ["Train", "Validation", "Test"]
    counts = [len(os.listdir(folder)) for folder in set_folders]

    # Plot the results
    plt.bar(set_names, counts, color=['blue', 'orange', 'green'])
    plt.xlabel("Dataset Type")
    plt.ylabel("Number of Images")
    plt.title("Number of Images in Train, Validation, and Test Sets")
    plt.show()

# Run the visualization
visualize_data_distribution()
