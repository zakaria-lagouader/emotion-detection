import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

# Specify the path to the directory containing your dataset folders
dataset_dir = "datasets/CK"

# Specify the path to create train, validation, and test directories
train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"


# Create train, validation, and test directories if they don't exist
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Loop through each emotion folder
emotions = os.listdir(dataset_dir)
for emotion in emotions:
    emotion_path = os.path.join(dataset_dir, emotion)

    # Get a list of all image files in the emotion folder
    all_images = [f for f in os.listdir(emotion_path) if f.endswith('.png')]

    # Randomly split the dataset into training (64%), validation (20%), and test (16%) sets
    train_images, test_images = train_test_split(all_images, test_size=0.16, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

    # Move images to the appropriate directories
    def move_images(image_list, src_dir, dest_dir):
        for image in image_list:
            src_path = os.path.join(src_dir, emotion, image)
            dest_path = os.path.join(dest_dir, emotion, image)
            
            # Create emotion subdirectories if they don't exist
            if not os.path.exists(os.path.join(dest_dir, emotion)):
                os.makedirs(os.path.join(dest_dir, emotion))
            
            copyfile(src_path, dest_path)

    move_images(train_images, dataset_dir, train_dir)
    move_images(val_images, dataset_dir, val_dir)
    move_images(test_images, dataset_dir, test_dir)