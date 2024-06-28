from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from enum import Enum
import h5py

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')

SEED = int(os.getenv('SEED'))


class PATIENT(Enum):
    HEALTHY = 0
    UNHEALTHY = 1

def create_dataset(raw_data_path = RAW_DATA_PATH, dataset_type = "Train"):
    if not os.path.isdir(RAW_DATA_PATH):
        print('Raw data folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Get all the filenames
    healthy_path = os.path.join(RAW_DATA_PATH, dataset_type, str(PATIENT.HEALTHY.value))
    healthy_files = os.listdir(healthy_path)
    unhealthy_path = os.path.join(RAW_DATA_PATH, dataset_type, str(PATIENT.UNHEALTHY.value))
    unhealthy_files = os.listdir(unhealthy_path)

    dataset = []

    for filename in healthy_files:
        img = Image.open(os.path.join(healthy_path, filename))
        img = img.convert("L")
        img = ImageOps.invert(img)
        signal = np.array(flatten(img), dtype=np.int32)
        dataset.append(np.append(signal, int(PATIENT.HEALTHY.value)))

    for filename in unhealthy_files:
        img = Image.open(os.path.join(unhealthy_path, filename))
        img = img.convert("L")
        img = ImageOps.invert(img)
        signal = np.array(flatten(img), dtype=np.int32)
        dataset.append(np.append(signal, int(PATIENT.UNHEALTHY.value)))

    # Set the random seed
    np.random.seed(SEED)
    # Randomly shuffle the dataset
    np.random.shuffle(dataset)

    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{dataset_type}.hdf5'), 'w')
    dset = f.create_dataset('data', data=dataset)

def invert_image(image):
    return ImageOps.invert(image.convert("L"))
def flatten(inverted_img):
    img_array = np.array(inverted_img)
    height, width = img_array.shape


    brightest_pixel_heights = []
    last_non_zero_left = None

    for col in range(width):
        column_data = img_array[:, col]

        # Check if there are any non-black pixels in the column
        if np.any(column_data > 0):
            brightest_pixel_height = np.argmax(column_data)

            adjusted_height = height - brightest_pixel_height - 1
            brightest_pixel_heights.append(adjusted_height)

            last_non_zero_left = adjusted_height
        else:
            brightest_pixel_heights.append(last_non_zero_left)

    last_non_zero_right = None
    # Fill in the remaining zeros
    for col in range(width - 1, -1, -1):
        if brightest_pixel_heights[col] is None:
            brightest_pixel_heights[col] = last_non_zero_right
        else:
            last_non_zero_right = brightest_pixel_heights[col]

    return brightest_pixel_heights


def plot_pixel_heights(pixel_heights, title):
    # Create a plot of the pixel heights
    plt.figure(figsize=(10, 5))
    plt.plot(pixel_heights, label='Brightest Pixel Height')
    plt.xlabel('Column Index')
    plt.ylabel('Pixel Height')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    create_dataset(dataset_type="Train")
    create_dataset(dataset_type="Test")
    # plot_pixel_heights(flatten(Image.open(os.path.join(RAW_DATA_PATH, 'Train', '0', 'healthy_1.png'))), 'Healthy Patient')
    # plot_pixel_heights(flatten(Image.open(os.path.join(RAW_DATA_PATH, 'Train', '1', 'unhealthy_1.png'))), 'Unhealthy Patient')

