import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SEGMENTED_DATA_PATH = os.getenv('SEGMENTED_DATA_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')

SEED = int(os.getenv('SEED'))
MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_INSTANCES'))

def create_dataset(max_amount=100):
    if not os.path.isdir(SEGMENTED_DATA_PATH):
        print('Segmented data folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Get all the filenames
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))

    # Clip the number of records to process
    if max_amount > len(files):
        max_amount = len(files)
    files = files[:max_amount]

    # Set the random seed
    np.random.seed(SEED)
    # Randomly shuffle the files now so we can split data later easier
    np.random.shuffle(files)

    data = []

    for filename in tqdm(files):
        ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
        sbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', filename), 'rb'))
        dbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', filename), 'rb'))

        for i in range(len(ppg)):
            data.append(np.append(ppg[i], [sbp[i], dbp[i]]))

    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'w')
    dset = f.create_dataset('data', data=data)

if __name__ == '__main__':
    create_dataset(max_amount=MAX_DATASET_SIZE)