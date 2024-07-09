import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

SEGMENTED_DATA_PATH = os.getenv('SEGMENTED_DATA_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')

SEED = int(os.getenv('SEED'))
MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))

def create_dataset(max_amount=-1):
    if not os.path.isdir(SEGMENTED_DATA_PATH):
        print('Segmented data folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Get all the filenames
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))

    # Clip the number of records to process
    if max_amount > len(files) or max_amount == -1:
        max_amount = len(files)
    files = files[:max_amount]

    # Set the random seed
    np.random.seed(SEED)
    # Randomly shuffle the files now so we can split data later easier
    np.random.shuffle(files)

    data = []

    empty_files = 0
    for filename in tqdm(files):
        ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
        if len(ppg) == 0:
            empty_files += 1
            print(f"Empty file: {filename}  Skipping...)")
            continue
        vpg = np.gradient(ppg, axis=1)
        apg = np.gradient(vpg, axis=1)
        sbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', filename), 'rb'))
        dbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', filename), 'rb'))

        for i in range(len(ppg)):
            data.append(np.concatenate([ppg[i], vpg[i], apg[i], [sbp[i], dbp[i]]]))

    print(f"Empty files: {empty_files}")
    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'w')
    dset = f.create_dataset('data', data=data)

if __name__ == '__main__':
    create_dataset(max_amount=MAX_DATASET_SIZE)