import os
from tqdm import tqdm
import h5py
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

DATASET_PATH = os.getenv('DATASET_PATH')
FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')

FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))
OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'
SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))

def fold_data():
    dataset_path = os.path.join(DATASET_PATH, f'dataset_Train.hdf5')
    if not os.path.isfile(dataset_path):
        print('Dataset file not found')
        return
    if not os.path.isdir(FOLDED_DATASET_PATH):
        os.mkdir(FOLDED_DATASET_PATH)

    training_amount = 0
    segment_amount = 0

    for fold_id in tqdm(range(FOLD_AMOUNT), desc='Folding Data'):
        # Load the dataset
        f = h5py.File(dataset_path, 'r')
        segment_amount = len(f['data'])

        validation_starts = {}
        for i in range(FOLD_AMOUNT):
            validation_starts[i] = i * int(segment_amount / FOLD_AMOUNT)
        validation_starts[FOLD_AMOUNT] = segment_amount

        # Split the data
        x_train = []
        x_val = []
        y_train = []
        y_val = []

        validation_start = validation_starts[fold_id]
        validation_end = validation_starts[fold_id + 1]

        if FOLD_AMOUNT == 1:
            validation_start = 0
            validation_end = 0

        max_ppg, min_ppg = -10000, 10000

        for i in tqdm(range(0, validation_start), desc=f'Fold {fold_id} - Training Data (left)'):
            x_train.append(f['data'][i][:-1].reshape(1, SIGNAL_LENGTH))
            y_train.append(f['data'][i][-1])

            max_ppg, min_ppg = max(max_ppg, max(f['data'][i][:-1])), min(min_ppg, min(f['data'][i][:-1]))

        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            x_val.append(f['data'][i][:-1].reshape(1, SIGNAL_LENGTH))
            y_val.append(f['data'][i][-1])

        for i in tqdm(range(validation_end, segment_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            x_train.append(f['data'][i][:-1].reshape(1, SIGNAL_LENGTH))
            y_train.append(f['data'][i][-1])

            max_ppg, min_ppg = max(max_ppg, max(f['data'][i][:-1])), min(min_ppg, min(f['data'][i][:-1]))

        f.close()
        del f

        # Normalize ppg
        x_train= np.array(x_train)
        x_train = (x_train - min_ppg) / (max_ppg - min_ppg)

        pickle.dump({'x': x_train, 'y': y_train}, open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'wb'))

        del x_train, y_train

        # Normalize ppg
        x_val = np.array(x_val)
        x_val = (x_val - min_ppg) / (max_ppg - min_ppg)

        pickle.dump({'x': x_val, 'y': y_val}, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        del x_val, y_val

        pickle.dump({
            'min_ppg': min_ppg, 'max_ppg': max_ppg,
        }, open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'wb'))

    print(f'Training contain {segment_amount} segments')

    dataset_path = os.path.join(DATASET_PATH, f'dataset_Test.hdf5')
    f = h5py.File(dataset_path, 'r')
    test_amount = len(f['data'])

    x_test = []
    y_test = []
    for i in tqdm(range(test_amount), desc='Testing Data'):
        x_test.append(f['data'][i][:-1].reshape(1, SIGNAL_LENGTH))
        y_test.append(f['data'][i][-1])

    pickle.dump({'x': x_test, 'y': y_test}, open(os.path.join(FOLDED_DATASET_PATH, f'test.pkl'), 'wb'))

    print(f'Testing contain {test_amount} segments')

if __name__ == '__main__':
    fold_data()