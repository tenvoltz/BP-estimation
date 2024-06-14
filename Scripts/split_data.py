import os
from tqdm import tqdm
import h5py
import numpy as np
import pickle

SIGNAL_LENGTH = 1024
DATASET_PATH = '../Data/raw_dataset/dataset.hdf5'
FOLDED_DATASET_PATH = '../Data/split_dataset'

TRAIN_TEST_SPLIT = 0.8
FOLD_AMOUNT = 5

def fold_data():
    if not os.path.isfile(DATASET_PATH):
        print('Dataset file not found')
        return
    if not os.path.isdir(FOLDED_DATASET_PATH):
        os.mkdir(FOLDED_DATASET_PATH)

    training_amount = 0
    testing_amount = 0
    validation_start = {}
    segment_amount = 0

    for fold_id in tqdm(range(FOLD_AMOUNT), desc='Folding Data'):
        # Load the dataset
        f = h5py.File(DATASET_PATH, 'r')
        segment_amount = len(f['data'])

        # Control the amount of data in each fold
        training_amount = int(TRAIN_TEST_SPLIT * segment_amount)
        testing_amount = segment_amount - training_amount
        validation_starts = {}
        for i in range(FOLD_AMOUNT):
            validation_starts[i] = i * int(training_amount / FOLD_AMOUNT)
        validation_starts[FOLD_AMOUNT] = training_amount

        # Split the data
        x_train = []
        x_val = []
        y_train = []
        y_val = []

        validation_start = validation_starts[fold_id]
        validation_end = validation_starts[fold_id + 1]

        max_ppg = -10000
        min_ppg = 10000
        '''Add max/min sbp/dbp later'''

        for i in tqdm(range(0, validation_start), desc=f'Fold {fold_id} - Training Data (left)'):
            ppg = f['data'][i][:-2]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            x_train.append(ppg.reshape(1, SIGNAL_LENGTH))
            y_train.append([sbp, dbp])

            max_ppg = max(max_ppg, max(ppg))
            min_ppg = min(min_ppg, min(ppg))

        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            ppg = f['data'][i][:-2]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            x_val.append(ppg.reshape(1, SIGNAL_LENGTH))
            y_val.append([sbp, dbp])

            max_ppg = max(max_ppg, max(ppg))
            min_ppg = min(min_ppg, min(ppg))

        for i in tqdm(range(validation_end, training_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            ppg = f['data'][i][:-2]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            x_train.append(ppg.reshape(1, SIGNAL_LENGTH))
            y_train.append([sbp, dbp])

            max_ppg = max(max_ppg, max(ppg))
            min_ppg = min(min_ppg, min(ppg))

        del f

        x_train = np.array(x_train)
        x_train -= min_ppg
        x_train /= (max_ppg - min_ppg)

        pickle.dump({'x': x_train, 'y': y_train}, open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'wb'))

        del x_train, y_train

        x_val = np.array(x_val)
        x_val -= min_ppg
        x_val /= (max_ppg - min_ppg)

        pickle.dump({'x': x_val, 'y': y_val}, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        del x_val, y_val

        pickle.dump({'min_ppg': min_ppg, 'max_ppg': max_ppg}, open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'wb'))

    f = h5py.File(DATASET_PATH, 'r')

    x_test = []
    y_test = []

    max_ppg = -10000
    min_ppg = 10000
    '''Add max/min sbp/dbp later'''

    for i in tqdm(range(training_amount, segment_amount), desc='Testing Data'):
        ppg = f['data'][i][:-2]
        sbp = f['data'][i][-2]
        dbp = f['data'][i][-1]

        x_test.append(ppg.reshape(1, SIGNAL_LENGTH))
        y_test.append([sbp, dbp])

        max_ppg = max(max_ppg, max(ppg))
        min_ppg = min(min_ppg, min(ppg))

    del f

    x_test = np.array(x_test)
    x_test -= min_ppg
    x_test /= (max_ppg - min_ppg)

    pickle.dump({'x': x_test, 'y': y_test}, open(os.path.join(FOLDED_DATASET_PATH, 'test.pkl'), 'wb'))

    del x_test, y_test

    print(f'\nThis dataset contains {segment_amount} segments')

if __name__ == '__main__':
    fold_data()