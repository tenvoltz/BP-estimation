import os
from tqdm import tqdm
import h5py
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))
DATASET_PATH = os.getenv('DATASET_PATH')
FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')

TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))

MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))

OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'

def fold_data():
    if not os.path.isfile(os.path.join(DATASET_PATH, 'dataset.hdf5')):
        print('Dataset file not found')
        return
    if not os.path.isdir(FOLDED_DATASET_PATH):
        os.mkdir(FOLDED_DATASET_PATH)

    training_amount = 0
    segment_amount = 0

    for fold_id in tqdm(range(FOLD_AMOUNT), desc='Folding Data'):
        # Load the dataset
        f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'r')
        segment_amount = len(f['data'])

        # Control the amount of data in each fold
        training_amount = int(TRAIN_TEST_SPLIT * segment_amount)
        validation_starts = {}
        for i in range(FOLD_AMOUNT):
            validation_starts[i] = i * int(training_amount / FOLD_AMOUNT)
        validation_starts[FOLD_AMOUNT] = training_amount

        data_train = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'targets': {
            'sbp': [], 'dbp': []
        }}
        data_val = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'targets': {
            'sbp': [], 'dbp': []
        }}

        validation_start = validation_starts[fold_id]
        validation_end = validation_starts[fold_id + 1]

        max_ppg, min_ppg = -10000, 10000
        max_vpg, min_vpg = -10000, 10000
        max_apg, min_apg = -10000, 10000

        max_sbp, min_sbp = -10000, 10000
        max_dbp, min_dbp = -10000, 10000

        for i in tqdm(range(0, validation_start), desc=f'Fold {fold_id} - Training Data (left)'):
            ppg = f['data'][i][0:SIGNAL_LENGTH]
            vpg = f['data'][i][SIGNAL_LENGTH:2*SIGNAL_LENGTH]
            apg = f['data'][i][2*SIGNAL_LENGTH:3*SIGNAL_LENGTH]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            data_train['signals']['ppg'].append(ppg.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpg.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apg.reshape(1, SIGNAL_LENGTH))
            data_train['targets']['sbp'].append(sbp)
            data_train['targets']['dbp'].append(dbp)

            max_ppg, min_ppg = max(max_ppg, max(ppg)), min(min_ppg, min(ppg))
            max_vpg, min_vpg = max(max_vpg, max(vpg)), min(min_vpg, min(vpg))
            max_apg, min_apg = max(max_apg, max(apg)), min(min_apg, min(apg))
            max_sbp, min_sbp = max(max_sbp, sbp), min(min_sbp, sbp)
            max_dbp, min_dbp = max(max_dbp, dbp), min(min_dbp, dbp)

        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            ppg = f['data'][i][0:SIGNAL_LENGTH]
            vpg = f['data'][i][SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
            apg = f['data'][i][2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            data_val['signals']['ppg'].append(ppg.reshape(1, SIGNAL_LENGTH))
            data_val['signals']['vpg'].append(vpg.reshape(1, SIGNAL_LENGTH))
            data_val['signals']['apg'].append(apg.reshape(1, SIGNAL_LENGTH))
            data_val['targets']['sbp'].append(sbp)
            data_val['targets']['dbp'].append(dbp)

        for i in tqdm(range(validation_end, training_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            ppg = f['data'][i][0:SIGNAL_LENGTH]
            vpg = f['data'][i][SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
            apg = f['data'][i][2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
            sbp = f['data'][i][-2]
            dbp = f['data'][i][-1]

            data_train['signals']['ppg'].append(ppg.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpg.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apg.reshape(1, SIGNAL_LENGTH))
            data_train['targets']['sbp'].append(sbp)
            data_train['targets']['dbp'].append(dbp)

            max_ppg, min_ppg = max(max_ppg, max(ppg)), min(min_ppg, min(ppg))
            max_vpg, min_vpg = max(max_vpg, max(vpg)), min(min_vpg, min(vpg))
            max_apg, min_apg = max(max_apg, max(apg)), min(min_apg, min(apg))
            max_sbp, min_sbp = max(max_sbp, sbp), min(min_sbp, sbp)
            max_dbp, min_dbp = max(max_dbp, dbp), min(min_dbp, dbp)

        f.close()
        del f

        # Convert to numpy arrays
        for key in data_train['signals'].keys():
            data_train['signals'][key] = np.array(data_train['signals'][key])
        for key in data_train['targets'].keys():
            data_train['targets'][key] = np.array(data_train['targets'][key])

        # Min-max Normalize PPG, VPG, and APG
        data_train['signals']['ppg'] = (data_train['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        data_train['signals']['vpg'] = (data_train['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        data_train['signals']['apg'] = (data_train['signals']['apg'] - min_apg) / (max_apg - min_apg)

        # Z-score Normalize PPG, VPG, and APG
        mean_ppg = np.mean(data_train['signals']['ppg'])
        std_ppg = np.std(data_train['signals']['ppg'])
        mean_vpg = np.mean(data_train['signals']['vpg'])
        std_vpg = np.std(data_train['signals']['vpg'])
        mean_apg = np.mean(data_train['signals']['apg'])
        std_apg = np.std(data_train['signals']['apg'])

        #data_train['signals']['ppg'] = (data_train['signals']['ppg'] - mean_ppg) / std_ppg
        #data_train['signals']['vpg'] = (data_train['signals']['vpg'] - mean_vpg) / std_vpg
        #data_train['signals']['apg'] = (data_train['signals']['apg'] - mean_apg) / std_apg

        # Get the mean of target
        mean_sbp = np.mean(data_train['targets']['sbp'])
        mean_dbp = np.mean(data_train['targets']['dbp'])

        if OUTPUT_NORMALIZED:
            data_train['targets']['sbp'] = (data_train['targets']['sbp'] - min_sbp) / (max_sbp - min_sbp)
            data_train['targets']['dbp'] = (data_train['targets']['dbp'] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump(data_train, open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'wb'))

        del data_train

        for key in data_val['signals'].keys():
            data_val['signals'][key] = np.array(data_val['signals'][key])
        for key in data_val['targets'].keys():
            data_val['targets'][key] = np.array(data_val['targets'][key])

        # Min-max Normalize PPG, VPG, and APG
        data_val['signals']['ppg'] = (data_val['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        data_val['signals']['vpg'] = (data_val['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        data_val['signals']['apg'] = (data_val['signals']['apg'] - min_apg) / (max_apg - min_apg)

        # Z-score Normalize PPG, VPG, and APG
        #data_val['signals']['ppg'] = (data_val['signals']['ppg'] - mean_ppg) / std_ppg
        #data_val['signals']['vpg'] = (data_val['signals']['vpg'] - mean_vpg) / std_vpg
        #data_val['signals']['apg'] = (data_val['signals']['apg'] - mean_apg) / std_apg

        if OUTPUT_NORMALIZED:
            data_val['targets']['sbp'] = (data_val['targets']['sbp'] - min_sbp) / (max_sbp - min_sbp)
            data_val['targets']['dbp'] = (data_val['targets']['dbp'] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump(data_val, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        pickle.dump(data_val, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        del data_val

        pickle.dump({
            'min_ppg': min_ppg, 'max_ppg': max_ppg,
            'min_vpg': min_vpg, 'max_vpg': max_vpg,
            'min_apg': min_apg, 'max_apg': max_apg,
            'mean_sbp': mean_sbp, 'mean_dbp': mean_dbp,
            'min_sbp': min_sbp, 'max_sbp': max_sbp,
            'min_dbp': min_dbp, 'max_dbp': max_dbp,
            'mean_ppg': mean_ppg, 'std_ppg': std_ppg,
            'mean_vpg': mean_vpg, 'std_vpg': std_vpg,
            'mean_apg': mean_apg, 'std_apg': std_apg
        }, open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'wb'))

    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'r')

    data_test = {'signals': {
        'ppg': [], 'vpg': [], 'apg': []
    }, 'targets': {
        'sbp': [], 'dbp': []
    }}

    for i in tqdm(range(training_amount, segment_amount), desc='Testing Data'):
        ppg = f['data'][i][0:SIGNAL_LENGTH]
        vpg = f['data'][i][SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
        apg = f['data'][i][2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
        sbp = f['data'][i][-2]
        dbp = f['data'][i][-1]

        data_test['signals']['ppg'].append(ppg.reshape(1, SIGNAL_LENGTH))
        data_test['signals']['vpg'].append(vpg.reshape(1, SIGNAL_LENGTH))
        data_test['signals']['apg'].append(apg.reshape(1, SIGNAL_LENGTH))
        data_test['targets']['sbp'].append(sbp)
        data_test['targets']['dbp'].append(dbp)

    del f

    pickle.dump(data_test, open(os.path.join(FOLDED_DATASET_PATH, 'test.pkl'), 'wb'))

    del data_test

    print(f'\nThis dataset contains {segment_amount} segments')

if __name__ == '__main__':
    fold_data()