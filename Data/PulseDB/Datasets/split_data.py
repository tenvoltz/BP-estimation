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

MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))
PATIENT_SIGNAL_RATIO = [float(i) for i in os.getenv('PATIENT_SIGNAL_RATIO').split(',')]
CURRENT_RATIO = float(os.getenv('CURRENT_RATIO'))
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))

OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'

def fold_data():
    dataset_path = os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}_Train_{CURRENT_RATIO}.hdf5')
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

        ppgs = f['signals']['ppg'][:MAX_DATASET_SIZE]
        vpgs = f['signals']['vpg'][:MAX_DATASET_SIZE]
        apgs = f['signals']['apg'][:MAX_DATASET_SIZE]

        ages = f['demographics']['age'][:MAX_DATASET_SIZE]
        genders = f['demographics']['gender'][:MAX_DATASET_SIZE]
        heights = f['demographics']['height'][:MAX_DATASET_SIZE]
        weights = f['demographics']['weight'][:MAX_DATASET_SIZE]
        bmis = f['demographics']['bmi'][:MAX_DATASET_SIZE]

        sbps = f['targets']['sbp'][:MAX_DATASET_SIZE]
        dbps = f['targets']['dbp'][:MAX_DATASET_SIZE]

        segment_amount = len(ppgs)

        validation_starts = {}
        for i in range(FOLD_AMOUNT):
            validation_starts[i] = i * int(segment_amount / FOLD_AMOUNT)
        validation_starts[FOLD_AMOUNT] = segment_amount

        # Split the data
        x_train = {'ppg': [], 'vpg': [], 'apg': [], 'demographics': []}
        x_val = {'ppg': [], 'vpg': [], 'apg': [], 'demographics': []}
        y_train = []
        y_val = []

        validation_start = validation_starts[fold_id]
        validation_end = validation_starts[fold_id + 1]

        max_ppg, min_ppg = -10000, 10000
        max_vpg, min_vpg = -10000, 10000
        max_apg, min_apg = -10000, 10000

        max_sbp, min_sbp = -10000, 10000
        max_dbp, min_dbp = -10000, 10000

        for i in tqdm(range(0, validation_start), desc=f'Fold {fold_id} - Training Data (left)'):
            x_train['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['demographics'].append(np.array([ages[i], genders[i], heights[i], weights[i], bmis[i]]))
            y_train.append([sbps[i], dbps[i]])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            x_val['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            x_val['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            x_val['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            x_val['demographics'].append(np.array([ages[i], genders[i], heights[i], weights[i], bmis[i]]))
            y_val.append([sbps[i], dbps[i]])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        for i in tqdm(range(validation_end, segment_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            x_train['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            x_train['demographics'].append(np.array([ages[i], genders[i], heights[i], weights[i], bmis[i]]))
            y_train.append([sbps[i], dbps[i]])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        f.close()
        del f

        x_train['ppg'] = np.array(x_train['ppg'])
        x_train['vpg'] = np.array(x_train['vpg'])
        x_train['apg'] = np.array(x_train['apg'])
        x_train['demographics'] = np.array(x_train['demographics'])
        # Min-max Normalize PPG, VPG, and APG
        x_train['ppg'] = (x_train['ppg'] - min_ppg) / (max_ppg - min_ppg)
        x_train['vpg'] = (x_train['vpg'] - min_vpg) / (max_vpg - min_vpg)
        x_train['apg'] = (x_train['apg'] - min_apg) / (max_apg - min_apg)
        # Normalize age
        mean_age = np.nanmean(x_train['demographics'][:, 0])
        std_age = np.nanstd(x_train['demographics'][:, 0])
        x_train['demographics'][:, 0] = (x_train['demographics'][:, 0] - mean_age) / std_age
        # Normalized height, weight, and BMI, where NA is excluded
        mean_height = np.nanmean(x_train['demographics'][:, 2])
        mean_weight = np.nanmean(x_train['demographics'][:, 3])
        mean_bmi = np.nanmean(x_train['demographics'][:, 4])

        std_height = np.nanstd(x_train['demographics'][:, 2])
        std_weight = np.nanstd(x_train['demographics'][:, 3])
        std_bmi = np.nanstd(x_train['demographics'][:, 4])

        x_train['demographics'][:, 2] = (x_train['demographics'][:, 2] - mean_height) / std_height
        x_train['demographics'][:, 3] = (x_train['demographics'][:, 3] - mean_weight) / std_weight
        x_train['demographics'][:, 4] = (x_train['demographics'][:, 4] - mean_bmi) / std_bmi

        # Replace NaN with 0
        x_train['demographics'][np.isnan(x_train['demographics'])] = 0

        if OUTPUT_NORMALIZED:
            y_train = np.array(y_train)
            y_train[:, 0] = (y_train[:, 0] - min_sbp) / (max_sbp - min_sbp)
            y_train[:, 1] = (y_train[:, 1] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump({'x': x_train, 'y': y_train}, open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'wb'))

        del x_train, y_train

        x_val['ppg'] = np.array(x_val['ppg'])
        x_val['vpg'] = np.array(x_val['vpg'])
        x_val['apg'] = np.array(x_val['apg'])
        x_val['demographics'] = np.array(x_val['demographics'])
        # Min-max Normalize PPG
        x_val['ppg'] = (x_val['ppg'] - min_ppg) / (max_ppg - min_ppg)
        x_val['vpg'] = (x_val['vpg'] - min_vpg) / (max_vpg - min_vpg)
        x_val['apg'] = (x_val['apg'] - min_apg) / (max_apg - min_apg)
        # Normalize age, height, weight, and BMI
        x_val['demographics'][:, 0] = (x_val['demographics'][:, 0] - mean_age) / std_age
        x_val['demographics'][:, 2] = (x_val['demographics'][:, 2] - mean_height) / std_height
        x_val['demographics'][:, 3] = (x_val['demographics'][:, 3] - mean_weight) / std_weight
        x_val['demographics'][:, 4] = (x_val['demographics'][:, 4] - mean_bmi) / std_bmi
        # Replace NaN with 0
        x_val['demographics'][np.isnan(x_val['demographics'])] = 0

        if OUTPUT_NORMALIZED:
            y_val = np.array(y_val)
            y_val[:, 0] = (y_val[:, 0] - min_sbp) / (max_sbp - min_sbp)
            y_val[:, 1] = (y_val[:, 1] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump({'x': x_val, 'y': y_val}, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        del x_val, y_val

        pickle.dump({
            'min_ppg': min_ppg, 'max_ppg': max_ppg,
            'min_sbp': min_sbp, 'max_sbp': max_sbp,
            'min_dbp': min_dbp, 'max_dbp': max_dbp,
            'mean_age': mean_age, 'std_age': std_age,
            'mean_height': mean_height, 'mean_weight': mean_weight, 'mean_bmi': mean_bmi,
            'std_height': std_height, 'std_weight': std_weight, 'std_bmi': std_bmi
        }, open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'wb'))

    print(f'\nThis dataset contains {segment_amount} segments')


if __name__ == '__main__':
    fold_data()