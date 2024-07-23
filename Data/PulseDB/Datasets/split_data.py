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
        data_train = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'demographics': {
            'age': [], 'gender': [], 'height': [], 'weight': [], 'bmi': []
        }, 'targets': {
            'sbp': [], 'dbp': []
        }}
        data_val = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'demographics': {
            'age': [], 'gender': [], 'height': [], 'weight': [], 'bmi': []
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
            data_train['signals']['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['demographics']['age'].append(ages[i])
            data_train['demographics']['gender'].append(genders[i])
            data_train['demographics']['height'].append(heights[i])
            data_train['demographics']['weight'].append(weights[i])
            data_train['demographics']['bmi'].append(bmis[i])
            data_train['targets']['sbp'].append(sbps[i])
            data_train['targets']['dbp'].append(dbps[i])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            data_val['signals']['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            data_val['signals']['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            data_val['signals']['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            data_val['demographics']['age'].append(ages[i])
            data_val['demographics']['gender'].append(genders[i])
            data_val['demographics']['height'].append(heights[i])
            data_val['demographics']['weight'].append(weights[i])
            data_val['demographics']['bmi'].append(bmis[i])
            data_val['targets']['sbp'].append(sbps[i])
            data_val['targets']['dbp'].append(dbps[i])

        for i in tqdm(range(validation_end, segment_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            data_train['signals']['ppg'].append(ppgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apgs[i].reshape(1, SIGNAL_LENGTH))
            data_train['demographics']['age'].append(ages[i])
            data_train['demographics']['gender'].append(genders[i])
            data_train['demographics']['height'].append(heights[i])
            data_train['demographics']['weight'].append(weights[i])
            data_train['demographics']['bmi'].append(bmis[i])
            data_train['targets']['sbp'].append(sbps[i])
            data_train['targets']['dbp'].append(dbps[i])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        f.close()
        del f

        # Convert to numpy arrays
        for key in data_train['signals'].keys():
            data_train['signals'][key] = np.array(data_train['signals'][key])
        for key in data_train['demographics'].keys():
            data_train['demographics'][key] = np.array(data_train['demographics'][key])
        for key in data_train['targets'].keys():
            data_train['targets'][key] = np.array(data_train['targets'][key])
        # Min-max Normalize PPG, VPG, and APG
        data_train['signals']['ppg'] = (data_train['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        data_train['signals']['vpg'] = (data_train['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        data_train['signals']['apg'] = (data_train['signals']['apg'] - min_apg) / (max_apg - min_apg)
        # Normalize age
        mean_age = np.nanmean(data_train['demographics']['age'])
        std_age = np.nanstd(data_train['demographics']['age'])
        data_train['demographics']['age'] = (data_train['demographics']['age'] - mean_age) / std_age
        # Normalized height, weight, and BMI, where NA is excluded
        mean_height = np.nanmean(data_train['demographics']['height'])
        mean_weight = np.nanmean(data_train['demographics']['weight'])
        mean_bmi = np.nanmean(data_train['demographics']['bmi'])

        std_height = np.nanstd(data_train['demographics']['height'])
        std_weight = np.nanstd(data_train['demographics']['weight'])
        std_bmi = np.nanstd(data_train['demographics']['bmi'])

        data_train['demographics']['height'] = (data_train['demographics']['height'] - mean_height) / std_height
        data_train['demographics']['weight']= (data_train['demographics']['weight'] - mean_weight) / std_weight
        data_train['demographics']['bmi'] = (data_train['demographics']['bmi'] - mean_bmi) / std_bmi

        # Replace NaN with 0
        for key in data_train['demographics'].keys():
            data_train['demographics'][key][np.isnan(data_train['demographics'][key])] = 0

        # Mean of SBP and DBP
        mean_sbp = np.nanmean(data_train['targets']['sbp'])
        mean_dbp = np.nanmean(data_train['targets']['dbp'])

        if OUTPUT_NORMALIZED:
            data_train['targets']['sbp'] = (data_train['targets']['sbp'] - min_sbp) / (max_sbp - min_sbp)
            data_train['targets']['dbp'] = (data_train['targets']['dbp'] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump(data_train, open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'wb'))

        del data_train

        # Convert to numpy arrays
        for key in data_val['signals'].keys():
            data_val['signals'][key] = np.array(data_val['signals'][key])
        for key in data_val['demographics'].keys():
            data_val['demographics'][key] = np.array(data_val['demographics'][key])
        for key in data_val['targets'].keys():
            data_val['targets'][key] = np.array(data_val['targets'][key])

        # Min-max Normalize PPG, VPG, and APG
        data_val['signals']['ppg'] = (data_val['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        data_val['signals']['vpg'] = (data_val['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        data_val['signals']['apg'] = (data_val['signals']['apg'] - min_apg) / (max_apg - min_apg)

        # Normalize age
        data_val['demographics']['age'] = (data_val['demographics']['age'] - mean_age) / std_age
        # Normalized height, weight, and BMI, where NA is excluded
        data_val['demographics']['height'] = (data_val['demographics']['height'] - mean_height) / std_height
        data_val['demographics']['weight'] = (data_val['demographics']['weight'] - mean_weight) / std_weight
        data_val['demographics']['bmi'] = (data_val['demographics']['bmi'] - mean_bmi) / std_bmi

        # Replace NaN with 0
        for key in data_val['demographics'].keys():
            data_val['demographics'][key][np.isnan(data_val['demographics'][key])] = 0

        if OUTPUT_NORMALIZED:
            data_val['targets']['sbp'] = (data_val['targets']['sbp'] - min_sbp) / (max_sbp - min_sbp)
            data_val['targets']['dbp'] = (data_val['targets']['dbp'] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump(data_val, open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'wb'))

        del data_val

        pickle.dump({
            'min_ppg': min_ppg, 'max_ppg': max_ppg,
            'min_sbp': min_sbp, 'max_sbp': max_sbp,
            'min_dbp': min_dbp, 'max_dbp': max_dbp,
            'mean_sbp': mean_sbp, 'mean_dbp': mean_dbp,
            'mean_age': mean_age, 'std_age': std_age,
            'mean_height': mean_height, 'mean_weight': mean_weight, 'mean_bmi': mean_bmi,
            'std_height': std_height, 'std_weight': std_weight, 'std_bmi': std_bmi
        }, open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'wb'))

    print(f'\nThis dataset contains {segment_amount} segments')


if __name__ == '__main__':
    fold_data()