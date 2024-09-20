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
CURRENT_DATASET_PATH = os.getenv('CURRENT_DATASET_PATH')
FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')

MAX_SEGMENT_AMOUNT = int(os.getenv('MAX_SEGMENT_AMOUNT'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))

OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'
CALIBRATION_FREE = os.getenv('CALIBRATION_FREE').lower() == 'true'

def fold_data():
    dataset_path = os.path.join(CURRENT_DATASET_PATH, 'dataset_train.hdf5')
    if not os.path.isfile(dataset_path):
        print('Dataset file not found')
        return
    if not os.path.isdir(FOLDED_DATASET_PATH):
        os.mkdir(FOLDED_DATASET_PATH)

    segment_amount = 0

    for fold_id in tqdm(range(FOLD_AMOUNT), desc='Folding Data'):
        # Load the dataset
        f = h5py.File(os.path.join(CURRENT_DATASET_PATH, 'dataset_train.hdf5'), 'r')
        segment_amount = len(f['signals']['ppg'])
        if segment_amount > MAX_SEGMENT_AMOUNT and MAX_SEGMENT_AMOUNT != -1:
            segment_amount = MAX_SEGMENT_AMOUNT

        ppgs = f['signals']['ppg'][:segment_amount]
        vpgs = f['signals']['vpg'][:segment_amount]
        apgs = f['signals']['apg'][:segment_amount]

        ages = f['demographics']['age'][:segment_amount]
        genders = f['demographics']['gender'][:segment_amount]
        heights = f['demographics']['height'][:segment_amount]
        weights = f['demographics']['weight'][:segment_amount]
        bmis = f['demographics']['bmi'][:segment_amount]

        sbps = f['targets']['sbp'][:segment_amount]
        dbps = f['targets']['dbp'][:segment_amount]
        subjects = f['targets']['subject'][:segment_amount]

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
            'sbp': [], 'dbp': [], 'subject': []
        }}
        data_val = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'demographics': {
            'age': [], 'gender': [], 'height': [], 'weight': [], 'bmi': []
        }, 'targets': {
            'sbp': [], 'dbp': [], 'subject': []
        }}

        validation_start = validation_starts[fold_id]
        validation_end = validation_starts[fold_id + 1]

        max_ppg, min_ppg = -10000, 10000
        max_vpg, min_vpg = -10000, 10000
        max_apg, min_apg = -10000, 10000

        max_sbp, min_sbp = -10000, 10000
        max_dbp, min_dbp = -10000, 10000

        for i in tqdm(range(0, validation_start), desc=f'Fold {fold_id} - Training Data (left)'):
            # min-max normalize each segment
            ppg_temp = (ppgs[i] - np.min(ppgs[i])) / (np.max(ppgs[i]) - np.min(ppgs[i]))
            vpg_temp = (vpgs[i] - np.min(vpgs[i])) / (np.max(vpgs[i]) - np.min(vpgs[i]))
            apg_temp = (apgs[i] - np.min(apgs[i])) / (np.max(apgs[i]) - np.min(apgs[i]))

            data_train['signals']['ppg'].append(ppg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['demographics']['age'].append(ages[i])
            data_train['demographics']['gender'].append(genders[i])
            data_train['demographics']['height'].append(heights[i])
            data_train['demographics']['weight'].append(weights[i])
            data_train['demographics']['bmi'].append(bmis[i])
            data_train['targets']['sbp'].append(sbps[i])
            data_train['targets']['dbp'].append(dbps[i])
            data_train['targets']['subject'].append(subjects[i])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])
        for i in tqdm(range(validation_start, validation_end), desc=f'Fold {fold_id} - Validation Data'):
            # min-max normalize each segment
            ppg_temp = (ppgs[i] - np.min(ppgs[i])) / (np.max(ppgs[i]) - np.min(ppgs[i]))
            vpg_temp = (vpgs[i] - np.min(vpgs[i])) / (np.max(vpgs[i]) - np.min(vpgs[i]))
            apg_temp = (apgs[i] - np.min(apgs[i])) / (np.max(apgs[i]) - np.min(apgs[i]))

            data_val['signals']['ppg'].append(ppg_temp.reshape(1, SIGNAL_LENGTH))
            data_val['signals']['vpg'].append(vpg_temp.reshape(1, SIGNAL_LENGTH))
            data_val['signals']['apg'].append(apg_temp.reshape(1, SIGNAL_LENGTH))
            data_val['demographics']['age'].append(ages[i])
            data_val['demographics']['gender'].append(genders[i])
            data_val['demographics']['height'].append(heights[i])
            data_val['demographics']['weight'].append(weights[i])
            data_val['demographics']['bmi'].append(bmis[i])
            data_val['targets']['sbp'].append(sbps[i])
            data_val['targets']['dbp'].append(dbps[i])
            data_val['targets']['subject'].append(subjects[i])
        for i in tqdm(range(validation_end, segment_amount), desc=f'Fold {fold_id} - Training Data (right)'):
            # min-max normalize each segment
            ppg_temp = (ppgs[i] - np.min(ppgs[i])) / (np.max(ppgs[i]) - np.min(ppgs[i]))
            vpg_temp = (vpgs[i] - np.min(vpgs[i])) / (np.max(vpgs[i]) - np.min(vpgs[i]))
            apg_temp = (apgs[i] - np.min(apgs[i])) / (np.max(apgs[i]) - np.min(apgs[i]))

            data_train['signals']['ppg'].append(ppg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['vpg'].append(vpg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['signals']['apg'].append(apg_temp.reshape(1, SIGNAL_LENGTH))
            data_train['demographics']['age'].append(ages[i])
            data_train['demographics']['gender'].append(genders[i])
            data_train['demographics']['height'].append(heights[i])
            data_train['demographics']['weight'].append(weights[i])
            data_train['demographics']['bmi'].append(bmis[i])
            data_train['targets']['sbp'].append(sbps[i])
            data_train['targets']['dbp'].append(dbps[i])
            data_train['targets']['subject'].append(subjects[i])

            max_ppg, min_ppg = max(max_ppg, max(ppgs[i])), min(min_ppg, min(ppgs[i]))
            max_vpg, min_vpg = max(max_vpg, max(vpgs[i])), min(min_vpg, min(vpgs[i]))
            max_apg, min_apg = max(max_apg, max(apgs[i])), min(min_apg, min(apgs[i]))

            max_sbp, min_sbp = max(max_sbp, sbps[i]), min(min_sbp, sbps[i])
            max_dbp, min_dbp = max(max_dbp, dbps[i]), min(min_dbp, dbps[i])

        f.close()
        del f

        ############################## Training Data ########################################

        # Convert to numpy arrays
        for key in data_train['signals'].keys():
            data_train['signals'][key] = np.array(data_train['signals'][key])
        for key in data_train['demographics'].keys():
            data_train['demographics'][key] = np.array(data_train['demographics'][key])
        for key in data_train['targets'].keys():
            data_train['targets'][key] = np.array(data_train['targets'][key])
        # Min-max Normalize PPG, VPG, and APG
        #data_train['signals']['ppg'] = (data_train['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        #data_train['signals']['vpg'] = (data_train['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        #data_train['signals']['apg'] = (data_train['signals']['apg'] - min_apg) / (max_apg - min_apg)

        # Z-score Normalize PPG, VPG, and APG
        # mean_ppg = np.mean(data_train['signals']['ppg'])
        # std_ppg = np.std(data_train['signals']['ppg'])
        # mean_vpg = np.mean(data_train['signals']['vpg'])
        # std_vpg = np.std(data_train['signals']['vpg'])
        # mean_apg = np.mean(data_train['signals']['apg'])
        # std_apg = np.std(data_train['signals']['apg'])
        #data_train['signals']['ppg'] = (data_train['signals']['ppg'] - mean_ppg) / std_ppg
        #data_train['signals']['vpg'] = (data_train['signals']['vpg'] - mean_vpg) / std_vpg
        #data_train['signals']['apg'] = (data_train['signals']['apg'] - mean_apg) / std_apg

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

        ############################## Validation Data ########################################

        # Convert to numpy arrays
        for key in data_val['signals'].keys():
            data_val['signals'][key] = np.array(data_val['signals'][key])
        for key in data_val['demographics'].keys():
            data_val['demographics'][key] = np.array(data_val['demographics'][key])
        for key in data_val['targets'].keys():
            data_val['targets'][key] = np.array(data_val['targets'][key])

        # Min-max Normalize PPG, VPG, and APG
        #data_val['signals']['ppg'] = (data_val['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        #data_val['signals']['vpg'] = (data_val['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        #data_val['signals']['apg'] = (data_val['signals']['apg'] - min_apg) / (max_apg - min_apg)

        #data_val['signals']['ppg'] = (data_val['signals']['ppg'] - mean_ppg) / std_ppg
        #data_val['signals']['vpg'] = (data_val['signals']['vpg'] - mean_vpg) / std_vpg
        #data_val['signals']['apg'] = (data_val['signals']['apg'] - mean_apg) / std_apg

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

        print(f'\nThis calibration-based folds contains {segment_amount} segments')

        f = h5py.File(os.path.join(CURRENT_DATASET_PATH, 'dataset_cal.hdf5'), 'r')
        segment_amount = len(f['signals']['ppg'])
        if segment_amount > MAX_SEGMENT_AMOUNT and MAX_SEGMENT_AMOUNT != -1:
            segment_amount = MAX_SEGMENT_AMOUNT

        ppgs = f['signals']['ppg'][:segment_amount]
        vpgs = f['signals']['vpg'][:segment_amount]
        apgs = f['signals']['apg'][:segment_amount]

        ages = f['demographics']['age'][:segment_amount]
        genders = f['demographics']['gender'][:segment_amount]
        heights = f['demographics']['height'][:segment_amount]
        weights = f['demographics']['weight'][:segment_amount]
        bmis = f['demographics']['bmi'][:segment_amount]

        sbps = f['targets']['sbp'][:segment_amount]
        dbps = f['targets']['dbp'][:segment_amount]

        # Split the data
        data_cal = {'signals': {
            'ppg': [], 'vpg': [], 'apg': []
        }, 'demographics': {
            'age': [], 'gender': [], 'height': [], 'weight': [], 'bmi': []
        }, 'targets': {
            'sbp': [], 'dbp': [], 'subject': []
        }}

        for i in tqdm(range(0, segment_amount), desc=f'Calibration Data'):
            ppg_temp = (ppgs[i] - np.min(ppgs[i])) / (np.max(ppgs[i]) - np.min(ppgs[i]))
            vpg_temp = (vpgs[i] - np.min(vpgs[i])) / (np.max(vpgs[i]) - np.min(vpgs[i]))
            apg_temp = (apgs[i] - np.min(apgs[i])) / (np.max(apgs[i]) - np.min(apgs[i]))

            data_cal['signals']['ppg'].append(ppg_temp.reshape(1, SIGNAL_LENGTH))
            data_cal['signals']['vpg'].append(vpg_temp.reshape(1, SIGNAL_LENGTH))
            data_cal['signals']['apg'].append(apg_temp.reshape(1, SIGNAL_LENGTH))
            data_cal['demographics']['age'].append(ages[i])
            data_cal['demographics']['gender'].append(genders[i])
            data_cal['demographics']['height'].append(heights[i])
            data_cal['demographics']['weight'].append(weights[i])
            data_cal['demographics']['bmi'].append(bmis[i])
            data_cal['targets']['sbp'].append(sbps[i])
            data_cal['targets']['dbp'].append(dbps[i])
            data_cal['targets']['subject'].append(subjects[i])

        f.close()
        del f

        # Convert to numpy arrays
        for key in data_cal['signals'].keys():
            data_cal['signals'][key] = np.array(data_cal['signals'][key])
        for key in data_cal['demographics'].keys():
            data_cal['demographics'][key] = np.array(data_cal['demographics'][key])
        for key in data_cal['targets'].keys():
            data_cal['targets'][key] = np.array(data_cal['targets'][key])

        # Min-max Normalize PPG, VPG, and APG
        #data_cal['signals']['ppg'] = (data_cal['signals']['ppg'] - min_ppg) / (max_ppg - min_ppg)
        #data_cal['signals']['vpg'] = (data_cal['signals']['vpg'] - min_vpg) / (max_vpg - min_vpg)
        #data_cal['signals']['apg'] = (data_cal['signals']['apg'] - min_apg) / (max_apg - min_apg)

        #data_cal['signals']['ppg'] = (data_cal['signals']['ppg'] - mean_ppg) / std_ppg
        #data_cal['signals']['vpg'] = (data_cal['signals']['vpg'] - mean_vpg) / std_vpg
        #data_cal['signals']['apg'] = (data_cal['signals']['apg'] - mean_apg) / std_apg

        # Normalize age
        data_cal['demographics']['age'] = (data_cal['demographics']['age'] - mean_age) / std_age
        # Normalized height, weight, and BMI, where NA is excluded
        data_cal['demographics']['height'] = (data_cal['demographics']['height'] - mean_height) / std_height
        data_cal['demographics']['weight'] = (data_cal['demographics']['weight'] - mean_weight) / std_weight
        data_cal['demographics']['bmi'] = (data_cal['demographics']['bmi'] - mean_bmi) / std_bmi

        # Replace NaN with 0
        for key in data_cal['demographics'].keys():
            data_cal['demographics'][key][np.isnan(data_cal['demographics'][key])] = 0

        if OUTPUT_NORMALIZED:
            data_cal['targets']['sbp'] = (data_cal['targets']['sbp'] - min_sbp) / (max_sbp - min_sbp)
            data_cal['targets']['dbp'] = (data_cal['targets']['dbp'] - min_dbp) / (max_dbp - min_dbp)

        pickle.dump(data_cal, open(os.path.join(FOLDED_DATASET_PATH, f'cal_{fold_id}.pkl'), 'wb'))

        del data_cal

        print(f'\nThis calibration-free folds contains {segment_amount} segments')

if __name__ == '__main__':
    fold_data()