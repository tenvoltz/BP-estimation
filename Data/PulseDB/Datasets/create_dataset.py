import os
import h5py
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from mat73 import loadmat
import pickle

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

SUBSETS_PATH = os.getenv('SUBSETS_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')

MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))
PATIENT_SIGNAL_RATIO = [float(i) for i in os.getenv('PATIENT_SIGNAL_RATIO').split(',')]

SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))

SEED = int(os.getenv('SEED'))

MATLAB_FIELD_NAME = 'Subset'
def create_dataset(input_path, max_amount=-1, dataset_type="Train", ratios=None):
    if ratios is None:
        ratios = [1]
    if not os.path.isdir(SUBSETS_PATH):
        print('Subsets folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    print(f'############### Loading Data {dataset_type} ###############')
    data = loadmat(input_path)
    print(f'############### Data Loaded ###############')
    # Get the index of all subjects
    subjects = data[MATLAB_FIELD_NAME]['Subject']
    subjects_start = [0]
    for i in range(1, len(subjects)):
        if subjects[i] != subjects[i-1]:
            subjects_start.append(i)
    subjects_start.append(len(subjects))

    if len(subjects) < max_amount or max_amount == -1:
        max_amount = len(subjects)

    del subjects

    ecg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 0, :]
    ppg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 1, :]
    sbps_all = data[MATLAB_FIELD_NAME]['SBP']
    dbps_all = data[MATLAB_FIELD_NAME]['DBP']
    age_all =  data[MATLAB_FIELD_NAME]['Age']
    gender_all = np.array(data[MATLAB_FIELD_NAME]['Gender']).squeeze()
    height_all = data[MATLAB_FIELD_NAME]['Height']
    weight_all = data[MATLAB_FIELD_NAME]['Weight']
    bmi_all = data[MATLAB_FIELD_NAME]['BMI']

    if max_amount == -1:
        max_amount = len(ppg_all)
    if ratios is None:
        ratios = [1]

    for ratio in ratios:
        print(f'############### Creating Dataset with ratio={ratio} ###############')
        if os.path.exists(os.path.join(DATASET_PATH, f'dataset_{max_amount}_{dataset_type}_{ratio}.hdf5')):
            print(f'Dataset with amount={max_amount} ratio={ratio} already exists')
            continue
        ecgs = []
        vecgs = []
        aecgs = []
        ppgs = []
        vpgs = []
        apgs = []
        sbps = []
        dbps = []
        age = []
        gender = []
        height = []
        weight = []
        bmi = []

        number_of_subjects = int((len(subjects_start) - 1) * ratio)
        if dataset_type == "Train":
            number_of_segments = (max_amount * TRAIN_TEST_SPLIT) // number_of_subjects
        else:
            number_of_segments = (max_amount * (1 - TRAIN_TEST_SPLIT)) // number_of_subjects
        if ratio == 1:
            number_of_segments = int(np.mean(np.diff(subjects_start)))

        dataset = []

        for i in tqdm(range(number_of_subjects), desc=f'Getting Data. Current Subject:'):

            start = int(subjects_start[i])
            end = int(subjects_start[i] + (number_of_segments if ratio != 1 else int(subjects_start[i+1])))

            if end > subjects_start[i+1]:
                end = int(subjects_start[i+1])

            ppg_temp = ppg_all[start:end]
            vpg_temp = np.gradient(ppg_temp, axis=1)
            apg_temp = np.gradient(vpg_temp, axis=1)

            ppgs.extend(ppg_temp)
            vpgs.extend(vpg_temp)
            apgs.extend(apg_temp)

            ecg_temp = ecg_all[start:end]
            vecg_temp = np.gradient(ecg_temp, axis=1)
            aecg_temp = np.gradient(vecg_temp, axis=1)

            ecgs.extend(ecg_temp)
            vecgs.extend(vecg_temp)
            aecgs.extend(aecg_temp)

            sbps.extend(sbps_all[start:end])
            dbps.extend(dbps_all[start:end])
            age.extend(age_all[start:end])
            gender.extend(gender_all[start:end])
            height.extend(height_all[start:end])
            weight.extend(weight_all[start:end])
            bmi.extend(bmi_all[start:end])

        pickle.dump({
            'number_of_subjects': number_of_subjects,
            'number_of_segments': number_of_segments,
        }, open(os.path.join(DATASET_PATH, f'stats_{max_amount}_{dataset_type}_{ratio}.pkl'), 'wb'))

        # Convert Gender into numerical values where M=1 and F=-1
        gender = np.where(np.array(gender) == 'M', 1, -1)

        # Shuffle the dataset
        dataset = np.concatenate((ecgs, vecgs, aecgs, ppgs, vpgs, apgs,
                                  np.stack((age, gender, height, weight, bmi, sbps, dbps), axis=1)), axis=1)
        np.random.seed(SEED)
        np.random.shuffle(dataset)

        f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{max_amount}_{dataset_type}_{ratio}.hdf5'), 'w')
        signal_group = f.create_group('signals')
        signal_group['ecg'] = dataset[:, :SIGNAL_LENGTH]
        signal_group['vecg'] = dataset[:, SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
        signal_group['aecg'] = dataset[:, 2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
        signal_group['ppg'] = dataset[:, :SIGNAL_LENGTH]
        signal_group['vpg'] = dataset[:, SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
        signal_group['apg'] = dataset[:, 2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
        demographic_group = f.create_group('demographics')
        demographic_group['age'] = dataset[:, -7]
        demographic_group['gender'] = dataset[:, -6]
        demographic_group['height'] = dataset[:, -5]
        demographic_group['weight'] = dataset[:, -4]
        demographic_group['bmi'] = dataset[:, -3]
        target_group = f.create_group('targets')
        target_group['sbp'] = dataset[:, -2]
        target_group['dbp'] = dataset[:, -1]

if __name__ == '__main__':
    #for file in os.listdir(SUBSETS_PATH):
    #    if file.endswith(".mat"):
            file = 'AAMI_Cal_Subset.mat'
            create_dataset(os.path.join(SUBSETS_PATH, file),
                           dataset_type=file.split('_')[0])
