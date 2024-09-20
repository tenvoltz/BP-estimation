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

TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT'))

SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))
SEED = int(os.getenv('SEED'))
MATLAB_FIELD_NAME = 'Subset'
CALIBRATION_FREE = os.getenv('CALIBRATION_FREE') == 'True'
PATIENT_RATIO = [float(i) for i in os.getenv('PATIENT_RATIO').split(',')]
SEGMENT_PER_PATIENT_RATIO = [float(i) for i in os.getenv('SEGMENT_PER_PATIENT_RATIO').split(',')]


def combine_data(start_subject_idx, end_subject_idx, number_of_segments, patient_ratio,
                 subjects_start, ppg_all, sbps_all, dbps_all, age_all, gender_all, height_all, weight_all, bmi_all):
    dataset = []

    for i in tqdm(range(start_subject_idx, end_subject_idx), desc='Getting Data. Current Subject:'):
        start = int(subjects_start[i])
        end = min(int(subjects_start[i] + (number_of_segments if patient_ratio != 1 else int(subjects_start[i + 1]))),
                  int(subjects_start[i + 1]))

        # Extract PPG, VPG, APG data
        ppg_temp = ppg_all[start:end]
        vpg_temp = np.gradient(ppg_temp, axis=1)
        apg_temp = np.gradient(vpg_temp, axis=1)

        # Extract other relevant data
        sbp_temp = sbps_all[start:end]
        dbp_temp = dbps_all[start:end]
        age_temp = age_all[start:end]
        gender_temp = np.where(np.array(gender_all[start:end]) == 'M', 1, -1)  # Convert gender
        height_temp = height_all[start:end]
        weight_temp = weight_all[start:end]
        bmi_temp = bmi_all[start:end]
        subject_idx = np.full((end - start), i)

        # Stack features for the current subject
        features_temp = np.stack((age_temp, gender_temp, height_temp, weight_temp, bmi_temp,
                                  sbp_temp, dbp_temp, subject_idx),
                                 axis=1)

        # Concatenate PPG, VPG, APG with the features
        subject_data = np.concatenate((ppg_temp, vpg_temp, apg_temp, features_temp), axis=1)

        # Append the subject's data to the dataset
        dataset.extend(subject_data)

    dataset = np.array(dataset)
    print(dataset.shape)
    return dataset

def write_to_hdf5(dataset_path, dataset, dataset_type):
    f = h5py.File(os.path.join(dataset_path, f'dataset_{dataset_type}.hdf5'), 'w')
    signal_group = f.create_group('signals')
    signal_group['ppg'] = dataset[:, :SIGNAL_LENGTH]
    signal_group['vpg'] = dataset[:, SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
    signal_group['apg'] = dataset[:, 2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
    signal_group['epg'] = dataset[:, 3 * SIGNAL_LENGTH:4 * SIGNAL_LENGTH]
    signal_group['vepg'] = dataset[:, 4 * SIGNAL_LENGTH:5 * SIGNAL_LENGTH]
    signal_group['aepg'] = dataset[:, 5 * SIGNAL_LENGTH:6 * SIGNAL_LENGTH]
    demographic_group = f.create_group('demographics')
    demographic_group['age'] = dataset[:, -8]
    demographic_group['gender'] = dataset[:, -7]
    demographic_group['height'] = dataset[:, -6]
    demographic_group['weight'] = dataset[:, -5]
    demographic_group['bmi'] = dataset[:, -4]
    target_group = f.create_group('targets')
    target_group['sbp'] = dataset[:, -3]
    target_group['dbp'] = dataset[:, -2]
    target_group['subject'] = dataset[:, -1]

def create_dataset(input_path, dataset_type="Train"):
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

    # ecg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 0, :]
    ppg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 1, :]
    sbps_all = data[MATLAB_FIELD_NAME]['SBP']
    dbps_all = data[MATLAB_FIELD_NAME]['DBP']
    age_all =  data[MATLAB_FIELD_NAME]['Age']
    gender_all = np.array(data[MATLAB_FIELD_NAME]['Gender']).squeeze()
    height_all = data[MATLAB_FIELD_NAME]['Height']
    weight_all = data[MATLAB_FIELD_NAME]['Weight']
    bmi_all = data[MATLAB_FIELD_NAME]['BMI']

    for segment_ratio in SEGMENT_PER_PATIENT_RATIO:
        for patient_ratio in PATIENT_RATIO:
            print(f'############### Creating Dataset with {segment_ratio} segments per patient and {patient_ratio} patient ratio ###############')
            dataset_path = os.path.join(DATASET_PATH, f'dataset_{dataset_type}_{segment_ratio}_{patient_ratio}')
            if not os.path.isdir(dataset_path):
                os.mkdir(dataset_path)

            number_of_subjects = int((len(subjects_start) - 1) * patient_ratio)
            number_of_segments = int(np.min(np.diff(subjects_start))) * segment_ratio

            subject_train = int(number_of_subjects * TRAIN_TEST_SPLIT)
            subject_val = number_of_subjects - subject_train

            # The two datasets are created separately based on subject
            train_dataset = combine_data(0, subject_train, number_of_segments, patient_ratio,
                                         subjects_start, ppg_all, sbps_all, dbps_all,
                                         age_all, gender_all, height_all, weight_all, bmi_all)
            val_dataset = combine_data(subject_train, subject_train + subject_val, number_of_segments, patient_ratio,
                                       subjects_start, ppg_all, sbps_all, dbps_all,
                                       age_all, gender_all, height_all, weight_all, bmi_all)

            np.random.seed(SEED)
            np.random.shuffle(train_dataset)
            np.random.shuffle(val_dataset)

            pickle.dump({
                'number_of_subjects': number_of_subjects,
                'number_of_segments': number_of_segments,
                'subject_train': subject_train,
                'subject_val': subject_val
            }, open(os.path.join(dataset_path, f'stats.pkl'), 'wb'))

            write_to_hdf5(dataset_path, train_dataset, 'train')
            write_to_hdf5(dataset_path, val_dataset, 'cal')

    """
    for segment_ratio in SEGMENT_PER_PATIENT_RATIO:
        for patient_ratio in PATIENT_RATIO:
            print(f'############### Creating Dataset with {segment_ratio} segments per patient and {patient_ratio} patient ratio ###############')
    
            number_of_subjects = int((len(subjects_start) - 1) * patient_ratio) 
            number_of_segments = int(np.min(np.diff(subjects_start))) * segment_ratio
    
            subject_train = int(number_of_subjects * TRAIN_TEST_SPLIT)
            subject_val = number_of_subjects - subject_train
    
            # ecgs, vecgs, aecgs = [], [], []
            ppgs, vpgs, apgs = [], [], []
            sbps, dbps = [], []
            age, gender, height, weight, bmi = [], [], [], [], []
    
            for i in tqdm(range(0, subject_train), desc=f'Getting Data. Current Subject:'):
    
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
    
                # ecg_temp = ecg_all[start:end]
                # vecg_temp = np.gradient(ecg_temp, axis=1)
                # aecg_temp = np.gradient(vecg_temp, axis=1)
                # ecgs.extend(ecg_temp)
                # vecgs.extend(vecg_temp)
                # aecgs.extend(aecg_temp)
    
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
                'subject_train': subject_train,
                'subject_val': subject_val
            }, open(os.path.join(DATASET_PATH, f'stats.pkl'), 'wb'))
    
            # Convert Gender into numerical values where M=1 and F=-1
            gender = np.where(np.array(gender) == 'M', 1, -1)
    
            # Shuffle the dataset
            dataset = np.concatenate((ppgs, vpgs, apgs, #ecgs, vecgs, aecgs,
                                      np.stack((age, gender, height, weight, bmi, sbps, dbps), axis=1)), axis=1)
            
            
            np.random.seed(SEED)
            np.random.shuffle(dataset)
    
            f = h5py.File(os.path.join(DATASET_PATH, f'dataset.hdf5'), 'w')
            signal_group = f.create_group('signals')
            signal_group['ppg'] = dataset[:, :SIGNAL_LENGTH]
            signal_group['vpg'] = dataset[:, SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
            signal_group['apg'] = dataset[:, 2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
            # signal_group['ecg'] = dataset[:, 3 * SIGNAL_LENGTH:4 * SIGNAL_LENGTH]
            # signal_group['vecg'] = dataset[:, 4 * SIGNAL_LENGTH:5 * SIGNAL_LENGTH]
            # signal_group['aecg'] = dataset[:, 5 * SIGNAL_LENGTH:6 * SIGNAL_LENGTH]
            demographic_group = f.create_group('demographics')
            demographic_group['age'] = dataset[:, -7]
            demographic_group['gender'] = dataset[:, -6]
            demographic_group['height'] = dataset[:, -5]
            demographic_group['weight'] = dataset[:, -4]
            demographic_group['bmi'] = dataset[:, -3]
            target_group = f.create_group('targets')
            target_group['sbp'] = dataset[:, -2]
            target_group['dbp'] = dataset[:, -1]
    
            ###### For calibration
    
            if CALIBRATION_FREE:
                ecgs, vecgs, aecgs = [], [], []
                ppgs, vpgs, apgs = [], [], []
                sbps, dbps = [], []
                age, gender, height, weight, bmi = [], [], [], [], []
    
                for i in tqdm(range(subject_train, subject_train + subject_val), desc=f'Getting Data. Current Subject:'):
    
                    start = int(subjects_start[i])
                    end = int(subjects_start[i] + (number_of_segments if ratio != 1 else int(subjects_start[i + 1])))
    
                    if end > subjects_start[i + 1]:
                        end = int(subjects_start[i + 1])
    
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
                }, open(os.path.join(DATASET_PATH, f'stats_cal.pkl'), 'wb'))
    
                # Convert Gender into numerical values where M=1 and F=-1
                gender = np.where(np.array(gender) == 'M', 1, -1)
    
                # Shuffle the dataset
                dataset = np.concatenate((ppgs, vpgs, apgs,
                                          np.stack((age, gender, height, weight, bmi, sbps, dbps), axis=1)), axis=1)
                np.random.seed(SEED)
                np.random.shuffle(dataset)
    
                f = h5py.File(os.path.join(DATASET_PATH, f'dataset_cal.hdf5'), 'w')
                signal_group = f.create_group('signals')
                signal_group['ppg'] = dataset[:, :SIGNAL_LENGTH]
                signal_group['vpg'] = dataset[:, SIGNAL_LENGTH:2 * SIGNAL_LENGTH]
                signal_group['apg'] = dataset[:, 2 * SIGNAL_LENGTH:3 * SIGNAL_LENGTH]
                signal_group['epg'] = dataset[:, 3 * SIGNAL_LENGTH:4 * SIGNAL_LENGTH]
                signal_group['vepg'] = dataset[:, 4 * SIGNAL_LENGTH:5 * SIGNAL_LENGTH]
                signal_group['aepg'] = dataset[:, 5 * SIGNAL_LENGTH:6 * SIGNAL_LENGTH]
                demographic_group = f.create_group('demographics')
                demographic_group['age'] = dataset[:, -7]
                demographic_group['gender'] = dataset[:, -6]
                demographic_group['height'] = dataset[:, -5]
                demographic_group['weight'] = dataset[:, -4]
                demographic_group['bmi'] = dataset[:, -3]
                target_group = f.create_group('targets')
                target_group['sbp'] = dataset[:, -2]
                target_group['dbp'] = dataset[:, -1]
    """

if __name__ == '__main__':
    #for file in os.listdir(SUBSETS_PATH):
    #    if file.endswith(".mat"):
            file = 'Train_Subset.mat'
            create_dataset(os.path.join(SUBSETS_PATH, file),
                           dataset_type=file.split('_')[0])
