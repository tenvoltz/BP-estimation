import os
from tqdm import tqdm
import numpy as np
from mat73 import loadmat
import csv

SUBSETS_PATH = r'C:\Users\jaria092\PycharmProjects\BP-estimation\Data\PulseDB\Subset_Files'
DATASET_PATH = r'C:\Users\jaria092\PycharmProjects\BP-estimation\Data\PulseDB\Dataset_Files'
MATLAB_FIELD_NAME = 'Subset'
def create_dataset(input_path, max_amount=1, dataset_type="Train"):
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

    ecg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 0, :]
    ppg_all = data[MATLAB_FIELD_NAME]['Signals'][:, 1, :]
    sbps_all = data[MATLAB_FIELD_NAME]['SBP']
    dbps_all = data[MATLAB_FIELD_NAME]['DBP']
    age_all =  data[MATLAB_FIELD_NAME]['Age']
    gender_all = np.array(data[MATLAB_FIELD_NAME]['Gender']).squeeze()
    height_all = data[MATLAB_FIELD_NAME]['Height']
    weight_all = data[MATLAB_FIELD_NAME]['Weight']
    bmi_all = data[MATLAB_FIELD_NAME]['BMI']


    if os.path.exists(os.path.join(DATASET_PATH, f'dataset_{max_amount}_{dataset_type}.hdf5')):
        print(f'Dataset with amount={max_amount} subjects already exists')
        return None
    subject_ids = []
    ecgs = []
    ppgs = []
    sbps = []
    dbps = []
    age = []
    gender = []
    height = []
    weight = []
    bmi = []

    number_of_subjects = max_amount

    for i in tqdm(range(number_of_subjects), desc=f'Getting Data. Current Subject:'):

        start = int(subjects_start[i])
        end = int(subjects_start[i+1])

        subject_ids.extend([i] * (end - start))
        ppg_temp = ppg_all[start:end]
        ppgs.extend(ppg_temp)
        ecg_temp = ecg_all[start:end]
        ecgs.extend(ecg_temp)

        sbps.extend(sbps_all[start:end])
        dbps.extend(dbps_all[start:end])
        age.extend(age_all[start:end])
        gender.extend(gender_all[start:end])
        height.extend(height_all[start:end])
        weight.extend(weight_all[start:end])
        bmi.extend(bmi_all[start:end])

    # Write the data to a .csv file
    with open(os.path.join(DATASET_PATH, f'dataset_{max_amount}_{dataset_type}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Age', 'Gender', 'Height', 'Weight', 'BMI', 'PPG', 'ECG', 'SBP', 'DBP'])
        for i in range(len(ppgs)):
            ppg_txt = '|'.join([str(x) for x in ppgs[i]])
            ecg_txt = '|'.join([str(x) for x in ecgs[i]])
            writer.writerow([subject_ids[i], age[i], gender[i], height[i], weight[i], bmi[i], ppg_txt, ecg_txt, sbps[i], dbps[i]])

def read_dataset(file):
    with open(file, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ID = row[0]
            demographics = row[1:6]
            ppg = row[6].split('|')
            ecg = row[7].split('|')
            bp = row[8:10]
            print(ID, demographics, ppg, ecg, bp)

if __name__ == '__main__':
    read_dataset(os.path.join(DATASET_PATH, 'dataset_1_AAMI.csv'))
    """
    file = 'AAMI_Cal_Subset.mat'
    create_dataset(os.path.join(SUBSETS_PATH, file),
                   dataset_type=file.split('_')[0])
    """