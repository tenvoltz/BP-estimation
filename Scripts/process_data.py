import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np

# Constants
RAW_DATA_PATH = '../Data/raw_data'
SEGMENTED_DATA_PATH = '../Data/segmented_data'
DATASET_PATH = '../Data/raw_dataset'

NUMPY_SEED = 42

fs = 125 # Sampling frequency
samples_per_segment = 1024 # 1024 samples per segment
stride_per_segment = 512 # 512 samples stride per segment

MAX_PROCESSED = 11
MAX_DATASET_SIZE = 11

# Signal enum
class Signal:
    PPG = 0
    ABP = 1
    ECG = 2
def process_data(max_processed = 1000):
    if not os.path.isdir(RAW_DATA_PATH):
        print('Raw data folder not found')
        return
    if not os.path.isdir(SEGMENTED_DATA_PATH):
        os.mkdir(SEGMENTED_DATA_PATH)
    if not os.path.isdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs')):
        os.mkdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))
    if not os.path.isdir(os.path.join(SEGMENTED_DATA_PATH, 'sbps')):
        os.mkdir(os.path.join(SEGMENTED_DATA_PATH, 'sbps'))
    if not os.path.isdir(os.path.join(SEGMENTED_DATA_PATH, 'dbps')):
        os.mkdir(os.path.join(SEGMENTED_DATA_PATH, 'dbps'))

    current_processed = 0
    for file_id in range(1, 5):   # 4 data files
        print(f'Processing file {file_id} out of 4')

        # Load the data
        f = h5py.File(os.path.join(RAW_DATA_PATH, f'Part_{file_id}.mat'), 'r')
        key = f'Part_{file_id}'

        # Clip the number of records to process
        record_amount = len(f[key])
        if current_processed + record_amount > max_processed:
            record_amount = max_processed - current_processed

        # Process each record
        for record_id in tqdm(range(record_amount), desc=f'Processing file {file_id}'):
            # Skip if the record is already processed
            if os.path.isfile(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', f'Part_{file_id}_{record_id}.pkl')) and \
               os.path.isfile(os.path.join(SEGMENTED_DATA_PATH, 'sbps', f'Part_{file_id}_{record_id}.pkl')) and \
               os.path.isfile(os.path.join(SEGMENTED_DATA_PATH, 'dbps', f'Part_{file_id}_{record_id}.pkl')):
                continue

            ppg = []
            abp = []

            sample_length = len(f[f[key][record_id][0]])
            for j in tqdm(range(sample_length), desc=f'Reading Samples from Record {record_id}/{record_amount}'):
                ppg.append(f[f[key][record_id][0]][j][Signal.PPG])
                abp.append(f[f[key][record_id][0]][j][Signal.ABP])

            ppg_segments = []
            sbp_segments = []
            dbp_segments = []

            for j in tqdm(range(0, len(ppg) - samples_per_segment, stride_per_segment), desc=f'Segmenting Record {record_id}/{record_amount}'):
                ppg_segments.append(ppg[j:j+samples_per_segment])
                sbp_segments.append(max(abp[j:j+samples_per_segment]))
                dbp_segments.append(min(abp[j:j+samples_per_segment]))

            pickle.dump(np.array(ppg_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(sbp_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(dbp_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))

        current_processed += record_amount

def create_dataset(max_amount=100):
    if not os.path.isdir(SEGMENTED_DATA_PATH):
        print('Segmented data folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Get all the filenames
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))

    # Clip the number of records to process
    if max_amount > len(files):
        max_amount = len(files)
    files = files[:max_amount]

    # Set the random seed
    np.random.seed(NUMPY_SEED)
    # Randomly shuffle the files now so we can split data later easier
    np.random.shuffle(files)

    data = []

    for filename in tqdm(files):
        ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
        sbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', filename), 'rb'))
        dbp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', filename), 'rb'))

        for i in range(len(ppg)):
            data.append(np.append(ppg[i], [sbp[i], dbp[i]]))

    f = h5py.File(os.path.join(DATASET_PATH, 'dataset.hdf5'), 'w')
    dset = f.create_dataset('data', data=data)

if __name__ == '__main__':
    process_data(max_processed=MAX_PROCESSED)
    create_dataset(max_amount=MAX_DATASET_SIZE)



