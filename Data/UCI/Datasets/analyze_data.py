import os
import pickle
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

class Signal:
    PPG = 0
    ABP = 1
    ECG = 2

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_INSTANCES'))
fs = int(os.getenv('SAMPLING_FREQUENCY'))

MODEL = os.getenv('MODEL')

def plot_random_signals(max_seconds=20):
    # Randomly select a file between 1 and 4
    file_id = np.random.randint(1, 5)
    # Load the data
    f = h5py.File(os.path.join(RAW_DATA_PATH, f'Part_{file_id}.mat'), 'r')
    key = f'Part_{file_id}'
    # Randomly select a record
    record_id = np.random.randint(len(f[key]))
    # Load the signals
    ppg = []
    abp = []
    sample_length = len(f[f[key][record_id][0]])
    if sample_length > max_seconds * fs:
        sample_length = max_seconds * fs
    for j in tqdm(range(sample_length), desc=f'Reading Samples from Record {record_id}'):
        ppg.append(f[f[key][record_id][0]][j][Signal.PPG])
        abp.append(f[f[key][record_id][0]][j][Signal.ABP])

    # Plot two signal as subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(ppg)
    plt.title('PPG')
    plt.subplot(2, 1, 2)
    plt.plot(abp)
    plt.title('ABP')
    plt.tight_layout()
    plt.show()

def plot_bp_histogram():
    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'r')
    segment_amount = len(f['data'])
    sbps = []
    dbps = []
    for i in tqdm(range(segment_amount), desc='Reading Data'):
        sbps.append(f['data'][i][-2])
        dbps.append(f['data'][i][-1])

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.hist(sbps, bins=50)

    plt.title('SBP')

    plt.subplot(2, 1, 2)
    plt.hist(dbps, bins=50)

    plt.title('DBP')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plot_random_signals()
    plot_bp_histogram()
