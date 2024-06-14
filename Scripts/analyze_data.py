import os
import pickle
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm

class Signal:
    PPG = 0
    ABP = 1
    ECG = 2

RAW_DATA_PATH = '../Data/raw_data'
RESULTS_PATH = '../Results'
FOLD_AMOUNT = 5
fs = 125 # Sampling frequency

def plot_mse_history():
    for fold_id in range(FOLD_AMOUNT):
        mse_history = pickle.load(open(os.path.join(RESULTS_PATH, f'mse_{fold_id}.pkl'), 'rb'))
        plt.plot(mse_history, label=f'Fold {fold_id}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

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

if __name__ == '__main__':
    plot_mse_history()
    plot_random_signals()
