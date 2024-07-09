import os
import pickle
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import scipy.signal as signal
import pywt

class Signal:
    PPG = 0
    ABP = 1
    ECG = 2

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))
fs = int(os.getenv('SAMPLING_FREQUENCY'))


def calc_baseline(signal):
    """
    Retrieved: https://github.com/spebern/py-bwr/blob/master/bwr.py#L5C1-L39C35
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

def plot_random_signals(max_seconds=20, record_id=None, file_id=None):
    # Randomly select a file between 1 and 4
    if file_id is None:
        file_id = np.random.randint(1, 5)
    # Load the data
    f = h5py.File(os.path.join(RAW_DATA_PATH, f'Part_{file_id}.mat'), 'r')
    key = f'Part_{file_id}'
    # Randomly select a record
    if record_id is None:
        record_id = np.random.randint(len(f[key]))
    # Load the signals
    ppg = []
    abp = []
    sample_length = len(f[f[key][record_id][0]])
    if sample_length > max_seconds * fs:
        sample_length = max_seconds * fs
    for j in tqdm(range(sample_length), desc=f'Reading Samples from Record {record_id} from File {file_id}'):
        ppg.append(f[f[key][record_id][0]][j][Signal.PPG])
        abp.append(f[f[key][record_id][0]][j][Signal.ABP])

    # Perform a band-pass fourth-
    # order Butterworth filter with cutoff frequencies of 0.5 Hz to 8 Hz

    b, a = signal.butter(N=4, Wn=[0.5, 8], btype='bandpass', fs=fs)
    new_ppg = signal.filtfilt(b, a, ppg)
    baseline = calc_baseline(new_ppg)
    new_ppg = new_ppg - baseline

    # Apply z-score normalization
    new_ppg = (new_ppg - np.mean(new_ppg)) / np.std(new_ppg)


    # Plot two signal as subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(ppg)
    plt.title('PPG')
    plt.subplot(3, 1, 2)
    plt.plot(new_ppg)
    plt.title('New PPG')
    plt.subplot(3, 1, 3)
    plt.plot(abp)
    plt.title('ABP')
    plt.tight_layout()

# Example 2542

if __name__ == '__main__':
    for file_id in range(1, 5):
        plot_random_signals(record_id=2542, file_id=file_id)
    plt.show()
