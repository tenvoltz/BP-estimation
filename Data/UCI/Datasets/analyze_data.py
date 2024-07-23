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
SEGMENTED_DATA_PATH = os.path.join(os.getenv('SEGMENTED_DATA_PATH'), 'PeakDetect_NoHRDetect')


def plot_peak_detect(max_seconds=20, record_id=None, filename=None):
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))
    if record_id is None and filename is None:
        record_id = np.random.randint(len(files))
    if filename is None:
        filename = files[record_id]

    ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
    abp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'abps', filename), 'rb'))
    sbp_peaks = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', filename), 'rb'))
    dbp_peaks = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', filename), 'rb'))

    if len(ppg) > max_seconds * fs:
        ppg = ppg[:max_seconds * fs]
        abp = abp[:max_seconds * fs]
        sbp_peaks = [peak for peak in sbp_peaks if peak < max_seconds * fs]
        dbp_peaks = [peak for peak in dbp_peaks if peak < max_seconds * fs]

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ppg)
    plt.scatter(sbp_peaks, [ppg[peak] for peak in sbp_peaks], c='r', label='SBP Peaks')
    plt.scatter(dbp_peaks, [ppg[peak] for peak in dbp_peaks], c='g', label='DBP Peaks')
    plt.title('PPG')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(abp)
    plt.scatter(sbp_peaks, [abp[peak] for peak in sbp_peaks], c='#F5C30E', label='SBP Peaks')
    plt.scatter(dbp_peaks, [abp[peak] for peak in dbp_peaks], c='#002D62', label='DBP Peaks')
    plt.title('ABP')
    plt.legend()
    plt.tight_layout()
    print(f'Filename: {filename}')


def plot_cheby2_filter():
    sos = signal.cheby2(N=4, rs=20, Wn=[0.5, 8], btype='bandpass', fs=fs, output='sos')
    w, h = signal.sosfreqz(sos, fs=fs)
    plt.figure(figsize=(10, 8))
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Cheby2 Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.axvline(0.5, color='green')  # cutoff frequency
    plt.axvline(8, color='green')  # cutoff frequency
    plt.axhline(-20, color='green')  # rs
    plt.grid(which='both', axis='both')
    plt.show()


def plot_random_signals(max_seconds=20, record_id=None, filename=None):
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))
    if record_id is None and filename is None:
        record_id = np.random.randint(len(files))
    if filename is None:
        filename = files[record_id]

    ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
    abp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'abps', filename), 'rb'))
    sos = signal.cheby2(N=4, rs=20, Wn=[0.5, 8], btype='bandpass', fs=fs, output='sos')
    filter_ppg = signal.sosfiltfilt(sos, ppg)

    if len(ppg) > max_seconds * fs:
        ppg = ppg[:max_seconds * fs]
        filter_ppg = filter_ppg[:max_seconds * fs]
        abp = abp[:max_seconds * fs]

    plt.figure(figsize=(10, 8), num=f'Random Signals {record_id}, which is {filename}')
    plt.subplot(2, 1, 1)
    plt.plot(ppg)
    plt.title('PPG')
    plt.subplot(2, 1, 2)
    plt.plot(filter_ppg)
    plt.title('Filter PPG')
    plt.tight_layout()
    print(f'Filename: {filename}')

# Example 2542

# Part_1_15.pkl
# Part_2_2118.pkl
# Part_3_586.pkl
# Part_2_741.pkl
# Part_2_434.pkl

if __name__ == '__main__':
    #plot_cheby2_filter()
    plot_random_signals(filename='Part_1_15.pkl')
    #for i in range(5):
    #    plot_random_signals()
    plt.show()
