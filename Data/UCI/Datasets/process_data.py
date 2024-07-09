import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np
from dotenv import load_dotenv
import scipy.signal as signal
import pywt

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
SEGMENTED_DATA_PATH = os.getenv('SEGMENTED_DATA_PATH')

fs = int(os.getenv('SAMPLING_FREQUENCY'))
SAMPLES_PER_SEGMENT = int(os.getenv('SAMPLES_PER_SEGMENT'))
SAMPLES_PER_STRIDE = int(os.getenv('SAMPLES_PER_STRIDE'))

MAX_PROCESSED = int(os.getenv('MAX_SEGMENTED_INSTANCES'))

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
    b, a = signal.butter(N=4, Wn=[0.5, 8], btype='bandpass', fs=fs)
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

            for j in tqdm(range(0, len(ppg) - SAMPLES_PER_SEGMENT, SAMPLES_PER_STRIDE), desc=f'Segmenting Record {record_id}/{record_amount}'):
                temp_ppg = ppg[j:j + SAMPLES_PER_SEGMENT]
                temp_ppg = signal.filtfilt(b, a, temp_ppg)
                baseline = calc_baseline(temp_ppg)
                temp_ppg = temp_ppg - baseline
                ppg_segments.append(temp_ppg)
                sbp_segments.append(max(abp[j:j + SAMPLES_PER_SEGMENT]))
                dbp_segments.append(min(abp[j:j + SAMPLES_PER_SEGMENT]))

            pickle.dump(np.array(ppg_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(sbp_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(dbp_segments), open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))

        current_processed += record_amount



if __name__ == '__main__':
    process_data(max_processed=MAX_PROCESSED)



