import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np
from dotenv import load_dotenv
import scipy.signal as signal

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

SEGMENTED_DATA_PATH = os.path.join(os.getenv('SEGMENTED_DATA_PATH'), 'PeakDetect_NoHRDetect')
DATASET_PATH = os.getenv('DATASET_PATH')

SEED = int(os.getenv('SEED'))
MAX_DATASET_SIZE = int(os.getenv('MAX_DATASET_SEGMENTS'))
fs = int(os.getenv('SAMPLING_FREQUENCY'))

SAMPLES_PER_SEGMENT = int(os.getenv('SAMPLES_PER_SEGMENT'))
SAMPLES_PER_STRIDE = int(os.getenv('SAMPLES_PER_STRIDE'))

def create_dataset(max_amount=-1):
    if not os.path.isdir(SEGMENTED_DATA_PATH):
        print('Segmented data folder not found')
        return
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Get all the filenames
    files = os.listdir(os.path.join(SEGMENTED_DATA_PATH, 'ppgs'))

    # Clip the number of records to process
    if max_amount > len(files) or max_amount == -1:
        max_amount = len(files)
    files = files[:max_amount]

    data = []

    # Chebychev filter
    sos = signal.cheby2(N=4, rs=20, Wn=[0.5, 8], btype='bandpass', fs=fs, output='sos')

    empty_files = 0
    out_of_range_segments = 0
    usable_segments = 0
    for filename in tqdm(files):
        ppg = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', filename), 'rb'))
        abp = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'abps', filename), 'rb'))
        if len(ppg) == 0:
            empty_files += 1
            continue
        sbp_peaks = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', filename), 'rb'))
        dbp_peaks = pickle.load(open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', filename), 'rb'))

        ppg = signal.sosfiltfilt(sos, ppg)
        vpg = np.gradient(ppg)
        apg = np.gradient(vpg)

        for j in tqdm(range(0, len(ppg) - SAMPLES_PER_SEGMENT, SAMPLES_PER_STRIDE),
                      desc=f'Segmenting Record {filename}'):
            sbps = [abp[peak] for peak in sbp_peaks if peak >= j and peak < j + SAMPLES_PER_SEGMENT]
            dbps = [abp[peak] for peak in dbp_peaks if peak >= j and peak < j + SAMPLES_PER_SEGMENT]

            if len(sbps) == 0 or len(dbps) == 0:
                out_of_range_segments += 1
                continue

            max_sbp, min_sbp = max(sbps), min(sbps)
            max_dbp, min_dbp = max(dbps), min(dbps)

            if min_sbp <= 80 or min_dbp <= 60:
                out_of_range_segments += 1
                continue
            if max_sbp >= 180 or max_dbp >= 130:
                out_of_range_segments += 1
                continue

            ppg_segment = ppg[j:j + SAMPLES_PER_SEGMENT]
            vpg_segment = vpg[j:j + SAMPLES_PER_SEGMENT]
            apg_segment = apg[j:j + SAMPLES_PER_SEGMENT]

            # Min-max normalization of ppg, vpg, apg
            ppg_segment = (ppg_segment - np.min(ppg_segment)) / (np.max(ppg_segment) - np.min(ppg_segment))
            vpg_segment = (vpg_segment - np.min(vpg_segment)) / (np.max(vpg_segment) - np.min(vpg_segment))
            apg_segment = (apg_segment - np.min(apg_segment)) / (np.max(apg_segment) - np.min(apg_segment))

            mean_sbp = np.mean(sbps)
            mean_dbp = np.mean(dbps)

            usable_segments += 1
            data.append(np.concatenate([ppg_segment, vpg_segment, apg_segment, [mean_sbp, mean_dbp]]))

    print(f"Empty files: {empty_files}")
    print(f"Out of range segments: {out_of_range_segments}")
    print(f"Usable segments: {usable_segments}")

    # Set the random seed
    np.random.seed(SEED)
    # Randomly shuffle the files now so we can split data later easier
    np.random.shuffle(data)

    f = h5py.File(os.path.join(DATASET_PATH, f'dataset_{MAX_DATASET_SIZE}.hdf5'), 'w')
    dset = f.create_dataset('data', data=data)

if __name__ == '__main__':
    create_dataset(max_amount=MAX_DATASET_SIZE)