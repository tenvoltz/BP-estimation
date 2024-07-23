import os
import h5py
from tqdm import tqdm
import pickle
import numpy as np
from dotenv import load_dotenv
import scipy.signal as signal
import pywt
import heartpy as hp
from itertools import compress

load_dotenv()
ENV_PATH = os.getenv('ENV_PATH')
load_dotenv(os.path.join(ENV_PATH))

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
SEGMENTED_DATA_PATH = os.path.join(os.getenv('SEGMENTED_DATA_PATH'), 'PeakDetect_NoHRDetect')

fs = int(os.getenv('SAMPLING_FREQUENCY'))
SAMPLES_PER_SEGMENT = int(os.getenv('SAMPLES_PER_SEGMENT'))
SAMPLES_PER_STRIDE = int(os.getenv('SAMPLES_PER_STRIDE'))

MAX_PROCESSED = int(os.getenv('MAX_SEGMENTED_INSTANCES'))

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
    if not os.path.isdir(os.path.join(SEGMENTED_DATA_PATH, 'abps')):
        os.mkdir(os.path.join(SEGMENTED_DATA_PATH, 'abps'))

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
               os.path.isfile(os.path.join(SEGMENTED_DATA_PATH, 'dbps', f'Part_{file_id}_{record_id}.pkl')) and \
               os.path.isfile(os.path.join(SEGMENTED_DATA_PATH, 'abps', f'Part_{file_id}_{record_id}.pkl')):
                continue

            ppg = []
            abp = []

            sample_length = len(f[f[key][record_id][0]])

            # Skip if the record is less than 8 minutes
            if sample_length < fs * 60 * 8:
                print(f"Skipping record {record_id} due to length less than 8 minutes")
                continue

            data = np.array(f[f[key][record_id][0]]).reshape(-1, 3)
            ppg = data[:, Signal.PPG]
            abp = data[:, Signal.ABP]

            try:
                abp_points = hp.process(abp, fs)
            except hp.exceptions.BadSignalWarning:
                continue

            is_valid_peaks = abp_points[0]['binary_peaklist']
            systolic_peaks = abp_points[0]['peaklist']
            systolic_peaks = list(compress(systolic_peaks, is_valid_peaks == 1))
            diastolic_peaks = find_minima(abp, systolic_peaks)

            pickle.dump(np.array(ppg),
                        open(os.path.join(SEGMENTED_DATA_PATH, 'ppgs', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(abp),
                        open(os.path.join(SEGMENTED_DATA_PATH, 'abps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(systolic_peaks),
                        open(os.path.join(SEGMENTED_DATA_PATH, 'sbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))
            pickle.dump(np.array(diastolic_peaks),
                        open(os.path.join(SEGMENTED_DATA_PATH, 'dbps', f'Part_{file_id}_{record_id}.pkl'), 'wb'))

            """
            for j in tqdm(range(0, len(ppg) - SAMPLES_PER_SEGMENT, SAMPLES_PER_STRIDE), desc=f'Segmenting Record {record_id}/{record_amount}'):
                ppg_segment = ppg[j:j + SAMPLES_PER_SEGMENT]
                abp_segment = abp[j:j + SAMPLES_PER_SEGMENT]

                # Skip if ppg is empty
                if len(ppg_segment) == 0:
                    continue
                # Skip if ppg has nan
                if np.isnan(ppg_segment).any():
                    continue
                # Skip if abp is empty
                if len(abp_segment) == 0:
                    continue
                # Skip if abp has nan
                if np.isnan(abp_segment).any():
                    continue

                if len(systolic_peaks) == 0 or len(diastolic_peaks) == 0:
                    continue
                max_sbp = max(abp_segment[systolic_peaks])
                max_dbp = min(abp_segment[diastolic_peaks])
                min_sbp = min(abp_segment[systolic_peaks])
                min_dbp = max(abp_segment[diastolic_peaks])
                if min_sbp <= 80 or min_dbp <= 60:
                    out_of_range_segments += 1
                    continue
                if max_sbp >= 180 or max_dbp >= 130:
                    out_of_range_segments += 1
                    continue

                #ppg_segments.append(ppg_segment)
                #abp_segments.append(abp_segment)
                # Calculate the mean of the systolic and diastolic peaks
                #sbp_segments.append(np.mean(abp_segment[systolic_peaks]))
                #dbp_segments.append(np.mean(abp_segment[diastolic_peaks]))
                # Get max and min of the segment
                #sbp_segments.append(max(abp_segment))
                #dbp_segments.append(min(abp_segment))
                usable_segments += 1

        current_processed += record_amount

    print(f'Processed {current_processed} records')
    print(f'Out of range segments: {out_of_range_segments}')
    print(f'Usable segments: {usable_segments}')
    """

def find_minima(signal, peaks):
    min_pks = []
    for i in range(0,len(peaks) - 1):
        pks_curr = peaks[i]
        pks_next = peaks[i+1]

        signal_window = signal[pks_curr:pks_next]
        min_pks.append(np.argmin(signal_window) + pks_curr)
    return min_pks

if __name__ == '__main__':
    process_data(max_processed=MAX_PROCESSED)



