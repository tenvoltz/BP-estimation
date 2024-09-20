import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from enum import Enum
from dotenv import load_dotenv
from torchvision import transforms

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

CURRENT_DATASET_PATH = os.getenv('CURRENT_DATASET_PATH')
FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))
CALIBRATION_FREE = os.getenv('CALIBRATION_FREE').lower() == 'true'
SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
DEMOGRAPHICS_LIST = [demographic.strip().lower() for demographic in os.getenv('DEMOGRAPHICS').split(',')] \
    if os.getenv('DEMOGRAPHICS') is not None else None
TARGETS_LIST = [target.strip().lower() for target in os.getenv('TARGETS').split(',')]

class Tensorize(object):
    def __call__(self, sample):
        for key in sample['inputs']:
            sample['inputs'][key] = torch.tensor(sample['inputs'][key], dtype=torch.float32)
        sample['targets'] = torch.tensor(sample['targets'], dtype=torch.float32)
        return sample
class TrimMethod(Enum):
    START = 'start'
    END = 'end'
    MIDDLE = 'middle'
class Trim(object):
    def __init__(self, length, method=TrimMethod.MIDDLE):
        self.length = length
        self.method = method

    def __call__(self, sample):
        if self.method == TrimMethod.START:
            sample['inputs']['signals'] = sample['inputs']['signals'][:, :self.length]
        elif self.method == TrimMethod.END:
            sample['inputs']['signals'] = sample['inputs']['signals'][:, -self.length:]
        elif self.method == TrimMethod.MIDDLE:
            sample['inputs']['signals'] = sample['inputs']['signals'][:, (sample['inputs']['signals'].shape[1] - self.length) // 2:
                                                                         (sample['inputs']['signals'].shape[1] + self.length) // 2]
        return sample
class PulseDBDataset(Dataset):
    def __init__(self, path_name, signals=None, demographics=None, targets=None, transform=None):
        data = pickle.load(open(path_name, 'rb'))
        self.inputs = []
        self.targets = []
        for idx in range(len(data['signals'][signals[0]])):
            input_sample = {'signals': np.concatenate([data['signals'][signal][idx] for signal in signals], axis=0)}
            if demographics is not None:
                input_sample['demographics'] = np.stack([data['demographics'][demographic][idx] for demographic in demographics], axis=0)
            self.inputs.append(input_sample)
        if targets is not None:
            self.targets = np.stack([data['targets'][target] for target in targets], axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'inputs': self.inputs[idx] if len(self.inputs) > 0 else None,
                  'targets': self.targets[idx] if len(self.targets) > 0 else None}
        if self.transform:
            sample = self.transform(sample)
        return sample

class UCIDataset(Dataset):
    def __init__(self, path_name, signals=None, targets=None, transform=None):
        data = pickle.load(open(path_name, 'rb'))
        self.inputs = []
        self.targets = []
        for idx in range(len(data['signals'][signals[0]])):
            input_sample = {'signals': np.concatenate([data['signals'][signal][idx] for signal in signals], axis=0)}
            self.inputs.append(input_sample)
        if targets is not None:
            self.targets = np.stack([data['targets'][target] for target in targets], axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {'inputs': self.inputs[idx] if len(self.inputs) > 0 else None,
                  'targets': self.targets[idx] if len(self.targets) > 0 else None}
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_patient_amount():
    if DATASET_NAME == 'PulseDB':
        return pickle.load(open(os.path.join(CURRENT_DATASET_PATH, 'stats.pkl'), 'rb'))['number_of_subjects']
    elif DATASET_NAME == 'UCI':
        raise ValueError(f'This dataset does not have a known number of patients: {DATASET_NAME}')
    else:
        raise ValueError(f'Unknown dataset name: {DATASET_NAME}')
def load_train_dataset(fold_id):
    if DATASET_NAME == 'PulseDB':
        transform = transforms.Compose([Trim(SIGNAL_LENGTH, TrimMethod.START), Tensorize()])
        train_dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'),
                                       signals=SIGNALS_LIST,
                                       demographics=DEMOGRAPHICS_LIST,
                                       targets=TARGETS_LIST,
                                       transform=transform)
        val_dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'),
                                     signals=SIGNALS_LIST,
                                     demographics=DEMOGRAPHICS_LIST,
                                     targets=TARGETS_LIST,
                                     transform=transform)
        cal_dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, f'cal_{fold_id}.pkl'),
                                     signals=SIGNALS_LIST,
                                     demographics=DEMOGRAPHICS_LIST,
                                     targets=TARGETS_LIST,
                                     transform=transform) if CALIBRATION_FREE else None
    elif DATASET_NAME == 'UCI':
        transform = transforms.Compose([Tensorize()])
        train_dataset = UCIDataset(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'),
                                   signals=SIGNALS_LIST,
                                   targets=TARGETS_LIST,
                                   transform=transform)
        val_dataset = UCIDataset(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'),
                                 signals=SIGNALS_LIST,
                                 targets=TARGETS_LIST,
                                 transform=transform)
        cal_dataset = None
    else:
        raise ValueError(f'Unknown dataset name: {DATASET_NAME}')
    return train_dataset, val_dataset, cal_dataset

def load_test_dataset(file_name):
    if DATASET_NAME == 'PulseDB':
        test_dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, file_name),
                                      signals=SIGNALS_LIST,
                                      demographics=DEMOGRAPHICS_LIST,
                                      targets=TARGETS_LIST,
                                      transform=transforms.Compose([Trim(SIGNAL_LENGTH, TrimMethod.START),
                                                                    Tensorize()]))
    elif DATASET_NAME == 'UCI':
        test_dataset = UCIDataset(os.path.join(FOLDED_DATASET_PATH, file_name),
                                  signals=SIGNALS_LIST,
                                  targets=TARGETS_LIST,
                                  transform=transforms.Compose([Tensorize()]))
    else:
        raise ValueError(f'Unknown dataset name: {DATASET_NAME}')
    return test_dataset

if __name__ == '__main__':
    dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, 'val_0.pkl'),
                             signals=['ppg', 'vpg'],
                             targets=['sbp', 'dbp', 'subject'],
                             transform=transforms.Compose([Trim(1248, TrimMethod.MIDDLE),
                                                           Tensorize()]))
    # Print 10 random samples
    for i in range(10):
        print(dataset[i])
    # Plot ppg and vpg signal
    import matplotlib.pyplot as plt
    plt.plot(dataset[4]['inputs']['signals'][0])
    plt.plot(dataset[4]['inputs']['signals'][1])
    plt.show()


