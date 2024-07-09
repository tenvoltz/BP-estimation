import os
import pickle
from tqdm import tqdm
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from helper import *
from metrics import evaluate_metrics
from Models import models
from data_loader import *

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')
OUTPUTS_PATH = os.getenv('OUTPUTS_PATH')

MODEL = os.getenv('MODEL')
MODEL_VERSION = os.getenv('MODEL_VERSION')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

SEED = int(os.getenv('SEED'))
IS_CUDA = torch.cuda.is_available() and os.getenv('CUDA').lower() == 'true'

OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'
SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
DEMOGRAPHICS_LIST = [demographic.strip().lower() for demographic in os.getenv('DEMOGRAPHICS').split(',')] \
    if os.getenv('DEMOGRAPHICS') is not None else None
TARGETS_LIST = [target.strip().lower() for target in os.getenv('TARGETS').split(',')]
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))

TEST_MODE = os.getenv('TEST_MODE').lower() == 'true'

def evaluate_model(writer=None, model_path=OUTPUTS_PATH, version=MODEL_VERSION, test_mode=TEST_MODE):
    set_all_seeds(SEED, IS_CUDA)
    print("Using GPU" if IS_CUDA else "Using CPU")
    print(f"########################## Evaluating model for fold {version} ##########################")

    if test_mode:
        print("Remember to add code to normalize the input")
        return None

    data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'stats_{version}.pkl'), 'rb'))
    if OUTPUT_NORMALIZED:
        min_sbp = data['min_sbp']
        max_sbp = data['max_sbp']
        min_dbp = data['min_dbp']
        max_dbp = data['max_dbp']

    # Load the test dataset
    file_name = f'test.pkl' if test_mode else f'val_{version}.pkl'
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
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load the model
    model = getattr(models, MODEL)().cuda() if IS_CUDA else getattr(models, MODEL)()
    model.load_state_dict(torch.load(os.path.join(model_path, 'Models', f'{MODEL}_{version}.pt')))
    model.eval()

    y_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            if IS_CUDA:
                for key in batch['inputs']:
                    batch['inputs'][key] = batch['inputs'][key].cuda()
            y_pred = model(batch['inputs'])
            if OUTPUT_NORMALIZED:
                # Denormalize the output
                y_pred[:, 0] = y_pred[:, 0] * (max_sbp - min_sbp) + min_sbp
                y_pred[:, 1] = y_pred[:, 1] * (max_dbp - min_dbp) + min_dbp
            y_preds.append(y_pred)
        if IS_CUDA:
            torch.cuda.empty_cache()
    y_preds = torch.cat(y_preds, dim=0).cpu()

    # Evaluate the model
    y_preds = y_preds.numpy()
    y_test = test_dataset.targets
    ieee_fig, aami_fig, bhs_fig, sample_fig = evaluate_metrics(y_test, y_preds)

    # Make histograms of the predictions and true values
    hist_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds[:, 0], bins=50, alpha=0.5, label='SBP Prediction', color='red')
    ax[0].legend()
    ax[0].set_title('SBP Histogram')
    ax[1].hist(y_preds[:, 1], bins=50, alpha=0.5, label='DBP Prediction', color='blue')
    ax[1].legend()
    ax[1].set_title('DBP Histogram')

    if writer is not None:
        writer.add_figure('IEEE', ieee_fig)
        writer.add_figure('AAMI', aami_fig)
        writer.add_figure('BHS', bhs_fig)
        writer.add_figure('Sample', sample_fig)
        writer.add_figure('Histogram', hist_fig)
        writer.add_histogram('SBP', y_pred[:, 0], 0)
        writer.add_histogram('DBP', y_pred[:, 1], 0)
    else:
        plt.show()


if __name__ == '__main__':
    model_path = os.path.join(OUTPUTS_PATH, 'Transformer_Only_2024-07-07_17-12-11')
    if MODEL_VERSION is not None:
        evaluate_model(model_path=model_path, test_mode=TEST_MODE)
    else:
        for i in range(FOLD_AMOUNT):
            evaluate_model(version=str(i), model_path = model_path, test_mode=TEST_MODE)