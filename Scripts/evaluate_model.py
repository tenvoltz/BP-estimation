import os
import pickle
from tqdm import tqdm
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.stats import boxcox

from helper import *
from metrics import *
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
CALIBRATION_FREE = os.getenv('CALIBRATION_FREE').lower() == 'true'

def evaluate_model(writer=None, model_path=OUTPUTS_PATH, version=MODEL_VERSION, test_mode=TEST_MODE):
    set_all_seeds(SEED, IS_CUDA)
    print("Using GPU" if IS_CUDA else "Using CPU")
    print(f"########################## Evaluating model for fold {version} ##########################")

    data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'stats_{version}.pkl'), 'rb'))
    min_sbp = data['min_sbp']
    max_sbp = data['max_sbp']
    min_dbp = data['min_dbp']
    max_dbp = data['max_dbp']

    # Load the test dataset
    file_name = f'cal_{version}.pkl' if CALIBRATION_FREE else f'test.pkl' if test_mode else f'val_{version}.pkl'
    test_dataset = load_test_dataset(file_name)
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
                batch['inputs'] = {key: tensor.cuda() for key, tensor in batch['inputs'].items()}
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
    ieee_fig, aami_fig, bhs_fig, sample_fig = evaluate_metrics(y_preds, y_test)
    hist_fig = plot_histogram(y_preds, y_test)
    bland_altman_fig = plot_bland_altman(y_preds, y_test)
    hist_error_fig = plot_error_histogram(y_preds, y_test)
    ppg_error_fig = plot_top_5_error_signals(y_preds, y_test, test_dataset)

    if writer is not None:
        writer.add_figure('IEEE', ieee_fig)
        writer.add_figure('AAMI', aami_fig)
        writer.add_figure('BHS', bhs_fig)
        writer.add_figure('Sample', sample_fig)
        writer.add_figure('Histogram', hist_fig)
        writer.add_figure('Error', hist_error_fig)
        writer.add_figure('Top 5 PPG error', ppg_error_fig)
        writer.add_figure('Bland-Altman SBP', bland_altman_fig)
        writer.add_histogram('SBP', y_pred[:, 0], 0)
        writer.add_histogram('DBP', y_pred[:, 1], 0)
    else:
        plt.show()


if __name__ == '__main__':
    model_path = os.path.join(OUTPUTS_PATH, 'IMSF_Net_AllSegment_FDiTrans_5Layer')
    if MODEL_VERSION is not None:
        evaluate_model(model_path=model_path, test_mode=TEST_MODE)
    else:
        for i in range(FOLD_AMOUNT):
            evaluate_model(version=str(i), model_path = model_path, test_mode=TEST_MODE)