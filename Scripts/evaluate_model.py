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

    """
    max_pred_sbp = np.max(y_preds[:, 0])
    min_pred_sbp = np.min(y_preds[:, 0])
    max_pred_dbp = np.max(y_preds[:, 1])
    min_pred_dbp = np.min(y_preds[:, 1])
    print(f"SBP: {min_pred_sbp} - {max_pred_sbp}")
    print(f"DBP: {min_pred_dbp} - {max_pred_dbp}")

    assert len(y_test) == len(y_preds)
    print(f"Number of samples: {len(y_test)}")
    print(f"Number of predictions: {len(y_preds)}")

    # Check for invalid predictions
    invalid_preds = np.where(np.isnan(y_preds))
    print(f"Invalid predictions: {len(invalid_preds[0])}")
    if len(invalid_preds[0]) > 0:
        print(f"Invalid predictions: {y_preds[invalid_preds]}")

    # Check for invalid dbp predictions like nan
    invalid_dbp_preds = np.where(np.isnan(y_preds[:, 1]))
    print(f"Invalid DBP predictions: {len(invalid_dbp_preds[0])}")
    if len(invalid_dbp_preds[0]) > 0:
        print(f"Invalid DBP predictions: {y_preds[invalid_dbp_preds]}")
    """
    # Make histograms of the predictions and true values
    hist_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds[:, 0], bins=50, alpha=0.5, label='SBP', color='red', density = True)
    #ax[0].hist(y_test[:, 0], bins=50, alpha=0.5, label='SBP True', color='green', density = True)
    ax[0].legend()
    ax[0].set_title('SBP Histogram')
    ax[1].hist(y_preds[:, 1], bins=50, alpha=0.5, label='DBP', color='blue', density = True)
    #ax[1].hist(y_test[:, 1], bins=50, alpha=0.5, label='DBP True', color='green', density = True)
    ax[1].legend()
    ax[1].set_title('DBP Histogram')
    """
    # Log-transform the histograms
    y_test_log = np.log2(y_test)
    y_preds_log = np.log2(y_preds)
    hist_fig2, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds_log[:, 0], bins=50, alpha=0.5, label='SBP', color='red', density=True)
    ax[0].hist(y_test_log[:, 0], bins=50, alpha=0.5, label='SBP True', color='green', density=True)
    ax[0].legend()
    ax[0].set_title('SBP Histogram 2')
    ax[1].hist(y_preds_log[:, 1], bins=50, alpha=0.5, label='DBP', color='blue', density=True)
    ax[1].hist(y_test_log[:, 1], bins=50, alpha=0.5, label='DBP True', color='green', density=True)
    ax[1].legend()
    ax[1].set_title('DBP Histogram 2')
    # Sqrt-transform the histograms
    y_test_sqrt = np.sqrt(y_test)
    y_preds_sqrt = np.sqrt(y_preds)
    hist_fig3, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds_sqrt[:, 0], bins=50, alpha=0.5, label='SBP', color='red', density=True)
    ax[0].hist(y_test_sqrt[:, 0], bins=50, alpha=0.5, label='SBP True', color='green', density=True)
    ax[0].legend()
    ax[0].set_title('SBP Histogram 3')
    ax[1].hist(y_preds_sqrt[:, 1], bins=50, alpha=0.5, label='DBP', color='blue', density=True)
    ax[1].hist(y_test_sqrt[:, 1], bins=50, alpha=0.5, label='DBP True', color='green', density=True)
    ax[1].legend()
    ax[1].set_title('DBP Histogram 3')
    # Box-Cox transform the histograms for sbp and dbp separately
    y_test_boxcox_sbp, y_test_lmbda_sbp = boxcox(y_test[:, 0])
    y_preds_boxcox_sbp = boxcox(y_preds[:, 0], y_test_lmbda_sbp)
    y_test_boxcox_dbp = boxcox(y_test[:, 1], -3)
    y_preds_boxcox_dbp = boxcox(y_preds[:, 1], -3)
    hist_fig4, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds_boxcox_sbp, bins=50, alpha=0.5, label=f'SBP with lambda: {y_test_lmbda_sbp}', color='red')
    ax[0].hist(y_test_boxcox_sbp, bins=50, alpha=0.5, label=f'SBP True with lambda: {y_test_lmbda_sbp}', color='green')
    ax[0].legend()
    ax[0].set_title(f'SBP Histogram 4')
    ax[1].hist(y_preds_boxcox_dbp, bins=50, alpha=0.5, label=f'DBP with lambda: {-3}', color='blue')
    ax[1].hist(y_test_boxcox_dbp, bins=50, alpha=0.5, label=f'DBP True with lambda: {-3}', color='green')
    ax[1].legend()
    ax[1].set_title(f'DBP Histogram 4')
    print(y_preds_boxcox_dbp[:10])
    print(y_test_boxcox_dbp[:10])
    # Assign box cox sbp and dbp
    y_preds = np.concatenate((y_preds_boxcox_sbp.reshape(-1, 1), y_preds_boxcox_dbp.reshape(-1, 1)), axis=1)
    y_test = np.concatenate((y_test_boxcox_sbp.reshape(-1, 1), y_test_boxcox_dbp.reshape(-1, 1)), axis=1)
    """
    # Make histogram of error
    error = y_preds - y_test
    hist_error_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(error[:, 0], bins=50, alpha=0.5, label='SBP Error', color='red')
    ax[0].legend()
    ax[0].set_title('SBP Error Histogram')
    ax[1].hist(error[:, 1], bins=50, alpha=0.5, label='DBP Error', color='blue')
    ax[1].legend()
    ax[1].set_title('DBP Error Histogram')

    # Get the top 5 errors
    top_10_error = np.argsort(np.abs(error.sum(axis=1)))[-5:]
    # Draw the ppg signals
    ppg_error_fig, ax = plt.subplots(5, 1, figsize=(10, 20))
    for i, idx in enumerate(top_10_error):
        ppg = test_dataset.inputs[idx]['signals'][0]
        ax[i].plot(ppg)
        ax[i].set_title(f'PPG Signal {idx} with error {error[idx]}')
    plt.tight_layout()


    if writer is not None:
        writer.add_figure('IEEE', ieee_fig)
        writer.add_figure('AAMI', aami_fig)
        writer.add_figure('BHS', bhs_fig)
        writer.add_figure('Sample', sample_fig)
        writer.add_figure('Histogram', hist_fig)
        writer.add_figure('Error', hist_error_fig)
        writer.add_figure('Top 5 PPG error', ppg_error_fig)
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