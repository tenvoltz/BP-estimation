import os
import pickle
from tqdm import tqdm
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import numpy as np
import matplotlib.pyplot as plt

from metrics import evaluate_metrics, evaluate_classifier
from Models import models

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
OUTPUT_TASK = os.getenv('OUTPUT_TASK').lower()

SIGNAL_AMOUNT = int(os.getenv('SIGNAL_AMOUNT'))
SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))

TEST_MODE = os.getenv('TEST_MODE').lower() == 'true'

def evaluate_model(writer=None, model_path = OUTPUTS_PATH, version=MODEL_VERSION, test_mode=TEST_MODE):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if IS_CUDA:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        print("Using GPU")
    else:
        print("Using CPU")

    print(f"########################## Evaluating model for fold {version} ##########################")

    file_name = 'test.pkl' if test_mode else f'val_{version}.pkl'
    data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, file_name), 'rb'))
    x_test = data['x']
    y_test = data['y']
    data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'stats_{version}.pkl'), 'rb'))
    min_ppg = data['min_ppg']
    max_ppg = data['max_ppg']
    if OUTPUT_NORMALIZED:
        min_sbp = data['min_sbp']
        max_sbp = data['max_sbp']
        min_dbp = data['min_dbp']
        max_dbp = data['max_dbp']


    if DATASET_NAME == 'PulseDB':
        if SIGNAL_AMOUNT == 1:
            x_test = x_test['ppg']
        elif SIGNAL_AMOUNT == 2:
            x_test = np.concatenate((x_test['ppg'], x_test['vpg']), axis=1)
        elif SIGNAL_AMOUNT == 3:
            x_test = np.concatenate((x_test['ppg'], x_test['vpg'], x_test['apg']), axis=1)
        else:
            raise ValueError("Invalid signal amount")
    # Cut the signal length
    x_test = x_test[:, :, :SIGNAL_LENGTH]

    if test_mode and DATASET_NAME == 'Rafita':
        x_test = (x_test - min_ppg) / (max_ppg - min_ppg)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32 if OUTPUT_TASK == 'regression' else torch.int64)

    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = getattr(models, MODEL)().cuda() if IS_CUDA else getattr(models, MODEL)()
    model.load_state_dict(torch.load(os.path.join(model_path, 'Models', f'{MODEL}_{version}.pt')))
    model.eval()

    y_pred = []
    with torch.no_grad():
        for x_batch, _ in tqdm(test_loader, desc='Testing'):
            if IS_CUDA:
                x_batch = x_batch.cuda()
            y_pred_batch = model(x_batch)
            # Denormalize the output
            if OUTPUT_NORMALIZED:
                y_pred_batch[:, 0] = y_pred_batch[:, 0] * (max_sbp - min_sbp) + min_sbp
                y_pred_batch[:, 1] = y_pred_batch[:, 1] * (max_dbp - min_dbp) + min_dbp
            y_pred.append(y_pred_batch)
        if IS_CUDA:
            torch.cuda.empty_cache()
    y_pred = torch.cat(y_pred, dim=0).cpu()
    if OUTPUT_TASK == 'classification':
        predicted = (torch.sigmoid(y_pred) >= 0.5).long().squeeze()
        confusion_fig, roc_fig = evaluate_classifier(y_test.numpy(), predicted.numpy())
        if writer is not None:
            writer.add_figure('Confusion Matrix', confusion_fig)
            writer.add_figure('ROC Curve', roc_fig)
        else:
            plt.show()
    else:
        y_pred = y_pred.numpy()
        y_test = y_test.numpy()
        ieee_fig, aami_fig, bhs_fig, sample_fig = evaluate_metrics(y_test, y_pred)

        if writer is not None:
            writer.add_figure('IEEE', ieee_fig)
            writer.add_figure('AAMI', aami_fig)
            writer.add_figure('BHS', bhs_fig)
            writer.add_figure('Sample', sample_fig)
            writer.add_histogram('SBP', y_pred[:, 0], 0)
            writer.add_histogram('DBP', y_pred[:, 1], 0)
        else:
            plt.show()

if __name__ == '__main__':
    model_path = os.path.join(OUTPUTS_PATH, 'Transformer_Only_2024-06-27_12-25-55')
    if MODEL_VERSION is not None:
        evaluate_model(model_path = model_path, test_mode=TEST_MODE)
    else:
        for i in range(FOLD_AMOUNT):
            evaluate_model(version=str(i), model_path = model_path, test_mode=TEST_MODE)