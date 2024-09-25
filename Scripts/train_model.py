import os
from tqdm import tqdm
import pickle
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np

from Models import models
from Models.loss import *
from helper import *
from data_loader import *

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))

OUTPUTS_PATH = os.getenv('OUTPUTS_PATH')
MODEL = os.getenv('MODEL')
BIAS_INIT = os.getenv('BIAS_INIT').lower() == 'true'

LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
HUBER_LOSS_DELTA = float(os.getenv('HUBER_LOSS_DELTA'))

EARLY_STOPPING = os.getenv('EARLY_STOPPING').lower() == 'true'
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE'))

LR_SCHEDULER_PATIENCE = int(os.getenv('LR_SCHEDULER_PATIENCE'))
LR_SCHEDULER_FACTOR = float(os.getenv('LR_SCHEDULER_FACTOR'))

SEED = int(os.getenv('SEED'))
IS_CUDA = torch.cuda.is_available() and torch.backends.cudnn.enabled and os.getenv('CUDA').lower() == 'true'

SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
DEMOGRAPHICS_LIST = [demographic.strip().lower() for demographic in os.getenv('DEMOGRAPHICS').split(',')] \
    if os.getenv('DEMOGRAPHICS') is not None else None
TARGETS_LIST = [target.strip().lower() for target in os.getenv('TARGETS').split(',')]
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))

OUTPUT_NORMALIZED = os.getenv('OUTPUT_NORMALIZED').lower() == 'true'

USED_FOLD_AMOUNT = int(os.getenv('USED_FOLD_AMOUNT'))
CALIBRATION_FREE = os.getenv('CALIBRATION_FREE').lower() == 'true'
LOGGING_GRADIENT_NORM = os.getenv('LOGGING_GRADIENT_NORM').lower() == 'true'


def train_model(outputs_path=OUTPUTS_PATH, writer=None):
    # Create the outputs folder
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    if not os.path.exists(os.path.join(outputs_path, 'Models')):
        os.makedirs(os.path.join(outputs_path, 'Models'))

    # Set the environment
    set_all_seeds(SEED, IS_CUDA)
    print("Using GPU" if IS_CUDA else "Using CPU")

    # Keep track of the best fold
    best_fold_loss = float('inf')
    best_fold_id = 0

    for fold_id in range(USED_FOLD_AMOUNT):
        print(f"########################## Training model for fold {fold_id} ##########################")
        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'rb'))
        # For bias initialization
        mean_sbp = data['mean_sbp']
        mean_dbp = data['mean_dbp']
        # For output normalization - If used
        min_sbp = data['min_sbp']
        max_sbp = data['max_sbp']
        min_dbp = data['min_dbp']
        max_dbp = data['max_dbp']

        train_dataset, val_dataset, cal_dataset = load_train_dataset(fold_id)
        del data

        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=IS_CUDA  # Pytorch Recommendation
        )
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=IS_CUDA  # Pytorch Recommendation
        )
        cal_loader = Data.DataLoader(
            dataset=cal_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=IS_CUDA  # Pytorch Recommendation
        ) if CALIBRATION_FREE else None

        # Define the model
        model = getattr(models, MODEL)(bias_init=torch.tensor([mean_sbp, mean_dbp])) if BIAS_INIT else getattr(models, MODEL)()
        model = model.cuda() if IS_CUDA else model

        # Loss function, optimizer and scheduler
        mse_loss_fn = MSELoss().cuda() if IS_CUDA else MSELoss()    # For the metrics
        mae_loss_fn = MAELoss().cuda() if IS_CUDA else MAELoss()    # For the metrics
        main_loss = nn.MSELoss().cuda() if IS_CUDA else nn.MSELoss()
        subject_loss = SubjectLoss(num_classes=get_patient_amount(), feat_dim=model.feature_dim).cuda() if IS_CUDA \
            else SubjectLoss(num_classes=get_patient_amount(), feat_dim=model.feature_dim)
        bp_ranges = {0: [min_sbp, max_sbp], 1: [min_dbp, max_dbp]}
        class_loss = BPClassLoss(num_bins=[10, 10], ranges=bp_ranges, feature_dim=model.feature_dim).cuda() if IS_CUDA \
            else BinLoss(num_bins=[10, 10], ranges=bp_ranges, feature_dim=model.feature_dim)

        loss_dict = {
            'Main': (main_loss, 1),
            'Subject (features)': (subject_loss, 0.05),
            #'Class (features)': (class_loss, 0.05),
        }
        loss_fn = CombinedLoss(loss_dict).cuda() if IS_CUDA else CombinedLoss(loss_dict)

        optimizers = [
            torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
            torch.optim.Adam(subject_loss.parameters(), lr=0.5),
            #torch.optim.Adam(class_loss.parameters(), lr=0.001)
        ]

        schedulers = {
            torch.optim.lr_scheduler.MultiStepLR(optimizers[1], milestones=[10, 20, 30, 40], gamma=0.5),
        }
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

        # Early stopping
        best_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE

        for epoch in tqdm(range(EPOCHS), desc=f'Fold {fold_id} - Training Epochs'):
            train_loss, train_mse, train_mae = 0, 0, 0
            eval_loss, eval_mse, eval_mae = 0, 0, 0
            cal_loss, cal_mse, cal_mae = 0, 0, 0

            model.train()
            for batch_idx, batch in enumerate(tqdm(train_loader, f'Fold {fold_id} Epoch {epoch} - Training')):
                if IS_CUDA:
                    batch['inputs'] = {key: tensor.cuda() for key, tensor in batch['inputs'].items()}
                    batch['targets'] = batch['targets'].cuda()
                model.zero_grad()
                for loss_func, _ in loss_dict.values():
                    loss_func.zero_grad()
                y_pred, features = model(batch['inputs'])
                loss = loss_fn(y_pred, features, batch['targets'])
                loss.backward()
                if batch_idx == 0 and LOGGING_GRADIENT_NORM and writer is not None:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            norm = param.grad.detach().norm().item()
                            writer.add_scalar(f'{MODEL}/{fold_id}/Gradient Norm/{name}', norm, epoch)

                for loss_func, weight in loss_dict.values():
                    for param in loss_func.parameters():
                        param.grad.data *= 1. / weight

                for optimizer in optimizers:
                    optimizer.step()

                with torch.no_grad():
                    mse_loss = mse_loss_fn(y_pred, batch['targets'])
                    mae_loss = mae_loss_fn(y_pred, batch['targets'])
                    train_loss += loss.item() * len(batch['targets'])
                    train_mse += mse_loss.item()
                    train_mae += mae_loss.item()

            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, f'Fold {fold_id} Epoch {epoch} - Validation'):
                    if IS_CUDA:
                        batch['inputs'] = {key: tensor.cuda() for key, tensor in batch['inputs'].items()}
                        batch['targets'] = batch['targets'].cuda()
                    y_pred, features = model(batch['inputs'])
                    # Calculate the metrics
                    loss = loss_fn(y_pred, features, batch['targets'])
                    mse_loss = mse_loss_fn(y_pred, batch['targets'])
                    mae_loss = mae_loss_fn(y_pred, batch['targets'])
                    eval_loss += loss.item() * len(batch['targets'])
                    eval_mse += mse_loss.item()
                    eval_mae += mae_loss.item()

            if CALIBRATION_FREE:
                with torch.no_grad():
                    for batch in tqdm(cal_loader, f'Fold {fold_id} Epoch {epoch} - Calibration'):
                        if IS_CUDA:
                            batch['inputs'] = {key: tensor.cuda() for key, tensor in batch['inputs'].items()}
                            batch['targets'] = batch['targets'].cuda()
                        y_pred, features = model(batch['inputs'])
                        # Calculate the metrics
                        loss = loss_fn(y_pred, features, batch['targets'])
                        mse_loss = mse_loss_fn(y_pred, batch['targets'])
                        mae_loss = mae_loss_fn(y_pred, batch['targets'])
                        cal_loss += loss.item() * len(batch['targets'])
                        cal_mse += mse_loss.item()
                        cal_mae += mae_loss.item()

            num_train_samples = len(train_loader.dataset)
            train_loss /= num_train_samples
            train_mae /= num_train_samples
            train_mse /= num_train_samples

            num_val_samples = len(val_loader.dataset)
            eval_loss /= num_val_samples
            eval_mae /= num_val_samples
            eval_mse /= num_val_samples

            if CALIBRATION_FREE:
                num_cal_samples = len(cal_loader.dataset)
                cal_loss /= num_cal_samples
                cal_mae /= num_cal_samples
                cal_mse /= num_cal_samples

            if writer is not None:
                for metric, value in zip(['Loss', 'MSE', 'MAE'], [train_loss, train_mse, train_mae]):
                    writer.add_scalar(f'{MODEL}/{fold_id}/{metric}/Train', value, epoch)
                for metric, value in zip(['Loss', 'MSE', 'MAE'], [eval_loss, eval_mse, eval_mae]):
                    writer.add_scalar(f'{MODEL}/{fold_id}/{metric}/Validation', value, epoch)
                if CALIBRATION_FREE:
                    for metric, value in zip(['Loss', 'MSE', 'MAE'], [cal_loss, cal_mse, cal_mae]):
                        writer.add_scalar(f'{MODEL}/{fold_id}/{metric}/Calibration', value, epoch)

            for scheduler in schedulers:
                scheduler.step()

            if eval_loss < best_fold_loss:
                best_fold_id = fold_id
                best_fold_loss = eval_loss

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))
                    patience = EARLY_STOPPING_PATIENCE  # Reset patience counter
                elif EARLY_STOPPING:
                    patience -= 1
                    if patience == 0:
                        print(f"Early stopping at epoch {epoch}")
                        break

            del batch, y_pred, loss
            if IS_CUDA:
                torch.cuda.empty_cache()

        print(f"########################## Fold {fold_id} - Training Complete ##########################")
        # If no model saved yet
        if not os.path.exists(os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt')):
            torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))

    return best_fold_id

if __name__ == '__main__':
    train_model()