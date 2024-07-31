import os
from tqdm import tqdm
import pickle
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np

from Models import models
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
def train_model(outputs_path=OUTPUTS_PATH, writer=None):
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    if not os.path.exists(os.path.join(outputs_path, 'Models')):
        os.makedirs(os.path.join(outputs_path, 'Models'))

    set_all_seeds(SEED, IS_CUDA)
    print("Using GPU" if IS_CUDA else "Using CPU")

    # Keep track of the best fold
    best_fold_loss = float('inf')
    best_fold_id = 0

    for fold_id in range(USED_FOLD_AMOUNT):
        print(f"########################## Training model for fold {fold_id} ##########################")
        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'stats_{fold_id}.pkl'), 'rb'))
        mean_sbp = data['mean_sbp']
        mean_dbp = data['mean_dbp']
        if OUTPUT_NORMALIZED:
            min_sbp = data['min_sbp']
            max_sbp = data['max_sbp']
            min_dbp = data['min_dbp']
            max_dbp = data['max_dbp']
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
            if CALIBRATION_FREE:
                cal_dataset = PulseDBDataset(os.path.join(FOLDED_DATASET_PATH, f'cal_{fold_id}.pkl'),
                                            signals=SIGNALS_LIST,
                                            demographics=DEMOGRAPHICS_LIST,
                                            targets=TARGETS_LIST,
                                            transform=transform)
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
        else:
            raise ValueError(f'Unknown dataset name: {DATASET_NAME}')
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
        if CALIBRATION_FREE:
            cal_loader = Data.DataLoader(
                dataset=cal_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=IS_CUDA  # Pytorch Recommendation
            )

        # Define the model
        if BIAS_INIT:
            model = getattr(models, MODEL)(bias_init=torch.tensor([mean_sbp, mean_dbp])).cuda() if IS_CUDA \
                else getattr(models, MODEL)(bias_init=torch.tensor([mean_sbp, mean_dbp]))
        else:
            model = getattr(models, MODEL)().cuda() if IS_CUDA else getattr(models, MODEL)()
        loss_fn = nn.L1Loss().cuda() if IS_CUDA else nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
        #                                                      factor=LR_SCHEDULER_FACTOR,
        #                                                       patience=LR_SCHEDULER_PATIENCE)

        # Early stopping
        best_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE

        # Train the model
        for epoch in tqdm(range(EPOCHS), desc=f'Fold {fold_id} - Training Epochs'):
            model.train()
            train_loss = 0
            train_mse = 0
            train_mae = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, f'Fold {fold_id} Epoch {epoch} - Training')):
                if IS_CUDA:
                    for key in batch['inputs']:
                        batch['inputs'][key] = batch['inputs'][key].cuda()
                    batch['targets'] = batch['targets'].cuda()
                # Zero the gradients (Optimization suggested by PyTorch)
                for param in model.parameters():
                    param.grad = None
                y_pred = model(batch['inputs'])                 # Forward pass
                loss = loss_fn(y_pred, batch['targets'])        # Compute the loss
                loss.backward()                                 # Backward pass

                if batch_idx == 0: # Start of batch
                    gradient_norm_dict = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradient_norm_dict[name] = param.grad.detach().norm().item()
                    if writer is not None:
                        for name, norm in gradient_norm_dict.items():
                            writer.add_scalar(f'{MODEL}/{fold_id}/Gradient Norm/{name}', norm, epoch)

                optimizer.step()                                # Update the weights
                with torch.no_grad():
                    train_loss += loss.item() * len(batch['targets'])
                    train_mse += torch.sum((y_pred - batch['targets']) ** 2).item()
                    train_mae += torch.sum(torch.abs(y_pred - batch['targets'])).item()

            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            train_mse /= len(train_loader.dataset)

            if writer is not None:
                writer.add_scalar(f'{MODEL}/{fold_id}/Loss/Train', train_loss, epoch)
                writer.add_scalar(f'{MODEL}/{fold_id}/MSE/Train', train_mse, epoch)
                writer.add_scalar(f'{MODEL}/{fold_id}/MAE/Train', train_mae, epoch)

            if FOLD_AMOUNT != 1:
                # If we are doing cross-validation
                model.eval()
                eval_loss = 0
                eval_mse = 0
                eval_mae = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, f'Fold {fold_id} Epoch {epoch} - Validation'):
                        if IS_CUDA:
                            for key in batch['inputs']:
                                batch['inputs'][key] = batch['inputs'][key].cuda()
                            batch['targets'] = batch['targets'].cuda()
                        y_pred = model(batch['inputs'])
                        eval_loss += loss_fn(y_pred, batch['targets']).item() * len(batch['targets'])
                        eval_mse += torch.sum((y_pred - batch['targets']) ** 2).item()
                        eval_mae += torch.sum(torch.abs(y_pred - batch['targets'])).item()

                eval_mse /= len(val_loader.dataset)
                eval_mae /= len(val_loader.dataset)
                eval_loss /= len(val_loader.dataset)
                #scheduler.step(eval_mse)

                if writer is not None:
                    writer.add_scalar(f'{MODEL}/{fold_id}/Loss/Validation', eval_loss, epoch)
                    writer.add_scalar(f'{MODEL}/{fold_id}/MSE/Validation', eval_mse, epoch)
                    writer.add_scalar(f'{MODEL}/{fold_id}/MAE/Validation', eval_mae, epoch)

                if CALIBRATION_FREE:
                    eval_loss = 0
                    eval_mse = 0
                    eval_mae = 0
                    with torch.no_grad():
                        for batch in tqdm(cal_loader, f'Fold {fold_id} Epoch {epoch} - Calibration'):
                            if IS_CUDA:
                                for key in batch['inputs']:
                                    batch['inputs'][key] = batch['inputs'][key].cuda()
                                batch['targets'] = batch['targets'].cuda()
                            y_pred = model(batch['inputs'])
                            eval_loss += loss_fn(y_pred, batch['targets']).item() * len(batch['targets'])
                            eval_mse += torch.sum((y_pred - batch['targets']) ** 2).item()
                            eval_mae += torch.sum(torch.abs(y_pred - batch['targets'])).item()

                    eval_mse /= len(cal_loader.dataset)
                    eval_mae /= len(cal_loader.dataset)
                    eval_loss /= len(cal_loader.dataset)
                    # scheduler.step(eval_mse)

                    if writer is not None:
                        writer.add_scalar(f'{MODEL}/{fold_id}/Loss/Calibration', eval_loss, epoch)
                        writer.add_scalar(f'{MODEL}/{fold_id}/MSE/Calibration', eval_mse, epoch)
                        writer.add_scalar(f'{MODEL}/{fold_id}/MAE/Calibration', eval_mae, epoch)

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

            del batch, y_pred
            if IS_CUDA:
                torch.cuda.empty_cache()

        print(f"########################## Fold {fold_id} - Training Complete ##########################")
        # If no model saved yet
        if not os.path.exists(os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt')):
            torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))

    return best_fold_id

if __name__ == '__main__':
    train_model()