import os
from tqdm import tqdm
import pickle
import torch.utils.data as Data
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np

from Models import models

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))

OUTPUTS_PATH = os.getenv('OUTPUTS_PATH')
MODEL = os.getenv('MODEL')

LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))

EARLY_STOPPING = os.getenv('EARLY_STOPPING').lower() == 'true'
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE'))

LR_SCHEDULER_PATIENCE = int(os.getenv('LR_SCHEDULER_PATIENCE'))
LR_SCHEDULER_FACTOR = float(os.getenv('LR_SCHEDULER_FACTOR'))

SEED = int(os.getenv('SEED'))
IS_CUDA = torch.cuda.is_available() and torch.backends.cudnn.enabled and os.getenv('CUDA').lower() == 'true'

SIGNAL_AMOUNT = int(os.getenv('SIGNAL_AMOUNT'))
SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))

OUTPUT_TASK = os.getenv('OUTPUT_TASK').lower()

def train_model(outputs_path=OUTPUTS_PATH, writer=None):
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    if not os.path.exists(os.path.join(outputs_path, 'Models')):
        os.makedirs(os.path.join(outputs_path, 'Models'))

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if IS_CUDA:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        print("Using GPU")
    else:
        print("Using CPU")

    # Keep track of the best fold
    best_fold_loss = float('inf')
    best_fold_accuracy = 0 # For classification
    best_fold_id = 0



    for fold_id in range(FOLD_AMOUNT):
        print(f"########################## Training model for fold {fold_id} ##########################")

        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'rb'))
        x_train = data['x']
        y_train = data['y']
        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'rb'))
        x_val = data['x']
        y_val = data['y']

        if DATASET_NAME == 'PulseDB':
            if SIGNAL_AMOUNT == 1:
                x_train = x_train['ppg']
                x_val = x_val['ppg']
            elif SIGNAL_AMOUNT == 2:
                x_train = np.concatenate((x_train['ppg'], x_train['vpg']), axis=1)
                x_val = np.concatenate((x_val['ppg'], x_val['vpg']), axis=1)
            elif SIGNAL_AMOUNT == 3:
                x_train = np.concatenate((x_train['ppg'], x_train['vpg'], x_train['apg']), axis=1)
                x_val = np.concatenate((x_val['ppg'], x_val['vpg'], x_val['apg']), axis=1)
            else:
                raise ValueError("Invalid signal amount")
        # Cut the signal length
        x_train = x_train[:, :, :SIGNAL_LENGTH]
        x_val = x_val[:, :, :SIGNAL_LENGTH]

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = Data.TensorDataset(x_train, y_train)
        val_dataset = Data.TensorDataset(x_val, y_val)

        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,      #2,
            pin_memory=IS_CUDA  # Pytorch Recommendation
        )
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,      # 2,
            pin_memory=IS_CUDA  # Pytorch Recommendation
        )

        # Define the model
        model = getattr(models, MODEL)().cuda() if IS_CUDA else getattr(models, MODEL)()
        if OUTPUT_TASK == 'classification':
            loss_fn = nn.BCEWithLogitsLoss().cuda() if IS_CUDA else nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss().cuda() if IS_CUDA else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               factor=LR_SCHEDULER_FACTOR,
                                                               patience=LR_SCHEDULER_PATIENCE)

        # Early stopping
        best_loss = float('inf')
        best_accuracy = 0 # For classification
        patience = EARLY_STOPPING_PATIENCE

        best_train_accuracy = 0 # For classification
        # Train the model
        for epoch in tqdm(range(EPOCHS), desc=f'Fold {fold_id} - Training Epochs'):
            model.train()
            train_mse = 0
            correct = 0  # For classification
            accuracy = 0  # For classification
            for x_batch, y_batch in tqdm(train_loader, f'Fold {fold_id} Epoch {epoch} - Training'):
                if IS_CUDA:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                # Zero the gradients (Optimization suggested by PyTorch)
                for param in model.parameters():
                    param.grad = None

                y_pred = model(x_batch)             # Forward pass
                if OUTPUT_TASK == 'classification':
                    y_pred = y_pred.squeeze(1)
                loss = loss_fn(y_pred, y_batch)     # Compute the loss
                loss.backward()                     # Backward pass
                optimizer.step()                    # Update the weights

                train_mse += loss.item()
                if OUTPUT_TASK == 'classification':
                    predicted = (torch.sigmoid(y_pred) >= 0.5).long()
                    correct += (predicted == y_batch).sum().item()
            if OUTPUT_TASK == 'classification':
                accuracy = correct / len(train_loader.dataset) * 100.0

            train_mse /= len(train_loader)
            if writer is not None:
                writer.add_scalar(f'{MODEL}/{fold_id}/MSE/Train', train_mse, epoch)
                if OUTPUT_TASK == 'classification':
                    writer.add_scalar(f'{MODEL}/{fold_id}/Accuracy', accuracy, epoch)

            if OUTPUT_TASK == 'classification':
                if accuracy >= best_train_accuracy:
                    best_train_accuracy = accuracy
                    torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))

            if FOLD_AMOUNT != 1:
                model.eval()
                eval_mse = 0
                correct = 0 # For classification
                accuracy = 0 # For classification
                with torch.no_grad():
                    for x_batch, y_batch in tqdm(val_loader, f'Fold {fold_id} Epoch {epoch} - Validation'):
                        if IS_CUDA:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                        y_pred = model(x_batch)
                        if OUTPUT_TASK == 'classification':
                            y_pred = y_pred.squeeze(1)
                        eval_mse += loss_fn(y_pred, y_batch).item()

                        if OUTPUT_TASK == 'classification':
                            predicted = (torch.sigmoid(y_pred) >= 0.5).long()
                            correct += (predicted == y_batch).sum().item()

                eval_mse /= len(val_loader)
                if OUTPUT_TASK == 'classification':
                    accuracy = correct / len(val_loader.dataset) * 100.0
                scheduler.step(eval_mse)

                if writer is not None:
                    writer.add_scalar(f'{MODEL}/{fold_id}/MSE/Validation', eval_mse, epoch)
                    if OUTPUT_TASK == 'classification':
                        writer.add_scalar(f'{MODEL}/{fold_id}/Accuracy', accuracy, epoch)

                if OUTPUT_TASK == 'classification':
                    if accuracy > best_fold_accuracy:
                        best_fold_id = fold_id
                        best_fold_accuracy = accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))
                        patience = EARLY_STOPPING_PATIENCE
                    elif EARLY_STOPPING:
                        patience -= 1
                        if patience == 0:
                            print(f"Early stopping at epoch {epoch}")
                            break
                else:
                    if eval_mse < best_fold_loss:
                        best_fold_id = fold_id
                        best_fold_loss = eval_mse
                    if eval_mse < best_loss:
                        best_loss = eval_mse
                        torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))
                        patience = EARLY_STOPPING_PATIENCE  # Reset patience counter
                    elif EARLY_STOPPING:
                        patience -= 1
                        if patience == 0:
                            print(f"Early stopping at epoch {epoch}")
                            break

            del x_batch, y_batch, y_pred
            if IS_CUDA:
                torch.cuda.empty_cache()

        print(f"########################## Fold {fold_id} - Training Complete ##########################")
        # If no model saved yet
        if not os.path.exists(os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt')):
            torch.save(model.state_dict(), os.path.join(outputs_path, 'Models', f'{MODEL}_{fold_id}.pt'))

    return best_fold_id

if __name__ == '__main__':
    train_model()