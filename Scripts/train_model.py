import os
from tqdm import tqdm
import pickle
import torch.utils.data as Data

from models import *

FOLDED_DATASET_PATH = '../Data/split_dataset'
MODEL_PATH = '../Models'
RESULTS_PATH = '../Results'

SIGNAL_LENGTH = 1024
FOLD_AMOUNT = 5
LEARNING_RATE = 0.001

BATCH_SIZE = 64
EPOCHS = 2

def train_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    for fold_id in range(FOLD_AMOUNT):
        print(f"Training model for fold {fold_id}")
        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'train_{fold_id}.pkl'), 'rb'))
        x_train = data['x']
        y_train = data['y']
        data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, f'val_{fold_id}.pkl'), 'rb'))
        x_val = data['x']
        y_val = data['y']

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = Data.TensorDataset(x_train, y_train)
        val_dataset = Data.TensorDataset(x_val, y_val)

        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=3
        )

        # Define the model
        model = CNN_Transformer()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Evaluate
        mse_history = []

        # Train the model
        for epoch in tqdm(range(EPOCHS), desc=f'Fold {fold_id} - Training Epochs'):
            model.train()
            for i, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()               # Zero the gradients
                y_pred = model(x_batch)             # Forward pass
                loss = loss_fn(y_pred, y_batch)     # Compute the loss
                loss.backward()                     # Backward pass
                optimizer.step()                    # Update the weights

            model.eval()
            mse = 0
            for i, (x_batch, y_batch) in enumerate(val_loader):
                y_pred = model(x_batch)
                mse += loss_fn(y_pred, y_batch).item()
            mse /= len(val_loader)
            mse_history.append(mse)

        print(f"Fold {fold_id} - Training Complete")

        # Save the model
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'model_{fold_id}.pt'))
        pickle.dump(mse_history, open(os.path.join(RESULTS_PATH, f'mse_{fold_id}.pkl'), 'wb'))

if __name__ == '__main__':
    train_model()