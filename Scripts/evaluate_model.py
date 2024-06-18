import os
import pickle
from tqdm import tqdm
import torch.utils.data as Data
from dotenv import load_dotenv

from metrics import evaluate_metrics
from models import *

load_dotenv()
FOLDED_DATASET_PATH = os.getenv('FOLDED_DATASET_PATH')
TRAINED_MODEL_PATH = os.getenv('TRAINED_MODEL_PATH')

def evaluate_model():
    data = pickle.load(open(os.path.join(FOLDED_DATASET_PATH, 'test.pkl'), 'rb'))
    x_test = data['x']
    y_test = data['y']

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )

    model = CNN_Transformer()
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model.eval()

    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc='Testing'):
            y_pred_batch = model(x_batch)
            y_pred.append(y_pred_batch)
    y_pred = torch.cat(y_pred, dim=0)

    y_pred = y_pred.numpy()
    y_test = y_test.numpy()

    evaluate_metrics(y_test, y_pred)

    # Print a sample prediction and ground truth
    print('Sample Prediction and Ground Truth')
    for i in range(5):
        print(f'Prediction: {y_pred[i]} Ground Truth: {y_test[i]}')

if __name__ == '__main__':
    evaluate_model()