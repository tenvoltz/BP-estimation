from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
import os
import inspect
import torchinfo

from evaluate_model import evaluate_model
from train_model import train_model
from Models import models

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

OUTPUTS_PATH = os.getenv('OUTPUTS_PATH')
ENV_PATH = os.getenv('ENV_PATH')
MODEL = os.getenv('MODEL')
FOLD_AMOUNT = int(os.getenv('FOLD_AMOUNT'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))
SIGNAL_AMOUNT = int(os.getenv('SIGNAL_AMOUNT'))
TEST_MODE = os.getenv('TEST_MODE').lower() == 'true'

LIST_OF_MODEL_NAMES = [name for name, obj in inspect.getmembers(models)
                       if inspect.isclass(obj) and obj.__module__ == models.__name__]

outputs_path = os.path.join(OUTPUTS_PATH, f'{MODEL}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

def setup_layout():
    layout = {"Fold-by-fold": {}}
    for fold_id in range(FOLD_AMOUNT):
        layout["Fold-by-fold"][f'Fold {fold_id} loss'] = ["Multiline", [f"{MODEL}/{fold_id}/MSE/Train",
                                                                   f"{MODEL}/{fold_id}/MSE/Validation"]]
    layout["Multiline Overall"] = {
        "training": ["Multiline", [f"{MODEL}/{fold_id}/MSE/Train" for fold_id in range(FOLD_AMOUNT)]],
        "validation": ["Multiline", [f"{MODEL}/{fold_id}/MSE/Validation" for fold_id in range(FOLD_AMOUNT)]]
    }

    return layout

def copy_metadata():
    shutil.copy(ENV_PATH, os.path.join(outputs_path, 'metadata.txt'))
    with open(os.path.join(outputs_path, 'metadata.txt'), 'a',  encoding="utf-8") as f:
        with open(os.path.join(DATA_PATH, DATASET_NAME, '.env'), 'r', encoding="utf-8") as env_file:
            shutil.copyfileobj(env_file, f)
    model = getattr(models, MODEL)()
    model_stat = str(torchinfo.summary(model, (BATCH_SIZE, SIGNAL_AMOUNT, SIGNAL_LENGTH), verbose=0))
    with open(os.path.join(outputs_path, 'metadata.txt'), 'a',  encoding="utf-8") as f:
        f.write("\n\n")
        f.write(model_stat)
        f.write("\n\n")
        f.write(inspect.getsource(getattr(models, MODEL)))


def pipeline():
    if not os.path.isdir(outputs_path):
        os.mkdir(outputs_path)
    copy_metadata()

    writer = SummaryWriter(
        log_dir=os.path.join(outputs_path, 'Results', f'{MODEL}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'),
        filename_suffix=f'_{MODEL}')
    writer.add_custom_scalars(setup_layout())

    best_fold_id = train_model(outputs_path=outputs_path, writer=writer)
    print(f"Best fold: {best_fold_id}")
    evaluate_model(writer=writer, model_path=outputs_path, version=best_fold_id, test_mode=TEST_MODE)

    writer.flush()

if __name__ == '__main__':
    pipeline()