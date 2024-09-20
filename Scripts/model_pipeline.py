from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
import os
import inspect
import torchinfo
import torch

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
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))
SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
DEMOGRAPHICS_LIST = [demographic.strip().lower() for demographic in os.getenv('DEMOGRAPHICS').split(',')] \
    if os.getenv('DEMOGRAPHICS') is not None else None
TARGETS_LIST = [target.strip().lower() for target in os.getenv('TARGETS').split(',')]

TEST_MODE = os.getenv('TEST_MODE').lower() == 'true'

LIST_OF_MODEL_NAMES = [name for name, obj in inspect.getmembers(models)
                       if inspect.isclass(obj) and obj.__module__ == models.__name__]

outputs_path = os.path.join(OUTPUTS_PATH, f'{MODEL}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

def setup_layout():
    layout = {}
    for model in LIST_OF_MODEL_NAMES:
        layout[model] = {}
        for fold_id in range(FOLD_AMOUNT):
            layout[model][f'Fold {fold_id} loss'] = ["Multiline", [f"{model}/{fold_id}/Loss/Train",
                                                                    f"{model}/{fold_id}/Loss/Validation",
                                                                    f"{model}/{fold_id}/Loss/Calibration"]]
            layout[model][f'Fold {fold_id} MSE'] = ["Multiline", [f"{model}/{fold_id}/MSE/Train",
                                                                  f"{model}/{fold_id}/MSE/Validation",
                                                                  f"{model}/{fold_id}/MSE/Calibration"]]
            layout[model][f'Fold {fold_id} MAE'] = ["Multiline", [f"{model}/{fold_id}/MAE/Train",
                                                                  f"{model}/{fold_id}/MAE/Validation",
                                                                  f"{model}/{fold_id}/MAE/Calibration"]]
    layout["Overall"] = {
        "training": ["Multiline", [f"{MODEL}/{fold_id}/Loss/Train" for fold_id in range(FOLD_AMOUNT)]],
        "validation": ["Multiline", [f"{MODEL}/{fold_id}/Loss/Validation" for fold_id in range(FOLD_AMOUNT)]],
        "calibration": ["Multiline", [f"{MODEL}/{fold_id}/Loss/Calibration" for fold_id in range(FOLD_AMOUNT)]],
        "training MSE": ["Multiline", [f"{MODEL}/{fold_id}/MSE/Train" for fold_id in range(FOLD_AMOUNT)]],
        "validation MSE": ["Multiline", [f"{MODEL}/{fold_id}/MSE/Validation" for fold_id in range(FOLD_AMOUNT)]],
        "calibration MSE": ["Multiline", [f"{MODEL}/{fold_id}/MSE/Calibration" for fold_id in range(FOLD_AMOUNT)]],
        "training MAE": ["Multiline", [f"{MODEL}/{fold_id}/MAE/Train" for fold_id in range(FOLD_AMOUNT)]],
        "validation MAE": ["Multiline", [f"{MODEL}/{fold_id}/MAE/Validation" for fold_id in range(FOLD_AMOUNT)]],
        "calibration MAE": ["Multiline", [f"{MODEL}/{fold_id}/MAE/Calibration" for fold_id in range(FOLD_AMOUNT)]]
    }
    layout["Between Models"] = {}
    for fold_id in range(FOLD_AMOUNT):
        layout["Between Models"][f'Fold {fold_id} Training'] = ["Multiline", [f"{model}/{fold_id}/Loss/Train" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Validation'] = ["Multiline", [f"{model}/{fold_id}/Loss/Validation" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Calibration'] = ["Multiline", [f"{model}/{fold_id}/Loss/Calibration" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Both'] = ["Multiline",
                                                            [f"{model}/{fold_id}/Loss/Train" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/Loss/Validation" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/Loss/Calibration" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Training MAE'] = ["Multiline", [f"{model}/{fold_id}/MAE/Train" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Validation MAE'] = ["Multiline", [f"{model}/{fold_id}/MAE/Validation" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Calibration MAE'] = ["Multiline", [f"{model}/{fold_id}/MAE/Calibration" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Both MAE'] = ["Multiline",
                                                            [f"{model}/{fold_id}/MAE/Train" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/MAE/Validation" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/MAE/Calibration" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Training MSE'] = ["Multiline", [f"{model}/{fold_id}/MSE/Train" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Validation MSE'] = ["Multiline", [f"{model}/{fold_id}/MSE/Validation" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Calibration MSE'] = ["Multiline", [f"{model}/{fold_id}/MSE/Calibration" for model in LIST_OF_MODEL_NAMES]]
        layout["Between Models"][f'Fold {fold_id} Both MSE'] = ["Multiline",
                                                            [f"{model}/{fold_id}/MSE/Train" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/MSE/Validation" for model in LIST_OF_MODEL_NAMES] +
                                                            [f"{model}/{fold_id}/MSE/Calibration" for model in LIST_OF_MODEL_NAMES]]


    return layout

def copy_metadata():
    shutil.copy(ENV_PATH, os.path.join(outputs_path, 'metadata.txt'))
    with open(os.path.join(outputs_path, 'metadata.txt'), 'a',  encoding="utf-8") as f:
        with open(os.path.join(DATA_PATH, DATASET_NAME, '.env'), 'r', encoding="utf-8") as env_file:
            shutil.copyfileobj(env_file, f)
    model = getattr(models, MODEL)()
    # Simulating running one sample through the model
    dummy_input = {'signals': torch.randn(1, len(SIGNALS_LIST), SIGNAL_LENGTH)}
    if DEMOGRAPHICS_LIST is not None:
        dummy_input['demographics'] = torch.randn(1, len(DEMOGRAPHICS_LIST))
    model_stat = str(torchinfo.summary(model, input_data=[dummy_input], verbose=2))
    input_shape = str({key: dummy_input[key].shape for key in dummy_input})
    with open(os.path.join(outputs_path, 'metadata.txt'), 'a',  encoding="utf-8") as f:
        f.write("\n\n")
        f.write(input_shape)
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
    with open(os.path.join(outputs_path, 'metadata.txt'), 'a', encoding="utf-8") as f:
        f.write(f"Best fold: {best_fold_id}\n")
    evaluate_model(writer=writer, model_path=outputs_path, version=best_fold_id, test_mode=TEST_MODE)
    writer.flush()

if __name__ == '__main__':
    pipeline()