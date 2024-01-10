import sys
import logging
import os
from box.exceptions import BoxValueError
import yaml
# from ...colorDetection import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pickle
import os


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("colorDetectionLogger")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    



class IncrementalPickleHandler:
    def __init__(self, base_name='model'):
        self.base_name = base_name
        self.highest_X = self._get_highest_X()

    def _get_highest_X(self):
        # Find the highest X in the existing pickle files
        existing_files = [f for f in os.listdir() if f.startswith(self.base_name)]
        numbers = [int(name[len(self.base_name):]) for name in existing_files if name[len(self.base_name):].isdigit()]

        return max(numbers, default=0)

    def save_pickle(self,data, path):
        # Increment X and save the pickle file
        self.highest_X += 1
        file_name = f'{self.base_name}{self.highest_X}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f"Pickle file saved as {file_name}")

    def load_latest_pickle(self):
        # Load the pickle file with the highest X
        file_name = f'{self.base_name}{self.highest_X}.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                data = pickle.load(file)
            logger.info(f"Loaded the latest pickle file: {file_name}")
            return data
        else:
            logger.info("No pickle file found.")
            return None

# Example usage:
pickle_handler = IncrementalPickleHandler(base_name='model')

# # Save pickle files with incrementing numbers
# pickle_handler.save_pickle({'example_data': 'some_data'})
# pickle_handler.save_pickle({'example_data': 'more_data'})

# # Load the latest pickle file
# loaded_data = pickle_handler.load_latest_pickle()
# print("Loaded Data:", loaded_data)
