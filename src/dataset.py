import os
import shutil
import splitfolders
import pandas as pd
import numpy as np
from tqdm import tqdm
from colorama import Fore

IMAGE_PATH = "image path"
TARGET_PATH = "target path" 

dataset_path = "dataset"
ladd_path = "ladd"
annotation_path = "/ladd/labels"
image_path = "ladd/images"

def create_dataset(data_path: str, target_path: str) -> pd.DataFrame:
    assert isinstance(data_path, str) 
    assert isinstance(target_path, str)
    
    dict_paths = {
        "image": [],
        "annotation": []
    }
    
    for dir_name, _, filenames in os.walk(data_path):
        for filename in tqdm(filenames):
            name = filename.split('.')[0]
            dict_paths["image"].append(f"{data_path}/{name}.jpg")
            dict_paths["annotation"].append(f"{target_path}/{name}.txt")

    
    dataframe = pd.DataFrame(
        data=dict_paths,
        index=np.arange(0, len(dict_paths["image"]))
    )
    
    return dataframe


df = create_dataset(
    data_path=IMAGE_PATH,
    target_path=TARGET_PATH
)

prepare_dirs(
    dataset_path=ladd_path,
    annotation_path=annotation_path,
    images_path=image_path
)

copy_dirs(
    dataframe=df, 
    data_path=image_path,
    target_path=annotation_path
)

splitfolders.ratio(
    input=ladd_path,
    output=dataset_path,
    seed=42,
    ratio=(0.80, 0.10, 0.10),
    group_prefix=None,
    move=True
)

finalizing_preparation(
    dataset_path,
    ladd_path
)
