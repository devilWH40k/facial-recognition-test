import os
import glob
import pickle
from typing import List, Tuple

from utils import generate_persons_photo_pairs


def make_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    if os.path.exists("dataset.obj"):
        with open("dataset.obj", "rb") as dataset_file:
            dataset = pickle.load(dataset_file)
            return dataset

    # Get person to samples map
    persons = glob.glob(os.path.join(dataset_dir, "*"))

    extensions = ["*.jpg", "*.png", "*.jpeg", "*.jfif"]
    persons_to_samples_dict = {}
    for p in persons:
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(p, ext)))
        persons_to_samples_dict[p] = files

    # Generate datasets for evaluation for each person
    dataset = generate_persons_photo_pairs(persons_to_samples_dict)

    # Store generated dataset in the file
    with open("dataset.obj", "wb") as dataset_file:
        pickle.dump(dataset, dataset_file)

    return dataset
