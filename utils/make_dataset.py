import os
import glob
from typing import List, Tuple

from utils import generate_speakers_file_pairs


def make_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    # Get speaker to samples map
    speakers = glob.glob(os.path.join(dataset_dir, "*"))
    speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*.wav")) for s in speakers }
    # Generate datasets for evaluation for each speaker
    dataset = generate_speakers_file_pairs(speaker_to_samples_dict)
    return dataset
