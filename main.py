import pandas as pd
from deepface import DeepFace
from core.face_model import FaceModel

# temporary
import os
import numpy as np

from utils import make_dataset

def main():

    dataset = make_dataset("./dataset")
    print(f"Number of pairs in dataset: {len(dataset)}")

    # Define models
    models = [
        # FaceModel("VGG-Face"), 
        # FaceModel("Facenet"), 
        # FaceModel("Facenet512"), 
        # FaceModel("OpenFace"), 
        # FaceModel("DeepFace"), 
        FaceModel("ArcFace"), 
        FaceModel("SFace"),
    ]

    results = {}

    # corupted_files = []

    # for files in dataset[1000:]:
    #     for file in files:
    #         try:
    #             embedding_objs = DeepFace.represent(img_path = file)
    #         except Exception:
    #             corupted_file = file
    #             if corupted_file not in corupted_files:
    #                 corupted_files.append(corupted_file)
    #                 print(corupted_file, "[corupted]")

    for model in models:
        result = model.evaluate(DeepFace.verify, dataset)
        results[model.model_name] = result

    table = pd.DataFrame(results)
    print(table)

    # Store resutls in a csv file
    # pd.DataFrame(results).transpose().to_csv("scores.csv")

    print("Done!")


if __name__ == "__main__":
    main()
