import pandas as pd
from deepface import DeepFace
from core.face_model import FaceModel

from utils import make_dataset

def main():

    dataset = make_dataset("./dataset")
    print(f"Number of pairs in dataset: {len(dataset)}")

    # Define models
    models = [
        FaceModel("VGG-Face"), 
        FaceModel("Facenet"), 
        FaceModel("Facenet512"), 
        FaceModel("OpenFace"), 
        FaceModel("DeepFace"), 
        FaceModel("ArcFace"), 
        FaceModel("SFace"),
    ]

    results = {}

    for model in models:
        result = model.evaluate(DeepFace.verify, dataset)
        results[model.model_name] = result

    table = pd.DataFrame(results)
    print(table)
    print("Done!")


if __name__ == "__main__":
    main()
