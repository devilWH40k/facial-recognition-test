import tqdm
import pandas as pd
from deepface import DeepFace

# from core import (
#     Pipeline,
#     Pyannote,
#     WavLM,
#     TitaNet,
#     Ecapa
# )
# from core.metrics import (
#     compute_eer, 
#     compute_min_dcf,
#     compute_far_frr,
# )
from utils import make_dataset


def get_label(file1: str, file2: str) -> int:
    """
    Return 0 if different speakers, 1 if same speakers.
    """
    def _get_name(x):
        return x.split("/")[-2]

    return int(_get_name(file1) == _get_name(file2))


def evaluate_model(
    model, 
    data, 
) -> pd.DataFrame:

    print("evaluating", model, "...")
    scores = []
    labels = []

    for file1, file2 in tqdm.tqdm(data, total=len(data)):
        result = DeepFace.verify(
            img1_path=file1,
            img2_path=file2,
            distance_metric="cosine",
            model_name=model
        )
        print("similarity:", result["distance"])
        # need similarity here
        # label = get_label(file1, file2)
        # scores.append(similarity)
        # labels.append(label)

    print("finished...")

    # ee_rate, thresh, fa_rate, fr_rate = compute_eer(scores, labels)
    # min_dcf = compute_min_dcf(fr_rate, fa_rate)
    # fa_score, fr_score = compute_far_frr(scores, labels, thresh)

    # result = {
    #     "pipeline": pipeline.name,
    #     "fa_score": fa_score,
    #     "fr_score": fr_score,
    #     "ee_rate": ee_rate,
    #     "dcf": min_dcf, 
    #     "threshold": thresh,
    # }
    # return result


def main():

    dataset = make_dataset("./dataset")
    print(f"Number of pairs in dataset: {len(dataset)}")

    # print(dataset[:40])

    # Define models
    models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "SFace",
    ]

    for model in models:
        evaluate_model(model, dataset[:3])

    # results = {}

    # for pipeline in pipelines:
    #     print(f"Evaluating pipeline: {pipeline.name}")
    #     results[pipeline.name] = evaluate_pipeline(pipeline, dataset)
    
    # # Store resutls in a csv file
    # pd.DataFrame(results).transpose().to_csv("scores.csv")

    # print("Done!")


if __name__ == "__main__":
    main()
