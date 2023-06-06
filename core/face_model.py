import tqdm
import os
from core.metrics import (
    compute_eer, 
    compute_min_dcf,
    compute_far_frr,
)


class FaceModel:
    def __init__ (self, model_name, distance_metric="cosine"):
        self.model_name = model_name
        self.distance_metric = distance_metric

    def evaluate(self, verify_func, data):
        print("evaluating", self.model_name, "...")
        scores = []
        labels = []

        for file_path1, file_path2 in tqdm.tqdm(data, total=len(data)):
            try:
                result = verify_func(
                    img1_path=file_path1,
                    img2_path=file_path2,
                    distance_metric=self.distance_metric,
                    model_name=self.model_name,
                )
                similarity = 1 - result["distance"]
                label = self._get_label(file_path1, file_path2)
                scores.append(similarity)
                labels.append(label)

            except Exception as e:
                print(e, file_path1, file_path2)

        result = self.calculate_metrics(scores, labels)
        return result
    
    def _get_label(self, file_path1, file_path2):
        """
        Return 0 if different persons, 1 if same persons.
        """
        label = self._get_persons_identifier(file_path1) == self._get_persons_identifier(file_path2)
        return int(label)
    
    @staticmethod
    def calculate_metrics(scores, labels):
        ee_rate, thresh, fa_rate, fr_rate = compute_eer(scores, labels)
        min_dcf = compute_min_dcf(fr_rate, fa_rate)
        fa_score, fr_score, acc = compute_far_frr(scores, labels, thresh)

        result = {
            "fa_score": round(fa_score, 3),
            "fr_score": round(fr_score, 3),
            "accuracy": acc,
            "ee_rate": round(ee_rate, 3),
            "dcf": round(min_dcf, 3), 
            "threshold": round(thresh, 3),
        }

        return result

    @staticmethod
    def _get_persons_identifier(file_path):
        return os.path.basename(os.path.dirname(file_path))
    
if __name__ == "__main__":
    model = FaceModel("some_model")
    