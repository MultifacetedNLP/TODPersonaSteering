import uuid

from dotmap import DotMap

import evaluate
import numpy as np
from torchmetrics import Metric


class NlgGleuMetric(Metric):
    def __init__(
        self,
        metric_name: str = "gleu",
    ):
        super().__init__()
        metric_maps = DotMap(
            {
                "gleu": {"metric": "google_bleu", "text": "Response GLEU"},
                "bleu": {"metric": "bleu", "text": "Response BLEU"},
            }
        )
        self.metric_map = metric_maps[metric_name]
        self.metric = evaluate.load(
            self.metric_map.metric, experiment_id=str(uuid.uuid4())
        )
        self.row_metric = evaluate.load(
            self.metric_map.metric, experiment_id=str(uuid.uuid4())
        )

    def update(self, predictions: list[str], references: list[str]) -> None:
        refs = np.expand_dims(references, axis=1)
        self.metric.add_batch(predictions=predictions, references=refs)

    def compute(self) -> float:
        try:
            res = self.metric.compute()[self.metric_map.metric]
        except Exception:
            res = 0.0
        return res

    def compute_row(self, pred: str, ref: str) -> None:
        try:
            res = self.row_metric.compute(predictions=[pred], references=[[ref]])[
                self.metric_map.metric
            ]
        except Exception:
            res = 0.0
        return round(res, 4)

    def __str__(self):
        res = self._compute()
        return f"{self.metric_map.text}: {res:.4f}"
