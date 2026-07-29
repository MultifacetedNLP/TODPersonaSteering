from dataclasses import asdict
import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from logger.inference_logger_dataclasses import (
    ApiCallInferenceLogData,
)
from evaluation.metrics.nlg_gleu_metric import NlgGleuMetric
from torchmetrics import MetricCollection

import utils
import torch
import numpy as np

# accelerator = Accelerator()


class NlgApiCallMetricManager:
    def __init__(self, logger, tokenizer=None):
        self.tokenizer = tokenizer
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.bert_score_metric = evaluate.load(
            "bertscore", experiment_id=str(uuid.uuid4())
        )
        self.logger = logger
        self.data: list[ApiCallInferenceLogData] = []

        self.response_metrics = MetricCollection(
            {
                "response_gleu": NlgGleuMetric(),
                "response_bleu": NlgGleuMetric("bleu"),
                # "response_bertscore": BertScoreMetric(tokenizer),
            }
        )

    def add_batch(
        self,
        input_tokens,
        label_tokens,
        pred_tokens,
        turn_row_types,
        is_retrievals,
        is_slot_fills,
        dialog_ids,
        turn_ids,
        is_multi_domain_api_calls,
        domains,
    ):
        if self.tokenizer:
            label_tokens = torch.where(label_tokens != -100, label_tokens, torch.tensor(self.tokenizer.pad_token_id))
            input_texts, labels, preds = [
                self.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for tokens in [input_tokens, label_tokens, pred_tokens]
            ]
        else:
            input_texts, labels, preds = input_tokens, label_tokens, pred_tokens

        (
            response_preds,
            response_labels,
        ) = ([], [])

        for (
            input_text,
            pred,
            label,
            turn_row_type,
            is_retrieval,
            is_slot_fill,
            dialog_id,
            turn_id,
            is_multi_domain_api_call,
            domain,
        ) in zip(
            input_texts,
            preds,
            labels,
            turn_row_types,
            is_retrievals,
            is_slot_fills,
            dialog_ids,
            turn_ids,
            is_multi_domain_api_calls,
            domains,
        ):
            row = ApiCallInferenceLogData(
                input_text=input_text,
                pred=pred,
                label=label,
                turn_row_type=int(turn_row_type),
                is_retrieval=int(is_retrieval),
                is_slot_fill=int(is_slot_fill),
                dialog_id=dialog_id.item(),
                turn_id=turn_id.item(),
                domains=domain, 
                is_multi_domain_api_call=int(is_multi_domain_api_call),
            )
            self.data.append(row)
            response_preds.append(row.pred)
            response_labels.append(row.label)
        self.response_metrics.update(
            references=response_labels, predictions=response_preds
        )

    def write_csv(self, csv_path):
        if not len(self.data):
            raise ValueError("Must call compute row wise metrics first")
        df = pd.DataFrame([asdict(d) for d in self.data])
        df.to_csv(csv_path, index=False, encoding="utf-8")
    
    def compute_bleu_and_gleu_metrics(self):
        for row in self.data:
            row_dict = DotMap(row.__dict__)
            for k, v in zip(
                list(self.response_metrics.keys()),
                list(self.response_metrics.values()),
            ):
                res = v.compute_row(row.pred, row.label)
                row_dict[k] = res  
            row.update(row_dict)
            
        all_data_df = pd.DataFrame(self.data)
        
        bleu = all_data_df.response_bleu.mean()
        gleu = all_data_df.response_gleu.mean()
        
        utils.log(self.logger, f"Retrieval BLEU: {bleu:.4f}")
        utils.log(self.logger, f"Retrieval GLEU: {gleu:.4f}")
    
    def compute_bert_score(self):
        preds = [row.pred for row in self.data]
        labels = [row.label for row in self.data]
        result = self.bert_score_metric.compute(
            predictions=preds,
            references=labels,
            model_type="distilbert-base-uncased",
        )
        avg_precision = np.mean(result["precision"])
        avg_recall = np.mean(result["recall"])
        avg_f1 = np.mean(result["f1"])
        score_str = f"BERT score: precision {avg_precision:.4f}, recall {avg_recall:.4f}, f1 {avg_f1:.4f}"
        self.logger.info(score_str)