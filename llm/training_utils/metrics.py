import inspect
import logging
from collections import defaultdict
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torchmetrics import F1Score, Metric, MetricCollection

logger = logging.getLogger(__name__)


def format_input(
    preds: torch.Tensor, target: torch.Tensor, threshold: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions
    """
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1:
        preds = preds.argmax(dim=-1)
    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    if threshold is not None:
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                preds = preds.sigmoid()
            preds = preds > threshold
        preds = preds.reshape(*preds.shape[:2], -1)
        target = target.reshape(*target.shape[:2], -1)

    return preds, target


class EarthMoversDistance(Metric):
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index

        self.add_state("cumulative_emd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Get label mask
        label_mask = target != self.ignore_index
        # Mask out predictions
        preds[~label_mask] = self.ignore_index
        # Get number of segments per sample
        num_segments = label_mask.sum(dim=1)

        # Calculate the absolute differences between the predicted and target positions
        abs_diff = torch.abs(preds - target)

        # Calculate the EMD for the current batch
        emd = torch.sum(abs_diff, dim=1).float()

        # Calculate the maximum EMD in case of worst predictions
        max_value = torch.max(num_segments)
        max_value_repeated = num_segments.unsqueeze(1).repeat(1, max_value)
        max_mask = target < (max_value_repeated // 2)
        worst_preds = torch.zeros_like(target)
        worst_preds[max_mask] = max_value_repeated[max_mask] - 1
        worst_preds[~label_mask] = self.ignore_index
        worst_abs_diff = torch.abs(worst_preds - target)
        max_emd = torch.sum(worst_abs_diff, dim=1).float()

        normalized_emd = emd / max_emd

        # remove nan values
        normalized_emd_masked = normalized_emd[~normalized_emd.isnan()]

        # Update the cumulative distance and the total count
        self.cumulative_emd += normalized_emd_masked.sum()
        self.total += normalized_emd_masked.numel()

    def compute(self):
        # Normalize the cumulative distance by the maximum possible distance
        return self.cumulative_emd / self.total.float()


class MaskedAccuracy(Metric):
    def __init__(
        self,
        ignore_index: int = -100,
        threshold: Optional[float] = None,
        mode: Literal["micro", "macro"] = "macro",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.mode = mode

        self.add_state(
            "correct", default=torch.tensor(0) if self.mode == "micro" else torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = format_input(preds, target, self.threshold)
        assert preds.shape == target.shape

        label_mask = target != self.ignore_index
        masked_preds = torch.masked_select(preds, label_mask)
        masked_target = torch.masked_select(target, label_mask)

        if self.mode == "micro":
            self.correct += torch.sum(torch.eq(masked_target, masked_preds))
            self.total += target.numel()
        else:
            self.correct += torch.eq(masked_target, masked_preds).float().mean()
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class MeanLoss(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        self.loss += loss
        self.count += 1

    def compute(self):
        return self.loss / self.count


class SklearnClassificationReport(Metric):
    def __init__(self, label_map: Dict[int, str]):
        super().__init__()
        self.label_map = label_map

        # states
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("predictions", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = format_input(preds, target)
        assert preds.shape == target.shape

        self.predictions.extend(preds.view(-1).detach().cpu().tolist())
        self.labels.extend(target.view(-1).detach().cpu().tolist())

    def compute(self):
        clf_report = classification_report(
            y_true=self.labels,
            y_pred=self.predictions,
            labels=list(self.label_map.values()),
            target_names=list(self.label_map),
        )
        return clf_report


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        # empty_target_action: str = "skip",
        ignore_index: int = -100,
        top_k: int = 5,
        label_map: Dict[int, str] | None = None,
        use_sustain_paper_map: bool = False,
    ):
        super().__init__()
        # self.empty_target_action = empty_target_action
        self.use_sustain_paper_map = use_sustain_paper_map
        self.ignore_index = ignore_index
        self.top_k = top_k
        self.label_map = {v: k for k, v in label_map.items()}
        self.num_labels = None

        # states
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("probabilities", default=[], dist_reduce_fx="cat")
        self.add_state("doc_ids", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, doc_ids: list[str]):
        if len(target.shape) > 2:
            target_mask = target != -100
            samples_per_doc = (target_mask).sum(dim=-1).count_nonzero(dim=-1)
            doc_ids = [i for doc_id, num_samples in zip(doc_ids, samples_per_doc) for i in [doc_id] * num_samples]
            target = target[target_mask].view(-1, target.shape[-1])

        if target.shape != preds.shape:
            target = target.view(preds.shape)

        if self.num_labels is None:
            self.num_labels = target.shape[-1]

        self.probabilities.append(preds.detach().cpu().numpy())
        self.labels.append(target.detach().cpu().numpy())
        self.doc_ids.append(np.array(doc_ids))

    def compute(self):
        # flatten list of batches to an array of shape num_blobs x num_labels
        targets = np.concatenate(self.labels, axis=0)
        probabilities = np.concatenate(self.probabilities, axis=0)
        doc_ids = np.concatenate(self.doc_ids, axis=0)

        if targets.sum() == 0:
            return {
                f"map_{self.top_k}_macro": 0.0,
                f"map_{self.top_k}_micro": 0.0,
                f"map_{self.top_k}_weighted": 0.0,
                f"map_{self.top_k}_per_req": 0.0,
            }

        # calculate r2b (requirement to blob) top k average precision
        report_results = defaultdict(lambda: defaultdict(list))
        for tar, pro, id_ in zip(targets, probabilities, doc_ids):
            report_results[id_]["probs"].append(pro)
            report_results[id_]["targets"].append(tar)

        ap_scores_per_req = defaultdict(list)
        annotated_ids_all = []
        for doc in report_results.values():
            targets_doc = np.array(doc["targets"])
            probs_doc = np.array(doc["probs"])

            if self.use_sustain_paper_map:
                average_precision_scores, annotated_ids = calculate_map_sustain(
                    targets_doc.T, probs_doc.T, top_k=self.top_k
                )
            else:
                average_precision_scores, annotated_ids = calculate_map(targets_doc.T, probs_doc.T, top_k=self.top_k)

            # sens_old.extend(sens.tolist())
            annotated_ids_all.extend(annotated_ids.tolist())
            if average_precision_scores.shape[0] != annotated_ids.shape[0]:
                logger.warning(
                    f"Average precision shape {average_precision_scores.shape[0]} is different from annotated ids shape"
                    f" {annotated_ids.shape[0]}."
                )
            for ap_score, id_ in zip(average_precision_scores, annotated_ids):
                idx = self.label_map[id_] if self.label_map else id_
                ap_scores_per_req[idx].append(ap_score)

        # Mean average precision per requirement
        mean_ap_per_req = {idx: np.mean(ap_scores) * 100 for idx, ap_scores in ap_scores_per_req.items()}

        # Mean average precision macro (Average over all per requirement scores)
        mean_average_precision_macro = np.mean(list(mean_ap_per_req.values()))

        # Mean average precision micro (Average over all per sample scores)
        mean_average_precision_micro = (
            np.mean([score for scores in ap_scores_per_req.values() for score in scores]) * 100
        )

        annotated_req_ids_sorted = sorted(set(annotated_ids_all))
        annotated_blobs_per_req = targets[:, annotated_req_ids_sorted].sum(axis=0)

        # Get support per requirement
        req_to_support = {
            self.label_map[id_] if self.label_map is not None else id_: int(support)
            for support, id_ in zip(annotated_blobs_per_req, annotated_req_ids_sorted)
        }

        # Mean average precision weighted (Weighted average by support over all per requirement scores)
        weights = annotated_blobs_per_req.tolist()
        mean_average_precision_weighted = np.average(list(mean_ap_per_req.values()), weights=weights)

        # Sort the requirements by support in descending order
        sorted_reqs = sorted(req_to_support, key=req_to_support.get, reverse=True)
        req_to_support = {req: int(req_to_support[req]) for req in sorted_reqs}
        mean_ap_per_req = {req: round(mean_ap_per_req[req], 2) for req in sorted_reqs}

        total_support = sum(req_to_support.values())
        df = pd.DataFrame({"MAP": mean_ap_per_req, "Support": req_to_support})
        df.loc["Macro"] = [round(mean_average_precision_macro, 2), total_support]
        df.loc["Micro"] = [round(mean_average_precision_micro, 2), total_support]
        df.loc["Weighted"] = [round(mean_average_precision_weighted, 2), total_support]
        df["Support"] = df["Support"].astype(int)

        # Convert the DataFrame to a string
        clf_report = df.to_string()

        if self.use_sustain_paper_map:
            return {
                f"map_{self.top_k}_sustain_macro": mean_average_precision_macro,
                f"map_{self.top_k}_sustain_micro": mean_average_precision_micro,
                f"map_{self.top_k}_sustain_weighted": mean_average_precision_weighted,
                f"map_{self.top_k}_sustain_per_req": clf_report,
            }
        else:
            return {
                f"map_{self.top_k}_macro": mean_average_precision_macro,
                f"map_{self.top_k}_micro": mean_average_precision_micro,
                f"map_{self.top_k}_weighted": mean_average_precision_weighted,
                f"map_{self.top_k}_per_req": clf_report,
            }


def init_metric_collection(metric_params: list[Dict], device: str | torch.device, **kwargs) -> Dict:  #
    metrics = {}
    for metric in metric_params:
        if "kwargs" not in metric:
            metric["kwargs"] = {}
        metric_name = metric["name"]
        metric_type = metric.get("type", metric_name)

        # update None metric kwargs if the key is present in **kwargs
        # -> needed for dynamically generated kwargs like label_map
        for k, v in metric["kwargs"].items():
            if v is None:
                metric["kwargs"][k] = kwargs.get(k)

        # init metrics
        metrics[metric_name] = METRICS[metric_type](**metric["kwargs"]).to(device)

    metrics_collection = MetricCollection(metrics)

    train_metrics = metrics_collection.clone(prefix="train-")
    valid_metrics = metrics_collection.clone(prefix="valid-")
    test_metrics = metrics_collection.clone(prefix="test-")

    return {"train": train_metrics, "valid": valid_metrics, "test": test_metrics}


def update_metrics(metrics: MetricCollection, step_output: Dict) -> Dict[str, torch.Tensor]:
    # rename keys to enable automatic evaluation
    step_output["target"] = step_output.pop("labels", None)
    step_output["preds"] = step_output.pop("logits", None)

    results = {}
    for metric_name, metric in metrics.items():
        # infer expected arguments to update metric
        expected_arguments = inspect.signature(metric.update).parameters.keys()

        # update metric_kwargs with other expected arguments
        metric_kwargs = {
            arg: step_output[arg]  # to_device(step_output[arg], device="cpu", detach=True)
            for arg in expected_arguments
            if arg in step_output
        }

        # get task name if one is defined
        task_name = metric_name.split("--", 1)[-1] if "--" in metric_name else None

        # if task name exists: Strip it from actual model output keys
        if task_name:
            for k, v in step_output.items():
                if k.startswith(task_name):
                    name = k.removeprefix(f"{task_name}_")
                    value = v  # to_device(v, device="cpu", detach=True)
                    if name.endswith("logits"):
                        name = name.replace("logits", "preds")
                        value = value.argmax(dim=-1)
                    if name.endswith("labels"):
                        name = name.replace("labels", "target")
                    if name in expected_arguments:
                        metric_kwargs[name] = value

        if "f1" in metric_name:
            if len(metric_kwargs["preds"].shape) == 3:
                metric_kwargs["preds"] = metric_kwargs["preds"].view(-1, metric_kwargs["preds"].shape[-1])
            if len(metric_kwargs["target"].shape) == 2:
                metric_kwargs["target"] = metric_kwargs["target"].view(-1)

        # update metric
        metric.update(**metric_kwargs)

        # update metric and compute results
        results[metric_name] = metric(**metric_kwargs)
    return results


def calculate_map(targets: np.ndarray, probabilities: np.ndarray, top_k=3) -> Tuple[np.ndarray, np.ndarray]:
    indices_annotated = np.where(np.sum(targets, axis=1) != 0)[0]
    probabilities = probabilities[indices_annotated, :]
    targets = targets[indices_annotated, :]

    prediction_indices_sorted = probabilities.argsort(axis=1)[:, ::-1]

    precision_scores = []
    relevant_items = []
    for k in range(1, top_k + 1):
        top_k_prediction_indices = prediction_indices_sorted[:, :k]
        intersection = np.take_along_axis(targets, indices=top_k_prediction_indices, axis=1)
        # check if current recommendation is relevant or not -> relevant: 1, irrelevant: 0
        indicator_fn = intersection[:, -1]

        # sum up the relevant recommendations up to k recommendations
        intersection_sum = intersection.sum(axis=1)
        # only keep precision if current recommendation is relevant (indicator fn)

        precision = (intersection_sum / k) * indicator_fn
        precision_scores.append(precision)
        relevant_items.append(indicator_fn)

    relevant_items_sum = sum(relevant_items)
    relevant_items_sum[relevant_items_sum == 0] = 1

    # minimum = np.minimum(top_k, targets.sum(axis=1))
    average_precision_scores = np.divide(sum(precision_scores), relevant_items_sum)  # minimum
    return average_precision_scores, indices_annotated


def calculate_map_sustain(targets: np.ndarray, probabilities: np.ndarray, top_k=3) -> Tuple[np.ndarray, np.ndarray]:
    indices_annotated = np.where(np.sum(targets, axis=1) != 0)[0]
    probabilities = probabilities[indices_annotated, :]
    targets = targets[indices_annotated, :]

    prediction_indices_sorted = probabilities.argsort(axis=1)[:, ::-1]

    precision_scores = []
    for k in range(1, top_k + 1):
        top_k_prediction_indices = prediction_indices_sorted[:, :k]
        intersection = np.take_along_axis(targets, indices=top_k_prediction_indices, axis=1)
        # check if current recommendation is relevant or not -> relevant: 1, irrelevant: 0
        indicator_fn = intersection[:, -1]

        # sum up the relevant recommendations up to k recommendations
        intersection_sum = intersection.sum(axis=1)
        # only keep precision if current recommendation is relevant (indicator fn)
        precision = intersection_sum / k * indicator_fn
        precision_scores.append(precision)

    minimum = np.minimum(top_k, targets.sum(axis=1))
    average_precision_scores = sum(precision_scores) / minimum
    return average_precision_scores, indices_annotated


METRICS = {
    "clf": SklearnClassificationReport,
    "f1": F1Score,
    "acc": MaskedAccuracy,
    "map": MeanAveragePrecision,
    "loss": MeanLoss,
    "emd": EarthMoversDistance,
}
