# 3rd Party
from object_detection.metrics.coco_tools import COCOWrapper,COCOEvalWrapper
# Python STL
import os
import shutil
import json
from time import time
# oriented-object-detection
from ood.detect import run_inference
from ood.evaluate import write_window_validation_file, write_window_results, evaluate


def coco_eval(data_path,
        annotations,
        images_np,
        images_dict,
        desired_ids,
        label_id_offsets,
        model,
        results_dir="/tmp"):

    timestamp   = int(time())
    results_dir = os.path.join(results_dir, f"coco_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "window_predictions.json")
    labels_path = os.path.join(results_dir, "window_groundtruths.json")
    
    test_images_dict, predicted_boxes, predicted_classes, predicted_scores = run_inference(images_np, images_dict, label_id_offsets, model)

    write_window_results(results_path, test_images_dict, min_threshold=.01)
    write_window_validation_file(labels_path, annotations, test_images_dict)
    
    with open(labels_path, "r") as r: cocoGt = COCOWrapper(json.load(r))
    cocoDt = cocoGt.loadRes(results_path)
    cocoEval = COCOEvalWrapper(cocoGt, cocoDt, iou_type="bbox", agnostic_mode=True)
    cocoEval.params.catIds = list(desired_ids) # set category ids we want to evaluate on
    metrics = cocoEval.ComputeMetrics()


    shutil.rmtree(results_dir)
    return metrics 

__all__ = ["coco_eval"]
