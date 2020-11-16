# 3rd Party
from object_detection.metrics.coco_tools import COCOWrapper,COCOEvalWrapper
# Python STL
import pdb
import os
import shutil
import json
from time import time
# oriented-object-detection
from ood.detect import run_inference
from ood.evaluate import write_window_validation_file, write_window_results, evaluate
from ood.utils import *
from ood.preprocess import *

def coco_eval(model, annotations_path, image_dir, desired_categories, win_set=None, min_coverage=.3, results_dir="/tmp"):
    timestamp   = int(time())
    results_dir = os.path.join(results_dir, f"coco_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "window_predictions.json")
    labels_path = os.path.join(results_dir, "window_groundtruths.json")

    annotations  = get_annotations(annotations_path)
    (images_dict, file_name_dict)   = construct_dicts(annotations)
    desired_ids = construct_desired_ids(desired_categories, annotations['categories'])
    category_index = construct_category_index(annotations, desired_categories)
    label_id_offsets = calculate_label_id_offsets(category_index)


    preprocessor = Preprocessor(images_dict,
                                file_name_dict,
                                image_dir,
                                annotations,
                                category_index,
                                win_set=win_set,
                                min_coverage=min_coverage)
    test_images_dict, predicted_boxes, predicted_classes, predicted_scores = run_inference(model, preprocessor, label_id_offsets)
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
