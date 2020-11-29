import pdb
import os
import sys
import tensorflow as tf
from pathlib import Path
import json
from ssd import SSD_Mobilenet
from common.eval import coco_eval, coco_by_category
from ood.preprocess import *
from ood.utils import *

data_root = Path("./dota_sports_data")
results_root = Path("./experiments_gridsearch/1606684100/4")

with open(results_root / "results.json", "r") as r:
    results_meta = json.load(r)
    records_dir  = Path( results_meta["dataset_dir"] )
with open(records_dir / "meta.json") as r:         dataset_meta = json.load(r)
with open(records_dir/"meta.json", "r") as r: dataset_meta = json.load(r)
min_coverage       = dataset_meta["min_coverage"]
win_set            = dataset_meta["win_set"]
desired_categories = dataset_meta["classes"]
n_categories = len(desired_categories)


image_dir = os.path.join(data_root, "validation/images")
annotations_path = os.path.join(data_root, "annotations/validation.json")
annotations = get_annotations(annotations_path)
desired_ids = construct_desired_ids(desired_categories, annotations['categories'])
# construct dictionaries containing info about images
(images_dict, file_name_dict) = construct_dicts(annotations)
# create category index in the correct format for retraining and detection
category_index = construct_category_index(annotations, desired_categories)
label_id_offsets = calculate_label_id_offsets(category_index)


#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

with tf.device("/device:GPU:2"):
    model = SSD_Mobilenet(n_categories)
    model.nms_redund_threshold = 0.75
    checkpoint_root = results_root / "bestpoints"
    checkpoint = tf.train.Checkpoint(model=model.variables)
    checkpoint.restore( tf.train.latest_checkpoint(checkpoint_root) )

    preprocessor = Preprocessor(images_dict, file_name_dict, image_dir, annotations, category_index, win_set=win_set, min_coverage=.3)
    for (window_np, box_set, class_set) in preprocessor.iterate():
        window_tensor = tf.convert_to_tensor(window_np, dtype=tf.float32)
        window_tensor = tf.expand_dims(window_tensor, axis=0)
        preproc, shapes = model.preprocess( window_tensor )
        raw_pred_dict   = model.predict(preproc, shapes)
        pred_dict       = model.postprocess(raw_pred_dict, shapes)
        pred_boxes = tf.squeeze(pred_dict["detection_boxes"]).numpy()
        pred_classes = tf.squeeze(pred_dict["detection_classes"]).numpy()
        pred_scores = tf.squeeze(pred_dict["detection_scores"]).numpy()
    
        plot_detections(window_np, pred_boxes, pred_classes, pred_scores, category_index)
        plt.show()
