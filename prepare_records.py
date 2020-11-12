#!/usr/bin/env python3

# Python STL
import pdb
import os
# 3rd Party
import tensorflow as tf
from PIL import Image
from object_detection.box_coders import faster_rcnn_box_coder
# Team-Internal
from ood.preprocess import * 
from ood.utils import get_annotations, visualize_image_set

# Local
from ssd import SSD512_VGG16
from ssd.targets import compute_ssd_targets
from ssd.data    import *


def main(out_dir,
        image_dir,
        annotations_path,
        desired_categories,
        win_set = None,
        num_out_files = 10):
    os.makedirs(out_dir,exist_ok=True)
    
    annotations = get_annotations(annotations_path)
    desired_ids = construct_desired_ids(desired_categories, annotations['categories'])
    # construct dictionaries containing info about images
    (images_dict, file_name_dict) = construct_dicts(annotations)
    # create category index in the correct format for retraining and detection
    category_index = construct_category_index(annotations, desired_categories)
    label_id_offsets = calculate_label_id_offsets(category_index)
    
    num_nonbackground_classes = len(desired_categories)
    # set windowing information (size of window and stride); these values taken from DOTA paper
    
    default_boxes = SSD512_VGG16.get_default_boxes()
    unmatched_class_target = SSD512_VGG16.get_unmatched_class_target(num_nonbackground_classes)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    
    fileprefix = "ssd512_examples"
    filenames = [os.path.join(out_dir, f"{fileprefix}_{i}.tfrecord") for i in range(num_out_files) ]
    writers   = [tf.io.TFRecordWriter(f) for f in filenames]
    
    count = 0    
    preprocessor = Preprocessor(images_dict, file_name_dict, image_dir, annotations, category_index)
    for (window_np, gt_boxes, gt_classes) in preprocessor.iterate(win_set):
        [gt_classes] = map_category_ids_to_index(label_id_offsets, [gt_classes])
        gt_classes   = tf.constant(gt_classes, dtype=tf.int32)
        gt_classes   = tf.one_hot(gt_classes, num_nonbackground_classes, dtype=tf.float32)
        zeros        = tf.zeros([gt_classes.shape[0], 1], gt_classes.dtype)
        gt_classes   = tf.concat([zeros, gt_classes], axis=1)
        targets_set  = compute_ssd_targets([gt_boxes], [gt_classes], default_boxes, box_coder, unmatched_class_target)
        targets_set  = [tf.squeeze(t) for t in targets_set]
        [cls_targets, cls_weights, reg_targets, reg_weights, matched] = targets_set
        serialized   = serialize_ssd_example(window_np, *targets_set)
        writers[count % len(writers)].write(serialized)
        count += 1
    for w in writers:
        w.close()

if __name__ == "__main__":
    win_height = 1024
    win_width  = 1024
    win_stride_vert  = 512
    win_stride_horiz = 512
    win_set = (win_height, win_width, win_stride_vert, win_stride_horiz) # windowing information
    desired_categories = {'tennis-court','soccer-ball-field','ground-track-field','baseball-diamond'}
    def write_dataset(partition):
        data_path = f"{os.environ['HOME']}/Datasets/dota_sports_data"
        image_dir = data_path + f"/{partition}/images"
        annotations_path = data_path + f"/annotations/{partition}.json"
        out_dir = f"./out/{partition}"
        main(out_dir, image_dir, annotations_path, desired_categories, win_set)
    write_dataset("train")
    write_dataset("validation")