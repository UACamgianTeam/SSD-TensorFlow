#!/usr/bin/env python3

# Python STL
import pdb
import os
import sys
import argparse
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
        desired_categories = None,
        win_set = None,
        min_coverage=.3,
        num_out_files = 30):
    os.makedirs(out_dir,exist_ok=True)
    
    annotations = get_annotations(annotations_path)
    if not desired_categories:
        desired_categories = ['plane','baseball-diamond','bridge','ground-track-field','small-vehicle','large-vehicle','ship','tennis-court','basketball-court',
                        'storage-tank','soccer-ball-field','roundabout','harbor','swimming-pool','helicopter']
    desired_ids = construct_desired_ids(desired_categories, annotations['categories'])
    assert len(desired_ids) == len(desired_categories)
    # construct dictionaries containing info about images
    (images_dict, file_name_dict) = construct_dicts(annotations)
    # create category index in the correct format for retraining and detection
    category_index = construct_category_index(annotations, desired_categories)
    label_id_offsets = calculate_label_id_offsets(category_index)
    
    num_nonbackground_classes = len(desired_categories)
    input_dims = SSD512_VGG16.get_input_dims()
    
    default_boxes = SSD512_VGG16.get_default_boxes()
    unmatched_class_target = SSD512_VGG16.get_unmatched_class_target(num_nonbackground_classes)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    
    fileprefix = "ssd512_examples"
    filenames = [os.path.join(out_dir, f"{fileprefix}_{i}.tfrecord") for i in range(num_out_files) ]
    writers   = [tf.io.TFRecordWriter(f) for f in filenames]
    
    count = 0    
    preprocessor = Preprocessor(images_dict, file_name_dict, image_dir, annotations, category_index)
    for (window_np, gt_boxes, gt_classes) in preprocessor.iterate(win_set, min_coverage=min_coverage):
        window_np = tf.image.resize(window_np, input_dims)

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

    out_meta = os.path.join(out_dir, "meta.json")
    with open(out_meta, "w") as w:
        json_string = json.dumps({
            "num_examples": count,
            "classes": list(desired_categories)
        }, indent=2)
        w.write(json_string + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TF Records from DOTA data for training an SSD")

    parser.add_argument("--in_dir", type=os.path.abspath, help="Input directory", required=True)
    parser.add_argument("--out_dir", type=os.path.abspath, help="Output directory", required=True)

    parser.add_argument("--sports", action="store_true", help="Limit to four sports categories")
    parser.add_argument("--partition", default="training", help="'train' or 'validation'")
    parser.add_argument("--out-files", type=int, default=30, help="Number of record files to generate")

    parser.add_argument("--min-coverage", type=float, default=0.3, help="For annotation to be preserved after windowing, at least this much must be present in the window")

    args = parser.parse_args()


    if args.sports:
        desired_categories = ["tennis-court", "baseball-diamond", "ground-track-field", "soccer-ball-field"]
    else:
        desired_categories = None
        

    image_dir        = os.path.join(args.in_dir, f"{args.partition}/images")
    annotations_path = os.path.join(args.in_dir, f"annotations/{args.partition}.json")
    partition_out = os.path.join(args.out_dir, args.partition)

    win_height = 1024
    win_width  = 1024
    win_stride_vert  = 512
    win_stride_horiz = 512
    win_set = (win_height, win_width, win_stride_vert, win_stride_horiz) # windowing information
    main(partition_out,
            image_dir,
            annotations_path,
            win_set=win_set,
            min_coverage=args.min_coverage,
            desired_categories=desired_categories,
            num_out_files=args.out_files
    )
