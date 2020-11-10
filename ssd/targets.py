# 3rd Party
import tensorflow as tf
from object_detection.matchers import hungarian_matcher
from object_detection.core import target_assigner, region_similarity_calculator
from object_detection.core.box_list import BoxList
from object_detection.box_coders import faster_rcnn_box_coder
# Python STL
from typing import List, Tuple

def compute_ssd_targets(gt_boxes_list: List[tf.Tensor],
                    gt_labels_list: List[tf.Tensor],
                    default_boxes,
                    box_coder,
                    unmatched_class_label) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:
    """
    """

    def concat_zero(arr):
        zero = tf.zeros( [*arr.shape[:-1], 1] )
        return tf.concat([zero, arr], axis=-1)
    gt_labels_list = list(map(concat_zero, gt_labels_list))


    assigner = target_assigner.TargetAssigner(
                region_similarity_calculator.IouSimilarity(),
                hungarian_matcher.HungarianBipartiteMatcher(),
                box_coder
    ) 
    cls_targets = []
    cls_weights = []
    reg_targets = []
    reg_weights = []
    matched = []
    for (gtb_arr, gtl_arr) in zip(gt_boxes_list, gt_labels_list):
        result = assigner.assign(
            anchors=default_boxes,
            groundtruth_boxes=BoxList(gtb_arr),
            groundtruth_labels=gtl_arr,
            unmatched_class_label=unmatched_class_label
        )
        cls_targets.append(result[0])
        cls_weights.append(result[1])
        reg_targets.append(result[2]) 
        reg_weights.append(result[3])
        matched.append(result[4])
    # By the way, one default box is matched to at most one gt box,
    #  but one gt box can have multiple default boxes matched.
    #  Thus, we produce a mapping from default boxes to gt boxes
    #  which is not one-to-one or onto.
    
    cls_weights = tf.stack(cls_weights, axis=0)
    # This assertion will pass every time. I don't know
    #  what the point of returning an array of all 1s is.
    #  Maybe some of the other assigner classes return
    #  more informative weights.
    # assert tf.math.reduce_all(cls_weights == 1.)
    cls_targets = tf.stack(cls_targets, axis=0)
    reg_targets = tf.stack(reg_targets, axis=0)
    reg_weights = tf.stack(reg_weights, axis=0)
    # An entry in [0,num_gt_boxes) indicates a match
    # An entry in {-2,-1} indicates a nonmatch
    matched = tf.stack(matched, axis=0)
    
    return (cls_targets, cls_weights, reg_targets, reg_weights, matched)

