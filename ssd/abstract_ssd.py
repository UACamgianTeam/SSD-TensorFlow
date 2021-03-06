# 3rd Party
from object_detection.core.box_list import BoxList
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import oriented_box_coder      # From our fork of TFObj
import tensorflow as tf
import numpy as np
# Python STL
from typing import Tuple,Dict,List
import pdb
# Local
from .components import horizontal_multibox_layer, class_multibox_layer, smooth_l1
from .targets import compute_ssd_targets

class AbstractSSD(object):


    ##### ABSTRACT: Must Implement in subclass #####

    @property
    def feature_shapes(self):
        raise NotImplementedError
    @property
    def ratios(self):
        raise NotImplementedError
    @property
    def scales(self):
        raise NotImplementedError
    def preprocess(self, image):
        raise NotImplementedError
    def predict(self, image, shapes):
        raise NotImplementedError
    @property
    def anchors(self) -> BoxList:
        raise NotImplementedError



    ##### CONCRETE #####

    def __init__(self,
            nonbackground_classes : int,
            loc_weight : float = 1.,
            conf_weight : float = 1.,
            nms_redund_threshold : float = 0.2,
            min_score_threshold : float  = 0.01,
            top_k_per_class : int = 100,
            ohem : bool = False,
            neg_loss_weight : float = 1.,
            obb             : bool  = False):

        self.num_nonbackground_classes = nonbackground_classes
        self.num_classes = nonbackground_classes + 1

        self.obb         = obb
        self.box_coder   = oriented_box_coder.OrientedBoxCoder() if obb else faster_rcnn_box_coder.FasterRcnnBoxCoder()

        self._ohem = ohem
        self._neg_loss_weight = neg_loss_weight
        self._loc_weight = loc_weight
        self._conf_weight = conf_weight
        self._nms_redund_threshold = nms_redund_threshold
        self._min_score_threshold = min_score_threshold
        self._top_k_per_class = top_k_per_class
        pass

    @staticmethod
    def get_unmatched_class_target(num_classes):
        return tf.constant([1] + [0]*num_classes, dtype=tf.float32)

    @property
    def n_anchors(self):
        return self.anchors.get().shape[0]

    @property
    def nms_redund_threshold(self):
        return self._nms_redund_threshold
    @nms_redund_threshold.setter
    def nms_redund_threshold(self, v):
        self._nms_redund_threshold = v

    @property
    def min_score_threshold(self):
        return self._min_score_threshold
    @min_score_threshold.setter
    def min_score_threshold(self, v):
        self._min_score_threshold = v

    @property
    def top_k_per_class(self):
        return self._top_k_per_class
    @top_k_per_class.setter
    def top_k_per_class(self, v):
        self._top_k_per_class = v

    @property
    def loc_weight(self) -> float:
        return self._loc_weight
    @loc_weight.setter
    def loc_weight(self, v: float):
        self._loc_weight = v

    @property
    def conf_weight(self) -> float:
        return self._conf_weight
    @conf_weight.setter
    def conf_weight(self, v: float):
        self._conf_weight = v

    @property
    def ohem(self) -> float:
        return self._ohem
    @ohem.setter
    def ohem(self, v: float):
        self._ohem = v
    @property
    def neg_loss_weight(self) -> float:
        return self._neg_loss_weight
    @neg_loss_weight.setter
    def neg_loss_weight(self, v: float):
        self._neg_loss_weight = v

    def postprocess(self, prediction_dict, shapes):
        logits = tf.squeeze(prediction_dict["class_predictions_with_background"])
        probs = tf.keras.layers.Softmax()(logits)
    
        encodings = tf.squeeze(prediction_dict["box_encodings"])

        hbb_boxes = self.box_coder.decode(encodings, self.anchors)
        hbb_boxes = hbb_boxes.get() # The actual array of boxes

        # We use the HBBs for non-max suppression, but we gather our results from either the HBBs
        # or the OBBs
        if self.obb:
            gather_boxes = self.box_coder.batch_obb_decode(encodings, self.anchors)
        else:
            gather_boxes = hbb_boxes
    
    
        probs_rank = tf.rank(probs)
        perm = tf.concat([[probs_rank-1], tf.range(probs_rank - 1, dtype=tf.int32)], axis=0)
        probs = tf.transpose(probs, perm)
    
        probs = probs[1:] # Exclude background class
        # Since we're excluding the background class, we go back to 0-indexing
        class_indices = tf.range(0, self.num_nonbackground_classes, dtype=tf.int32)
    
        def nms_class(scores, class_index):
            (best_inds, best_scores) = tf.image.non_max_suppression_with_scores(hbb_boxes,
                                                                                scores,
                                                                                max_output_size=self.top_k_per_class,
                                                                                iou_threshold=self.nms_redund_threshold,
                                                                                score_threshold=self.min_score_threshold)
            best_boxes = tf.gather(gather_boxes, best_inds, axis=0)
            best_labels = tf.fill( tf.shape(best_boxes)[:-1], class_index)
            return [best_boxes, best_scores, best_labels]
    
        def nms_accum(accum, args):
            [scores, class_index] = args
            [best_boxes, best_scores, best_labels] = nms_class(scores, class_index)
            return [
                    tf.concat([accum[0], best_boxes], axis=0),
                    tf.concat([accum[1], best_scores], axis=0),
                    tf.concat([accum[2], best_labels], axis=0)
            ]
    
        initializer = nms_class(probs[0], class_indices[0])
        if len(probs) == 1:
            [out_boxes, out_scores, out_classes] = initializer
        else:
            [out_boxes, out_scores, out_classes] = tf.foldl(nms_accum, [probs[1:],class_indices[1:]], initializer=initializer)
        
        
        sorted_indices = tf.argsort(out_scores, direction='DESCENDING')
        if len(sorted_indices) > self.top_k_per_class: sorted_indices = sorted_indices[:self.top_k_per_class]
        out_boxes = tf.gather(out_boxes, sorted_indices)
        out_scores = tf.gather(out_scores, sorted_indices)
        out_classes = tf.gather(out_classes, sorted_indices)
        
        # FIXME: These three lines assume we had a batch of one
        out_boxes = tf.expand_dims(out_boxes, axis=0)
        out_scores = tf.expand_dims(out_scores, axis=0)
        out_classes = tf.expand_dims(out_classes, axis=0)
        
        return {"detection_boxes": out_boxes,
                "detection_scores": out_scores,
                "detection_classes": out_classes} 

    def provide_groundtruth(self, groundtruth_boxes_list: List[tf.Tensor], groundtruth_labels_list: List[tf.Tensor]):
    
        unmatched_class_label = tf.constant([1] + [0 for _ in range(self.num_nonbackground_classes)], dtype=tf.float32)
        output = compute_ssd_targets(groundtruth_boxes_list,
                                groundtruth_labels_list,
                                self.anchors,
                                self.box_coder,
                                AbstractSSD.get_unmatched_class_target())
        self.provide_groundtruth_direct(*output)
    
    def provide_groundtruth_direct(self,
                                    cls_targets : tf.Tensor,
                                    cls_weights : tf.Tensor,
                                    reg_targets : tf.Tensor,
                                    reg_weights : tf.Tensor,
                                    matched     : tf.Tensor):
        self._cls_targets = cls_targets
        self._cls_weights = cls_weights
        self._reg_targets = reg_targets
        self._reg_weights = reg_weights
        self._matched = matched

    def loss(self, prediction_dict) -> Dict[str,tf.Tensor]:
        n_matched = tf.math.reduce_sum( tf.where(self._matched >= 0, 1, 0), axis=-1 )
        ####### Confidence/Class Loss #######
        logits = prediction_dict["class_predictions_with_background"]
        class_loss = tf.nn.softmax_cross_entropy_with_logits(self._cls_targets, logits, axis=-1)

        # Class loss for all the matched boxes (the easy part)
        pos_class_loss = tf.where(self._matched >= 0, class_loss, 0)
        pos_class_loss = tf.math.reduce_sum(pos_class_loss, axis=-1) # Reduce across (the matched) boxes
        
        # Hard negative mining (the hard part--pun intended)
        neg_class_loss = tf.sort( tf.where(self._matched >= 0, 0, class_loss), direction="DESCENDING", axis=-1)

        if self.ohem:
            top_k = tf.math.minimum(n_matched * 3, self.n_anchors - n_matched)
        else:
            top_k = self.n_anchors - n_matched # Equivalent to not doing OHEM

        # For a given image, we want the ratio of unmatched to matched default boxes to be at most 3:1 (See page 6 of original SSD paper)
        # Sum the loss of the top-k boxes for each image
        def top_k_loss(args):
            (image_loss, k) = args
            return tf.math.reduce_sum(image_loss[:k], axis=-1)
        mined_neg_loss = tf.map_fn(top_k_loss, (neg_class_loss, top_k), fn_output_signature=tf.float32)
        class_loss_by_image = pos_class_loss + self.neg_loss_weight*mined_neg_loss
        
        ####### Localization Loss #######
        pred_boxes = prediction_dict["box_encodings"]
        loc_loss = self._reg_targets - pred_boxes
        loc_loss = smooth_l1(loc_loss, alpha=1.)
        loc_loss = tf.math.reduce_sum(loc_loss, axis=-1) # Sum across box coordinates
        loc_loss = self._reg_weights * loc_loss 
        loc_loss = tf.math.reduce_sum(loc_loss, axis=-1) # Sum across default boxes

        cast_n = tf.cast(n_matched, tf.float32)
        total_loc_loss = tf.math.reduce_sum( loc_loss / cast_n )
        total_class_loss = tf.math.reduce_sum( class_loss_by_image / cast_n)
        losses_dict = {
            "Localization": total_loc_loss,
            "Confidence": total_class_loss,
            "WeightedTotal": self.conf_weight*total_class_loss + self.loc_weight*total_loc_loss
        }
        return losses_dict


__all__ = ["AbstractSSD"]
