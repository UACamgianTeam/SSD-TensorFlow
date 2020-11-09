# 3rd Party
from object_detection.core import target_assigner, region_similarity_calculator, box_list
from object_detection.matchers import hungarian_matcher
from object_detection.box_coders import faster_rcnn_box_coder
import tensorflow as tf
import numpy as np
# Python STL
from typing import Tuple,Dict,List
# Local
from .components import horizontal_multibox_layer, class_multibox_layer, smooth_l1
from .boxes import relative_box_coordinates

class SSD512_VGG16(object):

    @staticmethod
    def from_scratch(nonbackground_classes : int,
                     vgg_weights_path      : str,
                     quadrangles           : bool  = False,
                     loc_weight            : float = 1.,
                     nms_redund_threshold  : float = 0.2,
                     top_k_per_class       : int   = 100,
                     predictor_subset      : List[int] = None):
        return SSD512_VGG16(nonbackground_classes,
                vgg_weights_path=vgg_weights_path,
                quadrangles=quadrangles,
                loc_weight=loc_weight,
                nms_redund_threshold=nms_redund_threshold,
                predictor_subset=predictor_subset)

    @staticmethod
    def from_checkpoint(nonbackground_classes : int,
                        checkpoint_path       : str,
                        quadrangles           : bool  = False,
                        loc_weight            : float = 1.,
                        nms_redund_threshold  : float = 0.2,
                        top_k_per_class       : int   = 100,
                        predictor_subset      : List[int] = None):
        model = SSD512_VGG16(nonbackground_classes,
                vgg_weights_path=None,
                quadrangles=quadrangles,
                loc_weight=loc_weight,
                nms_redund_threshold=nms_redund_threshold,
                predictor_subset=predictor_subset)
        model.checkpoint.restore(checkpoint_path)
        return model

    def __init__(self,
            nonbackground_classes: int,
            vgg_weights_path = None,
            quadrangles=False,
            loc_weight=1.,
            nms_redund_threshold=0.2,
            top_k_per_class          : int   = 100,
            predictor_subset=None):
        self.input_dims = (512,512)
        self._nms_redund_threshold = nms_redund_threshold
        self._top_k_per_class = top_k_per_class
        self._loc_weight = loc_weight
        self._feature_shapes = [ (64,64), (32,32), (16,16), (8,8), (4,4), (2,2), (1,1) ]
        self.quadrangles = quadrangles
        self.unmatched_class_label = tf.constant([1]+[0 for _ in range(nonbackground_classes)], dtype=tf.float32)
        self.num_nonbackground_classes = nonbackground_classes
        self.num_classes = nonbackground_classes + 1

        self._load_box_descriptions()

        self._load_features(vgg_weights_path = vgg_weights_path)
        assert len(self._feature_shapes) == len(self.feature_maps)

        if not predictor_subset: predictor_subset = list(range(len(self._ratios)))
        self._feature_shapes = [self._feature_shapes[i] for i in predictor_subset]
        self.feature_maps = [self.feature_maps[i] for i in predictor_subset]
        self._ratios = [self._ratios[i] for i in predictor_subset]
        self._scales = [self._scales[i] for i in predictor_subset]

        # Network output
        self.coordinates = [] 
        self.logits = []
        for (feature_map, ratio_set) in zip(self.feature_maps, self.ratios):
            num_anchors = len(ratio_set)
            c = horizontal_multibox_layer(num_anchors)(feature_map)
            c = tf.reshape(c, [
                            -1, # Batch dimension,
                            c.shape[1] * c.shape[2] * num_anchors, # Y * X * box_shape
                            8 if self.quadrangles else 4
            ])
            self.coordinates.append(c)

            l = class_multibox_layer(num_anchors,self.num_classes)(feature_map)
            l = tf.reshape(l, [
                            -1, # Batch dimension
                            l.shape[1] * l.shape[2] * num_anchors, # Y * X * box_shape,
                            self.num_classes
            ])
            self.logits.append(l)
        self.coordinates = tf.concat(self.coordinates, axis=1) # [batch,box,coords]
        self.logits = tf.concat(self.logits, axis=1)           # [batch,box,classes]

        self.model = tf.keras.Model(
            inputs=[self.input],
            outputs={
                "bbox": self.coordinates,
                "logit": self.logits
            }
        )

        # Coordinates of default boxes
        self.default_boxes = []
        for (feature_shape, ratio_set, scale_set) in zip(self.feature_shapes, self.ratios, self.scales):
            coords = relative_box_coordinates(feature_shape,ratio_set,scale_set)
            coords = np.reshape(coords, [
                                        feature_shape[0]*feature_shape[1]*len(ratio_set),
                                        8 if self.quadrangles else 4
            ])
            self.default_boxes.append(coords)
        self.default_boxes = tf.constant(np.concatenate(self.default_boxes, axis=0), dtype=tf.float32)
        assert np.all(self.coordinates.shape[1:] == self.default_boxes.shape)
        
        self.n_default_boxes = self.default_boxes.shape[0] # For convenience
        self.default_boxes = box_list.BoxList( self.default_boxes  )
        
        self._checkpoint = tf.train.Checkpoint(model=self.model)
        
        self._top_k = tf.constant(200, dtype=tf.int32)
  
    @tf.function
    def preprocess(self, x) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.resize(x, self.input_dims), None 
        
    def predict(self, image, shapes):
        prediction_dict = self.model(image)
        return prediction_dict

    @tf.function
    def postprocess(self, prediction_dict, shapes):
        logits = tf.squeeze(prediction_dict["logit"])
        probs = tf.keras.layers.Softmax()(logits)
    
        boxes = tf.squeeze(prediction_dict["bbox"])
        boxes = faster_rcnn_box_coder.FasterRcnnBoxCoder().decode(boxes, self.default_boxes)
        boxes = boxes.get() # The actual array of boxes
    
    
        probs_rank = tf.rank(probs)
        perm = tf.concat([[probs_rank-1], tf.range(probs_rank - 1, dtype=tf.int32)], axis=0)
        probs = tf.transpose(probs, perm)
    
        probs = probs[1:] # Exclude background class
        # Since we're excluding the background class, we go back to 0-indexing
        class_indices = tf.range(0, self.num_nonbackground_classes, dtype=tf.int32)
    
        def nms_class(scores, class_index):
            (best_inds, best_scores) = tf.image.non_max_suppression_with_scores(boxes,scores,
                                                                                max_output_size=self.top_k_per_class,
                                                                                iou_threshold=self.nms_redund_threshold,
                                                                                score_threshold=0.01)
            best_boxes = tf.gather(boxes, best_inds, axis=0)
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
        if len(sorted_indices) > self._top_k: sorted_indices = sorted_indices[:self.top_k]
        out_boxes = tf.gather(out_boxes, sorted_indices)
        out_scores = tf.gather(out_scores, sorted_indices)
        out_classes = tf.gather(out_classes, sorted_indices)
        
        out_boxes = tf.expand_dims(out_boxes, axis=0)
        out_scores = tf.expand_dims(out_scores, axis=0)
        out_classes = tf.expand_dims(out_classes, axis=0)
        
        return {"detection_boxes": out_boxes,
                "detection_scores": out_scores,
                "detection_classes": out_classes} 
   

    def _load_features(self, vgg_weights_path : str = None, ckpt_path: str = None):
        self.input = tf.keras.Input([*self.input_dims, 3], dtype=tf.uint8)
        vgg_preprocessed = tf.keras.applications.vgg16.preprocess_input(self.input)
        
        ### Load VGG 16 Base Network ###
        if not ckpt_path:
            vgg_base = tf.keras.applications.VGG16(
                include_top = False,       # Don't need the fully-connected layers
                input_shape = (*self.input_dims,3), 
                pooling = None,            # SSD does not apply global max pooling on the last VGG layer
                weights = vgg_weights_path if vgg_weights_path else None # None causes random initilization
            )
        else:
            pass #TODO: Let users use ckpt_path to restore weights from checkpoint



        # Modify the VGG to only output some specific feature maps
        vgg_base = tf.keras.Model(
            inputs=vgg_base.input,
            outputs=[
                     vgg_base.get_layer("block4_conv3").output,
                     vgg_base.get_layer("block5_conv3").output
            ]
        )
        for layer in vgg_base.layers: layer.trainable = False


        [conv4, conv5] = vgg_base(vgg_preprocessed)
        # The SSD authors use a different pooling layer
        #  at the end of VGG than the original VGG authors
        pooled_conv5 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=[1,1], padding="same")(conv5)
        
        ### Layers added by SSD Authors ###
        # Redefine the factory functions for convenience
        def conv_factory(*args, **kwargs):
            return tf.keras.layers.Conv2D(*args, **kwargs, activation="relu")
        padding_factory = lambda: tf.keras.layers.ZeroPadding2D([1, 1])
        feature_map = pooled_conv5
        with tf.name_scope("ssd_blocks"):
            ## Strides of 1 ##
            with tf.name_scope("block6"):
                conv6_1 = conv_factory(filters=1024, kernel_size = [3,3], padding="same")(feature_map)
                conv6 = conv6_1
            with tf.name_scope("block7"):
                conv7_1 = conv_factory(filters=1024, kernel_size = [1,1], padding="same")(conv6)
                conv7 = conv7_1
            
            ## Strides of 2 ##
            with tf.name_scope("block8"):
                conv8_1 = conv_factory(filters=256, kernel_size = [1,1], padding="same")(conv7)
                padded_conv8_1 = padding_factory()(conv8_1)
                conv8_2 = conv_factory(filters=512, kernel_size = [3,3], strides=[2,2], padding="valid")(padded_conv8_1)
                conv8 = conv8_2
            with tf.name_scope("block9"):
                conv9_1 = conv_factory(filters=128, kernel_size = [1,1], padding="same")(conv8)
                padded_conv9_1 = padding_factory()(conv9_1)
                conv9_2 = conv_factory(filters=256, kernel_size = [3,3], strides=[2,2], padding="valid")(padded_conv9_1)
                conv9 = conv9_2
            with tf.name_scope("block10"):
                conv10_1 = conv_factory(filters=128, kernel_size = [1,1], padding="same")(conv9)
                padded_conv10_1 = padding_factory()(conv10_1)
                conv10_2 = conv_factory(filters=256, kernel_size = [3,3], strides=[2,2], padding="valid")(padded_conv10_1)
                conv10 = conv10_2
            with tf.name_scope("block11"):
                conv11_1 = conv_factory(filters=128, kernel_size = [1,1], padding="same")(conv10)
                padded_conv11_1 = padding_factory()(conv11_1)
                conv11_2 = conv_factory(filters=256, kernel_size = [3,3], strides=[2,2], padding="valid")(padded_conv11_1)
                conv11 = conv11_2
            with tf.name_scope("block12"):
                conv12_1 = conv_factory(filters=128, kernel_size = [1,1], padding="same")(conv11)
                padded_conv12_1 = padding_factory()(conv12_1)
                conv12_2 = conv_factory(filters=256, kernel_size = [4,4], strides=[2,2], padding="valid")(padded_conv12_1)
                conv12 = conv12_2
        self.feature_maps = [conv4, conv7, conv8, conv9, conv10, conv11, conv12]

    def _load_box_descriptions(self):
        ratios = [
            [1, 1, 2, 1/2],           # conv4
            [1, 1, 2, 1/2, 3, 1/3],   # conv7
            [1, 1, 2, 1/2, 3, 1/3],   # conv8_2
            [1, 1, 2, 1/2, 3, 1/3],   # conv9_2
            [1, 1, 2, 1/2],           # conv10_2
            [1, 1, 2, 1/2],           # conv11_2
            [1, 1, 2, 1/2, 3, 1/3],   # conv12_2
        ]
        scales = []
        s_min = 0.15
        s_max = 0.9
        m = len(ratios) - 1
        step_size = (s_max - s_min) / (m - 1)
        # Special set of ratios for conv4_3
        # (See footnote of SSD paper on page 7)
        scales.append([0.07*s_min] + [.07 for _ in range(len(ratios[0]) - 1)])
        for k in range(1, m + 1):
            scale_set  = []
            s_k = s_min + step_size * (k - 1)
            # The special ratio s_k * s_{k+1} for box 1:1 (pg. 6 of SSD paper)
            scale_set.append(s_k * (s_k + step_size))
            # Everyone else gets s_k
            scale_set.extend( (len(ratios[k]) - 1) * [s_k]  )
            scales.append(scale_set)
        self._ratios = ratios
        self._scales = scales

    @property
    def loc_weight(self) -> float:
        return self._loc_weight
    @loc_weight.setter
    def loc_weight(self, v: float):
        self._loc_weight = v

    @property
    def trainable_variables(self):
        return self.model.trainable_variables
    
    @property
    def ratios(self):
        return self._ratios
    @property
    def scales(self):
        return self._scales
    @property
    def feature_shapes(self):
        return self._feature_shapes
    @property
    def checkpoint(self) -> tf.train.Checkpoint:
        return self._checkpoint
    
    @property
    def top_k(self) -> tf.Tensor:
        return self._top_k
    @top_k.setter
    def top_k(self, v):
        self._top_k = tf.convert_to_tensor(v)

    @property
    def nms_redund_threshold(self):
        return self._nms_redund_threshold
    @nms_redund_threshold.setter
    def nms_redund_threshold(self, v):
        self._nms_redund_threshold = v

    @property
    def top_k_per_class(self):
        return self._top_k_per_class
    @top_k_per_class.setter
    def top_k_per_class(self, v):
        self._top_k_per_class = v

    def provide_groundtruth(self, groundtruth_boxes_list, groundtruth_labels_list):
        # Assume groundtruth_boxes_list has axes [image, box, coordinates]
        # Assume groundtruth_labels_list has axes [image, box]
        
        # From the user's perspective, foregound classes start at 0
        # We provide an extra background class for unmatched default boxes
        def concat_zero(arr):
            zero = tf.zeros( [*arr.shape[:-1], 1] )
            return tf.concat([zero, arr], axis=-1)
        groundtruth_labels_list = list(map(concat_zero, groundtruth_labels_list))
        
        if not self.quadrangles:
            self._provide_horizontal_groundtruth(groundtruth_boxes_list, groundtruth_labels_list)
        else:
          assert False

    def _provide_horizontal_groundtruth(self, groundtruth_boxes_list, groundtruth_labels_list):
        assigner = target_assigner.TargetAssigner(
                    region_similarity_calculator.IouSimilarity(),
                    hungarian_matcher.HungarianBipartiteMatcher(),
                    faster_rcnn_box_coder.FasterRcnnBoxCoder()
        ) 
        cls_targets = []
        cls_weights = []
        reg_targets = []
        reg_weights = []
        matched = []
        for (gtb_arr, gtl_arr) in zip(groundtruth_boxes_list, groundtruth_labels_list):
            result = assigner.assign(
                anchors=self.default_boxes,
                groundtruth_boxes=box_list.BoxList(gtb_arr),
                groundtruth_labels=gtl_arr,
                unmatched_class_label=self.unmatched_class_label
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
        
        self._cls_weights = tf.stack(cls_weights, axis=0)
        # This assertion will pass every time. I don't know
        #  what the point of returning an array of all 1s is.
        #  Maybe some of the other assigner classes return
        #  more informative weights.
        # assert tf.math.reduce_all(self._cls_weights == 1.)
        
        
        self._cls_targets = tf.stack(cls_targets, axis=0)
        self._reg_targets = tf.stack(reg_targets, axis=0)
        self._reg_weights = tf.stack(reg_weights, axis=0)
        
        
        # An entry in [0,num_gt_boxes) indicates a match
        # An entry in {-2,-1} indicates a nonmatch
        self._matched = tf.stack(matched, axis=0)
        self._n_matched = tf.math.reduce_sum( tf.where(self._matched >= 0, 1, 0), axis=-1 )

    def loss(self, prediction_dict, beta=0.001) -> Dict[str,tf.Tensor]:
        ####### Confidence/Class Loss #######
        logits = prediction_dict["logit"]
        class_loss = tf.nn.softmax_cross_entropy_with_logits(self._cls_targets, logits, axis=-1)

        # Class loss for all the matched boxes (the easy part)
        pos_class_loss = tf.where(self._matched >= 0, class_loss, 0)
        pos_class_loss = tf.math.reduce_sum(pos_class_loss, axis=-1) # Reduce across (the matched) boxes
        
        # Hard negative mining (the hard part--pun intended)
        neg_class_loss = tf.sort( tf.where(self._matched >= 0, 0, class_loss), direction="DESCENDING", axis=-1)
        # len(top_k) == num_images
        # For a given image, we want the ratio of unmatched to matched default boxes to be at most 3:1 (See page 6 of original SSD paper)
        #top_k = tf.math.minimum(self._n_matched * 3, self.n_default_boxes - self._n_matched)
        top_k = self.n_default_boxes - self._n_matched # Equivalent to not doing OHEM
        # Sum the loss of the top-k boxes for each image
        def top_k_loss(args):
            (image_loss, k) = args
            return tf.math.reduce_sum(image_loss[:k], axis=-1)
        mined_neg_loss = tf.map_fn(top_k_loss, (neg_class_loss, top_k), fn_output_signature=tf.float32)
        class_loss_by_image = pos_class_loss + beta*mined_neg_loss
        
        ####### Localization Loss #######
        pred_boxes = prediction_dict["bbox"]
        loc_loss = self._reg_targets - pred_boxes
        loc_loss = smooth_l1(loc_loss, alpha=1.)
        loc_loss = tf.math.reduce_sum(loc_loss, axis=-1) # Sum across box coordinates
        loc_loss = self._reg_weights * loc_loss 
        loc_loss = tf.math.reduce_sum(loc_loss, axis=-1) # Sum across default boxes

        cast_n = tf.cast(self._n_matched, tf.float32)
        total_loc_loss = tf.math.reduce_sum( loc_loss / cast_n )
        total_class_loss = tf.math.reduce_sum( class_loss_by_image / cast_n)
        losses_dict = {
            "Localization": total_loc_loss,
            "Confidence": total_class_loss,
            "WeightedTotal": total_class_loss + self.loc_weight*total_loc_loss
        }
        return losses_dict

__all__ = ["SSD512_VGG16"]
