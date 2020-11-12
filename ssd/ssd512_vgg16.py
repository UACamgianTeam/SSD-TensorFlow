# 3rd Party
from object_detection.core.box_list import BoxList
import tensorflow as tf
import numpy as np
# Python STL
from typing import Tuple,Dict,List
# Local
from .components import horizontal_multibox_layer, class_multibox_layer, smooth_l1
from .boxes import multilayer_default_boxes
from .abstract_ssd import AbstractSSD

class SSD512_VGG16(AbstractSSD):

    #################### STATIC METHODS ########################

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
                nms_redund_threshold=nms_redund_threshold)

    @staticmethod
    def from_checkpoint(nonbackground_classes : int,
                        checkpoint_path       : str,
                        quadrangles           : bool  = False,
                        loc_weight            : float = 1.,
                        nms_redund_threshold  : float = 0.2,
                        top_k_per_class       : int   = 100):
        model = SSD512_VGG16(nonbackground_classes,
                vgg_weights_path=None,
                quadrangles=quadrangles,
                loc_weight=loc_weight,
                nms_redund_threshold=nms_redund_threshold)
        model.checkpoint.restore(checkpoint_path)
        return model

    @staticmethod
    def get_input_dims():
        return (512, 512)

    @staticmethod
    def get_feature_shapes():                                                              
        return [ (64,64), (32,32), (16,16), (8,8), (4,4), (2,2), (1,1) ]

    @staticmethod
    def get_ratios():
        return [
            [1, 1, 2, 1/2],           # conv4
            [1, 1, 2, 1/2, 3, 1/3],   # conv7
            [1, 1, 2, 1/2, 3, 1/3],   # conv8_2
            [1, 1, 2, 1/2, 3, 1/3],   # conv9_2
            [1, 1, 2, 1/2],           # conv10_2
            [1, 1, 2, 1/2],           # conv11_2
            [1, 1, 2, 1/2, 3, 1/3],   # conv12_2
        ]
    @staticmethod
    def get_scales():
        ratios = SSD512_VGG16.get_ratios()
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
        return scales

    @staticmethod
    def get_default_boxes():
        feature_shapes = SSD512_VGG16.get_feature_shapes()
        ratios         = SSD512_VGG16.get_ratios()
        scales         = SSD512_VGG16.get_scales()
        default_boxes  = multilayer_default_boxes(feature_shapes, ratios, scales)
        default_boxes  = tf.constant(default_boxes, dtype=tf.float32)
        return BoxList(default_boxes)

    @staticmethod
    def get_unmatched_class_target(num_classes):
        return AbstractSSD.get_unmatched_class_target(num_classes)

    def __init__(self,
            nonbackground_classes: int,
            vgg_weights_path = None,
            quadrangles=False,
            loc_weight=1.,
            nms_redund_threshold=0.2,
            top_k_per_class          : int   = 100):

        super(SSD512_VGG16, self).__init__(nonbackground_classes=nonbackground_classes,
                                            top_k_per_class=top_k_per_class,
                                            nms_redund_threshold=nms_redund_threshold,
                                            loc_weight=loc_weight)


        self.input_dims      = SSD512_VGG16.get_input_dims()
        self._feature_shapes = SSD512_VGG16.get_feature_shapes()
        self._ratios         = SSD512_VGG16.get_ratios()
        self._scales         = SSD512_VGG16.get_scales()
        self.quadrangles     = quadrangles

        default_boxes = multilayer_default_boxes(self._feature_shapes, self._ratios, self._scales)
        default_boxes = tf.constant(default_boxes, dtype=tf.float32)
        self._default_boxes = BoxList(default_boxes)

        self._load_features(vgg_weights_path = vgg_weights_path)
        assert len(self._feature_shapes) == len(self._feature_maps)

        # Network output
        self.coordinates = [] 
        self.logits = []
        for (feature_map, ratio_set) in zip(self._feature_maps, self.ratios):
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

        self._checkpoint = tf.train.Checkpoint(model=self.model)
        
  
    @tf.function
    def preprocess(self, x) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.resize(x, self.input_dims), None 
        
    def predict(self, image, shapes):
        prediction_dict = self.model(image)
        return prediction_dict

   

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
        self._feature_maps = [conv4, conv7, conv8, conv9, conv10, conv11, conv12]

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
    def default_boxes(self):
        return self._default_boxes
    @property
    def checkpoint(self) -> tf.train.Checkpoint:
        return self._checkpoint
    

__all__ = ["SSD512_VGG16"]
