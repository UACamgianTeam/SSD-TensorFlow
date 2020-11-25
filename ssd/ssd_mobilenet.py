# 3rd Party
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
# Python STL
import os
# Local
from . import AbstractSSD

_pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssd_mobilenet_pipeline.config")

class SSD_Mobilenet(AbstractSSD):

    ############### STATIC METHODS ####################

    @staticmethod
    def get_input_dims():
        return (300, 300)

    @staticmethod
    def get_ratios():
        raise NotImplementedError
    @staticmethod
    def get_scales():
        raise NotImplementedError
    @staticmethod
    def get_anchors():
        configs = config_util.get_configs_from_pipeline_file(_pipeline_path)
        orig_model = model_builder.build(configs["model"], is_training=True)
        orig_model(tf.zeros([1,300,300,3])) # Force the anchors to be built
        return orig_model.anchors

    @staticmethod
    def get_unmatched_class_target(num_classes):
        return AbstractSSD.get_unmatched_class_target(num_classes)

    ################# PUBLIC MEMBERS #########################

    def __init__(self, nonbackground_classes):

        super(SSD_Mobilenet, self).__init__(nonbackground_classes=nonbackground_classes)

        configs = config_util.get_configs_from_pipeline_file(_pipeline_path)
        configs.ssd["num_classes"] = nonbackground_classes

        orig_model = model_builder.build(configs["model"], is_training=True)
        self._orig_model = orig_model

        self.input_dims = SSD_Mobilenet.get_input_dims()
        self.input = tf.keras.Input([*self.input_dims, 3], dtype=tf.uint8)

        
        # Only retraining the predictor heads
        self._trainable_variables = [v for v in orig_model.trainable_variables if v.name.startswith("BoxPredictor")]
        self._initializable_variables = [v for v in orig_model.variables if not v.name.startswith("BoxPredictor")]

    @tf.function
    def preprocess(self, x):
        return self._orig_model.preprocess(x)

    @tf.function
    def predict(self, preprocessed, shapes):
        return self._orig_model.predict(preprocessed, shapes)

    # This is redundant now, but we may need more flexibility
    #  later
    @tf.function
    def postprocess(self, prediction_dict, shapes):
        return super().postprocess(prediction_dict, shapes)

    @property
    def variables(self):
        raise NotImplementedError
    @property
    def trainable_variables(self):
        return self._trainable_variables
    @property
    def ratios(self):
        raise NotImplementedError
    @property
    def scales(self):
        raise NotImplementedError
    @property
    def feature_shapes(self):
        raise NotImplementedError
    @property
    def anchors(self):
        return self._orig_model.anchors

