# 3rd Party
import tensorflow as tf


def horizontal_multibox_layer(num_anchors) -> tf.Tensor:
    # (ymin or cy, xmin or cx, w, h) offsets for each default box
    return tf.keras.layers.Conv2D(
                                filters = num_anchors * 4,
                                kernel_size = [3,3],
                                padding="same",
                                activation=None
    )
def quadrangle_multibox_layer(num_anchors) -> tf.Tensor:
    # (x1,y1,x2,y2,x3,y3,x4,y4) offsets for each default box
    return tf.keras.layers.Conv2D(
                                filters = num_anchors * 8,
                                kernel_size = [3,3],
                                padding="same",
                                activation=None
    )
def class_multibox_layer(num_anchors, num_classes) -> tf.Tensor:
    # A class logit for each default box
    return tf.keras.layers.Conv2D(
                                filters = num_anchors * num_classes,
                                kernel_size = [3,3],
                                padding="same",
                                activation=None
    )

@tf.function
def smooth_l1(x, alpha):
    abs_x = tf.abs(x)
    return tf.where(abs_x > alpha, abs_x, x**2 / alpha)
