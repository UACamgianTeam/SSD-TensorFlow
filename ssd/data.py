# 3rd Party
import tensorflow as tf
# Python STL
import glob
import os

def serialize_ssd_example(image_arr, cls_targets, cls_weights, reg_targets, reg_weights, matched):
    feature = {
            "image"      : _tensor_feature(image_arr),
            "cls_targets": _tensor_feature(cls_targets),
            "cls_weights": _tensor_feature(cls_weights),
            "reg_targets": _tensor_feature(reg_targets),
            "reg_weights": _tensor_feature(reg_weights),
            "matched"    : _tensor_feature(matched)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def deserialize_ssd_example(example):
    example = tf.io.parse_single_example(example, _ssd_parse_dict)
    keys = ["image", "cls_targets", "cls_weights", "reg_targets", "reg_weights", "matched"]
    example = {k:tf.io.parse_tensor(example[k], out_type=tf.float32) for k in keys}
    return example

def ssd_tfrecords_dataset(records_dir, deterministic=True, num_parallel_calls=8):
    record_paths = glob.glob(os.path.join(records_dir, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(record_paths, num_parallel_reads=num_parallel_calls)
    dataset = dataset.map(deserialize_ssd_example,
                         deterministic=deterministic,
                         num_parallel_calls=num_parallel_calls)
    return dataset

################ Private ##################

_ssd_parse_dict = {
    "image"      : tf.io.FixedLenFeature([], dtype=tf.string),
    "cls_targets": tf.io.FixedLenFeature([], dtype=tf.string),
    "cls_weights": tf.io.FixedLenFeature([], dtype=tf.string),
    "reg_targets": tf.io.FixedLenFeature([], dtype=tf.string),
    "reg_weights": tf.io.FixedLenFeature([], dtype=tf.string),
    "matched"    : tf.io.FixedLenFeature([], dtype=tf.string),
}


def _tensor_feature(a):
    a = tf.cast(a, dtype=tf.float32)
    a = tf.io.serialize_tensor(a).numpy()
    a = tf.train.BytesList(value=[a])
    a = tf.train.Feature(bytes_list=a)
    return a

