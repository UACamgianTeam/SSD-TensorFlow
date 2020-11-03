# 3rd Party
import tensorflow as tf
# Python STL
import os
# Local
from .eval import coco_eval
from .checkpoints import save_checkpoint


class COCOEvalCallback(tf.keras.callbacks.Callback):
  def __init__(self, coco_eval_kwargs, log_dir):
    super(COCOEvalCallback, self).__init__()
    self._coco_eval_kwargs = coco_eval_kwargs
    os.makedirs(log_dir, exist_ok=True)
    self._writer = tf.summary.create_file_writer(log_dir)
    self._metrics = [
                     "Precision/mAP",
                     "Precision/mAP@.50IOU",
                     "Precision/mAP@.75IOU",
                     "Recall/AR@100"
    ]

  def on_epoch_begin(self, epoch, logs = None):
    pass

  def on_epoch_end(self, epoch, logs = None):
    metrics_dict = coco_eval(**self._coco_eval_kwargs)
    with self._writer.as_default():
      for m in self._metrics:
        tf.summary.scalar(m, metrics_dict[0][m], step=epoch)
    self._writer.flush()


class SummaryScalarCallback(tf.keras.callbacks.Callback):
  def __init__(self, log_dir):
    super(SummaryScalarCallback, self).__init__()
    os.makedirs(log_dir, exist_ok=True)
    self._writer = tf.summary.create_file_writer(log_dir)
  def on_epoch_begin(self, epoch, logs = None):
    pass
  def on_epoch_end(self, epoch, logs = None):
    if logs is None: return
    with self._writer.as_default():
      for (k,v) in logs.items():
        tf.summary.scalar(k, v, step=epoch)
    self._writer.flush()


class CheckpointCallback(tf.keras.callbacks.Callback):
  def __init__(self, ckpt: tf.train.Checkpoint, ckpt_path: str):
    super(CheckpointCallback, self).__init__()
    self._ckpt = ckpt
    self._ckpt_path = ckpt_path
  def on_epoch_begin(self, epoch, logs = None):
    pass
  def on_epoch_end(self, epoch, logs = None):
    save_checkpoint(self._ckpt, self._ckpt_path)

