# 3rd Party
import tensorflow as tf
# Python STL
import os

def save_checkpoint(ckpt: tf.train.Checkpoint, ckpt_path: str):
  dir_name = os.path.dirname(ckpt_path)
  os.makedirs(dir_name, exist_ok=True)
  ckpt.save(ckpt_path)
