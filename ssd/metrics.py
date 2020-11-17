import tensorflow as tf
# Python STL
from typing import Dict

ScalarDict = Dict[str,float]
MetricDict = Dict[str,tf.keras.metrics.Metric]

def make_metric_dict(names, types = None):
    d = dict()
    for (i, name) in enumerate(names):
        t = types[i] if types else tf.keras.metrics.Sum
        d[name] = t(name, dtype=tf.float32)
    return d

def update_metric_dict(m, u):
    for k in m:
        m[k](u[k])

def prepend_namespace(d, prefix):
    return {f"{prefix}/{k}":v for (k,v) in d.items()}

def metric2scalar_dict(m, prefix = None, v_func = None, reset_states=True):
    scalars = dict()
    for (k, v) in m.items():
        value = v.result()
        if v_func: value = v_func(value)
        scalars[k] = value
    if reset_states:
        for (k,v) in m.items():
            v.reset_states()
    if prefix: scalars = prepend_namespace(scalars, prefix)
    return scalars

def write_scalars(writer: tf.summary.SummaryWriter, scalar_dict, step):
    with writer.as_default():
        for (k, v) in scalar_dict.items():
            tf.summary.scalar(k, v, step=step)
    writer.flush()
