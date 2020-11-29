# 3rd Party
import tensorflow as tf
import numpy as np
# Python STL
import datetime
# Local
from .metrics import *
from .data    import *

################## Public #####################
def train_epoch(model, optimizer, dataset,
                epoch_index : int,
                batch_index : tf.Variable,
                log_freq : int = 250,
                writer = None):
    
    to_fine_tune = [v for v in model.trainable_variables]
    
    epoch_metrics = make_metric_dict(["Localization", "Confidence", "WeightedTotal"])
    era_metrics   = make_metric_dict(["Localization", "Confidence", "WeightedTotal"])
    
    for (_, met) in era_metrics.items():
        met.reset_states()

    epoch_samples = 0
    era_samples = 0
    _log("Started new training epoch")
    
    batch_start = batch_index.numpy()
    for batch in dataset:
        batch_index.assign_add(1)
        epoch_samples += len(batch["image"])
        era_samples += len(batch["image"])

        keys = ["cls_targets", "cls_weights", "reg_targets", "reg_weights", "matched"]
        images, shapes = model.preprocess(batch["image"])
        model.provide_groundtruth_direct(**{k:batch[k] for k in keys})
        with tf.GradientTape() as tape:
            prediction_dict = model.predict(images, shapes)
            loss_dict = model.loss(prediction_dict)
        gradients = tape.gradient(loss_dict["WeightedTotal"], to_fine_tune)
        optimizer.apply_gradients( zip(gradients, to_fine_tune) )
        update_metric_dict(epoch_metrics, loss_dict)
        update_metric_dict(era_metrics, loss_dict)
        
        if (batch_index - batch_start) % log_freq == 0:
            _log(f"Completed {batch_index - batch_start} batches")
            if writer:
                l_dict = metric2scalar_dict(era_metrics,
                                        prefix       = f"Loss/Train/Last_{log_freq}_Batches",
                                        v_func      = lambda v: v / era_samples,
                                        reset_states = True)
                write_scalars(writer, l_dict, step=batch_index)

    if writer:
        l_dict = metric2scalar_dict(epoch_metrics,
                                prefix       = f"Loss/Train/Epoch",
                                v_func      = lambda v: v / epoch_samples,
                                reset_states = True)
        write_scalars(writer, l_dict, step=epoch_index)

def validation_loss(model, dataset) -> ScalarDict:
    metrics = make_metric_dict(["Localization", "Confidence", "WeightedTotal"])
    n_samples = 0
    for batch in dataset:
        images, shapes = model.preprocess(batch["image"])
        prediction_dict = model.predict(images, shapes)
        keys = ["cls_targets", "cls_weights", "reg_targets", "reg_weights", "matched"]
        model.provide_groundtruth_direct(**{k:batch[k] for k in keys})
        loss_dict = model.loss(prediction_dict)
        update_metric_dict(metrics, loss_dict)
        n_samples += len(batch["image"])
    scalars = metric2scalar_dict(metrics, v_func = lambda v: v / n_samples, reset_states=True)
    return scalars


def make_valid_loss_improv_func(valid_records_dir, by="both", shuffle_buffer_size=1000, batch_size=4):
    assert by in {"both","conf","loc", "either"}

    best_conf_loss = tf.Variable(np.inf, shape=(), trainable=False, dtype=tf.float32)
    best_loc_loss  = tf.Variable(np.inf, shape=(), trainable=False, dtype=tf.float32)

    def f(model, writer=None, step=0):
        epoch_index   = step
        early_stop_by = by
        valid_dataset = ssd_tfrecords_dataset(valid_records_dir)
        valid_dataset = valid_dataset.shuffle(seed=epoch_index, buffer_size=shuffle_buffer_size)
        valid_dataset = valid_dataset.batch(batch_size)
        loss_dict     = validation_loss(model, valid_dataset)
        loc_loss      = loss_dict["Localization"]
        conf_loss     = loss_dict["Confidence"]


        if writer:
            loss_dict = prepend_namespace(loss_dict, "Loss/Validation")
            write_scalars(writer, loss_dict, step=epoch_index)

        loc_improved  = (loc_loss < best_loc_loss)
        conf_improved = (conf_loss < best_conf_loss)

        if loc_improved : best_loc_loss.assign(loc_loss)
        if conf_improved: best_conf_loss.assign(conf_loss)

        if by == "both"  : return (loc_improved and conf_improved, (conf_loss.numpy(),loc_loss.numpy()))
        if by == "either": return (loc_improved or  conf_improved, (conf_loss.numpy(),loc_loss.numpy()))
        if by == "conf"  : return (                 conf_improved,  conf_loss.numpy()                  )
        if by == "loc"   : return (loc_improved                  ,                    loc_loss.numpy() )

    return f



################## Private ####################
def _log(msg):
    print(f"[{_cur_date_string()}]: {msg}")

def _cur_date_string():
    return datetime.datetime.now().strftime("%m-%d-%y-%H:%M:%S")
