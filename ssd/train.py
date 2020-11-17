# 3rd Party
import tensorflow as tf
# Python STL
import datetime
# Local
from .metrics import *

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


################## Private ####################
def _log(msg):
    print(f"[{_cur_date_string()}]: {msg}")

def _cur_date_string():
    return datetime.datetime.now().strftime("%m-%d-%y-%H:%M:%S")
