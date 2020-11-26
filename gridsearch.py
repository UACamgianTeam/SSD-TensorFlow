# 3rd Party
import tensorflow as tf
import numpy as np
# Python STL
import glob
import os
from pathlib import Path
import sys
import json
import argparse
from itertools import product
import time
# Local
from ssd.train import *
from ssd.data  import *
from ssd import SSD512_VGG16, SSD_Mobilenet
from common.eval import coco_eval


def model_train(model,
                train_records_dir,
                valid_records_dir,
                model_dir,
                early_stop_by="both",
                max_epochs=100,
                batch_size=4,
                learning_rate=1e-6,
                patience=10):
    assert early_stop_by in {"both","loc","conf"}
    vars_dict = {
        "waited"               : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "epoch_index"          : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "best_model_index"     : tf.Variable(-1, trainable=False, shape=(), dtype=tf.int64),
        "batch_index"          : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "best_valid_loc_loss"  : tf.Variable(np.inf, trainable=False, shape=(), dtype=tf.float32),
        "best_valid_conf_loss" : tf.Variable(np.inf, trainable=False, shape=(), dtype=tf.float32),
        "model"                : model.variables,
        "optimizer"            : tf.keras.optimizers.Adam(learning_rate=learning_rate)
    }
    
    checkpoint       = tf.train.Checkpoint(**vars_dict)
    checkpoints_root = model_dir/"checkpoints"
    bestpoints_root  = model_dir/"bestpoints"
    logdir           = model_dir/"tb"
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    manager          = tf.train.CheckpointManager(checkpoint, directory=checkpoints_root, max_to_keep=patience + 1)
    best_manager     = tf.train.CheckpointManager(checkpoint, directory=bestpoints_root,  max_to_keep=1)

    epoch_index    = vars_dict["epoch_index"]
    optimizer      = vars_dict["optimizer"]
    best_loc_loss  = vars_dict["best_valid_loc_loss"]
    best_conf_loss = vars_dict["best_valid_conf_loss"]
    waited         = vars_dict["waited"]

    writer = tf.summary.create_file_writer(str(logdir))

    log_freq  = 100 // batch_size
    shuffle_buffer_size = 1000
    for _ in range(max_epochs):
        epoch_index.assign_add(1)
        dataset = ssd_tfrecords_dataset(train_records_dir)
        dataset = dataset.shuffle(seed=epoch_index, buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

        train_epoch(model,optimizer,dataset,
                                  epoch_index=vars_dict["epoch_index"],
                                  batch_index=vars_dict["batch_index"],
                                  log_freq=log_freq,
                                writer=writer)
        print("Completed an epoch")

        
        # Validation
        valid_dataset = ssd_tfrecords_dataset(valid_records_dir)
        valid_dataset = valid_dataset.shuffle(seed=epoch_index, buffer_size=shuffle_buffer_size)
        valid_dataset = valid_dataset.batch(batch_size)
        loss_dict = validation_loss(model, valid_dataset)
        loc_loss = loss_dict["Localization"]
        conf_loss = loss_dict["Confidence"]

        print(f"Validation loss: Conf={conf_loss}, Loc={loc_loss}")


        if (early_stop_by == "both") and (loc_loss < best_loc_loss) and (conf_loss < best_conf_loss):
            print("New best!")
            vars_dict["best_model_index"].assign(epoch_index)
            best_loc_loss.assign(loc_loss)
            best_conf_loss.assign(conf_loss)
            waited.assign(0)
        elif (early_stop_by == "loc") and (loc_loss < best_loc_loss):
            print("New best!")
            vars_dict["best_model_index"].assign(epoch_index)
            best_loc_loss.assign(loc_loss)
            waited.assign(1)
        elif (early_stop_by == "conf") and (conf_loss < best_conf_loss):
            print("New best!")
            vars_dict["best_model_index"].assign(epoch_index)
            best_conf_loss.assign(conf_loss)
            waited.assign(0)
        else:
            waited.assign_add(1)
            
        loss_dict = prepend_namespace(loss_dict, "Loss/Validation")
        write_scalars(writer, loss_dict, step=epoch_index)

        manager.save(checkpoint_number=epoch_index)
        if waited >= patience: break
    return (manager, best_manager)


def main(records_root,
        model_gen,
        weight_sets           = [(1,1)],
        ohem_sets             = [True],
        early_stop_by         = "both",
        nms_redund_thresholds = [.15, .25, .35, .45, .55, .65],
        learning_rates=[1e-6]):
    assert early_stop_by in {"both","loc","conf"}

    timestamp = int(time.time())
    print(f"Storing all results in {timestamp} directory")
    models_root = Path("./experiments_gridsearch") / str(timestamp)
    os.makedirs(models_root, exist_ok=True)

    data_root = Path("./dota_sports_data")
    dataset_dirs = [Path(p) for p in glob.glob( f"{records_root}/*") ]
    

    model_index = 1
    for [dataset_dir, (conf_weight, loc_weight), ohem, learning_rate] in product(dataset_dirs, weight_sets, ohem_sets, learning_rates):
        print(f"dataset_dir={dataset_dir}, conf_weight={conf_weight}, loc_weight={loc_weight}, ohem={ohem}, learning_rate={learning_rate}, early_stop_by={early_stop_by}")
        with open(dataset_dir/"meta.json", "r") as r:
            dataset_meta = json.load(r)

        train_records_dir  = dataset_dir/"train"
        valid_records_dir  = dataset_dir/"validation"
        model_dir          = models_root / f"{model_index}"
        os.makedirs(model_dir, exist_ok=True)

        win_set            = dataset_meta["win_set"]
        min_coverage       = dataset_meta["min_coverage"]
        desired_categories = dataset_meta["classes"]

        ############### TRAINING  ##################
        model             = model_gen()
        model.conf_weight = conf_weight
        model.loc_weight  = loc_weight
        model.ohem        = ohem

        (manager, best_manager) = model_train(model,
                                            train_records_dir,
                                            valid_records_dir,
                                            model_dir=model_dir,
                                            early_stop_by=early_stop_by,
                                            learning_rate=learning_rate)
        checkpoint              = best_manager.checkpoint
        checkpoint.restore(best_manager.latest_checkpoint)

        ############### EVALUATION ###################
        out_meta = dict()
        out_meta["dataset"]     = dataset_meta
        out_meta["loc_weight"]  = loc_weight
        out_meta["conf_weight"] = conf_weight
        out_meta["ohem"]        = ohem
        out_meta["early_stop_by"] = early_stop_by
        out_meta["learning_rate"] = learning_rate
        out_meta["dataset_dir"] = str(dataset_dir)
        out_meta["evaluations"] = []
        for thresh in nms_redund_thresholds:
            model.nms_redund_threshold = thresh
            (metrics_dict, by_class) = coco_eval(model,
                data_root / 'annotations/validation.json',
                data_root / 'validation/images/',
                min_coverage=min_coverage,
                desired_categories=desired_categories,
                win_set=win_set
            )
            out_meta["evaluations"].append({
                    "nms_redund_threshold" : model.nms_redund_threshold,
                    "top_k_per_class"      : model.top_k_per_class,
                    "score_threshold"      : model.min_score_threshold,
                    "metrics"              : metrics_dict
            })
        with open( model_dir / "results.json", "w") as w:
            w.write(json.dumps(out_meta, indent=2) + "\n")
        model_index += 1


def gen_ssd_mobilenet(n_categories, checkpoint_path):
    model = SSD_Mobilenet(n_categories)
    model.initialize_feature_extractor(checkpoint_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple models with different hyperparameters")

    parser.add_argument("--records-root", required=True, help="Directory containing directories of TFRecords")
    parser.add_argument("--model", default="ssd_mobilenet", help='"ssd_mobilenet" or "ssd512_vgg16"')

    args = parser.parse_args()

    n_categories = 4
    if args.model == "ssd_mobilenet":
        checkpoint_path = "/data/pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
        model_gen       = lambda: gen_ssd_mobilenet(n_categories, checkpoint_path)
    elif args.model == "ssd512_vgg16":
        weights_path = "/data/pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        model_gen    = lambda: SSD512_VGG16.from_scratch(n_categories, weights_path)

    with tf.device("/device:GPU:2"):
        main(args.records_root,
            model_gen,
            weight_sets           = [(1,1)],
            early_stop_by         = "conf",
            ohem_sets             = [True],
            nms_redund_thresholds = [.15,.25,.35,.45,.55,.65,.75],
            learning_rates        = [1e-4]
        )
