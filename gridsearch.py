# 3rd Party
import tensorflow as tf
import numpy as np
# Python STL
import glob
import os
from pathlib import Path
import sys
import pdb
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
                model_dir : Path,
                improv_func=None,
                max_epochs=100,
                batch_size=4,
                learning_rate=1e-4,
                patience=10):

    vars_dict = {
        "waited"               : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "epoch_index"          : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "best_model_index"     : tf.Variable(-1, trainable=False, shape=(), dtype=tf.int64),
        "batch_index"          : tf.Variable(0, trainable=False, shape=(), dtype=tf.int64),
        "model"                : model.variables,
        "optimizer"            : tf.keras.optimizers.Adam(learning_rate=learning_rate)
    }
    
    checkpoint       = tf.train.Checkpoint(**vars_dict)
    checkpoints_root = model_dir/"checkpoints"
    bestpoints_root  = model_dir/"bestpoints"
    logdir           = model_dir/"tb"
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(bestpoints_root, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    manager          = tf.train.CheckpointManager(checkpoint, directory=checkpoints_root, max_to_keep=patience + 1)
    best_manager     = tf.train.CheckpointManager(checkpoint, directory=bestpoints_root,  max_to_keep=1)

    epoch_index       = vars_dict["epoch_index"]
    optimizer         = vars_dict["optimizer"]
    waited            = vars_dict["waited"]

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


        if improv_func:
            (improved, result) = improv_func(model, writer=writer, step=epoch_index)
            print(f"Validation metric: {result}")
            if improved:
                print("New best!")
                waited.assign(0)
                vars_dict["best_model_index"].assign(epoch_index)
                best_manager.save()
            else:
                waited.assign_add(1)

        manager.save(checkpoint_number=epoch_index)
        if waited >= patience: break

    if improv_func:
        return best_manager
    else:
        return manager


def main(records_dir,
        model_gen,
        improv_funcs          = None,
        nms_redund_thresholds = [.15, .25, .35, .45, .55, .65, .75],
        learning_rates=[1e-6],
        patience=10,
        model_gen_kwargs_sets=None):

    timestamp = int(time.time())
    print(f"Storing all results in {timestamp} directory")
    models_root = Path("./experiments_gridsearch") / str(timestamp)
    os.makedirs(models_root, exist_ok=True)

    data_root = Path("./dota_sports_data")
    dataset_dir = records_dir
    


    model_index = 1

    hparams = [
        learning_rates,
    ]
    if improv_funcs:          hparams.append(improv_funcs.items())
    if model_gen_kwargs_sets: hparams.append(model_gen_kwargs_sets)
    grid = product(*hparams)

    for param_set in grid:
        #################### Parse Hyperparams ####################
        hp_index = 0
        learning_rate = param_set[hp_index]; hp_index += 1
        if improv_funcs:
            (improv_name, improv_func) = param_set[hp_index]
            hp_index += 1
        else:
            improv_name, improv_func = (None, None)

        if model_gen_kwargs_sets:
            model_gen_kwargs = param_set[hp_index]
            hp_index += 1
        else:
            model_gen_kwargs = dict()
        print("Searching a new grid cell")


        ############ Get Dataset Paths ##############
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
        model      = model_gen(**model_gen_kwargs)

        manager    = model_train(model,
                                            train_records_dir,
                                            model_dir=model_dir,
                                            improv_func=improv_func,
                                            learning_rate=learning_rate,
                                            patience=patience)
        checkpoint = manager.checkpoint
        checkpoint.restore(manager.latest_checkpoint)

        ############### EVALUATION ###################
        out_meta = dict()
        out_meta["model_params"]  = model_gen_kwargs
        out_meta["dataset"]       = dataset_meta
        out_meta["improv_func"]   = improv_name
        out_meta["learning_rate"] = learning_rate
        out_meta["dataset_dir"]   = str(dataset_dir)
        out_meta["patience"]      = patience
        out_meta["evaluations"]   = []
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple models with different hyperparameters")

    parser.add_argument("--records-dir", required=True, help="Directory containing directories of TFRecords")
    parser.add_argument("--model", default="ssd_mobilenet", help='"ssd_mobilenet" or "ssd512_vgg16"')

    args = parser.parse_args()

    n_categories = 4
    if args.model == "ssd_mobilenet":
        checkpoint_path = "/data/pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
        def model_gen(ohem, loc_weight, conf_weight, train_from_scratch):
            model             = SSD_Mobilenet(n_categories)
            if not train_from_scratch: model.initialize_feature_extractor(checkpoint_path)
            model.ohem        = ohem
            model.loc_weight  = loc_weight
            model.conf_weight = conf_weight
            return model

        arg_sets = [
                {"ohem": True, "loc_weight": 1., "conf_weight": 1., "train_from_scratch": False},
                #{"ohem": True, "loc_weight": 1., "conf_weight": 5., "train_from_scratch": False},
                #{"ohem": True, "loc_weight": 1., "conf_weight": 1., "train_from_scratch": True},
                #{"ohem": True, "loc_weight": 1., "conf_weight": 5., "train_from_scratch": True},
        ]
    elif args.model == "ssd512_vgg16":
        weights_path = "/data/pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        model_gen    = lambda: SSD512_VGG16.from_scratch(n_categories, weights_path)
        arg_sets     = None

    records_dir = Path(args.records_dir)

    with tf.device("/device:GPU:2"):

        valid_records_dir = records_dir / "validation"
        improv_funcs = {
            #"loc_and_conf": make_valid_loss_improv_func(valid_records_dir, by="both", batch_size=4, shuffle_buffer_size=1000)
            #"loc_or_conf": make_valid_loss_improv_func(valid_records_dir, by="either", batch_size=4, shuffle_buffer_size=1000)
            "conf": make_valid_loss_improv_func(valid_records_dir, by="conf", batch_size=4, shuffle_buffer_size=1000)
        }

        main(records_dir,
            model_gen,
            improv_funcs          = improv_funcs,
            nms_redund_thresholds = [.15,.25,.35,.45,.55,.65,.75],
            learning_rates        = [1e-4],
            patience              = 10,
            model_gen_kwargs_sets = arg_sets
        )
