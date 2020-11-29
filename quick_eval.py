import tensorflow as tf
from pathlib import Path
import json
from ssd import SSD_Mobilenet
from common.eval import coco_eval, coco_by_category

data_root = Path("./dota_sports_data")
results_root = Path("./experiments_gridsearch/1606684100/4")

with open(results_root / "results.json", "r") as r: records_dir  = Path( json.load(r)["dataset_dir"] )
with open(records_dir / "meta.json") as r:         dataset_meta = json.load(r)
with open(records_dir/"meta.json", "r") as r: dataset_meta = json.load(r)
min_coverage       = dataset_meta["min_coverage"]
win_set            = dataset_meta["win_set"]
desired_categories = dataset_meta["classes"]
n_categories = len(desired_categories)

with tf.device("/device:GPU:2"):
    model = SSD_Mobilenet(n_categories)
    checkpoint_root = results_root / "bestpoints"
    checkpoint = tf.train.Checkpoint(model=model.variables)
    checkpoint.restore( tf.train.latest_checkpoint(checkpoint_root) )

    results2 = coco_by_category(model,
                    data_root / 'annotations/validation.json',
                    data_root / 'validation/images/',
                    min_coverage=min_coverage,
                    desired_categories=desired_categories,
                    win_set=win_set
                )
    results = coco_eval(model,
                    data_root / 'annotations/validation.json',
                    data_root / 'validation/images/',
                    min_coverage=min_coverage,
                    desired_categories=desired_categories,
                    win_set=win_set
                )


