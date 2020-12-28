# SSD-TensorFlow

Implementation of Single-Shot Multibox Detector in TensorFlow 2

## Modules
One can run `pip install .` to make the `ssd` and `common`
modules both available for import in other programs.
Note that these have as a dependency the [Oriented Object Detection](https://github.com/UACamgianTeam/Oriented-Object-Detection)
package, which is also `pip install`able.

The `ssd` modules contains two SSD classes,
the classic SSD512 and the newer SSD-MobileNetV2.
The former is a custom implementation; the latter
is a thin veneer over the TensorFlow Object Detection
API's implementation.

The `common` module contains some utility functions
for training and evaluation.

# Scripts
The `gridsearch.py` script is used for training and evaluation.
Run `python gridsearch.py --help` to see the various hyperparameters
that can be tested.

The `prepare_records.py` script is used to convert a set of COCO annotations
to TFRecords to train these SSD models. This vastly speeds up the training
process, and the `gridsearch.py` script expects these records.

`quick_eval.py` and `quick_viz.py` are useful for evaluating the performance
of an already-trained model.
