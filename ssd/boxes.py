# Python STL
from typing import List, Tuple
# 3rd Party
import numpy as np

def multilayer_default_boxes(feature_shapes, ratios, scales, quadrangles=False)
    # Coordinates of default boxes
    default_boxes = []
    for (feature_shape, ratio_set, scale_set) in zip(feature_shapes, ratios, scales):
        coords = default_box_mesh(feature_shape,ratio_set,scale_set)
        coords = np.reshape(coords, [
                                    feature_shape[0]*feature_shape[1]*len(ratio_set),
                                    8 if quadrangles else 4
        ])
        default_boxes.append(coords)
    default_boxes = np.concatenate(default_boxes, axis=0)

def default_box_mesh(feature_shape: Tuple[int,int], ratios: List[float], scales: List[float]) -> np.ndarray:
    """
    Coordinates are vertex-based: (ymin,xmin, ymax, xmax)
    
    :returns: Bounding box array [anchor, coords]
    """
    assert len(ratios) == len(scales), f"{len(ratios)} vs. {len(scales)}"
    n_anchors = len(ratios)
    
    feature_shape = np.squeeze(np.array(feature_shape))
    if np.ndim(feature_shape) == 0: feature_shape = [1,1]
    
    ratios = np.squeeze(np.array(ratios))
    scales = np.squeeze(np.array(scales))
    Y_min,X_min = np.mgrid[ 0:feature_shape[0], 0:feature_shape[1] ]
    
    Y_min =  Y_min / feature_shape[0] # For whatever reason, /= caused errors
    X_min =  X_min / feature_shape[1]
    H = scales / np.sqrt(ratios)
    W = scales * np.sqrt(ratios)
    
    
    X_min = np.repeat(np.expand_dims(X_min,-1), n_anchors, axis=-1)
    Y_min = np.repeat(np.expand_dims(Y_min,-1), n_anchors, axis=-1)
    
    coordinates = np.stack([
                            Y_min,
                            X_min,
                            np.ones([*Y_min.shape]) * H,
                            np.ones([*Y_min.shape]) * W
    ],axis = -1)
    
    coordinates[...,2] += coordinates[...,0]
    coordinates[...,3] += coordinates[...,1]
    
    
    return coordinates


__all__ = ["multilayer_default_boxes, default_box_mesh"]
