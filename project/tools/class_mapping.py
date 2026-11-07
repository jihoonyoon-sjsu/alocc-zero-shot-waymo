"""Map ALOcc (Occ3D-nuScenes) and Occ3D-Waymo labels to shared merged classes."""
import numpy as np

MERGED_CLASSES = [
    'vehicle', 'pedestrian', 'two_wheeler', 'traffic_cone',
    'drivable_surface', 'sidewalk', 'vegetation', 'manmade', 'free',
]
OTHER = 255  # occupied, but in no merged class

NUSC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
NUSC_TO_MERGED = {
    'bus': 'vehicle', 'car': 'vehicle', 'construction_vehicle': 'vehicle',
    'trailer': 'vehicle', 'truck': 'vehicle',
    'pedestrian': 'pedestrian',
    'bicycle': 'two_wheeler', 'motorcycle': 'two_wheeler',
    'traffic_cone': 'traffic_cone',
    'driveable_surface': 'drivable_surface',
    'sidewalk': 'sidewalk',
    'vegetation': 'vegetation',
    'manmade': 'manmade',
    'free': 'free',
}

WAYMO_CLASSES = {
    0: 'general_object', 1: 'vehicle', 2: 'pedestrian', 3: 'sign',
    4: 'bicyclist', 5: 'traffic_light', 6: 'pole', 7: 'construction_cone',
    8: 'bicycle', 9: 'motorcycle', 10: 'building', 11: 'vegetation',
    12: 'tree_trunk', 13: 'road', 14: 'walkable', 23: 'free',
}
WAYMO_TO_MERGED = {
    'vehicle': 'vehicle',
    'pedestrian': 'pedestrian',
    'bicyclist': 'two_wheeler', 'bicycle': 'two_wheeler',
    'motorcycle': 'two_wheeler',
    'construction_cone': 'traffic_cone',
    'road': 'drivable_surface',
    'walkable': 'sidewalk',
    'vegetation': 'vegetation', 'tree_trunk': 'vegetation',
    'building': 'manmade', 'pole': 'manmade', 'sign': 'manmade',
    'traffic_light': 'manmade',
    'free': 'free',
}


def _lut(classes, to_merged):
    lut = np.full(256, OTHER, dtype=np.uint8)
    for idx, name in classes.items():
        if name in to_merged:
            lut[idx] = MERGED_CLASSES.index(to_merged[name])
    return lut


NUSC_LUT = _lut(dict(enumerate(NUSC_CLASSES)), NUSC_TO_MERGED)
WAYMO_LUT = _lut(WAYMO_CLASSES, WAYMO_TO_MERGED)


def nusc_to_merged(labels):
    return NUSC_LUT[labels]


def waymo_to_merged(labels):
    return WAYMO_LUT[labels]
