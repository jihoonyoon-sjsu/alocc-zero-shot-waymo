# Same Model, New Car?

Diagnosing zero-shot transfer of a pretrained 3D semantic occupancy model (ALOcc) from nuScenes to Waymo, with FlashOcc and OPUS as baselines.

CMPE 249 – Intelligent Autonomous Systems, SJSU. Project Track: Option 1 – Apply and Evaluate Pretrained Models (No Training).

Team: Simisola Winjobi, JiHoon Yoon, Tara Fathinejad

## Built on ALOcc

This repository includes the code of **ALOcc** (Chen et al., ICCV 2025), copied unmodified from [cdb342/ALOcc](https://github.com/cdb342/ALOcc) at commit `1bfd7de`, under the Apache 2.0 license (see `LICENSE`). ALOcc's original README is in [README_ALOcc.md](README_ALOcc.md).

```bibtex
@InProceedings{chen2025alocc,
    author    = {Chen, Dubing and Fang, Jin and Han, Wencheng and Cheng, Xinjing and Yin, Junbo and Xu, Chenzhong and Khan, Fahad Shahbaz and Shen, Jianbing},
    title     = {Alocc: adaptive lifting-based 3d semantic occupancy and cost volume-based flow prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```

## Repository layout

- `project/`: **our work** (code, run scripts, results). See [project/README.md](project/README.md) for a map of every folder.
- Everything else: ALOcc's code, unmodified.

## Results so far (Occ3D-nuScenes val, 6,019 frames)

| Run | mIoU | Occupied IoU |
|---|---|---|
| ALOcc-3D, published | 45.5 | |
| ALOcc-3D, our reproduction | 45.46 | 75.30 |
| ALOcc-3D, CAM_BACK removed | 40.24 | 73.18 |
