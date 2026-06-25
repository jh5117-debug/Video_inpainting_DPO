# Exp26 Shadow-Dev TC/VFID

TC and VFID are computed through the existing `inference.metrics.py` backend.
- I3D: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/i3d_rgb_imagenet.pt`
- OpenCLIP: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/open_clip_vit_h14`

| frame_range | step | variant | rows | tc_mean | tc_median | vfid |
| --- | --- | --- | --- | --- | --- | --- |
| no_first_frame | step0 | raw | 32 | 0.987396 | 0.989366 | 0.525803 |
| no_first_frame | step0 | comp | 32 | 0.986760 | 0.987468 | 0.531078 |
| no_first_frame | step50 | raw | 32 | 0.991770 | 0.992466 | 0.512713 |
| no_first_frame | step50 | comp | 32 | 0.991139 | 0.991918 | 0.499650 |

