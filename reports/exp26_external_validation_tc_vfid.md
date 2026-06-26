# Exp26 External Validation TC/VFID

TC and VFID are computed through the existing `inference.metrics.py` backend.
- I3D: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/i3d_rgb_imagenet.pt`
- OpenCLIP: `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/open_clip_vit_h14`

| frame_range | step | variant | rows | tc_mean | tc_median | vfid |
| --- | --- | --- | --- | --- | --- | --- |
| all49 | step0 | raw | 32 | 0.963981 | 0.969914 | 0.392721 |
| all49 | step0 | comp | 32 | 0.962269 | 0.965758 | 0.365418 |
| all49 | step10 | raw | 32 | 0.941309 | 0.945402 | 0.431728 |
| all49 | step10 | comp | 32 | 0.956647 | 0.960900 | 0.371666 |
| all49 | step30 | raw | 32 | 0.957420 | 0.964305 | 0.376376 |
| all49 | step30 | comp | 32 | 0.961475 | 0.965133 | 0.351153 |
| all49 | step50 | raw | 32 | 0.958679 | 0.964434 | 0.395202 |
| all49 | step50 | comp | 32 | 0.961295 | 0.965089 | 0.350629 |
| no_first_frame | step0 | raw | 32 | 0.964530 | 0.970483 | 0.451356 |
| no_first_frame | step0 | comp | 32 | 0.962637 | 0.965736 | 0.420941 |
| no_first_frame | step10 | raw | 32 | 0.942253 | 0.945985 | 0.542774 |
| no_first_frame | step10 | comp | 32 | 0.957412 | 0.962402 | 0.423166 |
| no_first_frame | step30 | raw | 32 | 0.957860 | 0.964719 | 0.447808 |
| no_first_frame | step30 | comp | 32 | 0.962044 | 0.967164 | 0.395327 |
| no_first_frame | step50 | raw | 32 | 0.959621 | 0.966116 | 0.465366 |
| no_first_frame | step50 | comp | 32 | 0.961672 | 0.965459 | 0.397402 |

