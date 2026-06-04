# VOCA: Instant Spatial Adaptation Agent

### Zero-shot Navigation Module for Human-Level Robot Intelligence (HLRI)

VOCA is a cutting-edge zero-shot navigation module designed to enhance the capabilities of Human-Level Robot Intelligence (HLRI) systems. VOCA enables robots to adapt instantly to new environments without the need for extensive retraining, allowing for seamless navigation and interaction in diverse settings.

## Checkpoints

Model checkpoints are not included in this GitHub repository. Download the checkpoint files from Google Drive and place them under the repository's `checkpoints/` directory.

Google Drive: <https://drive.google.com/drive/folders/1loNb1OdBP9HLqa5zP0RRj2qirIp1mEwr?usp=drive_link>

After downloading, the default ObjectNav benchmark expects the following files:

```text
checkpoints/
├── pointnav_weights.pth
├── yoloe-11l-seg.pt
└── vo/
    ├── act_forward.pth
    └── act_left_right_inv_joint.pth
```

The default paths are defined in `settings.py` and can be overridden with environment variables:

- `VOCA_CHECKPOINT_DIR`
- `VOCA_POINTNAV_CHECKPOINT`
- `VOCA_YOLOE_CHECKPOINT`
- `VOCA_POINTNAV_VO_FORWARD_CHECKPOINT`
- `VOCA_POINTNAV_VO_TURN_CHECKPOINT`
