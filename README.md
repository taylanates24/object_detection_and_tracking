# Object Detection and Tracking with SORT Algorithm, Kalman Filter and TensorRT

In this repository, a coco pre-trained YOLOX-x model is finetuned on BDD100K dataset. You can fine-tune any model that can be converted to tensorrt using the mmdetection-to-tensorrt repository. You can find a detailed expalanation about how to fine-tune an mmdetection model on a custom dataset in my medium blogpost:

### medium blog here

## Project Structure
```bash
project/
├── README.md
├── configs /
│   ├── _base_/
│   │    ├── schedules /
│   │    │   └──schedule_1x.py
│   │    └──default_runtime.py
│   ├── yolox_s_8x8_300e_coco.py
│   └── yolox_x_8x8_300e_coco.py
├── docker/
│   └── Dockerfile
├──tests /
│   ├── test_process.py
│   ├── test_tracker.py
│   └── test_utils.py
├── LICENSE
├── convert_tensorrt.py
├── detector.py
├── inference.py
├── inference.yaml
├── process.py
├── tracker.py
└── utils.py

