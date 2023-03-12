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
```

## Getting Started
To get started, follow these steps:

1- Clone this repository to your local machine.

```
git clone https://github.com/taylanates24/detection_and_tracking.git
```
2 - Build a docker image and create a container from docker image  (recommended)

```
docker build -t track:v1 -f docker/Dockerfile .
```

```
docker run -v $(pwd):/workspace -it --rm --ipc host track:v1
```
3 - Modify the configuration file inference.yaml to match your hyperparameters and input paths.

4 - Convert your trained mmdetection model to TensorRT:

```
python3 convert_tensorrt.py --config /path/to/config_file --checkpoint /path/to/checkpoint_file --save_path /save/path --device 'cuda:0' --fp16 True

```
5 - Start inference by running the following code :

```
python3 inference.py --infer_cfg inference.yaml
```



6 - Run 
```
python3 train.py --train_cfg training.yaml --dataset_cfg coco.yml
```
to start training the model.

