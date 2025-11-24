project_root/
│
├─ data/
│   ├─ raw_video/
│   │   ├─ A.mp4
│   │   ├─ B.mp4
│   │   └─ ...
│   └─ keypoint_img/
│       ├─ grayscale/
│       │   ├─ 224/
│       │   │   ├─ A/
│       │   │   ├─ B/
│       │   │   └─ ...
│       │   └─ 128/
│       │       └─ ...
│       └─ rgb/
│           └─ 224/
│               ├─ A/
│               └─ ...
│
├─ src/
│   ├─ keypoint_extractor/
│   │   ├─ __init__.py
│   │   └─ extractor.py
│   ├─ datasets/
│   │   └─ skeleton_dataset.py
│   ├─ models/
│   │   ├─ simple_cnn.py
│   │   ├─ resnet_wrapper.py
│   │   └─ efficientnet_wrapper.py
│   └─ train/
│       └─ train_skeleton_classifier.py
└─ requirements.txt

