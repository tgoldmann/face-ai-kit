# FaceAIKit: Face Detection and Recognition Library

FaceAIKit is a Python library designed for face detection and recognition application. With FaceAIKit, you can easily integrate state-of-the-art face detection and recognition capabilities into your applications and research projects. FaceAIKit offers a comprehensive solution to work with facial data. The library is designed for inference on various devices such as CPU (provided by onnx), GPU (provided by onnx), Rockchip RK3566 (RKNN framework), Rockchip RK3588 (RKNN framework).

<img src="doc/faceaikit.jpg" alt="drawing" width="200" style="margin-left: auto;margin-right: auto;  display: block;"/>



## Key Features
* Face Detection - Quickly locate faces within images or video streams using efficient algorithms for detecting faces in various contexts and orientations.

* Facial Landmark Detection - Identify key facial landmarks, such as eyes, nose, and mouth, to understand facial expressions.

* Face Recognition - Perform facial recognition to identify and verify individuals by comparing detected faces.

* Estimation of head rotation - Not available yet

Customizable: Fine-tune and customize the library's models to suit your specific needs and applications.


## News
* 27.12.2023 - FaceAIKit alpha version was introduced.

## Installation
You can install FaceAIKit using pip:

```bash
pip install face-ai-kit
```

## Configuration

Modules are configured using a yaml config file placed in the config folder. The library provides the ability to use your own config file, which must be placed in the folder linked by env FACEAIKITDIR.

Config file is split to following parts:

* retinaface_detector - Configuration for retinaface detector; supported modules ([RetinaFace](recognition_lib/modules/retinaface_detector/README.md))
* recognition: supported modules ([ArcFace](recognition_lib/modules/arcface/README.md))
* landmarks: supported modules ([N19](recognition_lib/modules/n19_landmarks/README.md))


## Models

Publicly available models are for CPU and GPU only. Models for other platforms are available on request. The trained model can be downloaded from [GDriver](test).

| Model name | Restriction | Note |
|------------|---------|------|
|ArcFace|For research purposes only |Trained on MS1M dataset, Resnet50 backbone|
|MagFace|Given by [MagFace](https://github.com/IrvingMeng/MagFace)|Converted model from [MagFace](https://github.com/IrvingMeng/MagFace)|
|N19|For research purposes only|Internal neural network, trained on WFLW dataset|
|RetinaFace|For research purposes only|Trained on WIDERFace dataset|

## Getting Started

To get started with FaceAIKit, please refer to the documentation and examples provided in this repository. You'll find detailed guides and sample code to help you integrate the library into your projects quickly.


Test images included in examples/data were obtained from https://vis-www.cs.umass.edu/fddb/.


### Init

To use the library, you need to perform initialization by creating a new instance of the FaceRecognition class. During initialization, the face recognition algorithm should be selected, and the path to a custom configuration file can be set as a class parameter.

Supported face recognition algorithms:

* ArcFace - 'arcface'
* MagFace - 'magface'


```python
lib = FaceRecognition(recognition='arcface')
```

### Face detection

Face detection is performed by calling the face_detection method. The function expects a NumPy image or a path to an image as an input parameter. In addition to this argument, the face alignment method can be set using the align parameter.


```python
lib.face_detection(image, align='square')
```

Supported face align algorithms:

* Square - 'square'
* Keypoints - 'keypoints'
* None - To use the output of the face detector directly


Output of face_detection is array cotaints dict with face, roi, score.


<img src="doc/detection.png" alt="drawing" width="400" style="margin-left: auto;margin-right: auto;  display: block;"/>



### Face recognition

Face recognition can be performed using several methods, each of which has a different use case.

#### Direct use face images for face recognition

In the input images, the faces are detected, whereby the two faces with the highest confidence scores are then compared.

```python
lib.verify(face_image1,face_image2)
```

#### Face recognition between batches consits of face images and corresponding ROIs

```python
lib.verify_batch(face_image1,face_image2)
```

#### Face recognition with defined face area in an image by ROI

```python
lib.verify_rois(face_image1, face_roi1,face_image2, face_roi2)
```

 
### Face landmarks

Detection of face landmarks can be perfomed by using following code:

```python
lib.landmarks(face_image1, face_roi1,face_image2, face_roi2)
```

The library supports N19 landmark (our model) and MediaPipi landmark detectors. With N19, it is possible to detect 98 keypoints defined by WFLW annotation. On the other hand, the MediaPipe detector allows to detect 192 face keypoints.

## Contributions
Contributions, bug reports, and feature requests are welcome! Feel free to submit issues to improve FaceAIKit and make it even more powerful and user-friendly.

## License
FaceAIKit is licensed under the MIT License, allowing you to use it in both open-source and commercial projects.

### Trained Models
Please note that the pre-trained models included in FaceAIKit are intended for research purposes only.


## Contact
If you have any questions, feedback, or inquiries about FaceAIKit, please don't hesitate to contact us at tgoldmann@seznam.cz or igoldmann@fit.vutbr.cz.


## References


* [ArcFace](https://arxiv.org/abs/1801.07698)

* [MagFace](https://github.com/IrvingMeng/MagFace)- source of MagFace model converted to onnx

* [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) - Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang. Look at Boundary: A Boundary-Aware Face Alignment Algorithm. 2018

* [Mediapipe](https://developers.google.com/mediapipe) - source of model for landmark detection