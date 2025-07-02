# 👋 Introduction

해당 연구는 학부생 3학년 2학기에 수강한 "딥러닝응용2" 수업에서 진행한 기말 프로젝트입니다.
수업에서는 Image Classification, Object Detection, Image Segmentation 등 Computer Vision에서 연구되는 분야에 대한 개념에 대해 학습하였으며, 저희는 그 중 Object Detection에 관심이 갔고 Face Detection에 초점을 맞추어 진행했습니다.

당시 Face Detection 분야의 SOTA 모델은 RetinaFace 이었습니다. 따라서, 본 SOTA 모델의 성능을 더 Develop 해보자라는 목표로 프로젝트를 시작하였습니다.
이후, Face Detection에서 RetinaFace를 기반으로 한 Develop 모델이 있는지를 살펴보았고, ACWFace라는 논문을 찾았습니다.
본 논문은 RetinaFace보다 더 뛰어난 Detection 성능을 보였지만, Flops가 높아 실시간 사용이 불가하였습니다.
따라서, 저희는 ACWFace의 성능을 유지하면서 Flops는 RetinaFace와 유사한 수준으로 낮추는 것을 최종 목표로 설정하여 진행했습니다.

# Approach

RetinaFace 자체에 대한 이해를 하기 위해 논문 스터디를 진행하였으며, [Code](https://github.com/kyeongha-git/Face_Detection-RetinaFace)를 한 줄씩 분석하며, 실제 데이터셋으로 훈련 및 평가를 진행했습니다.
또한, 경량화 방향을 잡기 위해 기존에 RetinaFace를 경량화한 논문을 찾아보았습니다. 그 결과, [FDLite]() 논문을 찾았으며 아래와 같은 결과를 얻을 수 있었습니다.
![image](https://github.com/user-attachments/assets/bb30e517-47d3-44cc-892c-564b693271ce)

위 이미지에서 보이는 것처럼 Detection Head 부분에서 가장 많은 Flops가 발생되는 것을 확인하였고, 이 부분을 개선하는 방향으로 설정하였습니다.

그러나, ACWFace 자체에서 Detection Head가 변경되었고 저희는 ACWFace 모델과 유사한 성능을 보여야 했기에 ACWFace에서 경량화할 수 있는 방향을 새로 모색하였습니다.

그 결과, 기존 Convolution 보다 Deformabel Convolution이 더 경량화되며 성능이 유지된다는 연구 결과를 찾았고 이를 적용하여 경량화를 진행했습니다.

# RetinaFace vs ACWFace

아래 이미지는 RetinaFace의 기본 구조입니다. 
- Backbone: 이미지의 특성을 추출하기 위해 기존 CNN 모델을 학습합니다. (논문에서는 ResNet50을 사용하였지만, 경량화를 위해 MobileNet 0.25를 사용합니다.)
- FPN: 이미지에서 다양한 스케일의 Feature Map을 추출합니다. 이후, Backbone에서 추출한 특성을 합하여 이미지의 특성을 유지하며 다양한 스케일을 학습합니다.
- Context Module: 3*3, 5*5, 7*7 Convolution을 이용해 얼굴의 다양한 크기를 탐지합니다.
![image](https://github.com/user-attachments/assets/159d12a1-00f9-4bc0-9150-f5e2663d6e30)



# 🚀 Presentation
