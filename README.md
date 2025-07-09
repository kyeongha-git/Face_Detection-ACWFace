# 👋 Introduction

해당 연구는 학부생 3학년 2학기에 수강한 "딥러닝응용2" 수업에서 진행한 기말 프로젝트입니다.
수업에서는 Image Classification, Object Detection, Image Segmentation 등 Computer Vision에서 연구되는 분야에 대한 개념을 학습하였으며, 그 중 **Object Detection**에 관심을 갖고 **Face Detection**에 초점을 맞추어 프로젝트를 진행했습니다.

당시 Face Detection 분야의 SOTA 모델은 [RetinaFace](https://arxiv.org/abs/1905.00641)였습니다. 이에 저희는 해당 SOTA 모델의 성능을 **개선(Develop)** 해보는 것을 목표로 프로젝트를 시작했습니다.

관련 문헌 조사 중, RetinaFace를 기반으로 한 향상 모델인 [ACWFace](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short) 논문을 발견하였고, 해당 모델이 **RetinaFace보다 뛰어난 Detection 성능**을 보였지만 **Flops가 높아 실시간 사용이 어려운 점**을 확인했습니다.

따라서 본 프로젝트는 **ACWFace의 성능을 유지하면서도, Flops는 RetinaFace 수준으로 낮추는 경량화 모델 설계**를 최종 목표로 설정하였습니다.

---

# 🤔 Approach

RetinaFace 구조에 대한 이해를 위해 논문 스터디를 진행하고, [기존 코드](https://github.com/kyeongha-git/Face_Detection-RetinaFace)를 한 줄씩 분석하며 데이터셋 기반으로 훈련 및 평가를 수행했습니다.

또한 경량화를 위한 방향성을 설정하기 위해, RetinaFace를 경량화한 연구 사례를 조사하던 중 [FDLite](https://arxiv.org/abs/2406.19107) 논문을 참고하게 되었고, 아래와 같은 분석 결과를 얻었습니다:

![image](https://github.com/user-attachments/assets/bb30e517-47d3-44cc-892c-564b693271ce)

해당 결과로부터, **Detection Head가 전체 연산량의 큰 비중을 차지**함을 알 수 있었고 이를 개선 포인트로 설정했습니다.
하지만 ACWFace는 이미 Detection Head에 변형이 가해진 상태였기에, **ACWFace 내에서 추가적으로 경량화할 수 있는 모듈을 분석**했습니다.

그 결과, 기존 Convolution보다 **Deformable Convolution이 경량화에 효과적이며 성능 유지 가능**하다는 점에 착안하여, 이를 최종 구조에 적용하였습니다.

---

# 📊 RetinaFace vs ACWFace

## ✅ RetinaFace 구조

* **Backbone**: 이미지 특성 추출 (ResNet50 → MobileNet 0.25 사용)
* **FPN**: 다양한 스케일의 Feature Map 학습
* **Context Module**: 다양한 크기의 얼굴 탐지를 위한 3x3, 5x5, 7x7 Conv

![image](https://github.com/user-attachments/assets/159d12a1-00f9-4bc0-9150-f5e2663d6e30)

## ✅ ACWFace 구조

* **WFPN**: 기존 FPN을 Weighted FPN으로 대체
* **EDAM**: Attention Layer 추가
* **SCM**: Context Module을 ShuffleNet 기반 모듈로 변경

![image](https://github.com/user-attachments/assets/e04b0c0b-40d0-4e68-9eac-9eb2bfd54edf)

---

# 📄 Detail

## 🔁 FPN vs WFPN

![image](https://github.com/user-attachments/assets/b13781a5-3e3d-471d-bb4d-bcfd6521b0bc)
![image](https://github.com/user-attachments/assets/7dd88189-7c5d-45cb-8eba-b8232df7d432)

* 상위 레벨 Feature Map을 하위로 전파할 때 **Upsampling(interpolate)** 과정을 거칩니다.
* WFPN에서는 **sigmoid를 통한 가중치 조절**을 추가하여 상위 Feature의 특성이 보존되도록 설계합니다.

## 🔃 SSH vs SCM

![image](https://github.com/user-attachments/assets/d13aac09-1cd7-4032-8e54-2d7fdb21dcbd)
![image](https://github.com/user-attachments/assets/3ddde5d2-f911-422b-99c2-6d6d21003250)

* SCM은 SSH 대비 채널 수를 줄여 **연산량을 감소**시켰으며,
* 채널 방향 concat 대신 **3x3 Conv 연산으로 통합**, **정보 혼합 및 성능 향상**에 기여합니다.

## ✨ EDAM (Enhanced Dual Attention Module)

![image](https://github.com/user-attachments/assets/43dc0a7b-015c-4aac-a23a-0af87755e215)
![image](https://github.com/user-attachments/assets/8e38990a-f3d6-4507-95ac-558871a86c9e)

* 기존 CBAM과 유사한 구조지만, Channel Attention에 **적응형 Conv1D** 구조를 적용
* Sigmoid 처리 방식에서 차별화하여 **성능 향상 및 연산 효율을 개선**합니다.

---

# 🚀 Result

아래는 각 모듈을 하나씩 추가하며 RetinaFace와의 성능, 연산량 차이를 비교한 결과입니다:

| Style                                            |    easy   |   medium  |    hard   |    Flops   |   parameter  |
| :----------------------------------------------- | :-------: | :-------: | :-------: | :--------: | :----------: |
| RetinaFace                                       |   90.7%   |   88.1%   |   73.4%   |   1.030G   |   426.608K   |
| + WFPN                                           |   90.8%   |   88.2%   |   74.0%   |   1.030G   |   426.738K   |
| + WFPN + SCM                                     |   91.0%   |   88.6%   |   74.7%   |   1.304G   |   523.986K   |
| + WFPN + SCM + EDAM (ACWFace)                    |   91.4%   |   89.2%   |   75.1%   |   1.427G   |   567.258K   |
| **+ WFPN + SCM + EDAM + Deformable Conv (Ours)** | **91.7%** | **89.8%** | **76.1%** | **1.124G** | **478.167K** |

✅ 최종적으로 **Detection 성능 1\~3% 향상**, **Flops 감소**, **ACWFace 경량화 성공**

📌 **향후 목표**: Flops를 더욱 낮추면서도 RetinaFace 수준 이상의 성능을 유지하는 모델 설계

---

# 📌 Reference

* RetinaFace: [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
* RetinaFace Code: [https://github.com/biubug6/Pytorch\_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
* FDLite: [https://arxiv.org/abs/2406.19107](https://arxiv.org/abs/2406.19107)
* ACWFace: [https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short)
