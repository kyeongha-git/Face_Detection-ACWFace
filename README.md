# 👋 Introduction

해당 연구는 학부생 3학년 2학기에 수강한 "딥러닝응용2" 수업에서 진행한 기말 프로젝트입니다.
수업에서는 Image Classification, Object Detection, Image Segmentation 등 Computer Vision에서 연구되는 분야에 대한 개념에 대해 학습하였으며, 저희는 그 중 Object Detection에 관심이 갔고 Face Detection에 초점을 맞추어 진행했습니다.

당시 Face Detection 분야의 SOTA 모델은 [RetinaFace](https://arxiv.org/abs/1905.00641) 이었습니다. 따라서, 본 SOTA 모델의 성능을 더 Develop 해보자라는 목표로 프로젝트를 시작하였습니다.
이후, Face Detection에서 RetinaFace를 기반으로 한 Develop 모델이 있는지를 살펴보았고, [ACWFace](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short)라는 논문을 찾았습니다.
본 논문은 RetinaFace보다 더 뛰어난 Detection 성능을 보였지만, Flops가 높아 실시간 사용이 불가하였습니다.
따라서, 저희는 ACWFace의 성능을 유지하면서 Flops는 RetinaFace와 유사한 수준으로 낮추는 것을 최종 목표로 설정하여 진행했습니다.

# 🤔 Approach

RetinaFace 자체에 대한 이해를 하기 위해 논문 스터디를 진행하였으며, [Code](https://github.com/kyeongha-git/Face_Detection-RetinaFace)를 한 줄씩 분석하며, 실제 데이터셋으로 훈련 및 평가를 진행했습니다.
또한, 경량화 방향을 잡기 위해 기존에 RetinaFace를 경량화한 논문을 찾아보았습니다. 그 결과, [FDLite]() 논문을 찾았으며 아래와 같은 결과를 얻을 수 있었습니다.
![image](https://github.com/user-attachments/assets/bb30e517-47d3-44cc-892c-564b693271ce)

위 이미지에서 보이는 것처럼 Detection Head 부분에서 가장 많은 Flops가 발생되는 것을 확인하였고, 이 부분을 개선하는 방향으로 설정하였습니다.

그러나, ACWFace 자체에서 Detection Head가 변경되었고 저희는 ACWFace 모델과 유사한 성능을 보여야 했기에 ACWFace에서 경량화할 수 있는 방향을 새로 모색하였습니다.

그 결과, 기존 Convolution 보다 Deformabel Convolution이 더 경량화되며 성능이 유지된다는 연구 결과를 찾았고 이를 적용하여 경량화를 진행했습니다.

# 📊 RetinaFace vs ACWFace

아래 이미지는 RetinaFace의 기본 구조입니다. 
- Backbone: 이미지의 특성을 추출하기 위해 기존 CNN 모델을 학습합니다. (논문에서는 ResNet50을 사용하였지만, 경량화를 위해 MobileNet 0.25를 사용합니다.)
- FPN: 이미지에서 다양한 스케일의 Feature Map을 추출합니다. 이후, Backbone에서 추출한 특성을 합하여 이미지의 특성을 유지하며 다양한 스케일을 학습합니다.
- Context Module: 3*3, 5*5, 7*7 Convolution을 이용해 얼굴의 다양한 크기를 탐지합니다.
  
![image](https://github.com/user-attachments/assets/159d12a1-00f9-4bc0-9150-f5e2663d6e30)

아래 이미지는 ACWFace의 기본 구조입니다.
- Backbone은 그대로 사용합니다.
- WFPN: FPN layer를 WFPN layer로 변경합니다.
- EDAM: 기존 RetinaFace에는 없던 Attetnion Layer를 추가합니다.
- Context Module: ShuffleNet에서 제안되었던 SCM 모듈로 변경합니다.

![image](https://github.com/user-attachments/assets/e04b0c0b-40d0-4e68-9eac-9eb2bfd54edf)

# 📄 Detail

## FPN vs WFPN
![image](https://github.com/user-attachments/assets/b13781a5-3e3d-471d-bb4d-bcfd6521b0bc)

![image](https://github.com/user-attachments/assets/7dd88189-7c5d-45cb-8eba-b8232df7d432)

stage1 → stage2 → stage3 순으로 상위 레벨 feature mapd 입니다.

가장 상위 레벨 feature map을 하위 레벨 feature map에 더해주는 과정에서 크기가 맞지 않기 때문에 interpolate layer를 한 번 거쳐줍니다. (논문에서는 Upsampling layer로 표현하였으며, Ups(O_i)로 표현.)

이 다음에는 하위 레벨 feature map과 더하고, conv_bn을 거쳐 output이 생성됩니다.

WFPN에서는 중간에 가중치를 구하는 과정이 있으며 이 가중치는 Ups(O_i+1)에 1*1 conv를 거치고 난 후 sigmoid를 통과한 값으로 결정됩니다.
이 가중치를 Ups(O_i+1)와 O_i에 각각 알파, 1-알파 값을 곱하고 더하는 과정이 추가되었습니다.

이를 통해, 기존 상위 레벨 feature map이 가지고 있던 특성을 덜 잃으면서 학습이 되도록 만들어준 것이 WFPN입니다.

## SSH vs SCM
![image](https://github.com/user-attachments/assets/d13aac09-1cd7-4032-8e54-2d7fdb21dcbd)

![image](https://github.com/user-attachments/assets/3ddde5d2-f911-422b-99c2-6d6d21003250)

SSH에서는 처음 3*3 conv 통과한 값의 채널이 C/2 였습니다. 근데, SCM에서는 C/4로 변경했습니다.

또한, Conv 레이어가 하나 더 추가되었습니다. → 모든 마지막 Conv 출력을 다 C/4로 맞추고, 마지막에 Conv를 하나 더 추가해서 모두 다 더하면 C가 되도록 설계했습니다.
이를 통해, 채널 수가 줄어들었으므로 연산량 감소하여 경량화 작업 중 하나에 포함됩니다.

원래 SSH는 출력값들을 모두 채널 방향으로 붙이는 작업(torch.cat)이었으나 → SCM에서는 이걸 3*3 Conv를 통해 붙입니다.
이를 통해, 채널 정보 간의 교차로 더 높은 성능을 달성 달성할 수 있습니다.

## EDAM
![image](https://github.com/user-attachments/assets/43dc0a7b-015c-4aac-a23a-0af87755e215)
![image](https://github.com/user-attachments/assets/8e38990a-f3d6-4507-95ac-558871a86c9e)

본 레이어는 RetinaFace에서 없었던 레이어입니다. ACWFace에서는 EDAM 모듈을 추가로 배치하였습니다.

CBAM 모듈에서 spatial attention 부분은 그대로 차용합니다.

EDAM에서는 Channel Attention 부분에서 MLP를 1*1 Conv2d → Relu → 1*1 Conv2d으로 구성했으나. 이를 Conv1D로 변경하고, kernel size를 채널마다 변경되도록 적응형으로 바꾸었습니다.

그리고, CBAM은 각 Convolution을 거친 결과를 다 더하고 나서 마지막에 sigmoid 한 번이었으나, EDAM은 Sigmoid를 각 Conv를 거친 결과에 취해주고, 합친 후 다시 Sigmoid를 취하는 것이 차이점입니다.

# 🚀 Result

ACWFace에서 추가된 레이어들을 RetinaFace에서 하나씩 추가하며 성능을 비교하였으며, 최종적으로는 Defomable Convolution을 추가하여 성능을 비교하였습니다.

| Style | easy | medium | hard | Flops | parameter |
|:-|:-:|:-:|:-:|:-:|:-:|
| RetinaFace | 90.7% | 88.1% | 73.4% | 1.030G | 426.608K |
| RetinaFace + WFPN | 90.8% | 88.2% | 74.0% | 1.030G | 426.738K |
| RetinaFace + WFPN + SCM | 91.0% | 88.6% | 74.7% | 1.304G | 523.986K |
| RetinaFace + WFPN + SCM + EDAM (ACWFace) | 91.4% | 89.2% | 75.1% | 1.427G | 567.258K |
| RetinaFace + WFPN + SCM + EDAM + Deformable Convolution (Ours) | 91.7% | 89.8% | 76.1% | 1.124G | 478.167K |

최종적으로 RetinaFace에서 Detection 성능을 1~3% 상승시켰으며, ACWFace 경량화에 성공하였습니다.

추후에는, 본 모델을 RetinaFace보다 Flops가 더 낮게 소모하면서 성능은 유지할 수 있도록 경량화할 예정입니다.

# Reference
RetinaFace paper: https://arxiv.org/abs/1905.00641
RetinaFace Code: https://github.com/biubug6/Pytorch_Retinaface
FDLite: https://arxiv.org/abs/2406.19107
ACWFace Paper: https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-1/013012/ACWFace-efficient-and-lightweight-face-detector-based-on-RetinaFace/10.1117/1.JEI.31.1.013012.short
