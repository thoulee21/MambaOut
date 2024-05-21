# [MambaOut: 시력을 위해 Mamba가 정말 필요한가요?](https://arxiv.org/abs/2405.07992)

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/whyu/MambaOut" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> "내가 무슨 말을 할 수 있겠는가, 맘바 아웃." —_코비 브라이언트, NBA 고별 연설, 2016_

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>

이것은 우리 논문에서 제안한 MambaOut의 PyTorch 구현입니다.[MambaOut: 시력을 위해 Mamba가 정말 필요한가요?](https://arxiv.org/abs/2405.07992)".

## 업데이트

-   2024년 5월 20일: 24개의 Gated CNN 블록이 포함된 MambaOut-Kobe 모델 버전 출시[이슈 #5](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019).**MambaOut-Kobe는 단 41%의 매개변수와 33%의 FLOP로 ViT-S보다 0.2%의 정확도를 능가합니다.**. 보다[모델](#models).

-   2024년 5월 18일: 추가[지도 시간](https://github.com/yuweihao/MambaOut/issues/210)Transformer FLOP 계산에 대해(논문의 방정식 6)

* * *

![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)그림 1: (a) Gated CNN 및 Mamba 블록의 아키텍처(정규화 및 단축키 생략). Mamba 블록은 추가 상태 공간 모델(SSM)을 사용하여 Gated CNN을 확장합니다. 섹션 3에서 개념적으로 논의할 것처럼 SSM은 ImageNet의 이미지 분류에 필요하지 않습니다. 이 주장을 경험적으로 검증하기 위해 우리는 Gated CNN 블록을 쌓아 MambaOut이라는 일련의 모델을 구축했습니다.(b) MambaOut은 ImageNet 이미지 분류에서 Vision Mamhba, VMamba 및 PlainMamba와 같은 시각적 Mamba 모델보다 성능이 뛰어납니다.

<br>

![MambaOut second figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_second_figure.png)그림 2: 메모리 관점에서 인과적 주의 및 RNN 유사 모델의 메커니즘 설명. 여기서 $x_i$는 $i$ 번째 단계의 입력 토큰을 나타냅니다. (a) 인과주의는 모든 이전 토큰의 키 $k$와 값 $v$를 메모리로 저장합니다. 현재 토큰의 키와 값을 지속적으로 추가하여 메모리가 업데이트되므로 메모리 손실은 없지만 시퀀스가 ​​길어질수록 이전 메모리와 현재 토큰을 통합하는 계산 복잡도가 증가한다는 단점이 있습니다. 따라서 Attention은 짧은 시퀀스를 효과적으로 관리할 수 있지만 긴 시퀀스에서는 어려움을 겪을 수 있습니다. (b) 대조적으로 RNN과 유사한 모델은 이전 토큰을 메모리 역할을 하는 고정 크기 숨겨진 상태 $h$로 압축합니다. 이 고정된 크기는 RNN 메모리가 본질적으로 손실이 많아 Attention 모델의 무손실 메모리 용량과 직접 경쟁할 수 없음을 의미합니다. 그럼에도 불구하고,**RNN과 유사한 모델은 시퀀스 길이에 관계없이 이전 메모리를 현재 입력과 병합하는 복잡성이 일정하게 유지되므로 긴 시퀀스를 처리하는 데 뚜렷한 이점을 보여줄 수 있습니다.**

<br>

![MambaOut third figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_third_figure.png)그림 3: (a) 토큰 혼합의 두 가지 모드. 총 $T$ 토큰의 경우 완전 표시 모드를 사용하면 $t$ 토큰이 모든 토큰의 입력을 집계할 수 있습니다(예: $ \\left).{엑스_난 \\맞아}_{i=1}^{T} $, 출력 $y를 계산합니다._티$. 대조적으로, 인과 모드는 $t$ 토큰을 이전 및 현재 토큰 $ \\left의 입력만 집계하도록 제한합니다.{x_i \\오른쪽}_{i=1}^{t} $. 기본적으로 Attention은 완전히 표시되는 모드에서 작동하지만 원인 주의 마스크를 사용하여 원인 모드로 조정할 수 있습니다. Mamba의 SSM과 같은 RNN 유사 모델은 반복적 특성으로 인해 본질적으로 인과 모드에서 작동합니다. (비)**우리는 ViT의 주의를 완전히 가시적인 모드에서 인과 모드로 수정하고 ImageNet의 성능 저하를 관찰했습니다. 이는 작업을 이해하는 데 인과 혼합이 불필요함을 나타냅니다.**

## 요구사항

PyTorch 및 timm 0.6.11(`pip install timm==0.6.11`).

데이터 준비: ImageNet은 다음과 같은 폴더 구조를 가지고 있으며, 이를 통해 ImageNet을 추출할 수 있습니다.[스크립트](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

    │imagenet/
    ├──train/
    │  ├── n01440764
    │  │   ├── n01440764_10026.JPEG
    │  │   ├── n01440764_10027.JPEG
    │  │   ├── ......
    │  ├── ......
    ├──val/
    │  ├── n01440764
    │  │   ├── ILSVRC2012_val_00000293.JPEG
    │  │   ├── ILSVRC2012_val_00002138.JPEG
    │  │   ├── ......
    │  ├── ......

## 모델

### ImageNet에서 훈련된 MambaOut

| 모델                                                                                                |  해결 |  매개변수 |  MAC  | Top1 Acc |
| :------------------------------------------------------------------------------------------------ | :-: | :---: | :---: | :------: |
| [mambaout_femto](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth) | 224 | H. 삼촌 |  1.2G |   78.9   |
| [mambaout_kobe](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth)\* | 224 |  910만 |  1.5G |   80.0   |
| [mambaout_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth)   | 224 | 26.5M |  4.5G |   82.7   |
| [mambaout_small](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth) | 224 | 48.5M |  9.0G |   84.1   |
| [mambaout_base](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth)   | 224 | 84. 댐 | 15.8G |   84.2   |

\*[고베 메모리얼 버전](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019)24개의 Gated CNN 블록이 있습니다.

#### 용법

또한 MambaOut으로 추론을 수행하는 단계를 실행하는 Colab 노트북도 제공됩니다.[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing).

## 데모 구축

웹 데모는 다음 위치에 표시됩니다.[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyu/MambaOut). 로컬에서 쉽게 그라디오 데모를 실행할 수도 있습니다. PyTorch 및 timm==0.6.11 외에도 Gradio를 설치하십시오.`pip install gradio`, 그런 다음 실행

```bash
python gradio_demo/app.py
```

## 확인

모델을 평가하려면 다음을 실행하세요.

```bash
MODEL=mambaout_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## 기차

기본적으로 배치 크기는 4096을 사용하고 8개의 GPU로 모델을 학습하는 방법을 보여줍니다. 다중 노드 학습의 경우 조정`--grad-accum-steps`당신의 상황에 따라.

```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/MambaOut # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=mambaout_tiny 
DROP_PATH=0.2


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH
```

다른 모델의 훈련 스크립트는 다음과 같습니다.[스크립트](/scripts/).

## Transformer FLOP 계산에 대한 튜토리얼

이것[지도 시간](https://github.com/yuweihao/MambaOut/issues/210)Transformer FLOP를 계산하는 방법을 보여줍니다(논문의 방정식 6). 피드백을 환영하며 지속적으로 개선해 나가겠습니다.

## 비브텍스

    @article{yu2024mambaout,
      title={MambaOut: Do We Really Need Mamba for Vision?},
      author={Yu, Weihao and Wang, Xinchao},
      journal={arXiv preprint arXiv:2405.07992},
      year={2024}
    }

## 승인

Weihao는 Snap Research Fellowship, Google TPU Research Cloud(TRC), Google Cloud Research Credits 프로그램의 일부 지원을 받았습니다. 귀중한 토론을 해주신 Dongze Lian, Qiuhong Shen, Xingyi Yang, Gongfan Fang에게 감사드립니다.

우리의 구현은 다음을 기반으로 합니다.[pytorch-이미지-모델](https://github.com/huggingface/pytorch-image-models),[풀포머](https://github.com/sail-sg/poolformer),[ConvNeXt](https://github.com/facebookresearch/ConvNeXt),[메타폼](https://github.com/sail-sg/metaformer)그리고[시작다음](https://github.com/sail-sg/inceptionnext).
