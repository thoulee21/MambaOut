# [MambaOut：我們真的需要 Mamba 來實現願景嗎？](https://arxiv.org/abs/2405.07992)

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/whyu/MambaOut" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> “我能說什麼，曼巴出去。” —_科比·布萊恩特，NBA 告別演說，2016_

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>

這是我們的論文提出的 MambaOut 的 PyTorch 實作“[MambaOut：我們真的需要 Mamba 來實現願景嗎？](https://arxiv.org/abs/2405.07992)".

## 更新

-   2024 年 5 月 20 日：發布 MambaOut-Kobe 模型版本，其中建議有 24 個門控 CNN 區塊[問題#5](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019).**MambaOut-Kobe 僅用 41% 的參數和 33% 的 FLOP 就比 ViT-S 的準確率高出 0.2%**。看[楷模](#models).

-   2024 年 5 月 18 日：新增[教學](https://github.com/yuweihao/MambaOut/issues/210)對 Transformer FLOP 進行計數（論文中的公式 6）。

* * *

![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)圖 1：(a) 門控 CNN 和 Mamba 區塊的架構（省略歸一化和捷徑）。 Mamba 模組透過附加的狀態空間模型 (SSM) 擴展了閘控 CNN。正如第 3 節中將在概念上討論的那樣，SSM 對於 ImageNet 上的圖像分類來說不是必需的。為了憑經驗驗證這一說法，我們堆疊門控 CNN 區塊來建立一系列名為 MambaOut 的模型。

<br>

![MambaOut second figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_second_figure.png)圖 2：從記憶角度來看因果注意力和類別 RNN 模型的機制說明，其中 $x_i$ 表示第 $i$ 步驟的輸入標記。 (a) 因果注意力將所有先前標記的鍵 $k$ 和值 $v$ 儲存為記憶體。透過不斷添加當前令牌的鍵和值來更新內存，因此內存是無損的，但缺點是隨著序列變長，整合舊內存和當前令牌的計算複雜度會增加。因此，注意力可以有效地管理短序列，但可能會遇到較長序列的困難。 (b) 相反，類似 RNN 的模型將先前的 token 壓縮為固定大小的隱藏狀態 $h$，用作記憶體。這種固定大小意味著 RNN 記憶體本質上是有損的，無法與注意力模型的無損記憶體容量直接競爭。儘管如此，**類似 RNN 的模型在處理長序列時可以表現出明顯的優勢，因為無論序列長度如何，將舊記憶體與當前輸入合併的複雜性保持不變。**

<br>

![MambaOut third figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_third_figure.png)圖 3：(a) 兩種代幣混合模式。對於總共 $T$ 代幣，完全可見模式允許代幣 $t$ 聚合來自所有代幣的輸入，即 $ \\left{X_我\\右}_{i=1}^{T} $，計算其輸出$y_t$。相反，因果模式將標記 $t$ 限制為僅聚合來自先前和當前標記 $ \\left 的輸入{x_i \\右}_{i=1}^{t} $.預設情況下，注意力在完全可見模式下運行，但可以使用因果注意力蒙版調整為因果模式。類似 RNN 的模型，例如 Mamba 的 SSM，由於其循環性質，本質上以因果模式運行。 (二)**我們將 ViT 的注意力從完全可見修改為因果模式，並觀察到 ImageNet 上的表現下降，這表明因果混合對於理解任務是不必要的。**

## 要求

PyTorch 和 timm 0.6.11 (`pip install timm==0.6.11`).

資料準備：ImageNet，資料夾結構如下，可以透過這個解壓縮ImageNet[腳本](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

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

## 楷模

### MambaOut 在 ImageNet 上訓練

| 模型                                                                                                |  解決 |   參數   |  MAC  | Top1 加速器 |
| :------------------------------------------------------------------------------------------------ | :-: | :----: | :---: | :------: |
| [mambaout_femto](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth) | 224 |   H叔叔  |  1.2G |   78.9   |
| [曼巴奧科比](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth)\*         | 224 |  9.1M  |  1.5G |   80.0   |
| [曼巴out_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth)      | 224 |  26.5M |  4.5G |   82.7   |
| [mambaout\_小](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth)    | 224 |  48.5M |  9.0G |   84.1   |
| [曼巴輸出基地](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth)          | 224 | 84. 水壩 | 15.8G |   84.2   |

\*[神戶紀念版](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019)有 24 個門控 CNN 塊。

#### 用法

我們還提供了一個 Colab 筆記本，它運行使用 MambaOut 執行推理的步驟：[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing).

## 建構演示

網路示範位於[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyu/MambaOut)。您也可以輕鬆地在本地運行 gradio 演示。除了 PyTorch 和 timm==0.6.11 之外，請安裝 gradio`pip install gradio`，然後運行

```bash
python gradio_demo/app.py
```

## 驗證

要評估模型，請運行：

```bash
MODEL=mambaout_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## 火車

我們預設使用 4096 的批次大小，並展示如何使用 8 個 GPU 訓練模型。對於多節點訓練，調整`--grad-accum-steps`根據你的情況。

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

其他模型的訓練腳本如圖[腳本](/scripts/).

## 計算 Transformer FLOP 次數的教學

這[教學](https://github.com/yuweihao/MambaOut/issues/210)展示如何計算 Transformer FLOP 數（論文中的公式 6）。歡迎回饋，我會不斷改進。

## 比布泰克斯

    @article{yu2024mambaout,
      title={MambaOut: Do We Really Need Mamba for Vision?},
      author={Yu, Weihao and Wang, Xinchao},
      journal={arXiv preprint arXiv:2405.07992},
      year={2024}
    }

## 致謝

Weihao was partly supported by Snap Research Fellowship, Google TPU Research Cloud (TRC), and Google Cloud Research Credits program. We thank Dongze Lian, Qiuhong Shen, Xingyi Yang, and Gongfan Fang for valuable discussions.

我們的實現是基於[pytorch-圖像模型](https://github.com/huggingface/pytorch-image-models),[池形成者](https://github.com/sail-sg/poolformer),[ConvNext](https://github.com/facebookresearch/ConvNeXt),[元形式](https://github.com/sail-sg/metaformer)和[下一個開始](https://github.com/sail-sg/inceptionnext).
