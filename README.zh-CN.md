# [MambaOut：我们真的需要 Mamba 来实现愿景吗？](https://arxiv.org/abs/2405.07992)

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/whyu/MambaOut" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> “我能说什么，曼巴出去。” —_科比·布莱恩特，NBA 告别演说，2016_

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>

这是我们的论文提出的 MambaOut 的 PyTorch 实现“[MambaOut：我们真的需要 Mamba 来实现愿景吗？](https://arxiv.org/abs/2405.07992)".

## 更新

-   2024 年 5 月 20 日：发布 MambaOut-Kobe 模型版本，其中建议有 24 个门控 CNN 块[问题#5](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019).**MambaOut-Kobe 仅用 41% 的参数和 33% 的 FLOP 就比 ViT-S 的准确率高出 0.2%**。看[楷模](#models).

-   2024 年 5 月 18 日：添加[教程](https://github.com/yuweihao/MambaOut/issues/210)对 Transformer FLOP 进行计数（论文中的公式 6）。

* * *

![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)图 1：(a) 门控 CNN 和 Mamba 块的架构（省略归一化和快捷方式）。 Mamba 模块通过附加的状态空间模型 (SSM) 扩展了门控 CNN。正如第 3 节中将在概念上讨论的那样，SSM 对于 ImageNet 上的图像分类来说不是必需的。为了凭经验验证这一说法，我们堆叠门控 CNN 块来构建一系列名为 MambaOut 的模型。(b) MambaOut 在 ImageNet 图像分类上优于视觉 Mamba 模型，例如 Vision Mamhba、VMamba 和 PlainMamba。

<br>

![MambaOut second figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_second_figure.png)图 2：从记忆角度来看因果注意力和类 RNN 模型的机制说明，其中 $x_i$ 表示第 $i$ 步骤的输入标记。 (a) 因果注意力将所有先前标记的键 $k$ 和值 $v$ 存储为内存。通过不断添加当前令牌的键和值来更新内存，因此内存是无损的，但缺点是随着序列变长，整合旧内存和当前令牌的计算复杂度会增加。因此，注意力可以有效地管理短序列，但可能会遇到较长序列的困难。 (b) 相反，类似 RNN 的模型将先前的 token 压缩为固定大小的隐藏状态 $h$，用作内存。这种固定大小意味着 RNN 内存本质上是有损的，无法与注意力模型的无损内存容量直接竞争。尽管如此，**类似 RNN 的模型在处理长序列时可以表现出明显的优势，因为无论序列长度如何，将旧内存与当前输入合并的复杂性保持不变。**

<br>

![MambaOut third figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_third_figure.png)图 3：(a) 两种代币混合模式。对于总共 $T$ 代币，完全可见模式允许代币 $t$ 聚合来自所有代币的输入，即 $ \\left{X_我\\右}_{i=1}^{T} $，计算其输出$y_t$。相反，因果模式将标记 $t$ 限制为仅聚合来自先前和当前标记 $ \\left 的输入{x_i \\右}_{i=1}^{t} $.默认情况下，注意力在完全可见模式下运行，但可以使用因果注意力蒙版调整为因果模式。类似 RNN 的模型，例如 Mamba 的 SSM，由于其循环性质，本质上以因果模式运行。 (二)**我们将 ViT 的注意力从完全可见修改为因果模式，并观察到 ​​ImageNet 上的性能下降，这表明因果混合对于理解任务是不必要的。**

## 要求

PyTorch 和 timm 0.6.11 (`pip install timm==0.6.11`).

数据准备：ImageNet，文件夹结构如下，可以通过这个解压ImageNet[脚本](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

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

### MambaOut 在 ImageNet 上训练

| 模型                                                                                                |  解决 |   参数   |  MAC  | Top1 加速器 |
| :------------------------------------------------------------------------------------------------ | :-: | :----: | :---: | :------: |
| [mambaout_femto](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth) | 224 |   H叔叔  |  1.2G |   78.9   |
| [曼巴奥科比](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth)\*         | 224 |  9.1M  |  1.5G |   80.0   |
| [曼巴out_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth)      | 224 |  26.5M |  4.5G |   82.7   |
| [mambaout\_小](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth)    | 224 |  48.5M |  9.0G |   84.1   |
| [曼巴输出基地](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth)          | 224 | 84. 水坝 | 15.8G |   84.2   |

\*[神户纪念版](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019)有 24 个门控 CNN 块。

#### 用法

我们还提供了一个 Colab 笔记本，它运行使用 MambaOut 执行推理的步骤：[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing).

## 构建演示

网络演示位于[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyu/MambaOut)。您还可以轻松地在本地运行 gradio 演示。除了 PyTorch 和 timm==0.6.11 之外，请安装 gradio`pip install gradio`，然后运行

```bash
python gradio_demo/app.py
```

## 验证

要评估模型，请运行：

```bash
MODEL=mambaout_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## 火车

我们默认使用 4096 的批量大小，并展示如何使用 8 个 GPU 训练模型。对于多节点训练，调整`--grad-accum-steps`根据你的情况。

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

其他模型的训练脚本如图[脚本](/scripts/).

## 计算 Transformer FLOP 次数的教程

这[教程](https://github.com/yuweihao/MambaOut/issues/210)展示了如何计算 Transformer FLOP 数（论文中的公式 6）。欢迎反馈，我会不断改进。

## 比布泰克斯

    @article{yu2024mambaout,
      title={MambaOut: Do We Really Need Mamba for Vision?},
      author={Yu, Weihao and Wang, Xinchao},
      journal={arXiv preprint arXiv:2405.07992},
      year={2024}
    }

## 致谢

伟豪得到了 Snap 研究奖学金、谷歌 TPU 研究云 (TRC) 和谷歌云研究学分计划的部分支持。我们感谢连东泽、沉秋红、杨兴一和方功凡的宝贵讨论。

我们的实现是基于[pytorch-图像模型](https://github.com/huggingface/pytorch-image-models),[池形成者](https://github.com/sail-sg/poolformer),[ConvNext](https://github.com/facebookresearch/ConvNeXt),[元形式](https://github.com/sail-sg/metaformer)和[下一个开始](https://github.com/sail-sg/inceptionnext).
