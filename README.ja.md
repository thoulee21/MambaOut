# [MambaOut: ビジョンに Mamba は本当に必要ですか?](https://arxiv.org/abs/2405.07992)

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/whyu/MambaOut" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center"><em>In memory of Kobe Bryant</em></p>

> 「何と言うか、マンバは出て行け。」 —_コービー・ブライアント、NBA お別れのスピーチ、2016 年_

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>

これは、私たちの論文で提案された MambaOut の PyTorch 実装です。[MambaOut: ビジョンに Mamba は本当に必要ですか?](https://arxiv.org/abs/2405.07992)”。

## アップデート

-   2024 年 5 月 20 日: によって提案された 24 個の Gated CNN ブロックを含む MambaOut-Kobe モデル バージョンをリリース[問題 #5](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019)。**MambaOut-Kobe は、わずか 41% のパラメーターと 33% の FLOP で ViT-S を 0.2% の精度で上回っています。**。見る[モデル](#models)。

-   2024 年 5 月 18 日: を追加[チュートリアル](https://github.com/yuweihao/MambaOut/issues/210)変圧器の FLOP の計算について (論文の式 6)。

* * *

![MambaOut first figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_first_figure.png)図 1: (a) Gated CNN ブロックと Mamba ブロックのアーキテクチャ (正規化とショートカットは省略)。 Mamba ブロックは、追加の状態空間モデル (SSM) を使用して Gated CNN を拡張します。セクション 3 で概念的に説明するように、SSM は ImageNet 上の画像分類には必要ありません。この主張を経験的に検証するために、Gated CNN ブロックを積み重ねて MambaOut という名前の一連のモデルを構築します。(b) MambaOut は、ImageNet 画像分類において、視覚的な Mamba モデル (たとえば、Vision Mamhba、VMamba、PlainMamba) よりも優れています。

<br>

![MambaOut second figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_second_figure.png)図 2: メモリの観点から見た因果的注意と RNN のようなモデルのメカニズムの図。$x_i$ は $i$ 番目のステップの入力トークンを示します。 (a) 因果的注意は、以前のすべてのトークンのキー $k$ と値 $v$ をメモリとして保存します。メモリは現在のトークンのキーと値を継続的に追加することによって更新されるため、メモリはロスレスですが、欠点は、古いメモリと現在のトークンを統合する計算の複雑さがシーケンスが長くなるにつれて増加することです。したがって、注意を払うことで短いシーケンスを効果的に管理できますが、長いシーケンスでは困難に遭遇する可能性があります。 (b) 対照的に、RNN のようなモデルは、以前のトークンを固定サイズの隠れ状態 $h$ に圧縮し、メモリとして機能します。この固定サイズは、RNN メモリが本質的に損失が多く、アテンション モデルの損失のないメモリ容量と直接競合できないことを意味します。それにもかかわらず、**RNN のようなモデルは、シーケンスの長さに関係なく、古いメモリと現在の入力をマージする複雑さが一定のままであるため、長いシーケンスの処理において明確な利点を実証できます。**

<br>

![MambaOut third figure](https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mambaout_third_figure.png)図 3: (a) トークン混合の 2 つのモード。合計 $T$ トークンの場合、完全可視モードでは、トークン $t$ がすべてのトークン、つまり $ \\left からの入力を集約できます。｛バツ_\\そうです｝_{i=1}^{T} $、出力 $y を計算します_t$。対照的に、因果モードでは、トークン $t$ が以前のトークンと現在のトークン $ \\left からの入力のみを集約するように制限されます。｛x_i \\右｝_{i=1}^{t} $。デフォルトでは、アテンションは完全に表示されたモードで動作しますが、因果的アテンション マスクを使用して因果的モードに調整できます。 Mamba の SSM などの RNN のようなモデルは、その反復的な性質により本質的に因果モードで動作します。 (b)**ViT のアテンションを完全可視モードから因果モードに変更し、ImageNet でのパフォーマンスの低下を観察しました。これは、タスクを理解するために因果混合が不要であることを示しています。**

## 要件

PyTorch と timm 0.6.11 (`pip install timm==0.6.11`）。

データの準備：以下のフォルダー構成のImageNet。これでImageNetを抽出できます。[脚本](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)。

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

## モデル

### ImageNet でトレーニングされた MambaOut

| モデル                                                                                             | Resolution | パラメータ |  MAC  | トップ1アクセス |
| :---------------------------------------------------------------------------------------------- | :--------: | :---: | :---: | :------: |
| [マンバアウト\_フェムト](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth) |     ２２４    |  H.叔父 |  1.2G |   ７８。９   |
| [マンバアウト神戸](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth)＊     |     ２２４    |  910万 |  1.5G |   ８０。０   |
| [mambaout_tiny](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth) |     ２２４    | 26.5M |  4.5G |   ８２。７   |
| [マンバアウト\_スモール](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth) |     ２２４    | 48.5M |  9.0G |   ８４。１   |
| [マンバアウトベース](https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth)     |     ２２４    | 84.ダム | 15.8G |   ８４。２   |

＊[神戸記念バージョン](https://github.com/yuweihao/MambaOut/issues/5#issuecomment-2119555019)24 個の Gated CNN ブロックを備えています。

#### 使用法

MambaOut で推論を実行するステップを実行する Colab ノートブックも提供します。[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing)。

## Ｇラヂオ でも

Web デモは次の場所で表示されます。[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyu/MambaOut)。 gradio デモをローカルで簡単に実行することもできます。 PyTorch と timm==0.6.11 以外に、gradio をインストールしてください。`pip install gradio`、実行します

```bash
python gradio_demo/app.py
```

## 検証

モデルを評価するには、次を実行します。

```bash
MODEL=mambaout_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## 電車

デフォルトではバッチ サイズ 4096 を使用し、8 GPU でモデルをトレーニングする方法を示します。マルチノードトレーニングの場合は、調整します`--grad-accum-steps`あなたの状況に応じて。

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

他のモデルのトレーニング スクリプトを以下に示します。[スクリプト](/scripts/)。

## 変圧器の FLOP のカウントに関するチュートリアル

これ[チュートリアル](https://github.com/yuweihao/MambaOut/issues/210)は、トランスの FLOP をカウントする方法を示しています (論文の式 6)。フィードバックを歓迎します。継続的に改善していきます。

## ビブテックス

    @article{yu2024mambaout,
      title={MambaOut: Do We Really Need Mamba for Vision?},
      author={Yu, Weihao and Wang, Xinchao},
      journal={arXiv preprint arXiv:2405.07992},
      year={2024}
    }

## 了承

Weihao は、Snap Research Fellowship、Google TPU Research Cloud (TRC)、および Google Cloud Research Credits プログラムによって部分的に支援されました。貴重な議論をしていただいた Dongze Lian、Qiuhong Shen、Xingyi Yang、Gongfan Fang に感謝します。

私たちの実装は以下に基づいています[pytorch-画像モデル](https://github.com/huggingface/pytorch-image-models)、[プールフォーマー](https://github.com/sail-sg/poolformer)、[次への変換](https://github.com/facebookresearch/ConvNeXt)、[メタフォーム](https://github.com/sail-sg/metaformer)そして[インセプションネクスト](https://github.com/sail-sg/inceptionnext)。
