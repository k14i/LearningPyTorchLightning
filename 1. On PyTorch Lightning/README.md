# On PyTorch Lightning / Learning PyTorch Lightning

## PyTorch Lightning とは

### 公式の説明

* [pytorch_lightning.LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)
    - PyTorch のコードを 5つのセクションに整理する
        1. Computations (init)
        2. Train loop (training_step)
        3. Validation loop (validation_step)
        4. Test loop (test_step)
        5. Optimizers (configure_optimizers)
    - 特徴は以下の通り
        1. PyTorch Lightning のコードは PyTorch のコードと同じである。
        2. PyTorch のコードが抽象化されるわけではなく、整理される。
        3. `LightningModule` に無いコードは `Trainer` によって自動化されている。
        4. Lightning が処理をするため `.cuda()` や `.to()` といったコールは不要。
        5. `DataLoader` において、デフォルトでは `DistributedSampler` が設定される。
        6. `LightningModule` は `torch.nn.Module` の一つであり、それに機能を追加したもの。

### 解釈

#### 1. PyTorch Lightning のコードは PyTorch のコードと同じである。

PyTorch Lightning を使って記述されるコードは PyTorch のコードであるということ。実際、 Computations セクションのコードは PyTorch のコードそのものである。

#### 2. PyTorch のコードが抽象化されるわけではなく、整理される。

PyTorch Lightning は PyTorch をシンプルなインターフェースで使うための所謂ラッパーではない。 PyTorch で従来記述していたエンジニアリングのためのコード（forループなど）に相当する機能を、 PyTorch Lightning は提供する。これにより、従来は自由に実装される反面煩雑になりがちだったエンジニアリング部分のコードが、 PyTorch Lightning の中でより整理された形で記述されることになる。

#### 3. `LightningModule` に無いコードは `Trainer` によって自動化されている。

PyTorch で従来 `torch.nn.Module` の外に記述していたエンジニアリングのためのコード（forループなど）に相当する機能は PyTorch Lightning の `LightningModule` 内に定義していくが、それらを適切な順序で適切な回数だけ呼び出す役割を担うのが `Trainer` である。

#### 4. Lightning が処理をするため `.cuda()` や `.to()` といったコールは不要。

一部、便利なラッパーのように隠蔽されている機能がある。

#### 5. `DataLoader` において、デフォルトでは `DistributedSampler` が設定される。

PyTorch における [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) の代わりに、 PyTorch Lightning では [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html) が用いられる。
PyTorch Lightning においてはデータ準備のプロセスを LightningDataModule に委ねることで、生産性と再利用性を高めることができる。その中で、サンプラーの仕様としては DistributedSampler がデフォルトとなっている。

#### 6. `LightningModule` は `torch.nn.Module` の一つであり、それに機能を追加したもの。

`LightningModule` は `torch.nn.Module` を継承したクラスであり、LightningModule が提供する各機能は、継承したクラスに対して追加されたメソッドである。


## サンプルコード

### ファイル

PyTorchLightningExampleOnMNIST.ipynb

### 免責

本リポジトリトップの README.md 参照

### 実行する場合のシステム要件

* Jupyter 実行環境がある
* Jupyter 実行環境に以下の Python パッケージがインストールされている
    - PyTorch Lightning
        - `pip install pytorch-lightning` または `conda install pytorch-lightning -c conda-forge`
    - torchvision
        - `pip install torchvision` または `conda install torchvision -c conda-forge`
