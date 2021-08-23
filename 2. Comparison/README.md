# Comparison / Learning PyTorch Lightning

ここでは PyTorch Lightning と比較するために、Python用の各種 Deep Learning ライブラリの実装例を示します。

## 比較対象

* PyTorch
* Keras
* Chainer


## 選定理由

### PyTorch

PyTorch Lightning と直接的に比較されるべき対象であるため。

### Keras

Deep Learning 用コードの省力化というコンセプトにおいて PyTorch Lightning と共通するため。

### Chainer

おまけ。既に Chainer はメンテナンス状態であるが、移行先として推奨されている PyTorch と比較した上で、 Chainer から PyTorch Lightning への移行のヒントとすべく掲載した。


## 見どころ

### PyTorch

* `LightningModule` と `torch.nn.Module` の共通点と相違点
* 各Module 外での実装
* ループを回す処理の記述

### Keras

* Keras での一般的な実装スタイル
* Keras が実現するシンプルさ vs. PyTorch Lightning における構造化

### Chainer

* PyTorch との共通点と相違点
