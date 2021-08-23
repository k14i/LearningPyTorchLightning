# Installation / Learning PyTorch Lightning

## Pip

```
pip install pytorch-lightning
```

## Conda

```
conda install pytorch-lightning -c conda-forge
```


## 実行環境ごとの留意点

### Apple Silicon (M1) Mac

Python のディストリビューションとして Anaconda を利用する場合、 miniforge を利用すること。 (2021/08時点)

### Azure Machine Learning Compute Instance

horovod を再インストールすること。（[参考](https://k14i.github.io/2021/08/19/Resolve-ImportError-to-use-PyTorch-Lightning-on-Azure-Machine-Learning-Compute-Instance/)）
