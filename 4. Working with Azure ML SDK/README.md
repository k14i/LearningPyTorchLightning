# Working with Azure ML SDK

[Azure Machine Learning SDK for Python](https://docs.microsoft.com/ja-jp/python/api/overview/azure/ml/?view=azure-ml-py) を使用して [Azure Machine Learning service](https://docs.microsoft.com/ja-jp/azure/machine-learning/overview-what-is-azure-machine-learning) の [Compute Instance](https://docs.microsoft.com/ja-jp/azure/machine-learning/concept-compute-instance) 上で PyTorch Lightning を利用したコードを実行する例を示します。


## ファイル構成

### WorkingWithAzureMLSDK.ipynb

実行用スクリプトを Jupyter Notebook 形式で実装したものです。

### scripts/*.py

Compute Instance にデプロイする PyTorch Lightning を利用したコードです。


## 環境構築と実行

### Azure Machine Learning service

1. Azure Machine Learning リソースを作成し、 Compute Instance を作成してください。
2. Compute Instance にシェルでログインし、[こちら](https://k14i.github.io/2021/08/19/Resolve-ImportError-to-use-PyTorch-Lightning-on-Azure-Machine-Learning-Compute-Instance/)の手順に沿って horovod を再インストールしてください。
3. AzureのサブスクリプションID，Azure Machine Learning のリソースグループ名，同じくワークスペース名をコピペできる状態にしておきます。

### ローカル

1. [公式](https://docs.microsoft.com/ja-jp/python/api/overview/azure/ml/install?view=azure-ml-py)の手順に沿って Azure ML SDK をインストールしてください。
2. WorkingWithAzureMLSDK.ipynb を開き、2番目のセル内にサブスクリプションID，リソースグループ名，ワークスペース名を記入（上書き）してください。
    - 認証情報ですので、記入済みのファイルが第三者の手に渡らないよう、取り扱いに注意してください。
3. `run` までセルを実行します。
