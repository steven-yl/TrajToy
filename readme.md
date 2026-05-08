## trainflow框架
conf/
    config.yaml
    training/toy.yaml
    trainflow_trainer/default.yaml
    trainflow_model/toy.yaml
    trainflow_data/toy.yaml

trainner是训练和评估模型的框架，主要用于训练和评估轨迹预测模型。
    - logger是日志，主要用于记录训练和评估过程中的日志。
    - callback是回调，主要用于在训练和评估过程中执行一些操作。
model是模型，主要用于定义模型结构。
    - loss
    - metrics
    - optimizer
    - scheduler
data是数据模块datamodule，主要用于定义数据集结构。
    - dataset是数据集，主要用于定义数据集结构。


## il框架
train.py
conf/
    config.yaml
    training/toy.yaml
    il_trainer/default.yaml
    il_model/toy.yaml
    il_data/toy.yaml

loss是损失函数，主要用于定义损失函数。
metrics是指标，主要用于定义指标。
dataset是数据集，主要用于定义数据集结构。

module是模块，主要用于定义模块结构。
    - model是模型，主要用于定义模型结构。

model是trainablemodel，主要用于定义可训练模型。

