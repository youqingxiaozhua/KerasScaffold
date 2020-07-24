#Keras深度学习脚手架
本项目是在完成深度学习作业时，因实验较多，所以代码逐渐积累形成。当时解决的问题主要是对于分类问题可以方便的切换数据集和模型，并
清晰记录实验结果。

目前实现的feature：

- data loader和model模块化
- 通用的分类数据集处理工具，一键生成并读取分类数据集
- 自动从最近的checkpoint恢复训练或直接读取网络参数文件


## data loader
代码的`dataset`文件夹下存放各个数据集，每个数据集下需在`data_loader.py`中实现`DataLoader`类的`get`方法，具体可参考
`dataset/mnist/data_loader.py`

### 分类任务通用data loader
针对分类任务，有封装好`Process`和`ClassifyDataset`两个类分别用于生成分类数据集和读取分类数据，具体可参考
`dataset/cat_dog/data_loader.py`

首先将样本按照类别放在不同的文件夹中，`Process`会读取每个文件夹并记录对应标签，按照划分比例划分为训练集、验证集和测试集。
如果运行正确，会在目标文件夹下生成一个`JPEGImages`文件夹存放所有图片，和四个csv文件，分别对应所有样本和划分的三个集合。

`ClassifyDataset`则会将上面生成的图片和csv文件封装成`tf.data.Dataset`，所以只需实现自己的数据增强部分即可。


## model
所有的model需要在`models/__init__.py`中导出

**注意：建议在实现时使用`__all__`来控制导出，如：```__all__ = ('ResNet50', 'ResNet50V2')```**


## 实验记录
每一次执行都需要指定实验名称，每次实验的log都保存在`dataset/${dataset}/exp/${exp_name}`下，每个log目录又分为如下结构：
```
exp/
├── ckpt    // 保存的checkpoint文件，当resume=ckpt的时候会自动从最近一次的ckpt恢复训练
├── log.txt // 训练日志
└── tb-logs // TensorBoard日志
```


## 运行试验
实验所需的全部参数在`utils/flags.py`中定义，快速开始可以参考`scripts`文件夹中的bash文件，例如：
```shell script
export CUDA_VISIBLE_DEVICES="0"
python3 main.py \
  --batch_size=128 \
  --exp_name="test" \
  --dataset="mnist" \
  --model="lenet" \
  --mode="train_test" \
  --early_stopping_patience=10 \
  --epoch=20 \
  --debug
```


