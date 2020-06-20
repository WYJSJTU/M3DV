# M3DV
EE228 ML Assignment
## 代码结构
* `mylib/train.py`为训练程序，包括数据加载，训练，模型保存，
* `mylib/test.py`为测试
* `mylib/models/densenet.py`为DenseNet 3D模型
* `mylib/dataloader/dataset.py`为数据集处理程序
* `mylib/dataloader/data_utilis.py`为数据集增强操作
* `mylib/checkpoint`存放模型权重
* `mylib/runs`存放训练loss,auc记录
## 模型下载
* 链接：https://pan.baidu.com/s/1vDt6PrhOjkwps1O7BOpMQg 
* 提取码：rqb0
* 下载后放入`mylib/checkpoint`
## 测试运行
* 在`mylib/test.py`中修改`model_path1`为模型的绝对路径，将测试集放入命名为`test`的文件夹，修改`test_path`为包含此文件夹的目录的绝对路径，之后运行即可。
