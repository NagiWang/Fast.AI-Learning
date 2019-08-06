# Fast.AI U-Net 图像语义分割初探

[TOC]

------

## Why Fast.AI ?

众所周知 Keras 因其简洁优美的 API 著称，特别是其函数式 API 的引入，更是将灵活性提升到了一个新的层次。Keras 可以在几行内定义一个简单的 CNN ，一两行行便可引入一个预训练模型。以下是 Keras 中一个基础的 U-Net（下文统称为 Unet ）的实现：

![](Pic\Chapter-1\keras_unet.png)

以上代码便简洁的定义了一个 Unet 。众所周知 Unet 在各类图像语义分割中占有重要地位，本文也将围绕 Unet 展开，若你还不知道 Unet 或者语义分割是什么的话，我认为这篇文章能帮助你理解它们：[深度学习笔记 | 第 8 讲：CNN 图像分割发家史之从 FCN 到 u-net](https://zhuanlan.zhihu.com/p/46214424)

好，现在该回归正题了——[Why Fast.AI ?](#why-fastai)  因为 Keras 已经很简洁了，不曾想  [Fast.AI](https://www.fast.ai/) （下文简称 fastai）比Keras 还要简洁得多，我们同样以 Unet 为例：

![fastai_unet_step_1](Pic/Chapter-1/fastai_unet_step_1.png)

抛开构造输入数据的代码不谈，仅用

```python
learn = unet_learner(data, models.resnet18, metrics=dice, wd=1e-2)
```

一行便构造了一个使用预训练的 Resnet-18 做特征提取器的 Unet ，Keras 则相形见绌。这也便是我告别 Keras 转投 fastai 阵营的一大原因。另一原因便是 fastai 自带的工具十分方便，例如 `learn.lr_find()` 就可以很方便的查找最优学习率，`learn.recorder.plot_lr()` 便可以绘制学习率变化的曲线等等。 fastai 内置了十分丰富的模块，是一个功能齐全的武器库，很多时候自己需要做的事就只是调用相关的 API 。这也是 fastai 的一大特点，fastai 能够很方便的使用余弦退火、SGDR 等 Tricks ，而 Keras 想要实现这些就得写长长的 Callback 函数了。所以，为啥不用 fastai 呢？

------

## 准备数据

我打算用一个图像语义分割的数据集来看看 fastai 到底有多少能耐。
- 下载数据

    先下载数据集，这个数据集是 Kaggle 上的一个公开数据集 [Brain MRI segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)，也是上个月 [FlyAI](https://www.flyai.com/) 平台上 [脑部MRI（磁共振）图像分割](https://www.flyai.com/d/MRISegmentation) 这个比赛所使用的数据集。至于为什么下载慢大家都心照不宣，我也就不多说了 (手动:dog:)
    
- 整理数据

    下载下来之后，解压出来的文件结构是这样的：

    ![mri_datasets_tree](Pic/Chapter-1/MRI_Dataset_tree.png)

    `lgg-mri-segmentation/kaggle_3m` 目录下每个文件夹都是单独的病案，以 `TCGA_CS_4941_19960909` 文件夹为例，里边都由 MRI （核磁共振影像）及其所对应的脑肿瘤标签所组成。如下所示：

    ![1564846924413](Pic/Chapter-1/MRI_img.png)

    要写程序读取这样文件结构的数据确实略显繁杂，故我需要对数据稍加整理。我所希望得到的是这样的一个文件结构：

    ![MRI_Dataset_tree](Pic/Chapter-1/MRI_Dataset_tree.png)

    所有的 MRI 数据在 `image` 文件夹下，所有的标签文件在 `label` 文件夹下。为此，我需要先创建一个 `MRI_Dataset` 文件夹，在里边再新建 `image` 和 `label` 两个文件夹，再将对应数据复制进去。作为猿猴我当然不会去手动复制啦，这就得靠一个简单的 Python 脚本来实现批量复制了。脚本程序如下：
    
    ![copy_py_script](Pic/Chapter-1/copy_py_script.png)
    
    在我的小笔记本上执行该脚本，复制那7000多个文件也就三分多钟而已，还是很快的（可比手动快多了）。
    
- 上传至 Google Drive（非必须）

    由于我的小笔记本显存太小，跑 Unet 很容易 OOM（Out of Memory）故选择在 Google Colab 上跑。如果你还不知道 Colab 这个神器的话请看这儿 [薅资本主义羊毛，用Google免费GPU](https://zhuanlan.zhihu.com/p/33344222) 。当然啦，要使用 Colab 还是需要梯子的，如果你本地 GPU 显存够大或者有其他 GPU 资源，那也就没必要用 Colab 了，跳过该步吧。

    1.  首先将数据上传至你自己的 [Google Drive](https://drive.google.com/) 上（怎么上传我就不演示了:neutral_face:）。再新建一个 Colab 文件，看看可爱的猫猫，再点击 `装载 GOOGLE 云端硬盘` 

        ![colab_step_1](Pic/Chapter-1/colab_step_1.png)

    2.  之后会弹出该提示，按提示操作即可

        ![colab_step_2](Pic/Chapter-1/colab_step_2.png)

    3.  点击 `Go to this URL in a browser` 那个网址，将会弹出一个网页，登陆你的 Google 帐号验证后会得到一长串字符序列，复制之后在图中白矩形框出粘贴并回车，随即装载 GOOGLE 云端硬盘成功。

        ![colab_step_3](Pic/Chapter-1/colab_step_3.png)

        成功后将会看到如下结果：

        ![colab_step_3_1](Pic/Chapter-1/colab_step_3_1.png)

------



## Fast.AI 启动！

自此，数据集的准备工作便已经完成了，接下来便是 Show Time ！

### 先看看薅到的资本主义羊毛

如果是使用 Colab 并启用 GPU 后端的话，执行以下指令

```python
!nvidia-smi
```

![fastai_step_1](Pic/Chapter-1/fastai_step_nvsmi.png)

执行上述指令后便可看到当前云服务器的 GPU 信息，不出意外的话都能连接到 [Tesla T4](https://www.nvidia.cn/data-center/tesla-t4/) GPU ，该卡有 16G 显存。Tesla T4 对混合精度运算有特殊加成，我接下来也将使用混合精度运算。



### 查看数据

1.  首先要导入必要的库（注：并不是所有导入的模块后文都会用到，最重要的是最后三句），如下所示：

    ![fastai_step_import](Pic/Chapter-1/fastai_step_import.png)

2.  定义 path 目录，path 为先前整理出来的 MRI_Dataset 数据集所在的目录，在此使用 pathlib.Path 模块以方便后续连接子目录的操作。 `get_image_files` 读取 `image` 和 `label` 下的 `.tif` 文件目录，用 Python 中的除号  “/”  连接目录字符串即可（例  `path / 'image' `），如下图所示：
    ![fastai_step_path](Pic/Chapter-1/fastai_step_path.png)

3.  查看 MRI 数据，输出第 1100 张 MRI 的路径并显示它的图像：

    ![fastai_step_imgshow](Pic/Chapter-1/fastai_step_imgshow.png)

4.  获取上图所对应的标签文件的路径。构造一个函数，实现输入 MRI 的路径，返回其对应的标签文件路径。

    -   /MRI_Dataset/image/TCGA_DU_A5TR_19970726_21.tif
    -   /MRI_Dataset/label/TCGA_DU_A5TR_19970726_21_mask.tif

    注意到标签的文件名与其对应的 MRI 文件名，就只是在扩展名之前多了个 `_mask` 而已。还有一点路径上的区别就是 `image` 和 `label` ，只需改动两处即可。该函数实现如下：

    ![fastai_step_func](Pic/Chapter-1/fastai_step_func.png)

    可见，该函数输出了正确的 MRI 所对应标签的路径。其中 `x.stem` 是获取 `x` 的文件名，`x.subfix` 是获取后缀名，在二者之间加上 `_mask` ，再在文件名前加上标签文件的目录即可。

### 数据生成器

1.   `SegmentationItemList` 是 `fastai.vision` 中自带的一个构造图像语义分割数据生成器的方法。 我们改写 `SegmentationItemList` 读取标签的方法 `open_mask` ，通过向其传入 `div=True` 参数使其读取标签文件后可以生成两个通道的标签文件，第一个通道无肿瘤部分的标签，第二个通道为有肿瘤部分的标签。不改动该处的话，在有的版本的 fastai 上就会出现莫名其妙的标签数组越界问题。我们构造一个新类，继承自 `SegmentationLabelList` ，重写其中的 `open_mask` 函数。具体实现代码如下：

    ![fastai_step_datalist](Pic/Chapter-1/fastai_step_datalist.png)

2.  接下来就可以使用我们上图中的 `MySegmentationItemList` 这个类来导入数据并创建数据生成器了，用于训练时候自动生成数据。

    ![fastai_step_myitemlist](Pic/Chapter-1/fastai_step_myitemlist.png)

下面为 Jupyter 中的代码：

![fastai_step_dataItem](Pic/Chapter-1/fastai_step_dataItem.png)

3.  显示一些数据看看我们的生成器是否创建正确。fastai 中内置了很方便的方法，如下所示：

    

    显然，我们的数据并没有