# Fast.AI U-Net 图像语义分割初探

---

<!-- TOC -->

- [Fast.AI U-Net 图像语义分割初探](#fastai-u-net-%e5%9b%be%e5%83%8f%e8%af%ad%e4%b9%89%e5%88%86%e5%89%b2%e5%88%9d%e6%8e%a2)
  - [Why Fast.AI ?](#why-fastai)
  - [准备数据](#%e5%87%86%e5%a4%87%e6%95%b0%e6%8d%ae)
  - [Fast.AI 启动！](#fastai-%e5%90%af%e5%8a%a8)
    - [先看看薅到的资本主义羊毛](#%e5%85%88%e7%9c%8b%e7%9c%8b%e8%96%85%e5%88%b0%e7%9a%84%e8%b5%84%e6%9c%ac%e4%b8%bb%e4%b9%89%e7%be%8a%e6%af%9b)
    - [查看数据](#%e6%9f%a5%e7%9c%8b%e6%95%b0%e6%8d%ae)
    - [数据生成器](#%e6%95%b0%e6%8d%ae%e7%94%9f%e6%88%90%e5%99%a8)
    - [构建模型](#%e6%9e%84%e5%bb%ba%e6%a8%a1%e5%9e%8b)
    - [微调模型](#%e5%be%ae%e8%b0%83%e6%a8%a1%e5%9e%8b)
  - [参考及引用](#%e5%8f%82%e8%80%83%e5%8f%8a%e5%bc%95%e7%94%a8)

<!-- /TOC -->

---

## Why Fast.AI ?

众所周知 Keras 因其简洁优美的 API 著称，特别是其函数式 API 的引入，更是将灵活性提升到了一个新的层次。Keras 可以在几行内定义一个简单的 CNN ，一两行行便可引入一个预训练模型。以下是 Keras 中一个基础的 U-Net（下文统称为 Unet ）的实现：

![keras_unet](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/keras_unet.png?raw=true)

以上代码便简洁的定义了一个 Unet 。众所周知 Unet 在各类图像语义分割中占有重要地位，本文也将围绕 Unet 展开，若你还不知道 Unet 或者语义分割是什么的话，我认为这篇文章能帮助你理解它们：[深度学习笔记 | 第 8 讲：CNN 图像分割发家史之从 FCN 到 u-net](https://zhuanlan.zhihu.com/p/46214424)

好，现在该回归正题了——[Why Fast.AI ?](#why-fastai) 因为 Keras 已经很简洁了，不曾想 [Fast.AI](https://www.fast.ai/) （下文简称 fastai）比 Keras 还要简洁得多，我们同样以 Unet 为例：

![fastai_unet_step_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_unet_step_1.png?raw=true)

抛开构造输入数据的代码不谈，仅用

```python
learn = unet_learner(data, models.resnet18, metrics=dice, wd=1e-2)
```

一行便构造了一个使用预训练的 Resnet-18 做特征提取器的 Unet ，Keras 则相形见绌。这也便是我告别 Keras 转投 fastai 阵营的一大原因。另一原因便是 fastai 自带的工具十分方便，例如 `learn.lr_find()` 就可以很方便的查找最优学习率，`learn.recorder.plot_lr()` 便可以绘制学习率变化的曲线等等。 fastai 内置了十分丰富的模块，是一个功能齐全的武器库，很多时候自己需要做的事就只是调用相关的 API 。这也是 fastai 的一大特点，fastai 能够很方便的使用余弦退火、SGDR 等 Tricks ，而 Keras 想要实现这些就得写长长的 Callback 函数了。所以，为啥不用 fastai 呢？

---

## 准备数据

我打算用一个图像语义分割的数据集来看看 fastai 到底有多少能耐。

- 下载数据

  先下载数据集，这个数据集是 Kaggle 上的一个公开数据集 [Brain MRI segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)，也是上个月 [FlyAI](https://www.flyai.com/) 平台上 [脑部 MRI（磁共振）图像分割](https://www.flyai.com/d/MRISegmentation) 这个比赛所使用的数据集。至于为什么下载慢大家都心照不宣，我也就不多说了 (手动:dog:)

- 整理数据

      下载下来之后，解压出来的文件结构是这样的：

  ![mri_datasets_tree](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/mri_datasets_tree.png?raw=true)

  `lgg-mri-segmentation/kaggle_3m` 目录下每个文件夹都是单独的病案，以 `TCGA_CS_4941_19960909` 文件夹为例，里边都由 MRI （核磁共振影像）及其所对应的脑肿瘤标签所组成。如下所示：

  ![MRI_img](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/MRI_img.png?raw=true)

  要写程序读取这样文件结构的数据确实略显繁杂，故我需要对数据稍加整理。我所希望得到的是这样的一个文件结构：

  ![MRI_Dataset_tree](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/MRI_Dataset_tree.png?raw=true)

  所有的 MRI 数据在 `image` 文件夹下，所有的标签文件在 `label` 文件夹下。为此，我需要先创建一个 `MRI_Dataset` 文件夹，在里边再新建 `image` 和 `label` 两个文件夹，再将对应数据复制进去。作为猿猴我当然不会去手动复制啦，这就得靠一个简单的 Python 脚本来实现批量复制了。脚本程序如下：

  ![copy_py_script](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/copy_py_script.png?raw=true)

  在我的小笔记本上执行该脚本，复制那 7000 多个文件也就三分多钟而已，还是很快的（可比手动快多了）。

- 上传至 Google Drive（非必须）

  由于我的小笔记本显存太小，跑 Unet 很容易 OOM（Out of Memory）故选择在 Google Colab 或 Kaggle Kernel 上跑。如果你还不知道 Colab 这个神器的话请看这儿 [薅资本主义羊毛，用 Google 免费 GPU](https://zhuanlan.zhihu.com/p/33344222) 。当然啦，要使用 Colab 还是需要梯子的，如果你本地 GPU 显存够大或者有其他 GPU 资源，那也就没必要用 Colab 了，跳过该步吧。

  1.  首先将数据上传至你自己的 [Google Drive](https://drive.google.com/) 上（怎么上传我就不演示了:neutral_face:）。再新建一个 Colab 文件，看看可爱的猫猫，再点击 `装载 GOOGLE 云端硬盘`

      ![colab_step_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/colab_step_1.png?raw=true)

  2.  之后会弹出该提示，按提示操作即可

      ![colab_step_2](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/colab_step_2.png?raw=true?raw=true)

  3.  点击 `Go to this URL in a browser` 那个网址，将会弹出一个网页，登陆你的 Google 帐号验证后会得到一长串字符序列，复制之后在图中白矩形框出粘贴并回车，随即装载 GOOGLE 云端硬盘成功。

      ![colab_step_3](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/colab_step_3.png?raw=true?raw=true)

      成功后将会看到如下结果：

      ![colab_step_3_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/colab_step_3_1.png?raw=true)

---

## Fast.AI 启动！

自此，数据集的准备工作便已经完成了，接下来便是 Show Time ！

### 先看看薅到的资本主义羊毛

如果是使用 Colab 或 Kaggle Kernel 并启用 GPU 后端的话，执行以下指令

```python
!nvidia-smi
```

> - 在 Colab 执行上述指令后便可看到当前云服务器的 GPU 信息，不出意外的话都能连接到 [Tesla T4](https://www.nvidia.cn/data-center/tesla-t4/) GPU ，该卡有 16G 显存（也有可能是 Tesla K80）。Tesla T4 对混合精度运算有特殊加成，我接下来也将使用混合精度运算。
>
>   ![fastai_step_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_nvsmi.png?raw=true)
>
> - 在 Kaggle Kernel 上执行的话，有一定可能申请到 [Tesla P100](https://www.nvidia.cn/data-center/tesla-p100/) ，这块卡的性能可比 Tesla T4 好太多了
>
>   ![kaggle_kernel](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/kaggle_kernel.png?raw=true)

### 查看数据

1.  首先要导入必要的库（注：并不是所有导入的模块后文都会用到，最重要的是最后三句），如下所示：

    ![fastai_step_import](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_import.png?raw=true)

2.  定义 path 目录，path 为先前整理出来的 MRI_Dataset 数据集所在的目录，在此使用 pathlib.Path 模块以方便后续连接子目录的操作。 `get_image_files` 读取 `image` 和 `label` 下的 `.tif` 文件目录，用 Python 中的除号 “/” 连接目录字符串即可（例 `path / 'image'`），如下图所示：
    ![fastai_step_path](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_path.png?raw=true)

3.  查看 MRI 数据，输出第 1100 张 MRI 的路径并显示它的图像：

    ![fastai_step_imgshow](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_imgshow.png?raw=true)

4.  获取上图所对应的标签文件的路径。构造一个函数，实现输入 MRI 的路径，返回其对应的标签文件路径。

    - /MRI_Dataset/image/TCGA_DU_A5TR_19970726_21.tif
    - /MRI_Dataset/label/TCGA_DU_A5TR_19970726_21_mask.tif

    注意到标签的文件名与其对应的 MRI 文件名，就只是在扩展名之前多了个 `_mask` 而已。还有一点路径上的区别就是 `image` 和 `label` ，只需改动两处即可。该函数实现如下：

    ![fastai_step_func](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_func.png?raw=true)

    可见，该函数输出了正确的 MRI 所对应标签的路径。其中 `x.stem` 是获取 `x` 的文件名，`x.subfix` 是获取后缀名，在二者之间加上 `_mask` ，再在文件名前加上标签文件的目录即可。

### 数据生成器

1.  `SegmentationItemList` 是 `fastai.vision` 中自带的一个构造图像语义分割数据生成器的方法。 我们改写 `SegmentationItemList` 读取标签的方法 `open` ，通过向其传入 `div=True` 参数使其读取标签文件后可以生成两个通道的标签文件，第一个通道无肿瘤部分的标签，第二个通道为有肿瘤部分的标签。不改动该处的话，在有的版本的 fastai 上就会出现莫名其妙的标签数组越界问题。我们构造一个新类，继承自 `SegmentationLabelList` ，重写其中的 `open` 函数。具体实现代码如下：


    ![fastai_step_datalist](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_datalist.png?raw=true)

2.  接下来就可以使用我们上图中的 `MySegmentationItemList` 这个类来导入数据并创建数据生成器了，用于训练时候自动生成数据。

    ![fastai_step_myitemlist](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_myitemlist.png?raw=true)

下面为 Jupyter 中的代码：

![fastai_step_dataItem](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_dataItem.png?raw=true)

3.  显示一些数据看看我们的生成器是否创建正确。fastai 中内置了很方便的方法，如下所示：

    ![fastai_step_showbatch](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_showbatch.png?raw=true)

    显然，我们的数据并没有任何问题，标签也正确地显示了。

### 构建模型

> _注：下述代码均为 Kaggle Kernel 上运行所得的结果_

1.  首先，我们通过一行代码来创建一个用 Resnet-34 做特征提取器的 Unet ，具体如下：

    ![fastai_step_create_unet](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_create_unet.png?raw=true)

    fastai 将会自动下载预训练 Resnet-34 并作为 Unet 的特征提取器。在此模型中我还使用了 `aelf_attention=True` 这将启用[自注意力机制](https://www.zhihu.com/topic/20682987/hot)。`metrics` 我指定为了 [`dice`](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)，这是医学影像分割中常用的一种评价指标。`wd=1e-2` 是指模型的权重衰减系数为 0.01。

2.  搜索最优学习率。执行下述指令即可搜索当前网络的学习率，并输出 fastai 建议的学习率：

    ![fastai_step_find_lr](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_find_lr.png?raw=true)

    从上图中可以知道建议的学习率为 5.25e-0.6，这是整个 lr-loss 曲线负斜率最大的点 ， [5.25e-0.6, 1e-03] 是一个不错的学习率范围。

3.  开始训练。

    ![fastai_step_train_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_train_1.png?raw=true)

    传入的回调函数是为了在每轮验证集 `loss` 下降时，保存我们的模型。由上图可见，经过五轮训练，dice 系数以提升至 `0.857` ，后续我还将进行微调，使网络性能再提高一些。

4.  查看训练期间 loss 、metric、学习率等随训练批次增加的变化。

    - `loss`

      ![fastai_step_loss_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_loss_1.png?raw=true)

    - `dice` 系数

      ![fastai_step_dice_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_dice_1.png?raw=true)

    - 学习率

      ![fastai_step_lr_1](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_lr_1.png?raw=true)

### 微调模型

> 注：下属操作大体同上，就不赘述了

1.  解冻模型，并将模型转为混合精度模式：

    ![fastai_step_unfreeze](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_unfreeze.png?raw=true)

2.  搜索最佳学习率

    ![fastai_step_find_lr_2](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_find_lr_2.png?raw=true)

3.  继续训练模型

    ![fastai_step_train_2](https://github.com/NagiSenbon/Fast.AI-Learning/blob/master/Pic/Chapter-1/fastai_step_train_2.png?raw=true)

    可见经过微调，`dice` 系数由原来的 `0.857` 提升到了 `0.579` 。如果更细致的调整模型的各个超参的话，我相信模型精度还能进一步提高。

---

## 参考及引用

> [FastAI Image Segmentation](https://towardsdatascience.com/fastai-image-segmentation-eacad8543f6f) > [Fast.ai Lesson 3 notes — Part 1 v3](https://medium.com/@lankinen/fast-ai-lesson-3-notes-part-1-v3-78d47bd11748) > [Semantic Segmentation on Aerial Images using fastai](https://medium.com/swlh/semantic-segmentation-on-aerial-images-using-fastai-a2696e4db127)
