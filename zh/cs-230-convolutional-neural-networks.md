**Convolutional Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; 卷积神经网络简明指南

<br>

**2. CS 230 - Deep Learning**

&#10230; CS230 - 深度学习

<br>

**3. [Overview, Architecture structure]**

&#10230; [概述，框架结构]

<br>

**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [网络层的类型，卷积，池化，全连接]

<br>

**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [卷积核超参数，维度，步长，填充]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [调整超参数，参数兼容性，模型复杂度，感受野]

<br>

**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [激活函数，线性修正单元，Softmax]

<br>

**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [目标检测，模型类型，检测，交并比，非极大值抑制，YOLO， R-CNN]

<br>

**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [人脸验证/识别，孪生神经网络，三重损失]

<br>

**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [神经风格迁移，激活，风格矩阵，风格/内容代价函数]

<br>

**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [计算技巧体系结构，生成对抗网络，ResNet，Inception网络]

<br>


**12. Overview**

&#10230; 概述

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; 传统CNN架构 - 卷积神经网络，也称之为CNN，是一种特殊类型的神经网络，通常由下面的网络层构成：

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; 下一节中将介绍通过相应的超参数来调整卷积层和池化层。

<br>

**15. Types of layer**

&#10230; 网络层类别

<br>

**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; 卷积层（CONV）―卷积层（OCNV）利用卷积核在相应的维度上扫描输入 I ，来实现卷积操作。它的超参数包括卷积核的大小 F 和步长 S 。

<br>

**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; 备注：卷积操作也可以推广到 1D 和 3D 的情况。

<br>

**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230;  池化层（POOL） ― 池化层（POOL）是一个下采样操作，通常在卷积层后使用，可以得到一些空间的不变形。特别的，最大和平均池化是特殊的池化方式，分别取池内的最大值和平均值。

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [类型，目的，图例，注释]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [最大池化，平均池化，每个池化操作选择当前窗口中的最大值，每个池化操作计算当前窗口内的平均值]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [保留检测特征，最常用，特征图的下采样，在LeNet中使用]

<br>

**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; 全连接层（FC） ― 全连接层（FC）是在展开的输入上操作的，其中每个输入都连接到所有的神经元。如果有的话，全连接层通常都是在CNN网络架构的末尾处，可以用于优化目标，比如说班级成绩。

<br>


**23. Filter hyperparameters**

&#10230; 卷积核超参数

<br>

**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; 卷积层中包含着卷积核，所以对于这些卷积核，了解其超参数的含义是很重要的。

<br>

**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; 卷积核的维度 ― 一个大小为 F×F 的卷积核应用于包含 C 个通道的输入得到的卷积核体积为 F×F×C ，在输入大小为  I×I×C 上做卷积操作生成大小为 O×O×1 的特征图（也称为激活图）。

<br>


**26. Filter**

&#10230; 卷积核

<br>

**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; 备注：使用 K 个 F×F 大小的卷积核得到大小为 O×O×K 的特征图。

<br>

**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; 步长 **―** 对于卷积或者池化操作，步长 S 表示的是卷积窗口在每次操作完成后移动的像素数。

<br>

**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; 零填充 **―** 零填充表示在输入的边界增加 P 个 0 。这个值可以手动指定，也可以通过下面三种模式自动设置：

<br>

**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [模式，值，图例，目的，Valid，Same，Full]

<br>

**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [无填充，如果维度不匹配丢弃最后的卷积操作，填充使得特征图大小为⌈IS⌉，输出大小在数学计算上很方便，也称之为“半”填充，最大填充使得末端卷积应用于输入的限制，卷积核“看到”端到端的输入]

<br>

**32. Tuning hyperparameters**

&#10230; 调整超参数

<br>

**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; 卷积层中的参数兼容性 ― 用 I 来表示输入输入大小，F 表示卷积核的长度，P 表示零填充的大小，S 表示步长，那么沿着维度的输出特征图的大小 O 公式为：

<br>

**34. [Input, Filter, Output]**

&#10230; [输入，卷积核，输出]

<br>

**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; 备注：通常情况下， Pstart=Pend≜P，这个场景下就可以将上面公式中的 Pstart+Pend 替换为 2P 。

<br>

**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; 理解模型的复杂度 ― 为了确定一个模型的复杂度，确定模型框架中包含的参数量通常很有帮助。在卷积神经网络的给顶层中，其操作如下：

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [图例，输入大小，输出大小，参数量，备注]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [每个卷积核一个偏置参数，在大多数情况下，S < F，常见的 K 为 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [池化操作是按通道进行的，大多数情况下，S=F]

<br>

**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [输入层展开，每个神经元一个偏置参数，FC 层的神经元数量不受结构限制]

<br>

**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; 感受野 ― 第 K 层的感受野是输入层中第 k 个激活图能够看到的像素，该区域记为 Rk×Rk 。Fj 表示第 j 层的卷积核大小，Si 表示第 i 层的步长值，约定S0=1，第 k 层的感受野可以用一下的公式来计算：

<br>

**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; 在下面这个例子中，F1=F2=3 以及 S1=S2=1，得到 R2=1+2⋅1+2⋅1=5 。

<br>

**43. Commonly used activation functions**

&#10230; 常用的激活函数

<br>

**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; 线性修正单元 ― 线性修正单元（ReLU）是一个激活函数，在所有的输入所有元素上使用。它是为了将非线性引入到网络中。下表总结了其变体：

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU， Leaky ReLU， ELU，其中]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [生物学上可以解释的非线性复杂度，解决ReLU负值死亡的问题，全局可导]

<br>

**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax  ― 在框架末尾的softmax步骤可以看成是广义的逻辑函数，输入一个分数向量，通过softmax函数输出概率向量。它的定义如下：

<br>


**48. where**

&#10230; 其中

<br>

**49. Object detection**

&#10230; 目标检测

<br>

**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; 模型类别 ― 有 3 中主要的目标检测算法，因此预测的目标是不同的。下表中描述了它们：

<br>

**51. [Image classification, Classification w. localization, Detection]**

&#10230; [图像分类，分类和定位，检测]

<br>


**52. [Teddy bear, Book]**

&#10230; [泰迪熊，书]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [分类一张图片，预测物体的概率，在图片中检测一个物体，预测物体的概率和所处位置，在图片中检测多个物体，预测每个物体的概率和所处位置]

<br>

**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [传统CNN，简化的YOLO，R-CNN，YOLO，R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 检测  ― 在目标检测的任务中，根据我们是指向定位物体的位置或者是检测图像中更复杂的形状，采用不同的方法。下表中总结了两个主要的方法：

<br>

**56. [Bounding box detection, Landmark detection]**

&#10230; [边框检测，关键点检测]

<br>

**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [检测物体所在的图片区域，检测一个形状或者物体的特征（比如眼睛），更精细]

<br>

**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [边框中心 (bx，by)，高度 bh 和宽度 bw，参考点 (l1x,l1y), ..., (lnx,lny)]

<br>

**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; 交并比 ― 交并比，也称之为 IoU，是一个用于衡量预测框 Bp 与实际边框 Ba 相交正确率的函数。定义如下：

<br>

**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 备注：总有 IoU∈[0,1] 。通常情况下，如果预测框 Bp 的IoU(Bp,Ba)⩾0.5，表明该预测结果相当不错了。

<br>

**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; 锚框  ― 锚框是一种用于预测重叠边框的技术。在实际中，允许神经网络同时预测多个框，每个预测框被约束为具有给定的几何属性集。比如说，第一个预测可能是给定形式的矩形框，而第二个可能是另一个几何形状的矩形框。

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 非极大值抑制  ― 非极大值抑制是一种用于移除相同物体的重叠边界框，通过选择最有代表性的那个边框。删除小于 0.6 预测概率的边框之后，重复指向以下不走，同时保留边框：

<br>

**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [给定类型，第一步：选择预测概率最大的边框。，第二步：去掉与上一个边框的 IoU⩾0.5 的边框]

<br>

**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [边框预测，选择最大概率边框，去除同一类别的重复边框，最终边框]

<br>

**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO ― You Only Look Once（YOLO）是一个目标检测算法，执行步骤如下：

<br>

**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [第一步：将输入图像分割为 G×G 的网格。，第二步：对于每个网格，运行CNN网络预测下面给定的 y ：，重复 k 次]

<br>

**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; pc 是检测物体的概率，bx,by,bh,bw 是检测边框的属性，c1,...,cp 是检测到 p 类别的 one-shot表示。k 是锚框的数量。

<br>

**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; 第三步：运行非极大值抑制算法去除潜在的重叠边框。

<br>

**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [原始图像，分成 G×G 网格，边框预测，非极大值抑制]

<br>

**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 备注：当 pc = 0，神经网络就没有检测到任何物体。这个情况下，对应的预测结果 bx,...,cp必须被忽略。

<br>

**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN ― Region with Convolutional Neural Networks (R-CNN) 是一个目标检测算法，首先对图像进行分割以找到潜在的相关边界框，然后运行检测算法以在那些边界框中找到最可能的对象。

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [原始图像，分割，边框预测，非极大值抑制]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 备注：尽管原始算法计算资源消耗大且速度慢，但是较新的体系结构能够使该算法运行得更快，例如Fast R-CNN和Faster R-CNN。

<br>


**74. Face verification and recognition**

&#10230; 人脸验证和识别

<br>

**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; 模型类别 ― 下表中总结两个主要的模型类别：

<br>

**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [人脸验证，人脸识别，查询，验证，数据库]

<br>

**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [这是正确的人吗？，一对一查询，这是数据库中 K 个人的其中一个吗？，一对多查询]

<br>

**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; 单样本学习  ― 单样本学习是一种脸部验证算法，使用有限的训练数据集来学习一个相识度函书，能够衡量两张给定图片的差异。通常将两张图片的相识度函数记为 d(image 1,image 2)。

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; 孪生神经网络

<br>

**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230;  Triplet loss ― Triplet loss ℓ 是一个损失函数，根据图像三元组 A（锚点），P（正样本），N（负样本）的嵌入表示进行计算。锚点和正样本属于同一个类别，负样本属于另外一个类别。通过调用 α∈R+ 边距参数，相应的损失定义如下：

<br>

**81. Neural style transfer**

&#10230; 神经风格迁移

<br>

**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; 动机  ― 神经风格迁移的目标是基于给定的内容 C 和 风格 S，生成图片 G 。

<br>

**83. [Content C, Style S, Generated image G]**

&#10230; [内容 C， 风格 S ，生成图片 G]

<br>

**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; 激活函数  ― 在给定的网络层 I ，激活函数记为 a[l] ，它的维度为 nH×nw×nc。

<br>

**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; 内容代价函数 ― 内容代价函数 Jcontent(C,G) 用于确定生成图片 G 和原始图片的内容差异。它的定义如下： 

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; 风格矩阵 ― 指定网络层的风格矩阵 G[I] 是一个Gram矩阵，矩阵中的每个元素 G[I]kk' 衡量通道 k 和 k' 之间的相关程度。它是关于激活函数 a[I] 的定义如下：

<br>

**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 备注：风格图片的风格矩阵和生成图片分别记为 G[l] (S) 和 G[l] (G) 。

<br>

**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; 风格代价函数  ― 风格代价函数 Jstyle(S, G) 用于衡量生成图片 G 和风格 S 之间的差异。定义如下：

<br>

**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 总的代价函数 ― 总的代价函数定义为内容和风格代价函数之和，权重参数为 α,β，公式如下：

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 备注：较高的 α 值会使模型更加关注内容，而较高的 β 值会使模型更加关注风格。

<br>


**91. Architectures using computational tricks**

&#10230; 使用计算技巧的架构

<br>

**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; 生成对抗网络 ― 生成对抗网络，也称之为 GANs，是由一个生成模型和判别模型构成，生成模型的目标是生成最真实的结果，这个结果输入到判别模型中，区分生成的图片和真实的图片。

<br>

**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [训练，噪声，真实世界的图片，生成器，判别器，真 假]

<br>

**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 备注：使用 GANs 变体的用例包括文本到图像，音乐生成和合成。

<br>

**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet ― 残差网络架构（也称为 ResNet）使用具有大量网络层的残差单元来减小训练误差。残差单元具有以下特征方程式：

<br>

**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception 网络 ― 这个结构使用了 inception 模块，旨在尝试不同的卷积，以便通过功能多样化来提高其性能。

<br>

**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; 深度学习简明指南现已提供 [中文] 

<br>

**98. Original authors**

&#10230; 原文作者

<br>


**99. Translated by X, Y and Z**

&#10230; 由 X, Y 和 Z 翻译

<br>

**100. Reviewed by X, Y and Z**

&#10230; 由 X, Y 和 Z 校对

<br>

**101. View PDF version on GitHub**

&#10230; 在 Github 上查看 PDF 版本

<br>

**102. By X and Y**

&#10230; 由 X 和 Y 完成

<br>