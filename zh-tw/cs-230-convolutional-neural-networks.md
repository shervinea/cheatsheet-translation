**Convolutional Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; 卷積類神經網路
<br>


**2. CS 230 - Deep Learning**

&#10230; CS290 - 深度學習

<br>


**3. [Overview, Architecture structure]**

&#10230; [概論, 架構結構]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [層的種類, 卷積, 池化, 全連接]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [卷積核超參數, 維度, 滑動間隔, 填充]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [調整超參數, 參數相容性, 模型複雜度, 接受區]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [激活函數, 線性整流函數, 歸一化指數函數]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [物體偵測, 模型種類, 偵測, 交併比, 非最大值抑制, YOLO, 區域卷積類神經網路]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [人臉驗證/辨別, 單樣本學習, 孿生網路, 三重損失函數]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [神經風格轉換, 激發, 風格矩陣/內容矩陣, 風格/內容成本函數]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [計算架構手法, 生成對抗網路, 殘差網路, inception網路]

<br>


**12. Overview**

&#10230; 概論

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; 傳統卷積類神經網路架構 - 卷積類神經網路, 簡稱為CNNs, 是一種類神經網路的變形，通常由下列的層組成：

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; 卷積層和池化層可利用超參數來優化，詳細內容由下個部分敘述。

<br>


**15. Types of layer**

&#10230; 層的種類

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; 卷積層 (CONV) - 卷積層利用卷積核沿著輸入數據的維度進行掃描。其超參數包含卷積核的尺寸F和滑動間隔S。輸出O稱為特徵圖或激發圖。

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; 備註：卷積之運算亦可推廣為一維或三維。

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; 池化層 (POOL) - 池化層用於降低取樣頻率，通常用於卷積層之後以處理空間變異性。其中，最大池化與平均池化，分別選取池中之最大值與平均值，為特別的池化種類。

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [種類, 目的, 圖示, 註解]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [最大池化層, 平均池化層, 每個池化計算該池中之最大值, 每個池化計算該池中平均值]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [保留偵測到之特徵, 最常使用, 降低特徵圖之採樣頻率,於LeNet中使用]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; 全連接層 (FC) - 全連接層之運作需要扁平的輸入，其中，所有的輸入數值與所有的神經元是全連接的。

<br>


**23. Filter hyperparameters**

&#10230; 卷積核的超參數

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; 卷積層有卷積核，而了解其中超參數的意義是重要的。

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; 卷積核的維度 - 一個尺寸為F×F的卷積核，套用在有C個頻道的輸入，是一個維度為F×F×C的體，計算卷積於輸入維度為I×I×C，輸出一個維度為O×O×1的特徵圖。

<br>


**26. Filter**

&#10230; 卷積核

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; 備註：應用K個維度為F×F的卷積核會得到維度為O×O×K的特徵圖。

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; 滑動間隔 - 對卷積或池化的運算，滑動間隔S表示每次運算結束後，視窗移動的像素數量。

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; 零填充 - 零填充表示將P個0填充於輸入數據的邊緣。此數值可手動指定，或是透過三種自動的模式。

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [模式, 數值, 圖示, 用途, Valid, Same, Full]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [無填充, 維度不相符則捨棄最後一個卷積, 填充使得特徵圖的維度為⌈IS⌉, 輸出維度是數學上方便的, 又稱為半填充, 最大的填充使終端的卷積運作於輸入之限度, 卷積核可端到端的「看到」整個輸入]

<br>


**32. Tuning hyperparameters**

&#10230; 優化超參數

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; 卷積層中的參數相容性 - 輸入數據維度I，卷積核維度F，零填充維度P，滑動間隔S，則輸出的特徵圖維度為O。

<br>


**34. [Input, Filter, Output]**

&#10230; [輸入, 卷積核, 輸出]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; 備註：時常Pstart=Pend≜P，則我們於上式中將Pstart+Pend以取2P代為。

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; 了解模型複雜度 - 為了了解模型的複雜度，我們時常計算模型中含有的參數量。給定一積類神經網路，定義為：

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [圖示, 輸入維度, 輸出維度, 參數數量, 備註]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [一個卷積核一個偏移值, 大部分來說 S<F, K常見的選擇為2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [池化運算以頻道為單位, 大部分來說S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [輸入需扁平化, 一個神經元一個偏差值, 全連接層中的神經元數量沒有結構限制]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; 接受區 - 在第k層的接受區表示為Rk×Rk，是輸入數據中，可被第k個激發圖所看見的像素。設Fj為第j層中卷積核的尺寸，Si為第i層的滑動間隔，通常為1；在第k層的接受區之運算為以下公式：

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; 以下範例中，F1=F2=3， S1=S2=1，因此R2=1+2⋅1+2⋅1=5。

<br>


**43. Commonly used activation functions**

&#10230; 常用的激發函數。

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; 線性整流函數 - 線性整流函數(ReLU)是一激發函數，可應用於所有體中的元素。用於增加非線性的性質到網路中。線性整流函數的變形如下：

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; 線性整流函數, 洩漏線性整流器,指數性線性函數, 其中

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [非線性複雜度生物可解釋性, 處理線性整流函數抑制負數問題, 全區間可微分]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; 歸一化指數函數 - 歸一化指數函數可被視為一廣義的邏輯函數，將一個分數的陣列x∈Rn輸出為一個機率的陣列p∈Rn，用於網路架構的終端。定義為：

<br>


**48. where**

&#10230; 其中

<br>


**49. Object detection**

&#10230; 物體偵測

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; 模型種類 - 有三種主要的物體辨別演算法，差別在於預測的目的不同。敘述於以下表格：

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [影像分類, 影像分類定位, 偵測]

<br>


**52. [Teddy bear, Book]**

&#10230; [泰迪熊, 書]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [分類一張圖, 預測可能為一物件的機率, 偵測一張圖中的物件, 預測可能為一物件的機率與物件的位置, 偵測一張圖中的數個物件, 預測可能為一物件的機率與物件的位置]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [傳統的卷積類神經網路. 簡化版YOLO, 區域卷積類神經網路, YOLO, 區域卷積類神經網路]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 偵測 - 於物件之中，選擇不同方法取決於是否想要定位物體的位置，或是偵測更複雜的形狀。兩個主要的介紹如下表：

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [定界框偵測, 特徵點偵測]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [偵測影像中有包含物件的部分, 偵測一物件之形狀或特性(如：眼睛), 更精準]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; 框的中心(bx,by), 高bh與寬bw, 參考點 (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; 交併比 - 交併比，簡稱為IoU，是一個用於評估定界框Bp預測位置與實際位置Ba比較正確性之函數。定義如下：

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 備註：交併比介於0到1之間。一般來說，一個好的定界框該有IoU(Bp,Ba)⩾0.5。

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; 錨框 - 錨框是一個用於預測重疊定界框的技術。實務上，網路可以同時預測多個定界框，而每個定界框有限制的幾何性質。例如：第一個預測定界框可能是一個正方形，而第二個可能是另一個有不同幾何性質的正方形。

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 非最大值抑制 - 非最大值抑制是一個用於移除重複、重疊選取同一物體定界框的方法，並選取最具代表性的。在去除預測機率小於0.6的定界框後，會重複以下的步驟:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [給定一類別, 步驟一: 選擇有最大機率的定界框, 步驟二: 拋棄與前一步驟選取的定界框有IoU⩾0.5的定界框]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [定界框預測, 選擇有最大機率的定界框, 移除同類別且重疊的定界框, 最終的定界框]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO - YOLO是一個物體偵測演算法, 流程如下：

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [步驟一：把輸入影像切成G×G個格子, 步驟二：對於每一個格子, 分別進行CNN的運算來預測以下所表示的y：, 重複k次]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; 其中, pc為預測物體之機率, bx,by,bh,bw為定界框的屬性, c1,...,cp為p個偵測類別的一位有效編碼, k為錨框的數量。

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; 步驟三: 計算非最大值抑制演算法來移除可能是重複、重疊的定界框。

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230;[原始影像, GxG的格子, 定界框的預測, 非最大值抑制]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 備註：當pc=0, 網路沒有預測到任何物件，而相關的預測bx,...,cp可忽略。

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; 區域卷積類神經網路 ― 區域卷積類神經網路是一個物件偵測演算法, 先將一個影像分割以找尋可能的定界框, 再執行偵測的演算法來預測最可能出現在該定界框的物件。

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [原始圖片, , 定界框預測, 非最大值抑制]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 備註：即使原始的演算法耗費很多計算資源且速度慢，新提出的架構提供更快的演算法，例如快速型區域卷積類神經網路與更快速型區域卷積類神經網路。

<br>


**74. Face verification and recognition**

&#10230; 人臉驗證與辨別

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; 模型的種類 - 有兩種主要的模型種類，如下表：

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [人臉驗證, 人臉辨別, , 對照, 資料庫]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [是否是正確的人？, 一對一查詢, 是否是K個存在資料庫中的其中一人？, 一對多查詢]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; 單樣本學習 - 單樣本學習是一種人臉驗證演算法，使用有數量限制的訓練資集來學習一個相似度函數，用來量化兩影像之間的差異。應用於兩影像之間的相似度函數時常標示為d(影像1, 影像2)。

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; 孿生網路 - 孿生網路之目的為學習如何將影像編碼，並用於後續量化兩影像之間得差異。給定一輸入影像x(i), 編碼後的輸出標示為f(x(i))。

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; 三重損失函數 - 三重損失函數ℓ是一個計算影像A(錨框), P (正向樣本) 和 N (負向樣本)間嵌入表徵的損失函數。錨框與正向樣本屬於同個類別，而與負向樣本不同。指定α∈R+為一範圍參數，此損失函數定義為：

<br>


**81. Neural style transfer**

&#10230; 神經風格轉換

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; 動機 - 神經風格轉換之目的為根據給定的內容C與風格S，產生一張圖片G。

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [內容C, 風格S, 生成影像G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; 激發 - 給定一層l, 它的激發可表示為a[l], 其維度為nH×nw×nc.

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; 內容成本函數 - 內容成本函數Jcontent(C,G)用於計算生成影像G與內容影像C之間的差異。定義如下：

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; 風格矩陣 - 於第l層的風格矩陣G[l]是一個格拉姆矩陣，矩陣中的每個元素G[l]kk′量化k與k′頻道之間的相關程度。此矩陣定義為基於其激發。

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 備註：風格影像S與生成影像G的風格矩陣分別表示為G[l]與G[l].

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; 風格成本函數 - 風格成本函數Jstyle(S,G)用於評估生成影像G與風格S之差別。定義如下：

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 總體成本函數 - 總體成本函數定義為內容成本函數與風格成本函數之組合，權重為α,β。

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 備註：越高的α值會使模型會較注重於內容，而較高的β值會使模型較注重風格。

<br>


**91. Architectures using computational tricks**

&#10230; 計算架構手法

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; 生成對抗網路 - 生成對抗網路，簡稱為GANs，是一個由生成網路與對抗網路所組成的模型，其中生成網路的目的為生成最貼近真實的輸出，並當作對抗網路之輸入，而對抗網路之目的為分辨輸入數據為真實或偽造。

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [訓練資料, 雜訊, 真實影像, 生成網路, 對抗網路, 真實 偽造]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 備註：不同生成對抗網路的用途包括：由文字生成影像, 生成或合成音樂。

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; 殘差網路 - 殘差網路(ResNet) 利用殘差架構連接更高層以減少訓練誤差。殘差架構可表示為下式：

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception 網路 - 此架構利用inception模組, 目的為嘗試不同的卷積運算, 透過特徵多樣化來提高模型的效能。特別的是, 此架構利用1×1卷積技術來限制計算負擔。

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; 深度學習參考手冊目前已有[目標語言]版。

<br>


**98. Original authors**

&#10230; 原始作者

<br>


**99. Translated by X, Y and Z**

&#10230; 由X, Y與Z翻譯

<br>


**100. Reviewed by X, Y and Z**

&#10230; 由X, Y與Z檢閱

<br>


**101. View PDF version on GitHub**

&#10230; 在GitHub上閱讀PDF版

<br>


**102. By X and Y**

&#10230; X, Y

<br>
