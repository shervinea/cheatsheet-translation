**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; 畳み込みニューラルネットワーク チートシート

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - 深層学習

<br>


**3. [Overview, Architecture structure]**

&#10230; [概要, アーキテクチャ構造]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [層のタイプ, 畳み込み, プーリング, 全結合]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230;

<br> [フィルタハイパーパラメータ, 大きさ, ストライド, パディング]


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [ハイパーパラメータの調整, パラメータの互換性, モデルの複雑さ, 受容野]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [活性化関数, 正規化線形ユニット, ソフトマックス]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [オブジェクト検出, モデルのタイプ, 検出, 積集合の和集合, 非最大抑制, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [顔認証/認識, 一発学習, シャムネットワーク, 三重項損失]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [ニューラルスタイル変換, 活性化, スタイル行列, スタイル/コンテンツコスト関数]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [計算トリックアーキテクチャ, 敵対的生成ネットワーク, ResNet, インセプションネットワーク]

<br>


**12. Overview**

&#10230; 概要

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; 伝統的な畳み込みニューラルネットワークのアーキテクチャ - CNNとしても知られる畳み込みニューラルネットワークは一般的に次の層で構成される特定タイプのニューラルネットワークです。

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; 畳み込み層とプーリング層は次のセクションで説明されるハイパーパラメータに関して微調整できます。

<br>


**15. Types of layer**

&#10230; 層のタイプ

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; 畳み込み層 (CONV) - 畳み込み層 (CONV)は入力Iを各次元に関して走査する時に、畳み込み演算を行うフィルタを使用します。畳み込み層のハイパーパラメータにはフィルタサイズFとストライドSが含まれます。結果出力Oは特徴マップまたは活性化マップと呼ばれます。

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; 注: 畳み込みステップは1次元や3次元の場合にも一般化できます。

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; プーリング (POOL) - プーリング層 (POOL)はダウンサンプリング操作で、通常は位置不変性をもつ畳み込み層の後に適用されます。特に、最大及び平均プーリングはそれぞれ最大と平均値が取られる特別な種類のプーリングです。

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [タイプ, 目的, 図, コメント]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [最大プーリング, 平均プーリング, 各プーリング操作は現在のビューの中から最大値を選ぶ, 各プーリング操作は現在のビューに含まれる値を平均する]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [検出された特徴を保持する, 最も一般的に利用される, 特徴マップをダウンサンプリングする, LeNetで利用される]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; 全結合 (FC) - 全結合 (FC) 層は平坦化された入力に対して演算を行います。各入力は全てのニューロンに接続されています。FC層が存在する場合、通常CNNアーキテクチャの末尾に向かって見られ、クラススコアなどの目的を最適化するため利用できます。

<br>


**23. Filter hyperparameters**

&#10230; フィルタハイパーパラメータ

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; 畳み込み層にはハイパーパラメータの背後にある意味を知ることが重要なフィルタが含まれています。

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; フィルタの大きさ - C個のチャネルを含む入力に適用されるF×Fサイズのフィルタの体積はF×F×Cで、それはI×I×Cサイズの入力に対して畳み込みを実行してO×O×1サイズの特徴マップ（活性化マップとも呼ばれる）出力を生成します。


<br>


**26. Filter**

&#10230; フィルタ

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; 注: F×FサイズのK個のフィルタを適用すると、O×O×Kサイズの特徴マップの出力を得られます。

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; ストライド - 畳み込みまたはプーリング操作において、ストライドSは各操作の後にウィンドウを移動させるピクセル数を表します。

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; ゼロパディング - ゼロパディングとは入力の各境界に対してP個のゼロを追加するプロセスを意味します。この値は手動で指定することも、以下に詳述する３つのモードのいずれかを使用して自動的に設定することもできます。

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [モード, 値, 図, 目的, Valid, Same, Full]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [パディングなし, もし大きさが合わなかったら最後の畳み込みをやめる, 特徴マップのサイズが[IS]になるようなパディング, 出力サイズは数学的に扱いやすい, 「ハーフ」パディングとも呼ばれる, 入力の一番端まで畳み込みが適用されるような最大パディング, フィルタは入力を端から端まで「見る」]

<br>


**32. Tuning hyperparameters**

&#10230; 調律ハイパーパラメータ

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; 畳み込み層内のパラメータ互換性 - Iを入力ボリュームサイズの長さ、Fをフィルタの長さ、Pをゼロパディングの量, Sをストライドとすると、その寸法に沿った特徴図の出力サイズOは次式で与えられる:

<br>


**34. [Input, Filter, Output]**

&#10230; [入力, フィルタ, 出力]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; 注意: しばしば、Pstart=Pend≜P、その場合、上記の式のようにPstart+Pendを2Pに置き換える事ができる。

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; モデルの複雑さを理解する - モデルの複雑さを評価する為モデルのアーキテクチャが持つことになるパラメータの数を決定することはしばしば有用です。畳み込みニューラルネットワーク内で、以下のように行なわれる。

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [図, 入力サイズ, 出力サイズ, 引数の数, 備考]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [フィルタにあたり1つのバイアスパラメータ, ほとんどの場合, S<F, Kの一般的な選択は2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [プール操作はチャネルごとに行われる, ほとんどの場合, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [入力は平坦化される, ニューラルごとにひとつのバイアスパラメータ, FCニューラルの数は構造制約がない]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; 受容的なフィルド - 層kの受容的なフィルドはk番目の活性化図の各ピックセルが見られる入力のRkxRkを表示されるエリアです。j層のフィルタサイズをFj、i層のストライド値をSi、規約S0=1とすると、k層での受容的なフィルドは式で計算される:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; 下記の例で、F1=F2=3、S1=S2=1となるのでR2=1+2⋅1+2⋅1=5となる。

<br>


**43. Commonly used activation functions**

&#10230; よく使われる活性化関数

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; 整流線形ユニット - 整流線形ユニット層(ReLU)はボリュームの全ての要素に利用される活性化関数gです。ReLUの目的は非線型性をネットワークに紹介する。ReLUの変種は以下の表でまとめられる:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230;[ReLU, Leaky ReLU, ELU, with]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [生物学的に解釈可能な非線形複雑性, 負の値の為dyingReLUの問題を示す,どこても差別化可能]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; ソフトマックス - ソフトマックスステップは入力としてx∈Rnスコアのベクターを取り、アーキテクチャの最後にソフトマックス関数を通じてp∈Rn出力確率のベクターを出して、一般化ロジスティック関数として見る事ができる。

<br>


**48. where**

&#10230; どこ

<br>


**49. Object detection**

&#10230; オブジェクト検出

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230;

<br> モデルの種類 - 物体認識アルゴリズムは主に三つのタイプがあり、予測されるものの性質は異なります。次の表で説明される。


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [画像分類, 分類 w. 局地化, 検出]

<br>


**52. [Teddy bear, Book]**

&#10230; [テディ熊, 本]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [画像分類, オブジェクトの確率予測, 画像内のオブジェクト検出, オブジェクトの確率と所在地予測, 画像内の複数オブジェクト検出, 複数オブジェクトの確率と所在地予測]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [伝統的なCNN, 単純されたYOLO, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 検出 - 物体検出の文脈では、画像内で物体を特定するのかそれとも複雑な形状を検出するのかによって、様々な方法は使用される。二つの主なものは次の表でまとめられる:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [物体検出, ランドマーク検出]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [物体が配置されている画像の部分検出, (例: 目)物体の特徴または形状検出, より粒状]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [センターのボックス(bx, by), 縦bhと幅bw, 各参照ポイント　(l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; 労働組合の交差点 - 労働組合の交差点(IoUとも呼ばれる)は予測バウンディングボックスBpが実際のバウンディングボックスBaに対してどれだけ正しくかを定量化する関数です。次のように定義される:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 注意: 常にIoU∈[0,1]を持ってます。規約により、予測されたバウンディングボックスBpはIoU(Bp,Ba)⩾0.5の場合適度に良いと見なされる。

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; アンカーボックス - アンカーボクシングは重複バウンディングボックスを予測する為利用される技術です。実際には、同時に複雑のボックスを予測すろことを許可されており、各ボックス予測は与えられた幾何学的なプロパーティのセットを持つように制約される。例えば、最初の予測は与えられたフォームの長方形のボックスになる可能性があり、二番目のボックスは異なる幾何学的なフォームの別の長方形になります。

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 非最大抑制 - 非最大抑制技術のねらいは最も代表的なもの選択によって同物体の重複する重なり合う境界ボックスを除去することです。0.6未満予測確率があるボックスを全て除去した後、残りのボックスがある間に以下のステップが繰り返される:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [与えられたクラス, ステップ1: 最大予測確率があるボックスを取り。, ステップ2: 前のボックスと一緒にIoU⩾0.5のボックスを切り捨てる。]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [ボックス予測, 最大確率のボックス選択, 同じクラスの重なり合う除去, 最後のバウンディングボックス]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO - 貴方は一度だけ見る (YOLO)は次のステップを実行するオブジェクト検出アルゴリズムです。

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [ステップ1: 入力画像をGxGグリッドに分ける。, ステップ2: 各グリッドセルに対して次の形式のyを予測するCNNを実行する:,k回繰り返す]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; ここで、pcは物体認識の確率、bx,by,bh,bwはバウンディングボックスのプロパーティ、c1, ..., cpはpクラスのうちどれが検出されたかのワンホット表現で、kはアンカーボックスの数です。

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; 潜在的な重複バウンディングボックスを除去する為に非最大抑制アルゴリズムを実行する。

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [元の画像, GxGグリッドでの分割, 物体検出, 非最大抑制]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 注意: pc=0時、ネットワークは物体を検出しません。その場合には適当な予測 bx, ..., cpそれぞれは無視する必要があります。

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN - 畳み込みニューラルネットワークを利用した領域は最初に潜在的な関連する境界ボックスを見つけるため画像を分割し、次にそれらの境界ボックス内の最も可能性の高いオブジェクトを見つけるため検出アルゴリズムを実行する物体検出アルゴリズムです。

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [元の画像, セグメンテーション, 物体予測, 非最大抑制]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 注意: 元のアルゴリズムは計算コストが高くて遅くても、より新たなアーキテクチャでは、Fast R-CNNやFaster R-CNNなど、アルゴリズムをより高い速度に実行できる。

<br>


**74. Face verification and recognition**

&#10230; 顔認証及び認識

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230;

<br> モデルのタイプ - 主な二つのモデルは次の表で要約される:


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [顔認証, 顔認識, クエリ, 参照, データベース]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [これは正しい人ですか?, 一対一見上げる, これはデータベース内のk人のうちの一人ですか, 一対多見上げる]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; ワンショット学習 - ワンショット学習は二つの与えられた画像の違いを定量かする類似性関数を学ぶ為有限トレーニングセットを利用する顔認証アルゴリズムです。二つの画像に適用される類似性関数はしばしばd(画像１、画像２)と記される。

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; シャムネットワー - シャムネットワーは2つの画像の違いを定量化して、画像暗号化方法を学ぶことを目的としている。与えられたインプット画像x(i)に対して暗号化された出力はしばしばf(x(i))と表示される。

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; トリプレット損失 - トリプレット損失ℓはトリプレットの画像A(アンカー)、P(ポジティブ)、N(負)の埋め込み表現で計算する損失関数です。アンカーとポジティブ例は同じクラスに属し、ネガティブ例は別のものに属する。マージンパラメータはα∈R+と呼ぶことによってこの損失は次のように定義される:

<br>


**81. Neural style transfer**

&#10230; 神経のスタイル転送

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230;

<br> モチベーション - 神経のスタイル転送の目的は与えられたコンテンツCとスタイルSに基づく画像Gを生成する。


**83. [Content C, Style S, Generated image G]**

&#10230; [コンテンツC, スタイルS, 生成された画像G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230;

<br> 活性化 - 与えられた層lで、活性化はa[l]と表示されて、nH×nw×ncの寸法。


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; コンテンツコスト関数 - Jcontent(C, G)というコンテンツコスト関数は元のコンテンツ画像Cと生成された画像Gとの違いを決定するため利用される。以下のように定義される:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; スタイル行列 - 与えられた層lのスタイル行列 G[l]はグラム配列で、各要素G[l]kk′がチャネルkとｋ′の相関関係を定量化する。

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 注意: スタイル画像及び生成された画像に対するスタイル行列はそれぞれG[l] (S)、G[l] (G)と表示される。

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230;　スタイルコスト関数 - スタイルコスト関数Jstyle(S,G)はスタイルSと生成された画像Gどう違うかを決定する為利用される。次のように定義される:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 全体コスト関数 - 全体コスト関数は以下のようにパラメータα,βによって重み付けされ、スタイルコスト関数とコンテンツの組み合わせた物として定義される:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 注意: αのより高い値はモデルが内容をより気にするようにさせ、βのより高い値はスタイルをより気にするようになる。

<br>


**91. Architectures using computational tricks**

&#10230; アーキテクチャは計算の詭計を利用している。

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; 生成型敵対的ネットワーク - 生成型敵対的ネットワーク、GANsとも呼ばれるは生成モデルと識別モデルで構成される、生成モデルの目的は生成された画像と実像を区別する目的とする識別にフィードされる最も真実の出力を生成する。

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [トレーニング, 騒音, 現実世界の画像, ジェネレータ, 弁別器, 偽のリアル]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 注意: GANsの変種を使用するユースケースには画像へのテキスト, 音楽生成及び合成があります。

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet - 残渣ネットワークアーキテクチャ（ResNetとも呼ばれる）はトレーニングエラーを減らすため多数の層がある残差ブロックを使用する。残差ブロックは次の特定方程式を有する。

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; インセプションネットワーク - このアーキテクチャはインセプションモジュールを利用し、特徴多様化を通じてパーフォーマンス改善の為別の畳み込みを試してみる目的とする。特に、計算負荷を限定する為1×1畳み込みトリックを使う。

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230;

<br> 深層学習チートシートは今[ターゲット言語]で利用可能です。


**98. Original authors**

&#10230; 原著者

<br>


**99. Translated by X, Y and Z**

&#10230; X, Y, Zによる翻訳された

<br>


**100. Reviewed by X, Y and Z**

&#10230; X, Y, Zによるレビューされた

<br>


**101. View PDF version on GitHub**

&#10230;

<br> GithubでPDFバージョン見る


**102. By X and Y**

&#10230; X, Yによる

<br>
