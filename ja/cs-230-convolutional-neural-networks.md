**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; 畳み込みニューラルネットワーク チートシート

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - ディープラーニング

<br>


**3. [Overview, Architecture structure]**

&#10230; [概要, アーキテクチャ構造]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [層の種類, 畳み込み, プーリング, 全結合]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [フィルタハイパーパラメータ, 次元, ストライド, パディング]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [ハイパーパラメータの調整, パラメータの互換性, モデルの複雑さ, 受容野]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [活性化関数, 正規化線形ユニット, ソフトマックス]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [物体検出, モデルの種類, 検出, IoU, 非極大抑制, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [顔認証/認識, One shot学習, シャムネットワーク, トリプレット損失]

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

&#10230; 伝統的な畳み込みニューラルネットワークのアーキテクチャ - CNNとしても知られる畳み込みニューラルネットワークは一般的に次の層で構成される特定種類のニューラルネットワークです。

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; 畳み込み層とプーリング層は次のセクションで説明されるハイパーパラメータに関してファインチューニングできます。

<br>


**15. Types of layer**

&#10230; 層の種類

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; 畳み込み層 (CONV) - 畳み込み層 (CONV)は入力Iを各次元に関して走査する時に、畳み込み演算を行うフィルタを使用します。畳み込み層のハイパーパラメータにはフィルタサイズFとストライドSが含まれます。結果出力Oは特徴マップまたは活性化マップと呼ばれます。

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; 注: 畳み込みステップは1次元や3次元の場合にも一般化できます。

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; プーリング (POOL) - プーリング層 (POOL)は位置不変性をもつ縮小操作で、通常は畳み込み層の後に適用されます。特に、最大及び平均プーリングはそれぞれ最大と平均値が取られる特別な種類のプーリングです。

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [種類, 目的, 図, コメント]

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

&#10230; フィルタの次元 - C個のチャネルを含む入力に適用されるF×Fサイズのフィルタの体積はF×F×Cで、それはI×I×Cサイズの入力に対して畳み込みを実行してO×O×1サイズの特徴マップ（活性化マップとも呼ばれる）出力を生成します。


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

&#10230; [パディングなし, もし次元が合わなかったら最後の畳み込みをやめる, 特徴マップのサイズが[IS]になるようなパディング, 出力サイズは数学的に扱いやすい, 「ハーフ」パディングとも呼ばれる, 入力の一番端まで畳み込みが適用されるような最大パディング, フィルタは入力を端から端まで「見る」]

<br>


**32. Tuning hyperparameters**

&#10230; ハイパーパラメータの調整

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; 畳み込み層内のパラメータ互換性 - Iを入力ボリュームサイズの長さ、Fをフィルタの長さ、Pをゼロパディングの量, Sをストライドとすると、その次元に沿った特徴マップの出力サイズOは次式で与えられます:

<br>


**34. [Input, Filter, Output]**

&#10230; [入力, フィルタ, 出力]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; 注: 多くの場合Pstart=Pend≜Pであり、上記の式のPstart+Pendを2Pに置き換える事ができます。

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; モデルの複雑さを理解する - モデルの複雑さを評価するために、モデルのアーキテクチャが持つパラメータの数を測定することがしばしば有用です。畳み込みニューラルネットワークの各層では、以下のように行なわれます。

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [図, 入力サイズ, 出力サイズ, パラメータの数, 備考]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [フィルタごとに1つのバイアスパラメータ, ほとんどの場合, S<F, Kの一般的な選択は2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [プール操作はチャネルごとに行われる, ほとんどの場合, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [入力は平坦化される, ニューロンごとにひとつのバイアスパラメータ, FCのニューロンの数には構造的制約がない]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; 受容野 - 層kにおける受容野は、k番目の活性化マップの各ピクセルが「見る」ことができる入力のRk×Rkの領域です。層jのフィルタサイズをFj、層iのストライド値をSiとし、慣例に従ってS0=1とすると、層kでの受容野は次の式で計算されます：

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; 下記の例のようにF1=F2=3、S1=S2=1とすると、R2=1+2⋅1+2⋅1=5となります。

<br>


**43. Commonly used activation functions**

&#10230; よく使われる活性化関数

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; 正規化線形ユニット - 正規化線形ユニット層(ReLU)はボリュームの全ての要素に利用される活性化関数gです。ReLUの目的は非線型性をネットワークに導入することです。変種は以下の表でまとめられています：

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230;[ReLU, Leaky ReLU, ELU, ただし]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [生物学的に解釈可能な非線形複雑性, 負の値に対してReLUが死んでいる問題に対処する,どこても微分可能]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; ソフトマックス - ソフトマックスのステップは入力としてスコアx∈Rnのベクトルを取り、アーキテクチャの最後にあるソフトマックス関数を通じて確率p∈Rnのベクトルを出力する一般化されたロジスティック関数として見ることができます。次のように定義されます。

<br>


**48. where**

&#10230; ここで

<br>


**49. Object detection**

&#10230; 物体検出

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; モデルの種類 - 物体認識アルゴリズムは主に3つの種類があり、予測されるものの性質は異なります。次の表で説明されています。

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [画像分類, 位置特定を伴う分類, 検出]

<br>


**52. [Teddy bear, Book]**

&#10230; [テディベア, 本]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [画像を分類する, 物体の確率を予測する, 画像内の物体を検出する, 物体の確率とその位置を予測する, 画像内の複数の物体を検出する, 複数の物体の確率と位置を予測する]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [伝統的なCNN, 単純されたYOLO, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 検出 - 物体検出の文脈では、画像内の物体の位置を特定したいだけなのかあるいは複雑な形状を検出したいのかによって、異なる方法が使用されます。二つの主なものは次の表でまとめられています：

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [バウンディングボックス検出, ランドマーク検出]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [物体が配置されている画像の部分を検出する, 物体（たとえば目）の形状または特徴を検出する, よりきめ細かい]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [中心(bx, by)、高さbh、幅bwのボックス, 参照点(l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Intersection over Union - Intersection over Union （IoUとしても知られる）は予測された境界ボックスBpが実際の境界ボックスBaに対してどれだけ正しく配置されているかを定量化する関数です。次のように定義されます：

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 注：常にIoU∈[0,1]となります。慣例では、IoU(Bp,Ba)⩾0.5の場合、予測された境界ボックスBpはそこそこ良いと見なされます。

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; アンカーボックス - アンカーボクシングは重なり合う境界ボックスを予測するために使用される手法です。 実際には、ネットワークは同時に複数のボックスを予測することを許可されており、各ボックスの予測は特定の幾何学的属性の組み合わせを持つように制約されます。例えば、最初の予測は特定の形式の長方形のボックスになる可能性があり、2番目の予測は異なる幾何学的形式の別の長方形のボックスになります。

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 非極大抑制 - 非極大抑制技術のねらいは、最も代表的なものを選択することによって、同じ物体の重複した重なり合う境界ボックスを除去することです。0.6未満の予測確率を持つボックスを全て除去した後、残りのボックスがある間、以下の手順が繰り返されます:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [特定のクラスに対して, ステップ1: 最大の予測確率を持つボックスを選ぶ。, ステップ2: そのボックスに対してIoU⩾0.5となる全てのボックスを破棄する。]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [ボックス予測, 最大確率のボックス選択, 同じクラスの重複除去, 最終的な境界ボックス]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO - You Only Look Once (YOLO)は次の手順を実行する物体検出アルゴリズムです。

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [ステップ1: 入力画像をGxGグリッドに分割する。, ステップ2: 各グリッドセルに対して次の形式のyを予測するCNNを実行する:,k回繰り返す]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; ここで、pcは物体を検出する確率、bx,by,bh,bwは検出された境界ボックスの属性、c1, ..., cpはp個のクラスのうちどれが検出されたかのOne-hot表現、kはアンカーボックスの数です。

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; ステップ3: 重複する可能性のある重なり合う境界ボックスを全て除去するため、非極大抑制アルゴリズムを実行する。

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [元の画像, GxGグリッドでの分割, 境界ボックス予測, 非極大抑制]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 注: pc=0のとき、ネットワークは物体を検出しません。その場合には、対応する予測 bx, ..., cpは無視する必要があります。

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN - Region with Convolutional Neural Networks (R-CNN)は物体検出アルゴリズムで、最初に画像をセグメント化して潜在的に関連する境界ボックスを見つけ、次に検出アルゴリズムを実行してそれらの境界ボックス内で最も可能性の高い物体を見つけます。

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [元の画像, セグメンテーション, 境界ボックス予測, 非極大抑制]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 注: 元のアルゴリズムは計算コストが高くて遅いですが、Fast R-CNNやFaster R-CNNなどの、より新しいアーキテクチャではアルゴリズムをより速く実行できます。

<br>


**74. Face verification and recognition**

&#10230; 顔認証及び認識

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; モデルの種類 - 2種類の主要なモデルが次の表にまとめられています:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [顔認証, 顔認識, クエリ, 参照, データベース]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [これは正しい人ですか?, 1対1検索, これはデータベース内のK人のうちの1人ですか, 1対多検索]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; ワンショット学習 - ワンショット学習は限られた学習セットを利用して、2つの与えられた画像の違いを定量化する類似度関数を学習する顔認証アルゴリズムです。2つの画像に適用される類似度関数はしばしばd(画像1, 画像2)と記されます。

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; シャムネットワーク - シャムネットワークは画像のエンコード方法を学習して2つの画像の違いを定量化することを目的としています。与えられた入力画像x(i)に対してエンコードされた出力はしばしばf(x(i))と記されます。

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; トリプレット損失 - トリプレット損失ℓは3つ組の画像A(アンカー)、P(ポジティブ)、N(ネガティブ)の埋め込み表現で計算される損失関数です。アンカーとポジティブ例は同じクラスに属し、ネガティブ例は別のクラスに属します。マージンパラメータをα∈R+と呼ぶことによってこの損失は次のように定義されます:

<br>


**81. Neural style transfer**

&#10230; ニューラルスタイル変換

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; モチベーション - ニューラルスタイル変換の目的は与えられたコンテンツCとスタイルSに基づく画像Gを生成することです。

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [コンテンツC, スタイルS, 生成された画像G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; 活性化 - 層lにおける活性化はa[l]と表記され、次元はnH×nw×ncです。

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; コンテンツコスト関数 - Jcontent(C, G)というコンテンツコスト関数は生成された画像Gと元のコンテンツ画像Cとの違いを測定するため利用されます。以下のように定義されます:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; スタイル行列 - 与えられた層lのスタイル行列G[l]はグラム行列で、各要素G[l]kk′がチャネルkとk′の相関関係を定量化します。活性化a[l]に関して次のように定義されます。

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 注: スタイル画像及び生成された画像に対するスタイル行列はそれぞれG[l] (S)、G[l] (G)と表記されます。

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230;　スタイルコスト関数 - スタイルコスト関数Jstyle(S,G)は生成された画像GとスタイルSとの違いを測定するため利用されます。以下のように定義されます:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 全体のコスト関数 - 全体のコスト関数は以下のようにパラメータα,βによって重み付けされたコンテンツ及びスタイルコスト関数の組み合わせとして定義されます:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 注: αの値を大きくするとモデルはコンテンツを重視し、βの値を大きくするとスタイルを重視します。

<br>


**91. Architectures using computational tricks**

&#10230; 計算トリックを使うアーキテクチャ

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; 敵対的生成ネットワーク - 敵対的生成ネットワーク（GANsとも呼ばれる）は生成モデルと識別モデルで構成されます。生成モデルの目的は、生成された画像と本物の画像を区別することを目的とする識別モデルに与えられる、最も本物らしい出力を生成することです。

<br>


**93. [Training set, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [学習セット, ノイズ, 現実世界の画像, 生成器, 識別器, 真 偽]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 注: GANsの変種を使用するユースケースにはテキストからの画像生成, 音楽生成及び合成があります。

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet - Residual Networkアーキテクチャ（ResNetとも呼ばれる）は学習エラーを減らすため多数の層がある残差ブロックを使用します。残差ブロックは次の特性方程式を有します。

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; インセプションネットワーク - このアーキテクチャはインセプションモジュールを利用し、特徴量の多様化を通じてパーフォーマンスを向上させるため、様々な畳み込みを試すことを目的としています。特に、計算負荷を限定するため1×1畳み込みトリックを使います。

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; ディープラーニングのチートシートが日本語で利用可能になりました。

<br>


**98. Original authors**

&#10230; 原著者

<br>


**99. Translated by X, Y and Z**

&#10230; X・Y・Z 訳

<br>


**100. Reviewed by X, Y and Z**

&#10230; X, Y, Z 校正

<br>


**101. View PDF version on GitHub**

&#10230; GitHubでPDF版を見る

<br>


**102. By X and Y**

&#10230; X・Y 著

<br>
