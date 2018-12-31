**Convolutional Neural Networks translation**

<br> 

**1. Convolutional Neural Networks cheatsheet**

&#10230; 합성곱 신경망 치트시트

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - 딥러닝

<br>


**3. [Overview, Architecture structure]**

&#10230; [개요, 구조]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [레이어, 합성곱, 풀링, 풀리커넥티드]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [필터 하이퍼파라미터, 차원, 스트라이드, 패딩]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [하이퍼파라미터 튜닝, 파라미터 호환성, 모델 복잡성, 수용 필드]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [활성화 함수, 수정된 선형 유닛, 소프트맥스]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [객체 탐지, 모델의 유형, 탐지, 합집합 분의 교집합, 논맥스 수프레션, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [얼굴 확인/인식, 원핫 학습, 샴 네트워크, 삼중항 손실]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [뉴럴 스타일 전달, 활성화, 스타일 행렬, 스타일/콘텐츠 비용 함수]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [계산 트릭 구조, GANs, 레스넷, 인셉션 네트워크]

<br>


**12. Overview**

&#10230; 개요

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; 전통적인 CNN의 구조 - CNN이라고도 하는 합성곱 신경망(convolution neural network)은 일반적으로 다음과 같은 계층으로 구성된 뉴럴네트워크의 특정 유형입니다.

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; 컨볼루션(convolution) 계층과 풀링(pooling) 계층은 다음 섹션에서 설명하는 하이퍼파라미터와 관련하여 세부 조정할 수 있습니다

<br>


**15. Types of layer**

&#10230; 계층 유형

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; 합성곱 계층 (CONV) - 합성곱 계층 (CONV)는 차원에 따라 입력값 I를 스캔할 때 합성곱(convolution) 연산을 수행하는 필터를 사용합니다. 그 하이퍼파라미터는 필터크기 F와 스트라이드(stride) S를 포함합니다. 그 결과 출력 O는 피쳐맵(feature map) 또는 활성화맵(activation map)이라고 합니다.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; 비고: 합성곱(convolution) 단계는 1D와 3D인 경우까지 잘 일반화 될 수 있습니다.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; 풀링 (POOL) - 풀링 계층 (POOL)은 일반적으로 몇가지 공간불변량을 수행하는 합성곱 계층 다음에 적용되는 다운샘플링(downsampling) 작업입니다. 특히, 최대풀링(max pooling)과 평균풀링(average pooling)은 최대값이나 평균값을 취하는 특별한 종류의 풀링(pooling)입니다.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [유형, 목적, 그림, 코멘트]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [최대 풀링, 평균 풀링, 각 풀링 작업은 현재 뷰(view)에서 최대값을 선택합니다. 각 풀링 작업은 현재 뷰(view)의 값들의 평균을 취합니다.

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [탐지된 특징을 보존, 가장 보편적으로 사용, 피쳐맵의 다운샘플링, LeNet에 사용]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; 풀리 커넥티드 (FC) - 풀리 커넥티드 계층 (FC)은 각 입력이 모든 뉴런에 연결된 평평한 입력에서 작동합니다. 존재한다면, FC 계층은 일반적으로 CNN구조의 끝부분에서 발견되며, 클래스 점수와 같은 목표를 최적화하는데 사용될 수 있습니다.
<br>


**23. Filter hyperparameters**

&#10230; 필터 하이퍼파라미터

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; 합성곱 계층은 필터의 하이퍼파라미터의 의미를 아는 것이 중요합니다.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; 필터의 차원 - C개의 채널을 포함하는 입력(input)에 적용되는 FxF 사이즈 필터(filter)는 IxIxC 크기의 입력(input)에 합성곱을 수행하고 OxOx1 크기의 피쳐맵 (또는 활성화 맵)을 출력으로 생성하는 FxFxC크기의 볼륨입니다.

<br>


**26. Filter**

&#10230; 필터

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; 비고: FxF 크기의 필터 K개를 적용하면 OxOxK 크기의 피처맵(feature map)이 출력으로 나온다.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; 스트라이드(stride) - 합성곱(convolution)이나 풀링(pooling) 작업의 경우, 스트라이드(stride) S는 각 작업 후에 윈도우(window)가 이동하는 픽셀 수를 나타냅니다.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; 제로-패딩(zero-padding) - 제로패딩은 입력(input)의 각 경계면에 P개의 제로(zero)를 더하는 과정을 말합니다. 이 값은 수동으로 지정하거나 아래에 설명된 세가지 모드 중 하나를 통해 자동으로 설정할 수 있습니다.

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [모드, 값, 그림, 목적, 유효한, 동일한, 전체]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; 패딩 없음, 차원이 맞지 않으면 마지막 합성곱(convolution)을 버림, 피처맵(feature map) 크기같은 패딩은 ⌈IS⌉ 크기를 갖음, 출력 크기는 수학적으로 편리함, 반쪽 패딩이라고도 부름, 마지막 합성곱이 입력(input)의 한계에 적용되도록 하는 최대패딩(maximum padding), 필터는 입력을 종단간(end-to-end)으로 확인합니다.

<br>


**32. Tuning hyperparameters**

&#10230; 하이퍼파라미터 튜닝

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; 합성곱 계층의 파리미터 호환성 - 입력 볼륨 크기의 길이 I, 필터의 길이 F, 제로패딩(zero-padding)의 양 P, 스트라이드 S에 대하여, 피처맵의 출력크기 O는 다음과 같이 주어진다.

<br>


**34. [Input, Filter, Output]**

&#10230; [입력, 필터, 출력]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; 비고: 빈번히, Pstart=Pend≜P를 만족하는 경우 위의 수식에서 Pstart+Pend를 2P로 바꿀 수 있습니다.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; 모델의 복잡성 이해 - 모델의 복잡성을 평가하기 위해, 그 구조의 파라미터 수를 결정하는 것이 유용합니다. 합성곱 신경망의 주어진 층에서, 그것은 다음과 같이 행해집니다.

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; 그림, 입력 크기, 출력 크기, 파라미터 수, 비고

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [필터 당 하나의 바이어스(bias) 파라미터, 대부분의 경우 S<F, K에 대핸 일반적인 선택은 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; 풀링 작업은 채널별로 수행, 대부분의 경우 S=F

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; 입력은 1차원, 뉴런 당 하나의 바이어스(bias) 파라미터, FC뉴런의 갯수는 구조적 제약이 없음

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; 수용 영역(receptive field) - k층에서 수용 영역은 k번째 활성화맵(activation map)의 각 픽셀이 볼 수 있는 입력의 RkxRk로 표시된 영역입니다. j층의 필터(filter)크기 Fj와 S0=1인 i층의 스트라이드(stride) 값 Si를 호출함으로써 k층의 수용영역은 다음 식으로 계산될 수 있습니다. 

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; 아래 예제에서, F1=F2=3이고 S1=S2=1이므로 R2=1+2⋅1+2⋅1=5이 됩니다.

<br>


**43. Commonly used activation functions**

&#10230; 일반적으로 사용되는 활성화함수(activation functions)

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; 수정된 선형 유닛 - rectified linear unit(ReLU) 레이어는 볼륨의 모든 요소에서 사용되는 활성화함수(activation function) g입니다. ReLU는 비선형성을 네트워크에 도입하는 것을 목표로 합니다. 그 변종은 아래 표에 요약되어 있습니다.

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; ReLU, Leaky ReLU, ELU

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [생물학적으로 해석가능한 비선형 복잡성, 음수일 때 ReLU가 죽는 문제, 모든점에서 미분가능]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; 소프트맥스(softmax) - 소프트맥스 단계는 점수벡터 x∈Rn를 입력(input)으로 취하고, 구조 끝의 소프트맥스(softmax) 함수를 통한 확률벡터 p∈Rn를 출력으로 취하는 일반화된 로지스틱(logistic) 함수로 볼 수 있습니다. 소프트맥스는 다음과 같이 정의됩니다.

<br>


**48. where**

&#10230; 여기서

<br>


**49. Object detection**

&#10230; 객체 탐지

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; 모델 유형 - 예측 대상의 특성이 다른 3가지 유형의 객체 인지 알고리즘이 아래 표에 설명되어 있습니다. 

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [이미지 분류, 분류 w. 지역화, 탐지]

<br>


**52. [Teddy bear, Book]**

&#10230; [곰인형, 책]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [그림 분류, 객체 가능성 예측, 사진에서 객체 탐지, 객체의 가능성 예측 및 위치 파악, 사진에서 여러개의 객체 탐지, 객체의 가능성 예측 및 위치 파악]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [전통적인 CNN, 단순화된 YOLO, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 탐지 - 물체 감지의 맥락에서, 물체의 위치를 찾고싶은지, 이미지의 더 복잡한 모양을 탐지하고 싶은지에 따라 다른 방법이 사용됩니다. 두가지 주요 방법이 아래 표에 요약되어 있습니다.

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; 바운딩 박스 탐지, 랜드마크 감지

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [객체가 위치한 이미지의 부분을 탐지, 객채의 모양이나 특징(예: 눈)을 탐지, 더 세분화 됨]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [박스의 센터 (bx, by), 높이 bh와 너비 bw, 참조포인트 (l1x,l1y), ..., (lnx,lny)

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; 합집합 분의 교집합 - IoU라고도 하는 합집합 분의 교집합은 예측된 바운딩박스(bounding box)가 실제 바운딩박스(bounding box)에 얼마나 정확하게 위치하는지를 정량화하는 함수로, 다음과 같이 정의됩니다.

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 비고: IoU∈[0,1]를 항상 만족합니다. 관습적으로, IoU(Bp,Ba)⩾0.5이면 예측 바운딩박스(bounding box) Bp는가 합리적으로 양호한 것으로 간주합니다.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; 앵커박스(Anchor box) - 앵커박싱(Anchor boxing)은 중첩 바운딩박스(bounding box)를 예측하는데 사용되는 기술입니다. 실제로, 네트워크는 하나 이상의 박스를 동시에 예측할 수 있으며, 각각의 박스 예측은 주어진 기하학적인 특정 세트를 갖도록 제한됩니다. 예를들어, 첫번째 예측은 주어진 양식의 직사각형 박스(box)일 수 있으며, 두번째는 다른 모양의 직사각형 박스(box)가 될 수 있습니다.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 비최대 억제 - 비최대 억제(non-max suppression) 기술은 가장 대표적인 것을 선택함으로써 동일한 객체에 대한 중복된 중첩바운딩박스(overlapping bounding box)를 제거하는 것을 목표로 한다. 확률 예측값이 0.6보다 작은 모든 박스를 제거한 후에, 박스가 남아있는 동안 다음 단계를 반복합니다.

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; 1단계 : 예측 확률이 가장 큰 박스를 선택합니다. 2단계 : 이전 상자에서 IoU⩾0.5를 만족하는 상자를 모두 버립니다.

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [박스 예측, 최대 확률을 갖는 박스 선택, 동일한 클래스에 대한 중복을 제거, 최종 바운딩박스(bounding box)]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO - You Only Look Once (YOLO)는 다음 단계를 수행하는 객체 탐지 알고리즘입니다.

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [1단계: 입력 이미지를 GxG 그리드(grid)로 분할합니다. 2단계: 각 그리드 셀(grid cell)에 대해 다음 형태의 y를 예측하는 CNN을 실행합니다., k번 반복]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; 여기서 pc는 객체를 탐지할 확률이고, bx,by,bh는 탐지된 바운딩박스(bounding box)의 속성이고, c1,...,cp는 p 클래스 중 어느것이 탐지되었는지를 나타내는 원핫(one-hot) 표현입니다. 그리고 k는 앵커박스(anchor box)의 갯수입니다.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; 3단계: 중복되는 중첩바운딩박스(overlapping bounding box)를 제거하기 위해 논맥스 수프레션 알고리즘을 실행합니다.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [원본 이미지, GxG 그리드 분할, 바운딩박스(bounding box) 예측, 논맥스 수프레션]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 비고: pc=0일 때, 네트워크는 어떠한 객체도 탐지하지 않습니다. 이 경우, 예측값 bx,...,cp는 무시되어야 합니다.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN - Region with Convolutional Neural Networks (R-CNN)은 잠재적으로 연관된 바운딩박스(bounding box)를 찾기 위해 이미지를 먼저 분할한 다음, 해당 바운딩박스(bounding box)에서 가장 가능성이 높은 객체를 찾기 위해 탐지(detection) 알고리즘을 실행하는 객체 탐지 알고리즘입니다.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [원본 이미지, 세그멘테이션, 바운딩박스 예측, 논맥스 수프레션]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 비고: 기존 알고리즘은 계산 비용이 많이 들고 느리지만, Fast R-CNN과 Faster R-CNN 같은 새로운 구조를 사용하면 알고리즘 실행 속도가 빨라집니다.  

<br>


**74. Face verification and recognition**

&#10230; 얼굴 확인 및 인식

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; 모델 유형 - 두가지 주요 모델 유형이 아래 표에 요약되어 있습니다.

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [얼굴 확인, 얼굴 인식, 질의 참조, 데이터베이스]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [이사람이 맞습니까? 일대일 조회, 데이터 베이스에 있는 K명의 사람 중 한명입니까?, 일대다 조회]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; 원 샷 학습 - 원 샷 학습은 제한된 훈련 셋(training set)을 사용하여 주어진 두 이미지가 얼마나 다른지를 정량화하는 유사도 함수를 학습시키는 얼굴 확인 알고리즘(face verification algorithm)입니다. 두 이미지에 적용된 유사도 함수는 d(image 1,image 2)로 씁니다.

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; 샴 네트워크 - 샴 네트워크는 두 이미지가 얼마나 다른지를 정량화하기 위해 이미지를 인코딩 하는 방법을 학습하는 것을 목표로 합니다. 주어진 입력 이미지 x(i)에 대해 인코딩된 출력은 f(x(i))라고 표현합니다.

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; 삼중항 손실 - The triplet loss ℓ는 이미지 A (anchor), P (positive) and N (negative)의 삼중합의 임베딩 표현을 계산한 손실함수입니다. anchor와 positive는 같은 클래스에 속하며, negative와는 다른 클래스에 속합니다. α∈R+라고 부르는 마진(margin) 파라미터에 의해 이 손실은 다음과 같이 정의됩니다.

<br>


**81. Neural style transfer**

&#10230; 뉴럴 스타일 전달

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; 동기 - 뉴럴 스타일 전달의 목표는 주어진 내용 C와 주어진 스타일 S에 기반하여 이미지 G를 생성하는 것입니다.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [내용 C, 스타일 S, 생성된 이미지 G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; 활성화 - 주어진 층 l에서, 활성화는 a[l]로 쓰며, nH×nw×nc의 차원을 갖는다.

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; 콘텐츠 비용 함수 - 콘텐츠 비용 함수 Jcontent(C,G)는 생성된 이미지 G가 원본 콘텐츠 이미지 C와 얼마자 다른지를 결정하는데 사용됩니다. 이 함수는 다음과 같이 정의됩니다.

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; 스타일 행렬 - 주어진 층 l의 스타일 행렬 G[l]은 그램행렬(Gram matrix)이며 각 요소 G[l]kk′는 채널 k와 채널 k'가 얼마나 상관되어 있는지를 정량화합니다.

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 비고: 스타일 이미지에 대하 스타일 행렬과 생성된 이미지는 각각 G[l] (S)와 G[l] (G)로 표시합니다.

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; 스타일 비용 함수 - 스타일 비용 함수 Jstyle(S,G)는 생성된 이미지 G가 스타일 S와 얼마나 다른지를 결정하는데 사용됩니다. 스타일 비용 함수는 다음과 같이 정의합니다.

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 전체 비용 함수 - 전체 비용 함수는 다음과 같이 파라미터 α,β로 가중치를 부여한 콘텐츠와 스타일 비용 함수의 조합으로서 정의됩니다.

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 비고: α 값이 높을수록 모델에 대한 관심이 높아지며, β 값이 높을수록 스타일에 대한 관심이 높아집니다.

<br>


**91. Architectures using computational tricks**

&#10230; 계산 트릭을 사용하는 구조

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Generative Adversarial Network - GANs라고도 불리는 Generative Adversarial Network는 생성모델(generative model)과 구분모델(discriminative model)로 구성된다. 생성모델은 구분모델에 투입될 가장 사실적인 결과물을 생성하는 것을 목표로 하며, 구분모델은 생성된 이미지와 실제 이미지를 구분짓는 것을 목표로 한다.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [훈련, 잡음, 실제 이미지, 생성자, 구분자, 실제 페이크]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 비고: GAN으로 생성된 변형이미지를 사용하는 유스케이스(use case)에는 텍스트, 이미지, 음악 생성자, 합성 등이 있다.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; 레스넷(ResNet) - ResNet이라고도 불리는 Residual Network architecture는 훈련 오차를 줄이기 위해 많은 레이어가 있는 잔여 블록(residual block)을 사용합니다. 잔여블록은 다음과같은 특성화 방정식을 만족합니다.

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; 인셉션 네트워크(Inception Network) - 이 구조는 인셉션(inception) 모듈을 사용하며 기능 다양화로 성능을 향상시키기 위해 다른 합성곱(convolution)을 시도합니다. 특히 계산 부담을 줄이기 위해 1x1 합성곱(convolution) 트릭을 사용합니다.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; 딥러닝 치트시트는 이제 [한국어]로 제공됩니다.

<br>


**98. Original authors**

&#10230; 원저자

<br>


**99. Translated by X, Y and Z**

&#10230; Soyoung Lee에 의해 번역됨

<br>


**100. Reviewed by X, Y and Z**

&#10230; X,Y,X에 의해 검토됨

<br>


**101. View PDF version on GitHub**

&#10230; GitHub에서 PDF버전 보기

<br>


**102. By X and Y**

&#10230; X와 Y

<br>
