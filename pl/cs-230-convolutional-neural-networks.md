Convolutional Neural Networks translation [webpage]

1. Convolutional Neural Networks cheatsheet

⟶Konwolucyjne Sieci Neuronowe - ściągawka

2. CS 230 - Deep Learning

⟶CS 230 - Uczenie Głębokie (ang. Deep Learning)

3. [Overview, Architecture structure]

⟶Przegląd, Struktura sieci

4. [Types of layer, Convolution, Pooling, Fully connected]

⟶Rodzaje warstw, Konwolucyjna, Redukująca, W pełni połączona

5. [Filter hyperparameters, Dimensions, Stride, Padding]

⟶Hiperparametry filtra, Wymiary, Krok, Margines z zerami

6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]

⟶Dostosowanie hiperparametrów, Zgodność parametrów, Złożoność modelu, Pole recepcyjne

7. [Activation functions, Rectified Linear Unit, Softmax]

⟶Funkcje aktywacji, ReLU, Softmax

8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]

⟶Wykrywanie obiektów, Modele, Wykrywanie, Współczynnik podobieństwa IoU, Tłumienie non-max, YOLO, R-CNN

9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]

⟶Rozpoznawanie twarzy, One-shot, Siamese network, Triplet loss

10. [Neural style transfer, Activation, Style matrix, Style/content cost function]

⟶Neural style transfer, aktywacja, macierz stylu, Funkcja straty styl/zawartość

11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]

⟶Architektury z użyciem trików obliczeniowych

12. Overview

⟶Przegląd

13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:

⟶Konwolucyjna sieć neuronowa - (ang. CNN ― Convolutional Neural Network) jest to rodzaj sieci neuronowch, które zazwyczaj składają się z następujących warstw: konwolucyjnej (ang. convolution), redukującej (ang. pooling) i w pełni połączonej (ang. fully connected).

14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.

⟶Warstwy konwolucyjna i redukująca mogą być dostrojone za pomocą tzw. hiperparametrów, które są opisane w kolejnych akapitach.

15. Types of layer

⟶Rodzaje warstw

16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.

⟶Warstwa Konwolucyjna (CONV) - (ang. convolution layer) działa w oparciu o filtry, które wykonują operację konwolucji podczas skanowania danych wejściowych (input - I). Wyróżnia się hiperparametry takie jak rozmiar filtra (filter size - F) oraz krok (stride - S). Otrzymane dane wyjściowe (output - O) noszą nazwę mapy cech (feature map/activation map).

17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.

⟶Uwaga: proces można uogólnić także dla przypadku 1D i 3D.

18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.

⟶Warstwa redukująca (POOL) - (ang. pooling layer) jest to operacja próbkowania, zazwyczaj następująca po warstwie konwolucyjnej. Najczęściej używane to Max Pooling - wybierająca wartość największą oraz Average Pooling - obliczająca średnią arytmetyczną.

19. [Type, Purpose, Illustration, Comments]

⟶Rodzaj, Działanie, Ilustracja, Komentarz

20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]

⟶Max pooling, Average pooling, Wybiera największą wartość w polu widzenia filtra, Oblicza wartość średnią z wartości w polu widzenia flitra

21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]

⟶Zachowuje wykryte cechy, Najczęściej używany, Próbkowanie w dół (downsampling) obrazu, Używany w LeNet

22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.

⟶Warstwa w pełni połączona (FC) ― (ang. fully connected layer) każda komórka danych wejściowych połączona jest każdym neuronem warstwy. Wymaga od danych wejściowych "spłaszczenia" (ang. flatten), czyli konwersji do 1-wymiarowej tablicy. Zazwyczaj umieszcza się je pod koniec sieci konwolucyjnej, gdzie odpowiadają za przyporządkowanie danych wejściowych do odpowiednich klas.

23. Filter hyperparameters

⟶Hiperparametry filtra

24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.

⟶Warstwa konwolucyjna wykorzystuje filtry, które zdefiniowane są poprzez hiperparametry. Poniżej przedstawiono ich znaczenie.

25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.

⟶Wymiary filtra ― Filtr o rozmiarze F×F zastosowany dla danych wejściowych zawierających liczbę C kanałów, stanowiący objętość F×F×C, wykonuje konwolucję na danych o rozmiarze I×I×C i daje na wyjściu mapę cech (zwaną mapą aktywacji) o rozmiarze O×O×1.

26. Filter

⟶Filtr

27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.

⟶Uwaga: stosowanie liczby K filtrów o rozmiarze F×F skutkuje na wyjściu mapą cech o rozmiarze O×O×K.

28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.

⟶Krok ― dla operacji konwolucji lub wybierania (poolingu), krok S oznacza liczbę pikseli, o którą przesuwa się okno filtra po każdej operacji.

29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:

⟶Margines z zerami ― (ang. zero-padding) oznacza dodanie na granicy obszaru dodatkowych pól wypełnionych zerami. Wartość można określić ręcznie lub wybrać spośród trzech trybów opisanych poniżej:

30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]

⟶Tryb, Wartość, Ilustracja, Działanie, Valid, Same, Full

31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]

⟶Brak marginesu, Porzuca ostatnią konwolucję jeśli wymiary sie nie zgadzają, Margines dopasowany tak, aby dane wyjściowe (output) miały wymiar ⌈IS⌉, Dogodny rozmiar danych wyjściowych, Nazywany także 'half' padding, Margines maksymalny dobrany tak, aby końcowe konwolucje obliczano na granicy obszaru, Filtr 'widzi' cały obszar

32. Tuning hyperparameters

⟶Dostosowanie hiperparametrów

33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:

⟶Zgodność parametrów w warstwie konwolucyjnej ― oznaczając przez I wielkość rozmiaru wejściowego, przez F wielkość filtra, przez P wielkość marginesu, przez S wielkość kroku, rozmiar danych wyjściowych O jest dany równaniem:

34. [Input, Filter, Output]

⟶Dane wejściowe, Filtr, Dane wyjściowe

35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.

⟶Uwaga: często Pstart=Pend≜P, wtedy człon Pstart+Pend można zastąpić przez 2P w wyrażeniu powyżej.

36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:

⟶Zrozumienie złożoności modelu ― w celu oszacowania złożoności modelu, często pomocne okazuje się określenie liczby parametrów, które ten model posiada. Dla danej warstwy konwolucyjnej sieci neuronowej, można to wykonać w poniższy sposób:

37. [Illustration, Input size, Output size, Number of parameters, Remarks]

⟶Ilustracja, Rozmiar danych wejściowych, Rozmiar danych wyjściowych, Liczba parametrów, Uwagi

38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]

⟶Jeden parametr odchylenia na filtr, W większości przypadków, S<F, Częsty wybór dla K to 2C

39. [Pooling operation done channel-wise, In most cases, S=F]

⟶Pooling wykonywany według kanałów, W większości przypadków, S=F

40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]

⟶Dane wejściowe "spłaszczone", Jedno odchylenie na neuron, Liczba neuronów sieci jest wolna od ograniczeń struktury

41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:

⟶Pole recepcyjne ― w odniesieniu do warstwy k jest to taka powierzchnia Rk×Rk danych wejściowych, którą 'widzi' każdy piksel k-tej mapy aktywacji. Nazywając Fj rozmiar filtra warstwy j oraz Si wartość kroku warstwy i a także zachowując konwencję, że S0=1, pole recepcyjne dla warstwy k można obliczyć za pomocą wyrażenia:

42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.

⟶W poniższym przykładzie F1=F2=3 oraz S1=S2=1, co daje R2=1+2⋅1+2⋅1=5.

43. Commonly used activation functions

⟶Powszechnie używane funkcje aktywacji

44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:

⟶Jednostronnie obcięta funkcja liniowa ― (ang. Rectified Linear Unit, ReLU) jest to funkcja aktywacji g, którą używa się dla każdego elementu obszaru. Jej celem jest wprowadzenie nieliniowości do sieci. Poniżej opisano jej warianty:

45. [ReLU, Leaky ReLU, ELU, with]

⟶ReLU, Leaky ("cieknąca") ReLU, ELU, dla

46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]

⟶Złożoności nieliniowe interpretowalne biologicznie, Rozwiązuje problem "umierającego" ReLU (dying ReLU) dla wartości ujemnych, W całości różniczkowalna

47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:

⟶Softmax (Znormalizowana funkcja wykładnicza) ― można ją interpretować jako uogólnioną funkcję logistyczną, która na wejściu pobiera wektor wartości x∈Rn i na wyjściu daje wektor prawdopodobieństw p∈Rn. Używana zazwyczaj w ostatniej warstwie sieci.

48. where

⟶gdzie

49. Object detection

⟶Wykrywanie obiektów

50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:

⟶Modele ― wyróżnia sie 3 główne rodzaje algorytmów wykrywania obiektów, dla których natura tego co jest przewidywane jest inna. Opisane sa w tabeli poniżej:

51. [Image classification, Classification w. localization, Detection]

⟶Klasyfikacja obrazu, Klasyfikacja z lokalizacją, Wykrywanie

52. [Teddy bear, Book]

⟶Miś, Książka

53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]

⟶Klasyfikuje obraz, Przewiduje prawdopodobieństwo obiektu, Wykrywa obiekt na obrazie, Przewiduje prawdopodobieństwo obiektu i jego lokalizację, Wykrywa do kilku obiektów na obrazie, Przewiduje prawdopodobieństwo obiektu i gdzie jest zlokalizowany

54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]

⟶Zwyczajne CNN, Uproszczone YOLO, R-CNN, YOLO, R-CNN

55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:

⟶Wykrywanie ― w kontekście wykrywania obiektów używa się różnych metod w zależności od tego czy chcemy tylko znaleźć obiekt czy wykryć bardziej skomplikowany kształt na obrazie. Dwa główne sposoby zestawiono w tabeli poniżej:

56. [Bounding box detection, Landmark detection]

⟶Wykrywanie ramką (bounding box detection), Wykrywanie poprzez punkty charakterystyczne (landmark detection)

57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]

⟶Wykrywa część obrazu gdzie znaleziono obiekt, Wykrywa kształt lub elementy charakterystyczne obiektu (np. oczy), Bardziej szczegółowe

58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]

⟶Ramka z centrum w (bx,by), wysokości bh i szerokości bw, punkty odniesienia (l1x,l1y), ..., (lnx,lny)

59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:

⟶Współczynnik podobieństwa IoU ― (ang. Intersection over Union, IoU) jest to funkcja, która ilościowo określa jak dobrze umiejscowiona jest przewidywana ramka Bp w stosunku do rzeczywistej ramki Ba. Zdefiniowany jest poprzez wyrażenie:

60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.

⟶Uwaga: współczynnik przyjmuje wartości IoU∈[0,1]. Umownie ramka Bp jest uważana za dobrze dopasowaną jeśli IoU(Bp,Ba)⩾0.5.

61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.

⟶Ramki kotwiczone ― (ang. anchor boxes) jest to technika używana do przewidywania nakładających się ramek. W praktyce, sieć jest zdolna do przewidzenia równocześnie więcej niż jednej ramki, gdzie każda z tych predykcji jest ograniczona zestawem własności geometrycznych. Dla przykładu pierwsza predykcja może być prostokątną ramką o danych wymiarach, podczas gdy druga będzie ramką o innych wymiarach.

62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:

⟶Tłumienie non-max ― (ang. non-max suppression) to technika, która ma na celu wyeliminowanie duplikatów ramek tego samego obiektu poprzez wybieranie najbardziej reprezetatywnych. Po usunięciu wszystkich ramek, których prawdopodobieństwo wynosiło poniżej 0,6, powtarzane są następujące kroki na pozostałuch ramkach:

63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]

⟶Dla danej klasy, Krok 1.: Wybierz ramkę z największym prawdopodobieństwem., Krok 2.: Odrzuć każdą ramkę posiadającą IoU⩾0.5 w stosunku do poprzedniej ramki.

64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]

⟶Predykcje ramki, Predykcje ramki z największym prawdopodobieństwem, Usunięcie nakładających się ramek dla tej samej klasy, Końcowe ramki

65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:

⟶YOLO ― akronim (ang. You Only Look Once, YOLO) w tłumaczeniu "patrzysz tylko raz", algorytm wykrywania obiektów, który wykonuje następujące kroki:

66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]

⟶

67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.

⟶

68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.

⟶

69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]

⟶

70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.

⟶

71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.

⟶

72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]

⟶

73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.

⟶

74. Face verification and recognition

⟶

75. Types of models ― Two main types of model are summed up in table below:

⟶

76. [Face verification, Face recognition, Query, Reference, Database]

⟶

77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]

⟶

78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).

⟶

79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).

⟶

80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:

⟶

81. Neural style transfer

⟶

82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.

⟶

83. [Content C, Style S, Generated image G]

⟶

84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc

⟶

85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:

⟶

86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:

⟶

87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.

⟶

88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:

⟶

89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:

⟶

90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.

⟶

91. Architectures using computational tricks

⟶

92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.

⟶

93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]

⟶

94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.

⟶

95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:

⟶

96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.

⟶

97. The Deep Learning cheatsheets are now available in [target language].

⟶

98. Original authors

⟶

99. Translated by X, Y and Z

⟶Tłumaczenie: Tomasz Pitala

100. Reviewed by X, Y and Z

⟶Korekta: 

101. View PDF version on GitHub

⟶Zobacz wersję PDF na GitHub

102. By X and Y

⟶
