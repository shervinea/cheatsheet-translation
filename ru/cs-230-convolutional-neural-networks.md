**Convolutional Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; Шпаргалка по Сверточным Нейронным Сетям

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Глубокое обучение

<br>


**3. [Overview, Architecture structure]**

&#10230; [Обзор, Структура архитектуры]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [Типы слоёв, Свертка, Пулинг, Полносвязный]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [Фильтрация гиперпараметров, Размеры, Шаг, Дополнение]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [Настройка гиперпараметров, Совместимость параметров, Сложность модели, Рецептивное поле]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [Функции активации, Блок линейной ректификации, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [Обнаружение объектов, Типы моделей, Обнаружение, Пересечение по объединению, Подавление немаксимумов, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [Проверка/распознавание лиц, Обучение с одного раза, Сиамская сеть, Triplet loss]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [Нейронный перенос стиля, Активация, Матрица стиля, Функция стоимости стиля/контента]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [Архитектуры с вычислительными трюками, Generative Adversarial Net, ResNet, Inception Network]

<br>


**12. Overview**

&#10230; Обзор

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Архитектура классической CNN ― Сверточные нейронные сети, также известные как Convolutional neural networks или CNNs, представляют собой особый тип нейронных сетей, которые обычно состоят из следующих слоев:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; Слой свертки и слой пулинга можно настроить с учетом гиперпараметров, которые описаны в следующих разделах.

<br>


**15. Types of layer**

&#10230; Типы слоёв

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Слой Свертки ― Convolution layer (CONV) использует фильтры, которые выполняют операции свертки при сканировании входа I относительно его размеров. Его гиперпараметры включают размер фильтра F и шаг S. Полученный результат O называется картой признаков или картой активации.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Примечание: шаг свертки также может быть обобщен на одномерные и трехмерные случаи.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Слой Пулинга ― Pooling (POOL) - это операция понижающей дискретизации, обычно применяемая после сверточного слоя, который обеспечивает пространственную инвариантность изображенных объектов. В частности, max-пулинг и усредненный пулинг - это особые виды пулинга, в которых берется максимальное и среднее значение соответственно.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Тип, Цель, Иллюстрация, Комментарии]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Max-пулинг, Усредненный пулинг, Каждая операция пулинга выбирает максимальное значение текущего представления, Каждая операция пулинга усредняет значения текущего представления]

<br>


**21. [Preserves detected features, Most commonly used, Down samples feature map, Used in LeNet]**

&#10230; [Сохраняет обнаруженные функции, Наиболее часто используется, Уменьшает размерность карты признаков, Используется в LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Полносвязный ― Fully Connected (FC) слой работает на сглаженном входе, где каждый вход подключен ко всем нейронам. Уровни FC, если они присутствуют, обычно находятся ближе к концу архитектур CNN и могут использоваться для оптимизации целевых метрик, таких как оценки классов.

<br>


**23. Filter hyperparameters**

&#10230; Фильтр гиперпараметров

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; Слой свертки содержит фильтры, для которых важно знать значение его гиперпараметров.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Размеры фильтра ― Фильтр размера F×F, применяемый к входу, содержащему каналы C, представляет собой объём F×F×C, который выполняет свертки на входе размера I×I×C и создает карту выходных признаков (также называемую картой активации) размера O×O×1.

<br>


**26. Filter**

&#10230; Фильтр

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Примечание: применение K фильтров размера F×F приводит к выходной карте признаков размером O×O×K.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Шаг ― Stride - Для сверточной операции или операции пулинга шаг S обозначает количество пикселей, на которое окно перемещается после каждой операции.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; Дополнение нулями ― Zero-padding означает процесс добавления P нулей к каждой стороне входного изображения. Это значение можно указать вручную или автоматически в одном из трех режимов, описанных ниже:

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Режим, Значение, Иллюстрация, Цель, Действительный, Такой же, Полный]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Без дополнения, Отбрасывает последнюю свертку при несовпадении размеров, Дополнения для карты признаков с размерами ⌈IS⌉, Размер вывода математически удобен, Также называется 'половинным' дополнением, Максимальное дополнение. С ним концевые свертки применяются к границам входа, Фильтр 'видит' вход от начала до конца]

<br>


**32. Tuning hyperparameters**

&#10230; Настройка гиперпараметров

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Совместимость параметров в сверточном слое ― Обозначим I длину входного размера объёма, F длину фильтра, P длину дополнения нулями, S шаг, затем выходной размер O карты признаков по этому измерению определяется как:

<br>


**34. [Input, Filter, Output]**

&#10230; [Вход, Фильтр, Выход]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Примечание: часто Pstart=Pend≜P, и в этом случае мы можем заменить Pstart+Pend на 2P в формуле выше.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Понимание сложности модели ― Чтобы оценить сложность модели, часто бывает полезно определить количество параметров, которые будет иметь её архитектура. В данном слое сверточной нейронной сети это делается следующим образом:

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Иллюстрация, Входной размер, Выходной размер, Количество параметров, Примечания]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [Один параметр смещения на фильтр, В большинстве случаев, S<F, Обычный выбор для K - это 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [Операция пулинга выполняется поканально, В большинстве случаев, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [Ввод сглаживается, Один параметр смещения на нейрон, Количество нейронов FC не имеет структурных ограничений]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Рецептивное поле ― Воспринимающее поле в слое k - это область, обозначенная Rk×Rk входа, которую может "видеть" каждый пиксель k-й карты активации. Называя Fj размером фильтра слоя j, а Si значением шага слоя i, и, согласно соглашению S0=1, рецептивное поле на слое k можно вычислить по формуле:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; В приведенном ниже примере у нас есть F1=F2=3 и S1=S2=1, который дает R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Часто используемые функции активации

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Блок линейной ректификации ― Rectified Linear Unit layer (ReLU) - это функция активации g, которая используется для всех элементов объёма. Он направлен на привнесение в сеть нелинейностей. Его варианты приведены в таблице ниже:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, ReLU с утечкой, ELU, с]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Сложности нелинейности поддаются биологической интерпретации, Решает проблему зануления ReLU отрицательных значений, Дифференцируема везде]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax ― Шаг softmax можно рассматривать как обобщенную логистическую функцию, которая принимает на вход вектор оценок x∈Rn и выводит вектор вероятностей классов p∈Rn через функцию softmax в конце архитектуры. Это определяется следующим образом:

<br>


**48. where**

&#10230; где

<br>


**49. Object detection**

&#10230; Обнаружение объектов

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Виды моделей ― Существует 3 основных типа алгоритмов распознавания объектов, для которых характер предсказаний различен. Они описаны в таблице ниже:

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Классификация изображений, Классификация с локализацией, Обнаружение]

<br>


**52. [Teddy bear, Book]**

&#10230; [Плюшевый мишка, Книга]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Классифицирует картинку, Прогнозирует вероятность объекта, Обнаруживает объект на картинке, Предсказывает вероятность объекта и его местонахождение, Обнаруживает до нескольких объектов на картинке, Прогнозирует вероятности появления объектов и их местонахождение]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [Классическая CNN, Упрощенный YOLO, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Обнаружение ― В контексте обнаружения объекта используются разные методы в зависимости от того, хотим ли мы просто найти объект или обнаружить более сложную форму на изображении (ориентир, Landmark, Reference point). Два основных из них суммированы в таблице ниже:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Обнаружение ограничивающей рамки, Обнаружение ориентира]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Обнаруживает часть изображения c объектом, Обнаруживает форму или характеристики объекта (например: глаза), Более детализировано]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Рамка с центром (bx,by), высота bh и ширина bw, Ориентиры (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Пересечение по объединению ― Intersection over Union (IoU) - это функция, которая количественно определяет, насколько правильно расположена предсказанная ограничивающая рамка Bp над фактической ограничительной рамкой Ba. Она определяется как:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Примечание: у нас всегда есть IoU∈[0,1]. По соглашению, прогнозируемая ограничивающая рамка Bp считается достаточно хорошей, если IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Якорные рамки ― Anchor box - это метод, используемый для прогнозирования перекрывающихся ограничивающих рамок. На практике сети позволяют прогнозировать более одной рамки одновременно, причем каждое предсказание рамки ограничивается заданным набором геометрических свойств. Например, первый прогноз потенциально может быть прямоугольной рамкой заданной формы, а второй будет другой прямоугольной рамкой с другими параметрами.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Подавление немаксимума ― Non-max suppression - Техника подавления немаксимумов направлена на удаление дублирующих перекрывающихся ограничивающих рамок одного и того же объекта путем выбора наиболее репрезентативных рамок. После удаления всех рамок, имеющих прогноз вероятности ниже 0.6, следующие шаги повторяются до тех пор, пока остаются рамки:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Для данного класса, Шаг 1: Выберите рамку с наибольшей вероятностью прогноза., Шаг 2: Отбросьте любую рамку с IoU⩾0.5 по сравнению с предыдущей рамкой.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Предсказание рамок, Выбор рамки с максимальной вероятностью, Удаление перекрытий того же класса, Окончательные ограничивающие рамки]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO ― You Only Look Once (YOLO) - это алгоритм обнаружения объектов, который выполняет следующие шаги:

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Шаг 1: Разделить входное изображение на сетку G×G., Шаг 2: Для каждой ячейки сетки запустите CNN, которая предсказывает y в следующей форме:, повторить k раз]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; где pc - вероятность обнаружения объекта, bx,by,bh,bw - свойства обнаруженной ограничивающий рамки, c1,...,cp - one-hot представление того, какой из p классов был обнаружен, а k - количество якорных рамок.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Шаг 3: Запустить алгоритм подавления немаксимальных значений, чтобы удалить любые потенциально повторяющиеся перекрывающиеся ограничивающие рамки.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Оригинальное изображение, Деление на сетку GxG, Предсказание ограничивающей рамки, Подавление немаксимумов]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Примечание: когда pc=0, сеть не обнаруживает никаких объектов. В этом случае соответствующие прогнозы bx,...,cp следует игнорировать.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN ― Region with Convolutional Neural Networks (R-CNN) - это алгоритм обнаружения объектов, который сначала сегментирует изображение, чтобы найти потенциально релевантные ограничивающие рамки, а затем запускает алгоритм обнаружения, чтобы найти наиболее вероятные объекты в этих ограничивающих рамках.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Оригинальное изображение, Сегментация, Прогнозирование ограничивающей рамки, Подавление немаксимумов]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; Примечание: хотя исходный алгоритм является дорогостоящим и медленным в вычислительном отношении, новые архитектуры позволили алгоритму работать быстрее, например Fast R-CNN и Faster R-CNN.

<br>


**74. Face verification and recognition**

&#10230; Проверка и распознавание лиц

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Типы моделей ― В таблице ниже приведены два основных типа моделей:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Проверка лица, Распознавание лиц, Запрос, Рекомендация, База данных]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [Это правильный человек?, Поиск one-to-one, Это одно из K лиц в базе данных?, Поиск one-to-many]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; Обучение с первого раза ― One Shot Learning - это алгоритм проверки лица, который использует ограниченный обучающий набор для изучения функции сходства, которая количественно определяет, насколько разные два заданных изображения. Функция подобия, применяемая к двум изображениям, часто обозначается d(image 1,image 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Сиамские сети ― Siamese Network стремятся научиться кодировать изображения, чтобы затем количественно оценить, насколько два изображения отличаются друг от друга. Для заданного входного изображения x(i) закодированный вывод часто обозначается как f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Потеря тройки ― Triplet loss ℓ - это функция потерь, вычисленная на представлении тройки изображений A (anchor), P (positive) и N (negative). Якорь (anchor) и положительный пример относятся к одному классу, а отрицательный пример - к другому. Называя α∈R+ параметром отступа, этот потеря определяется следующим образом:

<br>


**81. Neural style transfer**

&#10230; Нейронный перенос стиля

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Мотивация ― цель нейронного переноса стиля - создать изображение G на основе заданного контента C и заданного стиля S.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [Контент C, Стиль S, Сгенерированное изображение G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Активация ― В данном слое l активация обозначена a[l] и имеет размеры nH×nw×nc

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; Функция стоимости контента ― Функция стоимости контента Jcontent(C,G) используется для определения того, как сгенерированное изображение G отличается от исходного изображения C контента. Оно определяется следующим образом:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; Матрица стиля ― Матрица стиля G[l] данного слоя l является определителем Грама, где каждый из его элементов G[l]kk′ количественно определяет степень корреляции каналов k и k′. Она определяется по отношению к активациям a[l] следующим образом:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Примечание: матрица стиля для изображения стиля и сгенерированное изображение помечаются G[l] (S) и G[l] (G) соответственно.

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Функция стоимости стиля ― функция стоимости стиля Jstyle(S,G) используется для определения того, как сгенерированное изображение G отличается от стиля S. Он определяется следующим образом:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Функция общей стоимости ― функция общей стоимости определяется как комбинация функций стоимости контента и стиля, взвешенных параметрами α,β следующим образом:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Примечание: более высокое значение α заставит модель больше заботиться о контенте, а более высокое значение β заставит её больше заботиться о стиле.

<br>


**91. Architectures using computational tricks**

&#10230; Архитектуры с использованием вычислительных трюков

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Генеративные состязательные сети ― Generative adversarial networks, также известные как GANs, состоят из генеративной и дискриминативной моделей, где генеративная модель направлена на генерирование наиболее правдивого вывода, который будет передан дискриминативной модели, направленной на различение созданного и истинного изображения.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [Обучение, Шум, Реальное изображение, Генератор, Дискриминатор, Настоящее Подделка]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; Примечание: варианты использования с вариантами GAN включают текст в изображение, создание музыки и синтез.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; Residual Network architecture ― (также называется ResNet) использует остаточные блоки с большим количеством слоев, чтобы уменьшить ошибку обучения. Остаточный блок имеет следующее характеристическое уравнение:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception Network ― Эта архитектура использует начальные модули и нацелена на то, чтобы попробовать различные свертки, тем самым повышая распознавательную способность за счет комбинации карт признаков различных масштабов. В частности, она использует трюк свертки 1×1 для ограничения вычислительной нагрузки.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Шпаргалки по глубокому обучению теперь доступны на русском языке.

<br>


**98. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>


**99. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>


**100. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>


**101. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**102. By X and Y**

&#10230; По X и Y

<br>
