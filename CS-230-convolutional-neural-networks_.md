**Convolutional Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
Redes Neuronales Convolucionales
<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; Hoja de referencia de Redes Neuronales Convolucionales


<br>

**2. CS 230 - Deep Learning**

&#10230; CS - 230 Aprendizaje Profundo

<br>


**3. [Overview, Architecture structure]**

&#10230; [Resumen, Arquitectura de una Red Neuronal Convolucional tradicional] 
<br>
**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230;[Tipos de Capas, Convolucion, Pooling y Totalmente conectada]
    
<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230;[Hiperparámetros de filtro, Dimensiones, Tamaño de paso, Ceros de relleno ]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [Afinando hiperparámetros, Compatibilidad de Parámetros, Complejidad del modelo, Campo receptivo]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230;[Funciones de activación, Unidad Lineal Rectificada, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [Detección de objetos, Tipos de modelos, Detección, Intesección sobre la unión, supression non-max, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230;[Verificación facial/reconocimiento, aprendizaje de un solo intento, Red siamesa, Pérdida triple]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230;[Transferencia de estilo neuronal, Activación, Matriz de estilo, Función de costo contenido/estilo]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230;[Arquitecturas de trucos computacionales, Red Adversarial Generativa, ResNet, Red de inicio]

<br>


**12. Overview**

&#10230;[Resumen]

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Arquitectura de una Red Neuronal Convolucional tradicional – Las Redes Neuronales Convolucionales, también conocidas como CNN, por sus siglas en inglés, son un tipo especifico de red neuronal que generalmente esta compuesta de las siguientes capas:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; La capa de convolución y la capa de pooling pueden ser ajustadas con respecto a los hiperparámetros descritos en las siguientes secciones.

<br>


**15. Types of layer**

&#10230; Tipos de capas

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Capa convolucional (CONV) – La capa de Convolución (CONV) utiliza filtros que realizan operaciones de convolución conforme escanean la entrada I con respect a sus dimensiones. Sus hiperparámetros incluyen el tamaño del filtro F y el tamaño del paso S. La salida resultante O es llamada mapa de características o mapa de activación.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Nota: El paso de convolución puede ser generalizado al caso 1D y 3D.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Pooling(POOL) – La capa de pooling (POOL) es una operación de reducción, aplicada típicamente después de una capa de convolución, la cual realiza una invarianza especial. En particular, max pooling y avg pooling son tipos especiales de pooling en donde el máximo valor y el valor promedio son tomados, respectivamente.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230;[Tipo, Propósito, Explicación, Comentarios]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230;[Max pooling, Pooling promedio, Cada operación de pooling selecciona el valor máximo de la vista actual, Cada operación de pooling promedia los valores de la vista actual.]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230;[Conserva propiedades detectadas, Esta es la utilizada comunmente, Reduce el mapa de características, Utilizado en LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Totalmente Conectada (TC) – La capa totalmente conectada (TC) opera sobre una entrada aplanada en donde cada entrada está conectada a todos los neurones. Si las capas TC estan presente, usualmente se encuentran al final de la arquitectura RCN y pueden ser utilizadas para optimizar objetivos como la estimación de la categoría.

<br>


**23. Filter hyperparameters**

&#10230; Hiperparámetros de filtro

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; La capa de convolución contiene filtros para los cuales es importante saber el significado detrás de los hiperparámetros.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Dimensiones de un filtro – Un filtro de tamaño FxF aplicado a una entrada que contiene C canales genera un volumen FxFxC que opera convoluciones sobre una entrada de tamaño IxIxC y produce de salida un mapa de características (tambien llamado mapa de activación) de tamano OxOx1.

<br>


**26. Filter**

&#10230; Filtro

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Nota: La aplicación de K filtros de tamaño FxF produce un mapa de características de tamaño OxOxK.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Tamaño del paso – Para una operación de convolución o pooling, el tamaño del paso S denota el número de pixeles que la ventana se movera después de cada operación.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; Ceros de relleno – Ceros de relleno denota el proceso de agregar P ceros a cada lado de la frontera de la entrada. Este valor puede ser especificado ya sea manual o automaticamente por medio de una de las técnicas especificadas a continuacion.

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Modo, Valor, Explicación, Propósito, Válido, Mismo, Completo]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Sin relleno, Descarta la última convolución si las dimensiones no coinciden, Rellenar tal que el mapa de características tiene tamaño [IS], Tamaño de salida es matemáticamente conveniente. Tambien llamado medio relleno, Máximo relleno tal que las convoluciones del final son aplicadas en los límites de la entrada, El filtro ve la entrada de extremo a extremo.]

<br>


**32. Tuning hyperparameters**

&#10230; Afinando hiperparámetros

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Compatibilidad de hiperparámetros en la capa de convolución – Sea I la longitud del volumen de entrada, F la longitud del filtro, P la cantidad de zeros de relleno, S el tamaño del paso. Entonces el tamaño de la salida O del mapa de características a lo largo  de esa dimension esta dado por:

<br>


**34. [Input, Filter, Output]**

&#10230; [Entrada, Filtro, Salida]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Nota: Muchas veces Pinicio=Pfin<= P, en ese caso sustituimos Pinicio+Pfin=2P en la ecuación anterior.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Entendiendo la complejidad del modelo - Para evaluar la complejidad de un modelo es útil determinar el número de parámetros que tendra esta arquitectura. En una capa dada de una red neuronal convolucional, esto se hace de la siguient manera:

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Ilustración, Tamaño de entrada, Tamaño de salida, Número de parámetros, Observaciones]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230;[One parametro parcial por filtro. En la mayoria de los casos S<F. Una elección común de K es 2C]
 
<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [La operación de pooling se hace por el canal. En la mayoría de los casos, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [La entrada es aplanada. Un paramétro parcial por neurón. El número de neurones FC está libre de restricciones estructurales]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Campo Receptivo- El campo receptivo en la capa k es el área denotada por RkxRk de la entrada que cada pixel del K-ésimo mapa de activacion puede ver. Sea Fj el tamaño de filtro de la capa j y sea Si el tamaño del paso de la capa i-ésima y utilizando la convención S0=1, el campo receptor en la capa k puede ser calculado con la siguente fórmula:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; En el siguiente ejemplo F1=F2=3 y S1=S21=1, lo cual R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Funciones de activación comunmente utilizadas

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Unidad Lineal Rectificada- La capa de Unidad Lineal Rectificada (ReLU) es una función de activación g, que se utiliza en todos los elementos del volumen. Tiene como objetivo introducir no-linearidades a la red. Sus variantes son resumidas en la siguiente tabla:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, Leaky ReLU, ELU, con]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Complejidades de no-linealidad biológicamente interpretables, Aborda el problema de ReLu moribundo para valores negativos, Diferenciable en todas partes]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax – El paso softmax puede ser visto como una función f generalizada de logística que toma como entrada un vector de grado x∈Rn y da como salida un vector de probabilidad p∈Rn a traves de una función softmax al final de la arquitectura. Se define como:

<br>


**48. where**

&#10230; en donde

<br>


**49. Object detection**

&#10230; Detección de objetos

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Tipos de Modelos – Existen 3 tipos principales de algoritmos de reconocimiento de objetos, cada uno de ellos predice cosas diferentes. Se describen en la siguiente tabla.

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Clasificación de imagenes, Clasificación con localización, Deteccción]

<br>


**52. [Teddy bear, Book]**

&#10230; [Oso de peluche, Libro]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Clasifica una imagen, Predice la probabilidad del objeto, Detecta el objeto en una imagen, Predice la probabilidad de un objeto y en donde se encuentra localizado, Detecta varios objetos en una imagen, Predice las probabilidades de objetos y en donde se encuentran localizados.]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [CNN tradicional, YOLO simplificado, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Detección – En el contexto de detección de objetos, se utilizan diferentes métodos dependiendo de sí solamente deseamos localizar el objeto o si deseamos localizar una forma más compleja en la imagen. Los dos métodos principales son resumidos en la siguiente tabla:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Detección de cuadro delimitador, Detección de puntos de referencia]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Detecta la parte de la imagen en donde el objeto está localizado, Detecta la forma o características de un objeto (p.e. ojos), Más granular]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Cuadro con centro (bz,by), altura bh y ancho bw, Puntos de referencia (l1x,l1y),...,(lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; ntersección sobre la unión – Intersección sobre la unión, también conocida como IoU, es una función que cuantifica que tan bien se posicionó la caja delimitadora predicha Bp sobre la caja delimitadora actual Ba. Es definida a continuación:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Nota: Siempre se tiene que IoU∈[0,1]. Por convención, una caja delimitadora predicha Bp es considerada razonablemente buena si IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Cajas de anclaje – Las cajas de anclaje son una técnica utilizada para predecir cajas delimitadoras que se superponen/traslapan. En la práctica, se permite que la red prediga mas de una caja simultáneamente, en donde cada predicción de caja es restringida a ciertas propiedades geométricas. Por ejemplo, la primer predicción podría ser un rectángulo mientras que la segunda predicción podría ser otra caja rectangular de una forma geométrica diferente.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Supresión de non-max – La técnica de supressión de non-max tiene como objetivo remover las cajas delimitadoras duplicadas de un mismo objeto seleccionando las mas representativas. Despues de remover todas las cajas con una predicción de probabilidad menor a 0.6, se repiten los siguientes pasos mientras queden cajas:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Para una clase dada, Paso 1 : Escoja la caja con la predicción de probabilidad mayor., Paso 2 : Descarte cualquier caja que tenga una IoU⩾0.5  con la caja anterior.]


<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Predicciones de cuadro, selección de cuadro de máxima probabilidad, eliminación de superposición de la misma clase, cuadros delimitadores finales]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230;     • YOLO – You Only Look Once (Sólo miras una vez) es un algoritmo de detección de objetos que realiza los siguientes pasos:
    
<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Paso 1: Divide la imagen de entrada en una red de dimensiones GxG., Paso2 : Para cada célula de la red, crea una RCN para predecir y de la siguiente forma:, repetido k veces]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; en donde Pc es la probabilidad de detectar un objeto, bx,by,bn,bw son las propiedades de la caja delimitadora predicha. c1, …, cp es una representación one-hot de la cual p clases fueron detectadas y k es el número de cajas de anclaje.
    
<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Paso 3: Aplique el algoritmo de suppressión non-max para remover cualquier caja delimitadora duplicada que se traslape.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Imagen original, División en la cuadrícula GxG, Predicción de la caja delimitadora, supresion non-max]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Nota: Si pc =0 entonces la red no detecto ningun objeto. En tal caso, las correcciones correspondientes bx,…,cp deben ser ignoradas.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN – La Red Neuronal Convolucional con Región (Region-Convolutional Neural Net, R-CNN) es un algoritmo que primero segmenta a la imagen para encontrar posibles cajas delimitadoras reelevantes y posteriormente aplica el algoritmo de deteccción para encontrar los objetos más probables en esas cajas delimitadoras.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Imagen original, Segmentación, Predicción de la caja delimitadora, Supresión del non-max]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; a pesar de que el algoritmo original es computacionalmente costoso y lento, las arquitecturas más nuevas permiten que el algoritmo sea más rápido, por ejemplo RCNN rápida y RCNN más rápida.

<br>


**74. Face verification and recognition**

&#10230; Verificación facial y reconocimiento

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Tipos de modelos - Los dos tipos de modelos principales son resumidos en la siguiente tabla:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Verificación facial, Reconocimiento facial, Consulta, Referencia, Base de datos]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230;[¿Es esta la persona correcta?, Búsqueda uno a uno, ¿Es esta una de las K personas en la base de datos?, Búsqueda uno a muchos]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; Aprendizaje One-Shot: El aprendizaje One shot es un algoritmo de verificacion de cara que utiliza un conjunto de entrenamiento limitado para aprender una función de similaridad que cuantifíca las diferencias entre dos imágenes. La función de similaridad aplicada a 2 imagenes es comunmente denotada por d(imagen 1, imagen 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Red Siamesa: Las redes siamesas tienen como objetivo aprender a codificar imagenes para despues cuantificar qué tan disímiles son. Una imagen de entrada se denota como x(i), la salida codificada se denota como f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Perdida triple: La perdida triple es una función de perdida calculada en la representación anidada de la tercia de imagenes A(ancla), P(positivo) y N(negativo). Los ejemplos ancla y negativo pertenecen a la misma clase, mientras que el ejemplo positivo pertenece a una clase diferente. Sea alpha en R el parámetro de margen tal que la función de pérdida es definida como:
    
<br>


**81. Neural style transfer**

&#10230; Tansferencia estilo neural

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Motivación- El objetivo de la transferencia estilo neural es generar una imagen G basada en el contenido dado C y el estilo S.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [Contenido C, Estilo S, Imagen generada G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Activación: Sea l una capa dada, la función de activación es denotada como a[l] y sus dimensiones son nHxnWxnC

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; Función de costo-contenido: La función de costo-contenido Jcontenido(C,G) se utiliza para determinar qué tanto difiere la imagen generada G de la imagen de contenido original C. Se define como:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; La matriz de estílo - La matriz de estilo G[l] de una capa dada l es la matriz de Gram en donde cada uno de sus elementos G[l]kk′ cuantifica la relación entre los canales k y k’. Se define con respecto a las activaciones a[l]:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Nota: la matriz de estilo para la imagen de estilo y para la imagen generada son denotadas  como G[l] (S)  y G[l](G) respectivamente. 

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Función de costo-estílo: La función de costo-estílo Jestilo(S,G) se utiliza para estimar que tanto difiere la imagen generada G de la imagen estilo S. Se define como:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Función de costo total: La función de costo total es definida como una combinación de la función de costo-contenido y la función de costo-estilo ponderadas por los parámetros α,β,:			

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Nota: Un valor mayor de α hará que el modelo se vea impactado por el contenido , mientras que un valor mayor de β hará que el modelo se vea más afectado por el estilo.

<br>


**91. Architectures using computational tricks**

&#10230; Arquitecturas utilizando trucos computacionales

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Redes Adversarias Generativas - Las Redes Adversarias Generativas, tambien conocidas como GANs por sus siglas en inglés, están compuestas por un modelo generativo y por un modelo discriminativo, en donde el objetivo del *modelo generativo es generar la salida más exacta que se utilizará como entrada del modelo discriminativo, el cual tiene como objetivo diferenciar entre la imagen generada y la imagen real.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230;

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230;

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet - La arquitectura de Red Residual (también llamada ResNet por sus siglas en inglés) utiliza bloques residuales con una gran cantidad de capas destinadas a disminuir el error de entrenamiento. El bloque residual tiene la siguiente ecuación característica:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Red de inicio - Esta arquitectura utiliza módulos de inicio y tiene como objetivo probar diferentes circunvoluciones para aumentar su rendimiento a través de la diversificación de características. En particular, utiliza el truco de convolución 1×1 para limitar la carga computacional.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Las hojas de referencia de Deep Learning ahora están disponibles en [español].

<br>


**98. Original authors**

&#10230; Autores originales

<br>


**99. Translated by X, Y and Z**

&#10230; Traducida por Erika Muñoz Torres y José Nandez

<br>


**100. Reviewed by X, Y and Z**

&#10230; Revisada por X, Y y Z

<br>


**101. View PDF version on GitHub**

&#10230; Ver versión PDF en GitHub

<br>


**102. By X and Y**

&#10230; Por Afshine Amidi y Shervine AmidiX

<br>
