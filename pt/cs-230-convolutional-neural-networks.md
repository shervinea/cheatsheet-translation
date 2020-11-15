**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; Dicas de Redes Neurais Convolucionais

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Aprendizagem profunda

<br>


**3. [Overview, Architecture structure]**

&#10230; [Visão geral, Estrutura arquitetural]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [Tipos de camadas, Convolução, Pooling, Totalmente conectada]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [Hiperparâmetros de filtro, Dimensões, Passo, Preenchimento]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230;[Ajustando hiperparâmetros, Compatibilidade de parâmetros, Complexidade de modelo, Campo receptivo]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [Funções de Ativação, Unidade Linear Retificada, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230;[Detecção de objetos, Tipos de modelos, Detecção, Intersecção por União, Supressão não-máxima, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [Verificação / reconhecimento facial, Aprendizado de disparo único, Rede siamesa, Perda tripla]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [Transferência de estilo neural, Ativação, Matriz de estilo, Função de custo de estilo/conteúdo]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [Arquiteturas de truques computacionais, Rede Adversarial Generativa, ResNet, Rede de Iniciação]

<br>


**12. Overview**

&#10230; Visão geral

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Arquitetura de uma RNC tradicional (CNN) - Redes neurais convolucionais, também conhecidas como CNN (em inglês), são tipos específicos de redes neurais que geralmente são compostas pelas seguintes camadas:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; A camada convolucional e a camadas de pooling podem ter um ajuste fino considerando os hiperparâmetros que estão descritos nas próximas seções. 

<br>


**15. Types of layer**

&#10230; Tipos de camadas

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Camada convolucional (CONV) - A camada convolucional (CONV) usa filtros que realizam operações de convolução conforme eles escaneiam a entrada I com relação a suas dimensões. Seus hiperparâmetros incluem o tamanho do filtro F e o passo S. O resultado O é chamado de mapa de recursos (feature map) ou mapa de ativação.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Observação: o passo de convolução também pode ser generalizado para os casos 1D e 3D.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Pooling (POOL) - A camada de pooling (POOL) é uma operação de amostragem (downsampling), tipicamente aplicada depois de uma camada convolucional, que faz alguma invariância espacial. Em particular, pooling máximo e médio são casos especiais de pooling onde o máximo e o médio valor são obtidos, respectivamente.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Tipo, Propósito, Ilustração, Comentários]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Pooling máximo, Pooling médio, Cada operação de pooling seleciona o valor máximo da exibição atual, Cada operação de pooling calcula a média dos valores da exibição atual]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [Preserva os recursos detectados, Mais comumente usados, Mapa de recursos de amostragem (downsample), Usado no LeNet]


<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Totalmente Conectado (FC) - A camada totalmente conectada (FC opera em uma entrada achatada, onde cada entrada é conectada a todos os neurônios. Se estiver presente, as camadas FC geralmente são encontradas no final das arquiteturas da CNN e podem ser usadas para otimizar objetivos, como pontuações de classes.

<br>


**23. Filter hyperparameters**

&#10230; Hiperparâmetros de filtros

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; A camada de convolução contém filtros para os quais é importante conhecer o significado por trás de seus hiperparâmetros.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Dimensões de um filtro - Um filtro de tamanho F×F aplicado a uma entrada contendo C canais é um volume de tamanho F×F×C que executa convoluções em uma entrada de tamanho I×I×C e produz um mapa de recursos (também chamado de mapa de ativação) da saída de tamanho O×O×1.

<br>


**26. Filter**

&#10230; Filtros

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Observação: a aplicação de K filtros de tamanho F×F resulta em um mapa de recursos de saída de tamanho O×O×K.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Passo - Para uma operação convolucional ou de pooling, o passo S denota o número de pixels que a janela se move após cada operação.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; Zero preenchimento (Zero-padding) - Zero preenchimento denota o processo de adicionar P zeros em cada lado das fronteiras de entrada. Esse valor pode ser especificado manualmente ou automaticamente ajustado através de um dos três modelos abaixo:

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Modo, Valor, Ilustração, Propósito, Válido, Idêntico, Completo]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Sem preenchimento, Descarta a última convolução se as dimensões não corresponderem, Preenchimento de tal forma que o tamanho do mapa de recursos tenha tamanho ⌈IS⌉, Tamanho da saída é matematicamente conveniente, Também chamado de 'meio' preenchimento, Preenchimento máximo de tal forma que convoluções finais são aplicadas nos limites de a entrada, Filtro 'vê' a entrada de ponta a ponta]

<br>


**32. Tuning hyperparameters**

&#10230; Ajuste de hiperparâmetros

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Compatibilidade de parâmetro na camada convolucional - Considerando I o comprimento do tamanho do volume da entrada, F o tamanho do filtro, P a quantidade de preenchimento de zero (zero-padding) e S o tamanho do passo, então o tamanho de saída O do mapa de recursos ao longo dessa dimensão é dado por:


<br>


**34. [Input, Filter, Output]**

&#10230; [Entrada, Filtro, Saída]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Observação: diversas vezes, Pstart=Pend≜P, em cujo caso podemos substituir Pstart+Pen por 2P na fórmula acima.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Entendendo a complexidade do modelo - Para avaliar a complexidade de um modelo, é geralmente útil determinar o número de parâmetros que a arquitetura deverá ter. Em uma determinada camada de uma rede neural convolucional, ela é dada da seguinte forma: 

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Ilustração, Tamanho da entrada, Tamanho da saída, Número de parâmetros, Observações]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [Um parâmetro de viés (bias parameter) por filtro, Na maioria dos casos, S<F, Uma escolha comum para K é 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [Operação de pooling feita pelo canal, Na maior parte dos casos, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [Entrada é achatada, Um parâmetro de viés (bias parameter) por neurônio, O número de neurônios FC está livre de restrições estruturais]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Campo receptivo - O campo receptivo na camada k é a área denotada por Rk×Rk da entrada que cada pixel do k-ésimo mapa de ativação pode 'ver'. Ao chamar Fj o tamanho do filtro da camada j e Si o valor do passo da camada i e com a convenção S0=1, o campo receptivo na camada k pode ser calculado com a fórmula:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; No exemplo abaixo, temos que F1=F2=3 e S1=S2=1, o que resulta em R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Funções de ativação comumente usadas

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Unidade Linear Retificada (Rectified Linear Unit) - A camada unitária linear retificada (ReLU) é uma função de ativação g que é usada em todos os elementos do volume. Tem como objetivo introduzir não linearidades na rede. Suas variantes estão resumidas na tabela abaixo:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, Leaky ReLU, ELU, com]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Complexidades de não-linearidade biologicamente interpretáveis, Endereça o problema da ReLU para valores negativos, Diferenciável em todos os lugares]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax - O passo de softmax pode ser visto como uma função logística generalizada que pega como entrada um vetor de pontuações x∈Rn e retorna um vetor de probabilidades p∈Rn através de uma função softmax no final da arquitetura. É definida como:

<br>


**48. where**

&#10230; onde

<br>


**49. Object detection**

&#10230; Detecção de objeto

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Tipos de modelos - Existem 3 tipos de algoritmos de reconhecimento de objetos, para o qual a natureza do que é previsto é diferente para cada um. Eles estão descritos na tabela abaixo:

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Classificação de imagem, Classificação com localização, Detecção]

<br>


**52. [Teddy bear, Book]**

&#10230; [Urso de pelúcia, Livro]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Classifica uma imagem, Prevê a probabilidade de um objeto, Detecta um objeto em uma imagem, Prevê a probabilidade de objeto e onde ele está localizado, Detecta vários objetos em uma imagem, Prevê probabilidades de objetos e onde eles estão localizados]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [CNN tradicional, YOLO simplificado, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Detecção - No contexto da detecção de objetos, diferentes métodos são usados dependendo se apenas queremos localizar o objeto ou detectar uma forma mais complexa na imagem. Os dois principais são resumidos na tabela abaixo:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Detecção de caixa limite, Detecção de marco]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Detecta parte da imagem onde o objeto está localizado, Detecta a forma ou característica de um objeto (e.g. olhos), Mais granular]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Caixa central (bx,by), altura bh e largura bw, Pontos de referência (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Interseção sobre União (Intersection over Union) - Interseção sobre União, também conhecida como IoU, é uma função que quantifica quão corretamente posicionado uma caixa de delimitação predita Bp está sobre a caixa de delimitação real Ba. É definida por:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Observação: temos que IoU∈[0,1]. Por convenção, uma caixa de delimitação predita Bp é considerada razoavelmente boa se IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Caixas de ancoragem (Anchor boxes) - Caixas de ancoragem é uma técnica usada para predizer caixas de delimitação que se sobrepõem. Na prática, a rede tem permissão para predizer mais de uma caixa simultaneamente, onde cada caixa prevista é restrita a ter um dado conjunto de propriedades geométricas. Por exemplo, a primeira predição pode ser potencialmente uma caixa retangular de uma determinada forma, enquanto a segunda pode ser outra caixa retangular de uma forma geométrica diferente.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Supressão não máxima (Non-max suppression) - A técnica supressão não máxima visa remover caixas de delimitação de um mesmo objeto que estão duplicadas e se sobrepõem, selecionando as mais representativas. Depois de ter removido todas as caixas que contém uma predição menor que 0.6. os seguintes passos são repetidos enquanto existem caixas remanescentes:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Para uma dada classe, Passo 1: Pegue a caixa com a maior predição de probabilidade., Passo 2: Descarte todas as caixas que tem IoU⩾0.5 com a caixa anterior.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Predição de caixa, Seleção de caixa com máxima probabilidade, Remoção de sobreposições da mesma classe, Caixas de delimitação final]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO - Você Apenas Vê Uma Vez (You Only Look Once - YOLO) é um algoritmo de detecção de objeto que realiza os seguintes passos:

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Passo 1: Divide a imagem de entrada em uma grade G×G., Passo 2: Para cada célula da grade, roda uma CNN que prevê o valor y da seguinte forma:, repita k vezes]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; onde pc é a probabilidade de detecção do objeto, bx,by,bh,bw são as propriedades das caixas delimitadoras detectadas, c1,...,cp é uma representação única (one-hot representation) de quais das classes p foram detectadas, e k é o número de caixas de ancoragem.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Passo 3:  Rode o algoritmo de supressão não máximo para remover qualquer caixa delimitadora duplicada e que se sobrepõe.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Imagem original, Divisão em uma grade GxG, Caixa delimitadora prevista, Supressão não máxima]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Observação: Quando pc=0, então a rede não detecta nenhum objeto. Nesse caso, as predições correspondentes bx,...,cp devem ser ignoradas.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN - Região com Redes Neurais Convolucionais (R-CNN) é um algoritmo de detecção de objetos que primeiro segmenta a imagem para encontrar potenciais caixas de delimitação relevantes e então roda o algoritmo de detecção para encontrar os objetos mais prováveis dentro das caixas de delimitação.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Imagem original, Segmentação, Predição da caixa delimitadora, Supressão não-máxima]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; Observação: embora o algoritmo original seja computacionalmente caro e lento, arquiteturas mais recentes, como o Fast R-CNN e o Faster R-CNN, permitiram que o algoritmo fosse executado mais rapidamente.

<br>


**74. Face verification and recognition**

&#10230; Verificação facial e reconhecimento

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Tipos de modelos - Os dois principais tipos de modelos são resumidos na tabela abaixo:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Verificação facial, Reconhecimento facial, Consulta, Referência, Banco de dados]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [Esta é a pessoa correta?, Pesquisa um-para-um, Esta é uma das K pessoas no banco de dados?, Pesquisa um-para-muitos]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; Aprendizado de Disparo Único (One Shot Learning) - One Shot Learning é um algoritmo de verificação facial que utiliza um conjunto de treinamento limitado para aprender uma função de similaridade que quantifica o quão diferentes são as duas imagens. A função de similaridade aplicada a duas imagens é frequentemente denotada como  d(imagem 1, imagem 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Rede Siamesa (Siamese Network) - Siamese Networks buscam aprender como codificar imagens para depois quantificar quão diferentes são as duas imagens. Para uma imagem de entrada x(i), o resultado codificado é normalmente denotado como f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Perda tripla (Triplet loss) - A perda tripla ℓ é uma função de perda (loss function) computada na representação da encorporação de três imagens A (âncora), P (positiva) e N (negativa). O exemplo da âncora e positivo pertencem à mesma classe, enquanto o exemplo negativo pertence a uma classe diferente. Chamando o parâmetro de margem de α∈R+, essa função de perda é definida da seguinte forma:

<br>


**81. Neural style transfer**

&#10230; Transferência de estilo neural

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Motivação - O objetivo da transferência de estilo neural é gerar uma imagem G baseada num dado conteúdo C com um estilo S. 

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [Conteúdo C, Estulo S, Imagem gerada G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Ativação - Em uma dada camada l, a ativação é denotada como a[l] e suas dimensões são nH×nw×nc

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; Função de custo de conteúdo (Content cost function) - A função de custo de conteúdo Jcontent(C,G) é usada para determinar como a imagem gerada G difere da imagem de conteúdo original C. Ela é definida da seguinte forma:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; Matriz de estilo - A matriz de estilo G[l] de uma determinada camada l é a matriz de Gram em que cada um dos seus elementos G[l]kk′ quantificam quão correlacionados são os canais k e k′. Ela é definida com respeito às ativações a[l] da seguinte forma:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Observação: a matriz de estilo para a imagem estilizada e para a imagem gerada são denotadas como G[l] (S) e G[l] (G), respectivamente.

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Função de custo de estilo (Style cost function) - A função de custo de estilo Jstyle(S,G) é usada para determinar como a imagem gerada G difere do estilo S. Ela é definida da seguinte forma:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Função de custo geral (Overall cost function) é definida como sendo a combinação das funções de custo do conteúdo e do estilo, ponderada pelos parâmetros α,β, como mostrado abaixo:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Observação: um valor de α maior irá fazer com que o modelo se preocupe mais com o conteúdo enquanto um maior valor de β irá fazer com que ele se preocupe mais com o estilo.

<br>


**91. Architectures using computational tricks**

&#10230; Arquiteturas usando truques computacionais

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Rede Adversarial Gerativa (Generative Adversarial Network) - As Generaive Adversarial Networks, também conhecidas como GANs, são compostas de um modelo generativo e um modelo discriminativo, onde o modelo generativo visa gerar a saída mais verdadeira que será alimentada na discriminativa que visa diferenciar a imagem gerada e a imagem verdadeira.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [Treinamento, Ruído, Imagem real, Gerador, Discriminador, Falsa real]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; Observação: casos de uso usando variações de GANs incluem texto para imagem, geração de música e síntese.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet - A arquitetura de Rede Residual (também chamada de ResNet) usa blocos residuais com um alto número de camadas para diminuir o erro de treinamento. O bloco residual possui a seguinte equação caracterizadora:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Rede de Iniciação - Esta arquitetura utiliza módulos de iniciação e visa experimentar diferentes convoluções, a fim de aumentar seu desempenho através da diversificação de recursos. Em particular, ele usa o truque de convolução 1×1 para limitar a carga computacional.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Os resumos de Aprendizagem Profunda estão disponíveis em português.

<br>


**98. Original authors**

&#10230; Autores Originais

<br>


**99. Translated by X, Y and Z**

&#10230; Traduzido por Leticia Portella

<br>


**100. Reviewed by X, Y and Z**

&#10230; Revisado por Gabriel Fonseca

<br>


**101. View PDF version on GitHub**

&#10230; Ver versão em PDF no GitHub.

<br>


**102. By X and Y**

&#10230; Por X e Y

<br>
