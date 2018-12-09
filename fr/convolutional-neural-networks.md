**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230; Pense-bête de réseaux de neurones convolutionnels

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Apprentissage profond

<br>


**3. [Overview, Architecture structure]**

&#10230; [Vue d'ensemble, Structure de l'architecture]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [Types de couche, Convolution, Pooling, Fully connected]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [Paramètres du filtre, Dimensions, Stride, Padding]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [Réglage des paramètres, Compatibilité des paramètres, Complexité du modèle, Champ récepteur]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [Fonction d'activation, Unité linéaire rectifiée, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [Détection d'objet, Types de modèle, Détection, Intersection sur union, Suppression non-max, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [Vérification/reconnaissance de visage, Apprentissage par coup, Réseau siamois, Loss triple]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [Transfert de style de neurones, Activation, Matrice de style, Fonction de coût de style/contenu]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [Architectures à astuces calculatoires, Generative Adversarial Net, ResNet, Inception Network]

<br>


**12. Overview**

&#10230; Vue d'ensemble

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Architecture d'un CNN traditionnel ― Les réseaux de neurones convolutionnels (en anglais *Convolutional neural networks*), aussi connus sous le nom de CNNs, sont un type spécifique de réseaux de neurones qui sont généralement composés des couches suivantes :

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; La couche convolutionnelle et la couche de pooling peuvent être ajustées en utilisant des paramètres qui sont décrites dans les sections suivantes.

<br>


**15. Types of layer**

&#10230; Types de couche

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Couche convolutionnelle (CONV) ― La couche convolutionnelle (en anglais *convolution layer*) (CONV) utilise des filtres qui scannent l'entrée I suivant ses dimensions en effectuant des opérations de convolution. Elle peut être réglée en ajustant la taille du filtre F et le stride S. La sortie O de cette opération est appelée *feature map* ou aussi *activation map*.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Remarque : l'étape de convolution peut aussi être généralisée dans les cas 1D et 3D.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Pooling (POOL) ― La couche de pooling (en anglais *pooling layer*) (POOL) est une opération de sous-échantillonnage typiquement appliquée après une couche convolutionnelle. En particulier, les types de pooling les plus populaires sont le max et l'average pooling, où les valeurs maximales et moyennes sont prises, respectivement.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Type, But, Illustration, Commentaires]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Max pooling, Average pooling, Chaque opération de pooling sélectionne la valeur maximale de la surface. Chaque opération de pooling sélectionne la valeur moyenne de la surface.]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [Garde les caractéristiques détectées. Plus communément utilisé, Sous-échantillonne la *feature map*, Utilisé dans LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Fully Connected (FC) ― La couche de fully connected (en anglais *fully connected layer*) (FC) s'applique sur une entrée préalablement aplatie où chaque entrée est connectée à tous les neurones. Les couches de fully connected sont typiquement présentes à la fin des architectures de CNN et peuvent être utilisées pour optimiser des objectifs tels que les scores de classe.

<br>


**23. Filter hyperparameters**

&#10230; Paramètres du filtre 

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; La couche convolutionnelle contient des filtres pour lesquels il est important de savoir comment ajuster ses paramètres.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Dimensions d'un filtre ― Un filtre de taille F×F appliqué à une entrée contenant C canaux est un volume de taille F×F×C qui effectue des convolutions sur une entrée de taille I×I×C et qui produit un *feature map* de sortie (aussi appelé *activation map*) de taille O×O×1.

<br>


**26. Filter**

&#10230; Filtre

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Remarque : appliquer K filtres de taille F×F engendre un *feature map* de sortie de taille O×O×K.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Stride ― Dans le contexte d'une opération de convolution ou de pooling, la stride S est un paramètre qui dénote le nombre de pixels par lesquels la fenêtre se déplace après chaque opération.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; Zero-padding ― Le zero-padding est une technique consistant à ajouter P zeros à chaque côté des frontières de l'entrée. Cette valeur peut être spécifiée soit manuellement, soit automatiquement par le bias d'une des configurations détaillées ci-dessous :

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Configuration, Valeur, Illustration, But, Valide, Pareil, Total]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Pas de padding, Enlève la dernière opération de convolution si les dimensions ne collent pas, Le padding tel que la feature map est de taille ⌈IS⌉, La taille de sortie est mathématiquement satisfaisante, Aussi appelé 'demi' padding, Padding maximum tel que les dernières convolutions sont appliquées sur les bords de l'entrée, Le filtre 'voit' l'entrée du début à la fin]

<br>


**32. Tuning hyperparameters**

&#10230; Ajuster les paramètres

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Compatibilité des paramètres dans la couche convolutionnelle ― En notant I le côté du volume d'entrée, F la taille du filtre, P la quantité de zero-padding, S la stride, la taille O de la feature map de sortie suivant cette dimension est telle que :

<br>


**34. [Input, Filter, Output]**

&#10230; [Entrée, Filtre, Sortie]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Remarque : on a souvent Pstart=Pend≜P, auquel cas on remplace Pstart+Pend par 2P dans la formule au-dessus.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Comprendre la complexité du modèle ― Pour évaluer la complexité d'un modèle, il est souvent utile de déterminer le nombre de paramètres que l'architecture va avoir. Dans une couche donnée d'un réseau de neurones convolutionnels, on a :

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Illustration, Taille d'entrée, Taille de sortie, Nombre de paramètres, Remarques]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [Un paramètre de biais par filtre, Dans la plupart des cas, S<F, 2C est un choix commun pour K]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [L'opération de pooling est effectuée pour chaque canal, Dans la plupart des cas, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [L'entrée est aplatie, Un paramètre de bias par neurone, Le choix du nombre de neurones de FC est libre]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Champ récepteur ― Le champ récepteur à la couche k est la surface notée Rk×Rk de l'entrée que chaque pixel de la k-ième *activation map* peut 'voir'. En notant Fj la taille du filtre de la couche j et Si la valeur de stride de la couche i et avec la convention S0=1, le champ récepteur à la couche k peut être calculé de la manière suivante :

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; Dans l'exemple ci-dessous, on a F1=F2=3 et S1=S2=1, ce qui donne R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Fonctions d'activation communément utilisées

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Unité linéaire rectifiée ― La couche d'unité linéaire rectifiée (en anglais *rectified linear unit layer*) (ReLU) est une fonction d'activiation g qui est utilisée sur tous les éléments du volume. Elle a pour but d'introduire des complexités non-linéaires au réseau. Ses variantes sont récapitulées dans le tableau suivant :

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, Leaky ReLU, ELU, avec]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Complexités non-linéaires intereprétables d'un point de vue biologique, Repond au problème de *dying ReLU*, Dérivable partout]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; 

<br>


**48. where**

&#10230; 

<br>


**49. Object detection**

&#10230; 

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; 

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; 

<br>


**52. [Teddy bear, Book]**

&#10230; 

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; 

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; 

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; 

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; 

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; 

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; 

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; 

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; 

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; 

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; 

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; 

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; 

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; 

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; 

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; 

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; 

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; 

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; 

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; 

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; 

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; 

<br>


**74. Face verification and recognition**

&#10230; 

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; 

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; 

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; 

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; 

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; 

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; 

<br>


**81. Neural style transfer**

&#10230; 

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; 

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; 

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; 

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; 

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; 

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; 

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; 

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; 

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; 

<br>


**91. Architectures using computational tricks**

&#10230; 

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; 

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; 

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; 

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; 

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; 

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; 

<br>


**98. Original authors**

&#10230; 

<br>


**99. Translated by X, Y and Z**

&#10230; 

<br>


**100. Reviewed by X, Y and Z**

&#10230; 

<br>


**101. View PDF version on GitHub**

&#10230; 

<br>


**102. By X and Y**

&#10230; 

<br>
