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

&#10230; Architecture d'un CNN traditionnel ― Les réseaux de neurones convolutionnels (en anglais <i>Convolutional neural networks</i>), aussi connus sous le nom de CNNs, sont un type spécifique de réseaux de neurones qui sont généralement composés des couches suivantes :

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; La couche convolutionnelle et la couche de pooling peuvent être ajustées en utilisant des paramètres qui sont décrites dans les sections suivantes.

<br>


**15. Types of layer**

&#10230; Types de couche

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Couche convolutionnelle (CONV) ― La couche convolutionnelle (en anglais <i>convolution layer</i>) (CONV) utilise des filtres qui scannent l'entrée I suivant ses dimensions en effectuant des opérations de convolution. Elle peut être réglée en ajustant la taille du filtre F et le stride S. La sortie O de cette opération est appelée *feature map* ou aussi *activation map*.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Remarque : l'étape de convolution peut aussi être généralisée dans les cas 1D et 3D.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Pooling (POOL) ― La couche de pooling (en anglais <i>pooling layer</i>) (POOL) est une opération de sous-échantillonnage typiquement appliquée après une couche convolutionnelle. En particulier, les types de pooling les plus populaires sont le max et l'average pooling, où les valeurs maximales et moyennes sont prises, respectivement.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Type, But, Illustration, Commentaires]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Max pooling, Average pooling, Chaque opération de pooling sélectionne la valeur maximale de la surface. Chaque opération de pooling sélectionne la valeur moyenne de la surface.]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [Garde les caractéristiques détectées. Plus communément utilisé, Sous-échantillonne la <i>feature map</i>, Utilisé dans LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Fully Connected (FC) ― La couche de fully connected (en anglais <i>fully connected layer</i>) (FC) s'applique sur une entrée préalablement aplatie où chaque entrée est connectée à tous les neurones. Les couches de fully connected sont typiquement présentes à la fin des architectures de CNN et peuvent être utilisées pour optimiser des objectifs tels que les scores de classe.

<br>


**23. Filter hyperparameters**

&#10230; Paramètres du filtre 

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; La couche convolutionnelle contient des filtres pour lesquels il est important de savoir comment ajuster ses paramètres.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Dimensions d'un filtre ― Un filtre de taille F×F appliqué à une entrée contenant C canaux est un volume de taille F×F×C qui effectue des convolutions sur une entrée de taille I×I×C et qui produit un <i>feature map</i> de sortie (aussi appelé <i>activation map</i>) de taille O×O×1.

<br>


**26. Filter**

&#10230; Filtre

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Remarque : appliquer K filtres de taille F×F engendre un <i>feature map</i> de sortie de taille O×O×K.

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

&#10230; Champ récepteur ― Le champ récepteur à la couche k est la surface notée Rk×Rk de l'entrée que chaque pixel de la k-ième <i>activation map</i> peut 'voir'. En notant Fj la taille du filtre de la couche j et Si la valeur de stride de la couche i et avec la convention S0=1, le champ récepteur à la couche k peut être calculé de la manière suivante :

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; Dans l'exemple ci-dessous, on a F1=F2=3 et S1=S2=1, ce qui donne R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Fonctions d'activation communément utilisées

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Unité linéaire rectifiée ― La couche d'unité linéaire rectifiée (en anglais <i>rectified linear unit layer</i>) (ReLU) est une fonction d'activiation g qui est utilisée sur tous les éléments du volume. Elle a pour but d'introduire des complexités non-linéaires au réseau. Ses variantes sont récapitulées dans le tableau suivant :

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, Leaky ReLU, ELU, avec]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Complexités non-linéaires intereprétables d'un point de vue biologique, Repond au problème de <i>dying ReLU</i>, Dérivable partout]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax ― L'étape softmax peut être vue comme une généralisation de la fonction logistique qui prend comme argument un vecteur de scores x∈Rn et qui renvoie un vecteur de probabilités p∈Rn à travers une fonction softmax à la fin de l'architecture. Elle est définie de la manière suivante :

<br>


**48. where**

&#10230; où

<br>


**49. Object detection**

&#10230; Détection d'objet

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Types de modèles ― Il y a 3 principaux types d'algorithme de reconnaissance d'objet, pour lesquels la nature de ce qui est prédit est different. Ils sont décrits dans la table ci-dessous :

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Classification d'image, Classification avec localisation, Détection]

<br>


**52. [Teddy bear, Book]**

&#10230; [Ours en peluche, Livre]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Classifie une image, Predit la probabilité d'un objet, Détecte un objet dans une image, Prédit la probabilité de présence d'un objet et où il est situé, Peut détecter plusieurs objets dans une image, Prédit les probabilités de présence des objets et où ils sont situés]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [CNN traditionnel, YOLO simplifié, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Détection ― Dans le contexte de la détection d'objet, des methodes différentes sont utilisées selon si l'on veut juste localiser l'objet ou alors détecter une forme plus complexe dans l'image. Les deux méthodes principales sont résumées dans le tableau ci-dessous :

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Détection de zone délimitante, Detection de forme complexe]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Détecte la partie de l'image où l'objet est situé, Détecte la forme ou les caractéristiques d'un objet (e.g. yeux), Plus granulaire]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Zone de centre (bx,by), hauteur bh et largeur bw, Points de référence (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Intersection sur Union ― Intersection sur Union (en anglais <i>Intersection over Union</i>), aussi appelé IoU, est une fonction qui quantifie à quel point la zone délimitante prédite Bp est correctement positionnée par rapport à la zone délimitante vraie Ba. Elle est définie de la manière suivante :

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Remarque : on a toujours IoU∈[0,1]. Par convention, la prédiction Bp d'une zone délimitante est considérée comme étant satisfaisante si l'on a IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Zone d'accroche ― La technique des zones d'accroche (en anglais <i>anchor boxing</i>) sert à prédire des zones délimitantes qui se chevauchent. En pratique, on permet au réseau de prédire plus d'une zone délimitante simultanément, où chaque zone prédite doit respecter une forme géométrique particulière. Par example, la première prédiction peut potentiellement être une zone rectangulaire d'une forme donnée, tandis qu'une seconde prédiction doit être une zone rectangulaire d'une autre forme.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Suppression non-max ― La technique de suppression non-max (en anglais <i>non-max suppression</i>) a pour but d'enlever des zones délimitantes qui se chevauchent et qui prédisent un seul et même objet, en sélectionnant les zones les plus representatives. Après avoir enlevé toutes les zones ayant une probabilité prédite de moins de 0.6, les étapes suivantes sont répétées pour éliminer les zones redondantes :

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Pour une classe donnée, Étape 1 : Choisir la zone ayant la plus grande probabilité de prédiction., Étape 2 : Enlever toute zone ayant IoU⩾0.5 avec la zone choisie précédemment.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Zones prédites, Sélection de la zone de probabilité maximum, Suppression des chevauchements d'une même classe, Zones délimitantes finales]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO ― L'algorithme You Only Look Once (YOLO) est un algorithme de détection d'objet qui fonctionne de la manière suivante :

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Étape 1 : Diviser l'image d'entrée en une grille de taille G×G., Étape 2 : Pour chaque cellule, faire tourner un CNN qui prédit y de la forme suivante :, répété k fois]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; où pc est la probabilité de détecter un objet, bx,by,bh,bw sont les propriétés de la zone délimitante détectée, c1,...,cp est une répresentation binaire (en anglais <i>one-hot representation</i>) de l'une des p classes détectée, et k est le nombre de zones d'accroche.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Étape 3 : Faire tourner l'algorithme de suppression non-max pour enlever des doublons potentiels qui chevauchent des zones délimitantes.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Image originale, Division en une grille de taille GxG, Prediction de zone délimitante, Suppression non-max]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Remarque : lorsque pc=0, le réseau ne détecte plus d'objet. Dans ce cas, les prédictions correspondantes bx,...,cp doivent être ignorées.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN ― L'algorithme de région avec des réseaux de neurones convolutionnels (en anglais <i>Region with Convolutional Neural Networks</i>) (R-CNN) est un algorithme de détection d'objet qui segmente l'image d'entrée pour trouver des zones délimitantes pertinentes, puis fait tourner un algorithme de détection pour trouver les objets les plus probables d'apparaître dans ces zones délimitantes.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Image originale, Segmentation, Prédiction de zone délimitante, Suppression non-max]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; Remarque : bien que l'algorithme original soit lent et coûteux en temps de calcul, de nouvelles architectures ont permis de faire tourner l'algorithme plus rapidement, tels que le Fast R-CNN et le Faster R-CNN.

<br>


**74. Face verification and recognition**

&#10230; Vérification et reconnaissance de visage

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Types de modèles ― Deux principaux types de modèle sont récapitulés dans le tableau ci-dessous :

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Vérification de visage, Reconnaissance de visage, Requête, Référence, Base de données]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [Est-ce la bonne personne ?, , Est-ce une des K personnes dans la base de données ?, ]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; Apprentissage par coup ― L'apprentissage par coup (en anglais <i>One Shot Learning</i>) est un algorithme de vérification de visage qui utilise un training set de petite taille pour apprendre une fonction de similarité qui quantifie à quel point deux images données sont différentes. La fonction de similarité appliquée à deux images est souvent notée d(image 1,image 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Réseaux siamois ― Les réseaux siamois (en anglais <i>Siamese Networks</i>) ont pour but d'apprendre comment encoder des images pour quantifier le degré de difference de deux images données. Pour une image d'entrée donnée x(i), l'encodage de sortie est souvent notée f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Loss triple ― Le loss triple (en anglais <i>triplet loss</i>) ℓ est une fonction de loss calculée sur une représentation encodée d'un triplet d'images A (accroche), P (positif), et N (négatif). L'exemple d'accroche et l'exemple positif appartiennent à la même classe, tandis que l'exemple négatif appartient à une autre. En notant α∈R+ le paramètre de marge, le loss est défini de la manière suivante :

<br>


**81. Neural style transfer**

&#10230; Transfert de style neuronal

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Motivation ― Le but du transfert de style neuronal (en anglais <i>neural style transfer</i>) est de générer une image G à partir d'un contenu C et d'un style S.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [Contenu C, Style S, Image générée G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Activation ― Dans une couche l donnée, l'activation est notée a[l] et est de dimensions nH×nw×nc

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; Fonction de coût de contenu ― La fonction de coût de contenu (en anglais <i>content cost function</i>), notée Jcontenu(C,G), est utilisée pour quantifier à quel point l'image générée G diffère de l'image de contenu original C. Elle est définie de la manière suivante :

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; Matrice de style ― La matrice de style (en anglais <i>style matrix</i>) G[l] d'une couche l donnée est une matrice de Gram dans laquelle chacun des éléments G[l]kk′ quantifie le degré de corrélation des canaux k and k′. Elle est définie en fonction des activations a[l] de la manière suivante :

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Remarque : les matrices de style de l'image de style et de l'image générée sont notées G[l] (S) and G[l] (G) respectivement.

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Fonction de coût de style ― La fonction de coût de style (en anglais <i>style cost function</i>), notée Jstyle(S,G), est utilisée pour quantifier à quel point l'image générée G diffère de l'image de style S. Elle est définie de la manière suivante :

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Fonction de coût totale ― La fonction de coût totale (en anglais <i>overall cost function</i>) est définie comme étant une combinaison linéaire des fonctions de coût de contenu et de style, pondérées par les paramètres α,β, de la manière suivante :

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Remarque : plus α est grand, plus le modèle privilégiera le contenu et plus β est grand, plus le modèle sera fidèle au style.

<br>


**91. Architectures using computational tricks**

&#10230; Architectures utilisant des astuces de calcul

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Réseau antagoniste génératif ― Les réseaux antagonistes génératifs (en anglais <i>generative adversarial networks</i>), aussi connus sous le nom de GANs, sont composés d'un modèle génératif et d'un modèle discriminatif, où le modèle génératif a pour but de générer des prédictions aussi réalistes que possibles, qui seront ensuite envoyées dans un modèle discriminatif qui aura pour but de différencier une image générée d'une image réelle.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [Training, Bruit, Image réelle, Générateur, Discriminant, Vrai faux]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; Remarque : les GANs sont utilisées dans des applications pouvant aller de la génération de musique au traitement de texte vers image.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet ― L'architecture du réseau résiduel (en anglais <i>Residual Network</i>), aussi appelé ResNet, utilise des blocs résiduels avec un nombre élevé de couches et a pour but de réduire l'erreur de training. Le bloc résiduel est caractérisé par l'équation suivante :

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception Network ― Cette architecture utilise des modules d'<i>inception</i> et a pour but de tester toute sorte de configuration de convolution pour améliorer sa performance en diversifiant ses attributs. En particulier, elle utilise l'astuce de la convolution 1x1 pour limiter sa complexité de calcul.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'apprentissage profond sont maintenant disponibles en français.

<br>


**98. Original authors**

&#10230; Auteurs

<br>


**99. Translated by X, Y and Z**

&#10230; Traduit par X, Y et Z

<br>


**100. Reviewed by X, Y and Z**

&#10230; Relu par X, Y et Z

<br>


**101. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>


**102. By X and Y**

&#10230; Par X et Y

<br>
