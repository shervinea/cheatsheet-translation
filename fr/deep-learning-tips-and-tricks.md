**Deep Learning Tips and Tricks translation**

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; Pense-bête de petites astuces d'apprentissage profond

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Apprentissage profond

<br>


**3. Tips and tricks**

&#10230; Petites astuces

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [Traitement des données, Augmentation des données, Normalization de lot]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [Entrainement d'un réseau de neurones, Epoch, Mini-lot, Entropie croisée, Rétropropagation du gradient, Algorithme du gradient, Mise à jour des coefficients, Vérification de gradient]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [Ajustement de paramètres, Initialisation de Xavier, Apprentissage par transfert, Taux d'apprentissage, Taux d'apprentissage adaptifs]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [Régularisation, Abandon, Régularisation des coefficients, Arrêt prématuré]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [Bonnes pratiques, Surapprentissage d'un mini-lot, Vérification de gradient]

<br>


**9. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>


**10. Data processing**

&#10230; Traitement des données

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; Augmentation des données - Les modèles d'apprentissage profond ont typiquement besoin de beaucoup de données afin d'être entrainés convenablement. Il est souvent utile de générer plus de données à partir de celles déjà existantes à l'aide de techniques d'augmentation de données. Celles les plus souvent utilisées sont résumées dans le tableau ci-dessous. À partir d'une image, voici les techniques que l'on peut utiliser :

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [Original, Symmétrie axiale, Rotation, Recadrage aléatoire]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [Image sans aucune modification, Symmetrie par rapport à un axe pour lequel le sens de l'image est conservé, Rotation avec un petit angle, Reproduit une calibration imparfaite de l'horizon, Concentration aléatoire sur une partie de l'image, Plusieurs rognements aléatoires peuvent être faits à la suite]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [Changement de couleur, Addition de bruit, Perte d'information, Changement de contraste]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; [Nuances de RGB sont légèrement changées, Capture le bruit qui peut survenir avec de l'exposition lumineuse, Addition de bruit, Plus de tolérance envers la variation de la qualité de l'entrée, Parties de l'image ignorées, Imite des pertes potentielles de parties de l'image, Changement de luminosité, Contrôle la différence de l'exposition dû à l'heure de la journée]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; Remarque : les données sont normalement augmentées à la volée durant l'étape de training.

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalisation de lot ― La normalisation de lot (en anglais <i>batch normalization</i>) est une étape qui normalise le lot {xi} avec un choix de paramètres γ,β. En notant μB,σ2B la moyenne et la variance de ce que l'on veut corriger du lot, on a :

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Ceci est couramment fait après un fully connected/couche de convolution et avant une couche non-linéaire. Elle vise à permettre d'avoir de plus grands taux d'apprentissages et de réduire la dépendance à l'initialisation.

<br>


**19. Training a neural network**

&#10230; Entraîner un réseau de neurones

<br>


**20. Definitions**

&#10230; Définitions

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; Epoch ― Dans le contexte de l'entraînement d'un modèle, l'<i>epoch</i> est un terme utilisé pour réferer à une itération où le modèle voit tout le training set pour mettre à jour ses coefficients.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; Gradient descent sur mini-lots ― Durant la phase d'entraînement, la mise à jour des coefficients n'est souvent basée ni sur tout le training set d'un coup à cause de temps de calculs coûteux, ni sur un seul point à cause de bruits potentiels. À la place de cela, l'étape de mise à jour est faite sur des mini-lots, où le nombre de points dans un lot est un paramètre que l'on peut régler.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; Fonction de loss ― Pour pouvoir quantifier la performance d'un modèle donné, la fonction de loss (en anglais <i>loss function</i>) L est utilisée pour évaluer la mesure dans laquelle les sorties vraies y sont correctement prédites par les prédictions du modèle z.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Entropie croisée ― Dans le contexte de la classification binaire d'un réseau de neurones, l'entropie croisée (en anglais <i>cross-entropy loss</i>) L(z,y) est couramment utilisée et est définie par :

<br>


**25. Finding optimal weights**

&#10230; Recherche de coefficients optimaux

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; Backpropagation ― La backpropagation est une méthode de mise à jour des coefficients d'un réseau de neurones en prenant en compte les sorties vraies et désirées. La dérivée par rapport à chaque coefficient w est calculée en utilisant la règle de la chaîne.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; En utilisant cette méthode, chaque coefficient est mis à jour par :

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Mettre à jour les coefficients ― Dans un réseau de neurones, les coefficients sont mis à jour par :

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [Étape 1 : Prendre un lot de training data et effectuer une forward propagation pour calculer le loss, Étape 2 : Backpropaguer le loss pour obtenir le gradient du loss par rapport à chaque coefficient, Étape 3 : Utiliser les gradients pour mettre à jour les coefficients du réseau.]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [Forward propagation, Backpropagation, Mise à jour des coefficients]

<br>


**31. Parameter tuning**

&#10230; Réglage des paramètres

<br>


**32. Weights initialization**

&#10230; Initialisation des coefficients

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Initialization de Xavier ― Au lieu de laisser les coefficients s'initialiser de manière purement aléatoire, l'initialisation de Xavier permet d'avoir des coefficients initiaux qui prennent en compte les caractéristiques uniques de l'architecture.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; Apprentissage de transfert ― Entraîner un modèle d'apprentissage profond requière beaucoup de données et beaucoup de temps. Il est souvent utile de profiter de coefficients pre-entraînés sur des données énormes qui ont pris des jours/semaines pour être entraînés, et profiter de cela pour notre cas. Selon la quantité de données que l'on a sous la main, voici différentes manières d'utiliser cette methode :

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [Taille du training, Illustration, Explication]

<br>


**36. [Small, Medium, Large]**

&#10230; [Petit, Moyen, Grand]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [Gèle toutes les couches, entraîne les coefficients du softmax, Gèle la plupart des couches, entraîne les coefficients des dernières couches et du softmax, Entraîne les coefficients des couches et du softmax en initialisant les coefficients sur ceux qui ont été pré-entraînés]

<br>


**38. Optimizing convergence**

&#10230; Optimisation de la convergence

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Taux d'apprentissage ― Le taux d'apprentissage (en anglais <i>learning rate</i>), souvent noté α ou η, indique la vitesse à laquelle les coefficients sont mis à jour. Il peut être fixe ou variable. La méthode actuelle la plus populaire est appelée Adam, qui est une méthode faisant varier le taux d'apprentissage.

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; Taux d'apprentissage adaptatifs ― Laisser le taux d'apprentissage varier pendant la phase d'entraînement du modèle peut réduire le temps d'entraînement et améliorer la qualité de la solution numérique optimale. Bien que la méthode d'Adam est la plus utilisée, d'autres peuvent aussi être utiles. Les différentes méthodes sont récapitulées dans le tableau ci-dessous :

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [Méthode, Explication, Mise à jour de b, Mise à jour de b]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [Momentum, Amortit les oscillations, Amélioration par rapport à la méthode SGD, 2 paramètres à régler]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, Root Mean Square propagation, Accélère l'algorithme d'apprentissage en contrôlant les oscillations]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [Adam, Adaptive Moment estimation, Méthode la plus populaire, 4 paramètres à régler]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; Remarque : parmi les autres méthodes existantes, on trouve Adadelta, Adagrad et SGD.

<br>


**46. Regularization**

&#10230; Régularisation

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; Dropout ― Le dropout est une technique qui est destinée à empêcher le sur-ajustement sur les données de training en abandonnant des unités dans un réseau de neurones avec une probabilité p>0. Cela force le modèle à éviter de trop s'appuyer sur un ensemble particulier de features.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; Remarque : la plupart des frameworks d'apprentissage profond paramètrent le dropout à travers le paramètre 'garder' 1-p.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; Régularisation de coefficient ― Pour s'assurer que les coefficients ne sont pas trop grands et que le modèle ne sur-ajuste pas sur le training set, on utilise des techniques de régularisation sur les coefficients du modèle. Les techniques principales sont résumées dans le tableau suivant :

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230; [LASSO, Ridge, Elastic Net]

<br>

**50 bis. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Réduit les coefficients à 0, Bon pour la sélection de variables, Rend les coefficients plus petits, Compromis entre la sélection de variables et la réduction de la taille des coefficients]

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230; Arrêt prématuré ― L'arrêt prématuré (en anglais <i>early stopping</i>) est une technique de régularisation qui consiste à stopper l'étape d'entraînement dès que le loss de validation atteint un plateau ou commence à augmenter.

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230; [Erreur, Validation, Training, arrêt prématuré, Epochs]

<br>


**53. Good practices**

&#10230; Bonnes pratiques

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230; Sur-ajuster un mini-lot ― Lorsque l'on débugge un modèle, il est souvent utile de faire de petits tests pour voir s'il y a un gros souci avec l'architecture du modèle lui-même. En particulier, pour s'assurer que le modèle peut être entraîné correctement, un mini-lot est passé dans le réseau pour voir s'il peut sur-ajuster sur lui. Si le modèle ne peut pas le faire, cela signifie que le modèle est soit trop complexe ou pas assez complexe pour être sur-ajusté sur un mini-lot.

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230; Gradient checking ― La méthode de gradient checking est utilisée durant l'implémentation d'un backward pass d'un réseau de neurones. Elle compare la valeur du gradient analytique par rapport au gradient numérique au niveau de certains points et joue un rôle de vérification élementaire.

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230; [Type, Gradient numérique, Gradient analytique]

<br>


**57. [Formula, Comments]**

&#10230; [Formule, Commentaires]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230; [Coûteux; le loss doit être calculé deux fois par dimension, Utilisé pour vérifier l'exactitude d'une implémentation analytique, Compromis dans le choix de h entre pas trop petit (instabilité numérique) et pas trop grand (estimation du gradient approximative)]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230; [Resultat 'exact', Calcul direct, Utilisé dans l'implémentation finale]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'appentissage profond sont maintenant disponibles en français.

<br>

**61. Original authors**

&#10230; Auteurs

<br>

**62. Translated by X, Y and Z**

&#10230; Traduit par X, Y et Z

<br>

**63. Reviewed by X, Y and Z**

&#10230; Relu par X, Y et Z

<br>

**64. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>

**65. By X and Y**

&#10230; Par X et Y

<br>
