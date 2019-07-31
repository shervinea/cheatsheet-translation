**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**

&#10230; Modèles basés sur le réflex à l'aide de l'apprentissage automatique

<br>


**2. Linear predictors**

&#10230; Prédicteurs linéaires

<br>


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

&#10230; Dans cette section, nous allons explorer les modèles basés sur le réflex qui peuvent s'améliorer avec l'expérience s'appuyant sur des données ayant une correspondance entrée-sortie.

<br>


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

&#10230; Vecteur caractéristique - Le vecteur caractéristique (en anglais feature vector) d'une entrée x est noté ϕ(x) et se décompose en :

<br>


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

&#10230; Score - Le score s(x,w) d'un exemple (ϕ(x),y)∈Rd×R associé à un modèle linéaire de paramètres w∈Rd est donné par le produit scalaire :

<br>


**6. Classification**

&#10230; Classification

<br>


**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

&#10230; Classifieur linéaire - Étant donnés un vecteur de paramètres w∈Rd et un vecteur caractéristique ϕ(x)∈Rd, le classifieur linéaire binaire est donné par :

<br>


**8. if**

&#10230; si

<br>


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

&#10230; Marge - La marge (en anglais margin) m(x,y,w)∈R d'un exemple (ϕ(x),y)∈Rd×{−1,+1} associée à un modèle linéaire de paramètre w∈Rd quantifie la confiance associée à une prédiction : plus cette valeur est grande, mieux c'est. Cette quantité est donnée par :

<br>


**10. Regression**

&#10230; Régression

<br>


**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**

&#10230; Régression linéaire - Étant donnés un vecteur de paramètres w∈Rd et un vecteur caractéristique ϕ(x)∈Rd, le résultat d'une régression linéaire de paramètre w, notée fw, est donné par :

<br>


**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**

&#10230; Résidu - Le résidu res(x,y,w)∈R est défini comme étant la différence entre la prédiction fw(x) et la vraie valeur y.

<br>


**13. Loss minimization**

&#10230; Minimisation de la fonction objectif

<br>


**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**

&#10230; Fonction objectif - Une fonction objectif (en anglais loss function) Loss(x,y,w) traduit notre niveau d'insatisfaction avec les paramètres w du modèle dans la tâche de prédiction de la sortie y à partir de l'entrée x. C'est une quantité que l'on souhaite minimiser pendant la phase d'entraînement.

<br>


**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**

&#10230; Cas de la classification - Trouver la classe d'un exemple x appartenant à y∈{−1,+1} peut être faite par le biais d'un modèle linéaire de paramètre w à l'aide du prédicteur fw(x)≜sign(s(x,w)). La qualité de cette prédiction peut alors être évaluée au travers de la marge m(x,y,w) intervenant dans les fonctions objectif suivantes :

<br>


**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**

&#10230; [Nom, Illustration, Fonction objectif zéro-un, Fonction objectif de Hinge, Fonction objectif logistique]

<br>


**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**

&#10230; Cas de la régression - Prédire la valeur y∈R associée à l'exemple x peut être faite par le biais d'un modèle linéaire de paramètre w à l'aide du prédicteur fw(x)≜s(x,w). La qualité de cette prédiction peut alors être évaluée au travers du résidu res(x,y,w) intervenant dans les fonctions objectif suivantes :

<br>


**18. [Name, Squared loss, Absolute deviation loss, Illustration]**

&#10230; [Nom, Erreur quadratique, Erreur absolue, Illustration]

<br>


**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**

&#10230; Processus de minimisation de la fonction objectif - Lors de l'entraînement d'un modèle, on souhaite minimiser la valeur de la fonction objectif évaluée sur l'ensemble d'entraînement :

<br>


**20. Non-linear predictors**

&#10230; Prédicteurs non linéaires

<br>


**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k plus proches voisins - L'algorithme des k plus proches voisins (en anglais k-nearest neighbors ou k-NN) est une approche non paramétrique où la réponse associée à un exemple est déterminée par la nature de ses k plus proches voisins de l'ensemble d'entraînement. Cette démarche peut être utilisée pour la classification et la régression.

<br>


**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Remarque : plus le paramètre k est grand, plus le biais est élevé. À l'inverse, la variance devient plus élevée lorsque l'on réduit la valeur k.

<br>


**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Réseaux de neurones - Les réseaux de neurones (en anglais neural networks) constituent un type de modèle basés sur des couches (en anglais layers). Parmi les types de réseaux populaires, on peut compter les réseaux de neurones convolutionnels et récurrents (abbréviés respectivement en CNN et RNN en anglais). Une partie du vocabulaire associé aux réseaux de neurones est détaillée dans la figure ci-dessous :

<br>


**24. [Input layer, Hidden layer, Output layer]**

&#10230; [Couche d'entrée, Couche cachée, Couche de sortie]

<br>


**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; En notant i la i-ème couche du réseau et j son j-ième neurone, on a :

<br>


**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**

&#10230; où l'on note w, b, x, z le coefficient, le biais ainsi que la variable de sortie respectivement.

<br>


**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**

&#10230; Pour un aperçu plus détaillé des concepts ci-dessus, rendez-vous sur le pense-bête d'apprentissage supervisé !

<br>


**28. Stochastic gradient descent**

&#10230; Algorithme du gradient stochastique

<br>


**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**

&#10230; Descente de gradient - En notant η∈R le taux d'apprentissage (en anglais learning rate ou step size), la règle de mise à jour des coefficients pour cet algorithme utilise la fonction objectif Loss(x,y,w) de la manière suivante :

<br>


**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**

&#10230; Mises à jour stochastiques - L'algorithme du gradient stochastique (en anglais stochastic gradient descent ou SGD) met à jour les paramètres du modèle en parcourant les exemples (ϕ(x),y)∈Dtrain de l'ensemble d'entraînement un à un. Cette méthode engendre des mises à jour rapides à calculer mais qui manquent parfois de robustesse.

<br>


**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**

&#10230; Mises à jour par lot - L'algorithme du gradient par lot (en anglais batch gradient descent ou BGD) met à jour les paramètre du modèle en utilisant des lots entiers d'exemples (e.g. la totalité de l'ensemble d'entraînement) à la fois. Cette méthode calcule des directions de mise à jour des coefficients plus stable au prix d'un plus grand nombre de calculs.

<br>


**32. Fine-tuning models**

&#10230; Peaufinage de modèle

<br>


**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**

&#10230; Classe d'hypothèses - Une classe d'hypothèses F est l'ensemble des prédicteurs candidats ayant un ϕ(x) fixé et dont le paramètre w peut varier.

<br>


**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**

&#10230; Fonction logistique - La fonction logistique σ, aussi appelée en anglais sigmoid function, est définie par :

<br>


**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**

&#10230; Remarque : la dérivée de cette fonction s'écrit σ′(z)=σ(z)(1−σ(z)).

<br>


**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**

&#10230; Rétropropagation du gradient (en anglais backpropagation) - La propagation avant (en anglais forward pass) est effectuée via fi, valeur correspondant à l'expression appliquée à l'étape i. La propagation de l'erreur vers l'arrière (en anglais backward pass) se fait via gi=∂out∂fi et décrit la manière dont fi agit sur la sortie du réseau.

<br>


**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**

&#10230; Erreur d'approximation et d'estimation - L'erreur d'approximation ϵapprox représente la distance entre la classe d'hypothèses F et le prédicteur optimal g∗. De son côté, l'erreur d'estimation quantifie la qualité du prédicteur ^f par rapport au meilleur prédicteur f∗ de la classe d'hypothèses F.

<br>


**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Régularisation - Le but de la régularisation est d'empêcher le modèle de surapprendre (en anglais overfit) les données en s'occupant ainsi des problèmes de variance élevée. La table suivante résume les différents types de régularisation couramment utilisés :

<br>


**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Réduit les coefficients à 0, Bénéfique pour la sélection de variables, Rapetissit les coefficients, Compromis entre sélection de variables et coefficients de faible magnitude]

<br>


**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**

&#10230; Hyperparamètres - Les hyperparamètres sont les paramètres de l'algorithme d'apprentissage et incluent parmi d'autres le type de caractéristiques utilisé ainsi que le paramètre de régularisation λ, le nombre d'itérations T, le taux d'apprentissage η.

<br>


**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulaire ― Lors de la sélection d'un modèle, on divise les données en 3 différentes parties :

<br>


**42. [Training set, Validation set, Testing set]**

&#10230; [Données d'entraînement, Données de validation, Données de test]

<br>


**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**

&#10230; [Le modèle est entrainé, Constitue normalement 80% du jeu de données, Le modèle est évalué, Constitue normalement 20% du jeu de données, Aussi appelé données de développement (en anglais hold-out ou development set), Le modèle donne ses prédictions, Données jamais observées]

<br>


**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Une fois que le modèle a été choisi, il est entrainé sur le jeu de données entier et testé sur l'ensemble de test (qui n'a jamais été vu). Ces derniers sont représentés dans la figure ci-dessous :

<br>


**45. [Dataset, Unseen data, train, validation, test]**

&#10230; [Jeu de données, Données inconnues, entrainement, validation, test]

<br>


**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**

&#10230; Pour un aperçu plus détaillé des concepts ci-dessus, rendez-vous sur le pense-bête de petites astuces d'apprentissage automatique !

<br>


**47. Unsupervised Learning**

&#10230; Apprentissage non supervisé

<br>


**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**

&#10230; Les méthodes d'apprentissage non supervisé visent à découvrir la structure (parfois riche) des données.

<br>


**49. k-means**

&#10230; k-moyennes (en anglais k-means)

<br>


**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**

&#10230; Partitionnement - Étant donné un ensemble d'entraînement Dtrain, le but d'un algorithme de partitionnement (en anglais clustering) est d'assigner chaque point ϕ(xi) à une partition zi∈{1,...,k}.

<br>


**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**

&#10230; Fonction objectif - La fonction objectif d'un des principaux algorithmes de partitionnement, k-moyennes, est donné par :

<br>


**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Après avoir aléatoirement initialisé les centroïdes de partitions μ1,μ2,...,μk∈Rn, l'algorithme k-moyennes répète l'étape suivante jusqu'à convergence :

<br>


**53. and**

&#10230; et

<br>


**54. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Initialisation des moyennes, Assignation de partition, Mise à jour des moyennes, Convergence]

<br>


**55. Principal Component Analysis**

&#10230; Analyse des composantes principales

<br>


**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Étant donnée une matrice A∈Rn×n, λ est dite être une valeur propre de A s'il existe un vecteur z∈Rn∖{0}, appelé vecteur propre, tel que :

<br>


**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Théorème spectral ― Soit A∈Rn×n. Si A est symétrique, alors A est diagonalisable par une matrice réelle orthogonale U∈Rn×n. En notant Λ=diag(λ1,...,λn), on a :

<br>


**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Remarque : le vecteur propre associé à la plus grande valeur propre est appelé le vecteur propre principal de la matrice A.

<br>


**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algorithme ― La procédure d'analyse des composantes principales (en anglais PCA - Principal Component Analysis) est une technique de réduction de dimension qui projette les données sur k dimensions en maximisant la variance des données de la manière suivante :

<br>


**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Étape 1: Normaliser les données pour avoir une moyenne de 0 et un écart-type de 1.

<br>


**61. [where, and]**

&#10230; [où, et]

<br>


**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**

&#10230; [Étape 2: Calculer Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, qui est symmétrique avec des valeurs propres réelles., Étape 3: Calculer u1,...,uk∈Rn les k valeurs propres principales orthogonales de Σ, i.e. les vecteurs propres orthogonaux des k valeurs propres les plus grandes., Étape 4: Projeter les données sur spanR(u1,...,uk).]

<br>


**63. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Cette procédure maximise la variance sur tous les espaces à k dimensions.

<br>


**64. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Données dans l'espace initial, Trouve les composantes principales, Données dans l'espace des composantes principales]

<br>


**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**

&#10230; Pour un aperçu plus détaillé des concepts ci-dessus, rendez-vous sur le pense-bête d'apprentissage non supervisé !

<br>


**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin]**

&#10230; [Prédicteurs linéaires, Vecteur caractéristique, Classification/régression linéaire, Marge]

<br>


**67. [Loss minimization, Loss function, Framework]**

&#10230; [Minimisation de la fonction objectif, Fonction objectif, Cadre]

<br>


**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**

&#10230; [Prédicteurs non linéaires, k plus proches voisins, Réseaux de neurones]

<br>


**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**

&#10230; [Algorithme du gradient stochastique, Gradient, Mises à jour stochastiques, Mises à jour par lots]

<br>


**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**

&#10230; [Peaufiner les modèles, Classe d'hypothèses, Rétropropagation du gradient, Régularisation, Vocabulaire]

<br>


**71. [Unsupervised Learning, k-means, Principal components analysis]**

&#10230; [Apprentissage non supervisé, k-means, Analyse des composantes principales]

<br>


**72. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>


**73. Original authors**

&#10230; Auteurs d'origine

<br>


**74. Translated by X, Y and Z**

&#10230; Traduit par X, Y et Z

<br>


**75. Reviewed by X, Y and Z**

&#10230; Revu par X, Y et Z

<br>


**76. By X and Y**

&#10230; De X et Y

<br>


**77. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'intelligence artificielle sont maintenant disponibles en français.
