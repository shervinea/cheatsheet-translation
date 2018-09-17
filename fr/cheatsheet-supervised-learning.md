d'apprentissage l'ensemble d'apprentissage **1. Supervised Learning cheatsheet**

&#10230; Pense-bête d'apprentissage supervisé

<br>

**2. Introduction to Supervised Learning**

&#10230; Introduction à l'apprentissage supervisé

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Étant donné un ensemble de points {x(1),...,x(m)} associés à un ensemble d'issues {y(1),...,y(m)}, on veut construire un classfieur qui apprend comment prédire y à partir de x.  

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Type de prédiction ― Les différents types de modèles prédictifs sont récapitulés dans le tableau ci-dessous :

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Régression, Classifieur, Issue, Exemples]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Continu, Classe, Régression linéaire, Régression logistique, SVM, Naive Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Type de modèle ― Les différents modèles sont récapitulés dans le tableau ci-dessous :

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Modèle discriminatif, Modèle génératif, But, Ce qui est appris, Illustration, Exemples]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Estimer directement P(y|x), Estimer P(x|y) puis déduire P(y|x), Frontière de décision, Distribution de probabilité des données, Régressions, SVMs, GDA Naive Bayes]

<br>

**10. Notations and general concepts**

&#10230; Notations et concepts généraux

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hypothèse ― Une hypothèse est noté hθ et est le modèle que l'on choisit. Pour un input donné x(i), la prédiction donnée par le modèle est hθ(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Fonction de loss ― Une fonction de loss est une fonction L:(z,y)∈R×Y⟼L(z,y)∈R qui prend comme entrée une valeur prédite z correspondant à une valeur réelle y et donne une indication de la mesure dans laquelle ils diffèrent. Les fonctions de loss principales sont récapitulées dans le tableau ci-dessous :

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Erreur des moindres carrés, Logistic loss, Hinge loss, Cross-entropie]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Régression linéaire, régression logistique, SVM, Réseau de neurones]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Fonction de cost ― La fonction de cost J est communément utilisée pour évaluer la performance d'un modèle, et est définie avec la fonction de loss L par :

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Algorithme du gradient ― En notant α∈R le learning rate, la règle de mise à jour pour l'algorithme est exprimé avec le learning rate de la fonction de cost J de la manière suivante :

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Remarque : L'algorithme du gradient stochastique (SGC) met à jour le paramètre à partir de chaque exemple de l'ensemble d'apprentisage , tandis que l'algorithme du gradient de batch le fait à partir de chaque batch d'exemples.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Vraisemblance ― La vraisemblance d'un modèle L(θ) de paramètre θ est utilisée pour trouver le paramètre optimal θ par le biais du maximum de vraisemblance. En pratique, on utilise la log vraisemblance ℓ(θ)=log(L(θ)) qui est plus facile à optimiser. On a :

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Algorithme de Newton ― L'algorithme de Newton est une méthode numerique qui trouve θ tel que ℓ′(θ)=0. La règle de mise à jour est :

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Remarque : la généralisation multidimensionnelle, aussi connue sous le nom de la méthode de Newton-Raphson, a la règle de mise à jour suivante :

<br>

**21. Linear models**

&#10230; Modèles linéaires

<br>

**22. Linear regression**

&#10230; Régression linéaire

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; On suppose ici que y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Équations normales ― En notant X la matrice de design, la valeur de θ qui minimize la fonction de cost a une solution de forme fermée tel que :

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Algorithme LMS ― En notant α le learning rate, la règle de mise à jour d'algorithme des moindres carrés (LMS) pour un ensemble d'apprentissage  de m points, qui est aussi connu sous le nom de règle de Widrow-Hoff, est :

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Remarque : la règle de mise à jour est un cas particulier de l'algorithme du gradient.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR - Locally Weighted Regression, souvent noté LWR, est une variante de la simple régression linéaire qui applique un coefficient à chaque exemple par sa fonction de cost via w(i)(x), qui est défini par le paramètre τ∈R de la manière suivante :

<br>

**28. Classification and logistic regression**

&#10230; Classification et régression logistique

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Sigmoïde ― La sigmoïde g, aussi connue sous le nom de fonction logistique, est définie par :

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Régression logistique ― On suppose ici que y|x;θ∼Bernoulli(ϕ). On a la forme suivante :

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Remarque : il n'y a pas de solution fermée dans le cas de la régression logistique.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Régression softmax ― Une régression softmax, aussi appelée un régression logistique multiclasse, est utilisée pour généraliser la régression logistique lorsqu'il y a plus de 2 classes à prédire. Par convention, on fixe θK=0, ce qui oblige le paramètre de Bernoulli ϕi de chaque classe i à être égal à :

<br>

**33. Generalized Linear Models**

&#10230; Modèles linéaires généralisés

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Famille exponentielle ― Une classe de distributions est dite d'appartenir à la famille exponentielle lorsqu'elle peut être écrite en terme de paramètre naturel, aussi appelé paramètre canonique ou fonction de lien η, une statistique suffisante T(y) et une fonction de log-partition a(η) de la manière suivante :

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Remarque : on aura souvent T(y)=y. Aussi, exp(−a(η)) peut être vu comme un paramètre normalisant qui s'assurera que les probabilités somment à un.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Voici les distributions exponentielles les plus communémment rencontrées, récapitulées dans le tableau ci-dessous :

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Distribution, Bernoulli, Gaussian, Poisson, Geometric]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Hypothèses pour les GLMs ― Les modèles linéaires généralisés (GLM) ont pour but de prédire une variable aléatoire y comme une fonction de x∈Rn+1 et repose sur les 3 hypothèses suivantes :

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Remarque : la méthode des moindres carrés ordinaires et la régression logistique sont des cas spéciaux des modèle linéaires généralisés.

<br>

**40. Support Vector Machines**

&#10230; Support Vector Machines

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; Le but des support vector machines est de trouver la ligne qui maximise la distance minimum à la ligne.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Classifieur à marges optimales ― Le classifier à marges optimales h est tel que :

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; où (w,b)∈Rn×R est une solution du problème d'optimisation suivant :

<br>

**44. such that**

&#10230; tel que

<br>

**45. support vectors**

&#10230; support vectors

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Remarque : la ligne est définie par wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Hinge loss ― Le hinge loss est utilisé dans le cadre des SVMs et est défini de la manière suivante :

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Noyau ― Étant donné un feature mapping ϕ, on définit le noyau K par :

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; En pratique, le noyau K défini par K(x,z)=exp(−||x−z||22σ2) est appelé le noyau gaussien, et est communément utilisé.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Séparabilité non-linéaire, Utilisation d'un kernel mapping, Frontière de décision dans l'espace original]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Remarque : on dit que l'on utilise le "kernel trick" pour calculer la fonction de cost en utilisant le noyau parce que l'on a pas besoin de savoir le mapping explicite, qui est souvent compliquée, mais on a juste besoin d'avoir les valeurs de K(x,z).

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangien ― On définit le lagrangien L(w,b) par :

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Remarque : les coefficients βi sont appelés les multiplicateurs de Lagrange.

<br>

**54. Generative Learning**

&#10230; Generative learning

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Un modèle génératif essaie d'abord d'apprendre comment les données sont générées en estimant P(x|y), qui va nous servir à estimer P(y|x) en utilisant le théorème de Bayes.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Gaussian Discriminant Analysis

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Cadre ― Le Gaussian Discriminant Analysis suppose que y et x|y=0 et x|y=1 sont tel que :

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Estimation ― Le tableau suivant récapitule les estimations que l'on a trouvées lors de la maximisation de la vraisemblance :

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Hypothèse ― Le modèle de Naive Bayes suppose que les caractéristiques de chaque point sont tous indépendants :

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Solutions ― Maximiser la log vraisemblance donne les solutions suivantes, avec k∈{0,1},l∈[[1,L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Remarque : Naive Bayes est couramment utilisé pour la classification de texte et pour la détection de spams.

<br>

**63. Tree-based and ensemble methods**

&#10230; Méthode à base d'arbres et d'ensembles

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Ces méthode peuvent être utilisées pour des problèmes de régression et de classification.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART ― Classification and Regression Trees (CART), aussi connu sous le nom d'arbre de décision, peut être représenté sous la forme d'arbres binaires. Ils ont l'avantage d'être très interprétable.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Random forest ― C'est une technique à base d'arbres qui utilise un très grand nombre d'arbres de décisions construits à partir d'ensembles de caractéristiques aléatoirement sélectionnés. Contrairement à un simple arbre de décision, il n'est pas interprétable du tout mais le fait qu'il ait une bonne performance en fait un algorithme populaire.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Remarque : les random forests sont un type de méthodes d'ensemble.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Boosting ― L'idée des méthodes de boosting est de combiner plusieurs modèles faibles pour former un modèle meilleurs. Les principales méthodes de boosting sont récapitulées dans le tableau ci-dessous :

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Adaptive boosting, Gradient boosting]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; De grands coefficients sont mis sur les erreurs pour s'améliorer à la prochaine étape de boosting

<br>

**71. Weak learners trained on remaining errors**

&#10230; Les modèles faibles sont trainés sur les erreurs restantes

<br>

**72. Other non-parametric approaches**

&#10230; Autres approches non-paramétriques

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-nearest neighbors ― L'algorithme des k-nearest neighbors, aussi connu sous le nom de k-NN, est une approche non-paramétrique où la réponse d'un point est déterminée par la nature de ses k voisins dans l'ensemble d'apprentissage . Il peut être utilisé dans des cadres de classification et de régression.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Remarque : Plus le paramètre k est élevé, plus le biais est élevé, et plus le paramètre k est faible, plus la variance est élevée.

<br>

**75. Learning Theory**

&#10230; Learning Theory

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Inégalité de Boole ― Soit A1,...,Ak k évènements. On a :

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Inégalité d'Hoeffding ― Soit Z1,..,Zm m variables iid tirées d'une distribution de Bernoulli de paramètre ϕ. Soit ˆϕ leur moyenne empirique et γ>0 fixé. On a :

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Remarque : cette inégalité est aussi connue sous le nom de borne de Chernoff.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Erreur d'apprentissage  ― Pour un classifieur donné h, on définit l'erreur d'apprentissage  ˆϵ(h), aussi connu sous le nom de risque empirique ou d'erreur empirique, par :

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; Probablement Approximativement Correct (PAC) ― PAC est un cadre dans lequel de nombreux résultats sur la learning theory ont été prouvé, et a les ensembles d'hypothèses suivants :

<br>

**81: the training and testing sets follow the same distribution **

&#10230; l'ensemble d'apprentissage et l'ensemble d'évaluation  suivent la même distribution

<br>

**82. the training examples are drawn independently**

&#10230; les exemples d'apprentissage  sont tirés indépendamment

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering ― Étant donné un ensemble S={x(1),...,x(d)}, et un ensemble de classifieurs H, on dit que H brise S si pour tout ensemble de labels {y(1),...,y(d)}, on a :

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Théorème de la borne supérieure ― Soit H une hypothèse finie de classe tel que |H|=k et soit δ et la taille de l'échantillon m fixes. Alors, avec un probabilité d'au moins 1−δ, on a :

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; Dimension VC ― La dimension de Vapnik-Chervonenkis (VC) d'une classe d'hypothèses de classes infinies donnée H, que l'on note VC(H), est la taille de l'ensemble le plus grand qui est brisé par H.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Remarque : la dimension VC de H={ensemble of classifieurs linéaires à 2 dimensions} est 3.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Théorème (Vapnik) ― Soit H donné, avec VC(H) = d avec m le nombre d'exemples d'apprentissage . Avec une probabilité d'au moins 1−δ, on a :
