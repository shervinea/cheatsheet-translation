**1. Unsupervised Learning cheatsheet**

&#10230; Pense-bête d'apprentissage non-supervisé

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Introduction à l'apprentissage non-supervisé

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivation ― Le but de l'apprentissage non-supervisé est de trouver des formes cachées dans un jeu de données non-labelées {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Inégalité de Jensen ― Soit f une fonction convexe et X une variable aléatoire. On a l'inégalité suivante :

<br>

**5. Clustering**

&#10230; Partitionnement

<br>

**6. Expectation-Maximization**

&#10230; Espérance-Maximisation

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Variables latentes ― Les variables latentes sont des variables cachées/non-observées qui posent des difficultés aux problèmes d'estimation, et sont souvent notées z. Voici les cadres dans lesquelles les variables latentes sont le plus fréquemment utilisées :

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Cadre, Variable latente z, Commentaires]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Mixture de k gaussiennes, Analyse factorielle]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algorithme ― L'algorithme d'espérance-maximisation (EM) est une méthode efficace pour estimer le paramètre θ. Elle passe par le maximum de vraisemblance en construisant un borne inférieure sur la vraisemblance (E-step) et optimisant cette borne inférieure (M-step) de manière successive :

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step : Évaluer la probabilité postérieure Qi(z(i)) que chaque point x(i) provienne d'une partition particulière z(i) de la manière suivante :

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step : Utiliser les probabilités postérieures Qi(z(i)) en tant que coefficients propres aux partitions sur les points x(i) pour ré-estimer séparemment chaque modèle de partition de la manière suivante :

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Initialisation de gaussiennes, Étape d'espérance, Étape de maximisation, Convergence]

<br>

**14. k-means clustering**

&#10230; Partitionnement k-means

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; On note c(i) la partition du point i et μj le centre de la partition j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algorithme ― Après avoir aléatoirement initialisé les centroïdes de partitions μ1,μ2,...,μk∈Rn, l'algorithme k-means répète l'étape suivante jusqu'à convergence :

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Initialisation des moyennes, Assignation de la partition, Mise à jour des moyennes, Convergence]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Fonction de distortion ― Pour voir si l'algorithme converge, on regarde la fonction de distortion définie de la manière suivante :

<br>

**19. Hierarchical clustering**

&#10230; Regroupement hiérarchique

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algorithme ― C'est un algorithme de partitionnement avec une approche hiérarchique qui construit des partitions intriqués de manière successive.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Types ― Il y a différents types d'algorithme de regroupement hiérarchique qui ont pour but d'optimiser différents fonctions objectif, récapitulés dans le tableau ci-dessous :

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ward linkage, Average linkage, Complete linkage]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Minimiser la distance au sein d'une partition, Minimiser la distance moyenne entre chaque paire de partitions, Minimiser la distance maximale entre les paires de partition]

<br>

**24. Clustering assessment metrics**

&#10230; Indicateurs d'évaluation de clustering

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; Dans le cadre de l'apprentissage non-supervisé, il est souvent difficile d'évaluer la performance d'un modèle vu que les vrais labels ne sont pas connus (contrairement à l'apprentissage supervisé).

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Coefficient silhouette ― En notant a et b la distance moyenne entre un échantillon et tous les autres points d'une même classe, et entre un échantillon et tous les autres points de la prochaine partition la plus proche, le coefficient silhouette s d'un échantillon donné est défini de la manière suivante :

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Index de Calinski-Harabaz ― En notant k le nombre de partitions, Bk et Wk les matrices de dispersion entre-partitions et au sein d'une même partition sont définis respectivement par :

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; l'index de Calinski-Harabaz s(k) renseigne sur la qualité des partitions, de sorte à ce qu'un score plus élevé indique des partitions plus denses et mieux séparées entre elles. Il est défini par :

<br>

**29. Dimension reduction**

&#10230; Réduction de dimension

<br>

**30. Principal component analysis**

&#10230; Analyse des composantes principales

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; C'est une technique de réduction de dimension qui trouve les directions maximisant la variance, vers lesquelles les données sont projetées.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Valeur propre, vecteur propre ― Soit une matrice A∈Rn×n, λ est dit être une valeur propre de A s'il existe un vecteur z∈Rn∖{0}, appelé vecteur propre, tel que l'on a :

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Théorème spectral ― Soit A∈Rn×n. Si A est symmétrique, alors A est diagonalisable par une matrice réelle orthogonale U∈Rn×n. En notant Λ=diag(λ1,...,λn), on a :

<br>

**34. diagonal**

&#10230; diagonal

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Remarque : le vecteur propre associé à la plus grande valeur propre est appelé le vecteur propre principal de la matrice A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algorithme ― La procédure d'analyse des composantes principales (en anglais *PCA - Principal Component Analysis*) est une technique de réduction de dimension qui projette les données sur k dimensions en maximisant la variance des données de la manière suivante :

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Étape 1 : Normaliser les données pour avoir une moyenne de 0 et un écart-type de 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Étape 2 : Calculer Σ=1mm∑i=1x(i)x(i)T∈Rn×n, qui est symmétrique et aux valeurs propres réelles.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Étape 3 : Calculer u1,...,uk∈Rn les k valeurs propres principales orthogonales de Σ, i.e. les vecteurs propres orthogonaux des k valeurs propres les plus grandes.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Étape 4 : Projeter les données sur spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Cette procédure maximise la variance sur tous les espaces à k dimensions.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Données dans l'espace initial, Trouver les composantes principales, Données dans l'espace des composantes principales]

<br>

**43. Independent component analysis**

&#10230; Analyse en composantes indépendantes

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; C'est une technique qui vise à trouver les sources génératrices sous-jacentes.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Hypothèses ― On suppose que nos données x ont été générées par un vecteur source à n dimensions s=(s1,...,sn), où les si sont des variables aléatoires indépendantes, par le biais d'une matrice de mélange et inversible A de la manière suivante : 

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; Le but est de trouver la matrice de démélange W=A−1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Algorithme d'ICA de Bell and Sejnowski ― Cet algorithme trouve la matrice de démélange W en suivant les étapes ci-dessous :

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Écrire la probabilité de x=As=W−1s par :

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Écrire la log vraisemblance de notre ensemble d'apprentissage {x(i),i∈[[1,m]]} et en notant g la fonction sigmoïde par :

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Par conséquent, l'algorithme du gradient stochastique est tel que pour chaque example de ensemble d'apprentissage x(i), on met à jour W de la manière suivante :
