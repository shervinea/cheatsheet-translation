**1. Probabilities and Statistics refresher**

&#10230; Rappel de probabilités et de statistiques

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Introduction aux probabilités et combinatoires

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Univers de probabilités ― L'ensemble de toutes les issues possibles d'une expérience aléatoire est appelé l'univers de probabilités d'une expérience aléatoire et est noté S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Évènement ― Toute partie E d'un univers est appelé un évènement. Ainsi, un évènement est un ensemble d'issues possibles d'une expérience aléatoire. Si l'issue de l'expérience aléatoire est contenue dans E, alors on dit que E s'est produit.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Axiomes de probabilités. Pour chaque évènement E, on note P(E) la probabilité que l'évènement E se produise.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Axiome 1 ― Toute probabilité est comprise entre 0 et 1 inclus, i.e.

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Axiome 2 ― La probabilité qu'au moins un des évènements élementaires de tout l'univers se produise est 1, i.e. 

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Axiome 3 ― Pour toute séquence d'évènements mutuellement exclusifs E1, ..., En, on a :

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Permutation ― Une permutation est un arrangement de r objets parmi n objets, dans un ordre donné. Le nombre de tels arrangements est donné par P(n,r), défini par :

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Combinaison ― Une combinaison est un arrangement de r objets parmi n objets, où l'ordre ne compte pas. Le nombre de tels arrangements est donné par C(n,r), défini par : 

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Remarque : on note que pour 0⩽r⩽n, on a P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Probabilité conditionnelle 

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Théorème de Bayes ― Pour des évènements A et B tel que P(B)>0, on a :

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Remarque : on a P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Partition ― Soit {Ai,i∈[[1,n]]} tel que pour tout i, Ai≠∅. On dit que {Ai} est une partition si l'on a :

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Remarque : pour tout évènement B dans l'univers de probabilités, on a P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Formule étendue du théorème de Bayes ― Soit {Ai,i∈[[1,n]]} une partition de l'univers de probabilités. On a :

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Indépendance ― Deux évènements A et B sont dits indépendants si et seulement si on a :

<br>

**19. Random Variables**

&#10230; Variable aléatoires

<br>

**20. Definitions**

&#10230; Définitions

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Variable aléatoire ― Une variable aléatoire, souvent notée X, est une fonction qui associe chaque élement de l'univers de probabilité à la droite des réels.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Fonction de répartition (CDF) ― La fonction de répartition F, qui est croissante monotone et telle que limx→−∞F(x)=0 et limx→+∞F(x)=1, est définie de la manière suivante :

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Remarque : on a P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Densité de probabilité (PDF) ― La densité de probabilité f est une probabilité que X prend sur les valeurs entre deux réalisations adjacentes d'une variable aléatoire.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Relations avec le PDF et CDF ― Voici les propriétés importantes à savoir dans les cas discret (D) et continu (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Cas, CDF F, PDF F, Propriétés du PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Espérance et Moments de la distribution ― Voici les expressions de l'espérance E[X], l'espérance généralisée E[g(X)], kième moment E[Xk] et fonction caractéristique ψ(ω) dans les cas discret et continu.

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Variance ― La variance d'une variable aléatoire, souvent notée Var(X) ou σ2, est une mesure de la dispersion de ses fonctions de distribution. Elle est déterminée de la manière suivante :

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Écart-type ― L'écart-type d'une variable aléatoire, souvent notée σ, est une mesure de la dispersion de sa fonction de distribution, qui est compatible avec les unités de la variable aléatoire. Il est déterminé de la manière suivante :

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Transformation de variables aléatoires ― Soit X, Y des variables liées par une certaine fonction. En notant fX et fY les fonctions de distribution de X et Y respectivement, on a :

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Loi d'intégration de Leibniz ― Soit g une fonction de x et potentiellement c, et a, b, les limites de l'intervalle qui peuvent dépendre de c. On a :

<br>

**32. Probability Distributions**

&#10230; Distributions de probabilité

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Inégalité de Tchebychev ― Soit X une variable aléatoire de moyenne μ. Pour k,σ>0, on a l'inégalité suivante :

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Distributions importantes ― Voici les distributions importantes à savoir :

<br>

**35. [Type, Distribution]**

&#10230; [Type, Distribution]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Variables aléatoires conjointement distribuées

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Densité marginale et fonction de répartition ― A partir de la densité de probabilité fXY, on a :

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Cas, Densité marginale, Fonction de répartition]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Densité conditionnelle ― La densité conditionnelle de X par rapport à Y, souvent notée fX|Y, est définie de la manière suivante :

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Indépendance ― Deux variables aléatoires X et Y sont dits indépendants si on a :

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Covariance ― On définit la covariance de deux variables aléatoires X et Y, que l'on note σ2XY ou plus souvent Cov(X,Y), de la manière suivante :

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Corrélation ― En notant σX,σY les écart-types de X et Y, on définit la corrélation entre les variables aléatoires X et Y, que l'on note ρXY, de la manière suivante :

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;

<br>

**45. Parameter estimation**

&#10230;

<br>

**46. Definitions**

&#10230;

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;

<br>

**51. Estimating the mean**

&#10230;

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230;

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230;

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;

<br>

**55. Estimating the variance**

&#10230;

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;

<br>
