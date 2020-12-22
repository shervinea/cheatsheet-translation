**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**

&#10230; Памятка: Вероятности и Статистики

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Введение в вероятность и комбинаторику

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Пространство выборки ― Набор всех возможных результатов эксперимента известен как пространство выборки эксперимента и обозначается S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Событие ― Любое подмножество E пространства выборки известно как событие. То есть событие - это набор, состоящий из возможных результатов эксперимента. Если результат эксперимента содержится в E, то мы говорим, что E произошло.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Аксиомы вероятности Для каждого события E мы обозначаем P(E) как вероятность наступления события E.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Аксиома 1 ― Каждая вероятность находится в диапазоне от 0 до 1, то есть:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Аксиома 2 ― Вероятность того, что произойдет хотя бы одно из элементарных событий во всем пространстве выборки, равна 1, то есть:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Аксиома 3 ― Для любой последовательности взаимоисключающих событий E1,...,En, у нас есть:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Перестановка ― Permutation - это расположение r объектов из пула n объектов в заданном порядке. Количество таких расположений определяется как P(n,r), определяемое как:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Комбинация ― Combination - это расположение r объектов из пула, состоящего из n объектов, где порядок не имеет значения. Количество таких расположений дается C(n,r), определяемым как:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Примечание: отметим, что для 0⩽r⩽n, у нас есть P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Условная вероятность

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Правило Байеса ― Для событий A и B таких, что P(B)>0, у нас есть:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Примечание: у нас есть P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Разбиение ― Пусть {Ai,i∈[[1,n]]} таково, что для всех i Ai≠∅. Мы говорим, что {Ai} - разбиение, если у нас есть:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Примечание: для любого события B в пространстве выборки мы имеем P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Расширенная форма правила Байеса ― Пусть {Ai,i∈[[1,n]]} - раздел пространства выборки. У нас есть:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Независимость ― два события A и B независимы тогда и только тогда, когда у нас есть:

<br>

**19. Random Variables**

&#10230; Случайные величины

<br>

**20. Definitions**

&#10230; Определения

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Случайная величина ― случайная величина, которую часто называют X, представляет собой функцию, которая отображает каждый элемент в пространстве выборки на действительную линию.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; (Кумулятивная) функция распределения ― Сumulative distribution function (CDF) F, которая монотонно не убывает и такова, что limx→−∞F(x)=0 и limx→+∞F(x)=1, определяется как:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Примечание: у нас есть P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Функция плотности вероятности ― Probability density function (PDF) f - это вероятность того, что X принимает значения между двумя смежными реализациями случайной величины.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Отношения с участием PDF и CDF ― Вот важные свойства, которые необходимо знать в дискретном (D) и непрерывном (C) случаях.

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Случай, CDF F, PDF f, Свойства PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Ожидание и моменты распределения ― вот выражения ожидаемого значения E[X], обобщенного ожидаемого значения E[g(X)], k-го момента E[Xk] и характеристической функции ψ(ω) для дискретного и непрерывного случаев:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Дисперсия ― Дисперсия случайной величины (Variance), которую часто называют Var(X) или σ2, является мерой разброса её функции распределения. Она определяется следующим образом:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Среднеквадратическое отклонение ― Стандартное отклонение случайной величины (Standard deviation), часто обозначаемой σ, является мерой разброса её функции распределения, которая совместима с единицами измерения фактической случайной величины. Оно определяется следующим образом:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Преобразование случайных величин ― пусть переменные X и Y связаны некоторой функцией. Обозначим fX и fY функцию распределения X и Y соответственно у нас есть:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Интегральное правило Лейбница ― Пусть g является функцией x и потенциально c, и границы a,b могут зависеть от c. У нас есть:

<br>

**32. Probability Distributions**

&#10230; Распределения вероятностей

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Неравенство Чебышева ― Пусть X - случайная величина с математическим ожиданием μ. Для k,σ>0 имеет место неравенство:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Основные распределения ― вот основные распределения, о которых следует помнить:

<br>

**35. [Type, Distribution]**

&#10230; [Тип, Распределения]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Совместно распределенные случайные переменные

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Предельная плотность и кумулятивное распределение ― Из функции вероятности совместной плотности fXY мы имеем

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Случай, Предельная плотность, Кумулятивная функция]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Условная плотность ― Условная плотность X по отношению к Y, часто обозначаемая как fX|Y, определяется следующим образом:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Независимость ― две случайные величины X и Y называются независимыми, если у нас есть:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Ковариация ― мы определяем ковариацию двух случайных величин X и Y, которые мы обозначаем σ2XY или более часто Cov(X,Y), следующим образом:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Корреляция ― Обозначим σX,σY стандартных отклонений X и Y, мы определяем корреляцию между случайными величинами X и Y, отмеченными ρXY, следующим образом:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; Замечание 1: отметим, что для любых случайных величин X, Y мы имеем ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; Замечание 2: Если X и Y независимы, то ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Оценка параметров

<br>

**46. Definitions**

&#10230; Определения

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Случайная выборка ― Случайная выборка представляет собой набор из n случайных величин X1,...,Xn, которые независимы и одинаково распределены с X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Оценщик ― Estimator - это функция данных, которая используется для определения значения неизвестного параметра в статистической модели.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Смещение ― смещение оценки ^θ определяется как разница между ожидаемым значением распределения ^θ и истинным значением, то есть:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Примечание: оценщик считается беспристрастным, если имеется E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Оценка среднего

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Среднее значение выборки ― Среднее значение случайной выборки используется для оценки истинного среднего μ распределения, часто обозначается как ¯¯¯¯¯X и определяется следующим образом:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Примечание: выборочное среднее несмещенное, т. е E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Центральная предельная теорема ― Возьмем случайную выборку X1,...,Xn следуя заданному распределению со средним μ и дисперсией σ2, тогда у нас есть:

<br>

**55. Estimating the variance**

&#10230; Оценка дисперсии

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Выборочная дисперсия ― Дисперсия случайной выборки используется для оценки истинной дисперсии σ2 распределения, часто обозначается как s2 или ^σ2 и определяется следующим образом:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Примечание: дисперсия выборки несмещенная, т. е E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Отношение хи-квадрат с дисперсией выборки ― пусть s2 будет дисперсией выборки случайной выборки. У нас есть:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; [Введение, Пространство выборки, Событие, Перестановка]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; [Условная вероятность, Правило Байеса, Независимость]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Случайные величины, Определения, Ожидание, Дисперсия]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Распределения вероятностей, Неравенство Чебышева, Основные распределения]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Совместно распределенные случайные величины, Плотность, Ковариация, Корреляция]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Оценка параметров, Среднее, Дисперсия]

<br>

**65. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/shervinea

<br>

**66. Translated by X, Y and Z**

&#10230; Российская адаптация: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>

**67. Reviewed by X, Y and Z**

&#10230; Проверено X, Y и Z

<br>

**68. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>

**69. By X and Y**

&#10230; По X и Y

<br>
