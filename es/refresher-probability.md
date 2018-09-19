**1. Probabilities and Statistics refresher**

&#10230; 1. Repaso de Probabilidades y Estadística

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; 2. Introducción a la Probabilidad y Combinatorias

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; 3. Espacio muestral ― El conjunto de todos los posibles resultados de un experimento es conocido como el espacio muestral del experimento y se denota por S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; 4. Evento ― Cualquier subconjunto E del espacio muestral se conoce como evento. Es decir, un evento es un conjunto de posibles resultados del experimento. Si el resultado del experimento está contenido en E, entonces decimos que E ha ocurrido.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; 5. Axiomas de Probabilidad Para cada evento E, denotaremos P(E) como la probabilidad de que ocurra el evento E.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; 6. Axioma 1 ― Cada probabilidad está entre 0 y 1 incluida, i.e:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; 7. Axioma 2 ― La probabilidad de que ocurra al menos uno de los eventos elementales en todo el espacio de la muestra es 1, i.e:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; 8. Axioma 3 ― Para cualquier secuencia de eventos mutuamente excluyentes E1,...,En, tenemos:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; 9. Permutación ― Una permutación es un arreglo de r objetos de un grupo de n objetos, en un orden dado. El número de tales arreglos viene dado por P(n,r), definido como:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; 10. Combinación ― Una combinación es un arreglo de r objetos de un grupo de n objetos, donde el orden no importa. El número de tales arreglos viene dado por C(n,r), definido como:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; 11. Observación: observamos que para 0⩽r⩽n, tenemos P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; 12. Probabilidad Condicionada

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; 13. Regla de Bayes ― Para los eventos A y B de tal manera que P(B)>0, tenemos:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; 14. Observación: tenemos P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; 15. Partición ― Sea {Ai,i∈[[1,n]]} tal que para todo i, Ai≠∅. Decimos que {Ai} es una partición si tenemos:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; 16. Observación: para cualquier evento B en el espacio muestral, tenemos P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; 17. Forma extendida de la regla de Bayes ― Sea {Ai,i∈[[1,n]]} una partición del espacio muestral. Tenemos:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; 18. Independencia ― Dos eventos A y B son independientes si y solo si tenemos:

<br>

**19. Random Variables**

&#10230; 19. Variables Aleatorias

<br>

**20. Definitions**

&#10230; 20. Definiciones

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; 21. Variable aleatoria ― Una variable aleatoria, a menudo denominada X, es una función que asigna cada elemento de un espacio muestral a una línea real. 

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; 22. Función de distribución acumulada (CDF) ― La función de distribución acumulada F, que es monótonamente no decreciente y es tal que limx→−∞F(x)=0 y limx→+∞F(x)=1, se define como:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; 23. Observación: tenemos P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; 24. Función de densidad de probabilidad (PDF) ― La función de densidad de probabilidad f es la probabilidad de que X asuma valores entre dos realizaciones adyacentes de la variable aleatoria.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; 25. Relaciones entre PDF y CDF ― Aquí están las propiedades importantes a conocer en los casos discretos (D) y continuos (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; 26. [Caso, CDF F, PDF f, Propiedades de PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; 27. Esperanza y Momentos de la Distribución ― Aquí están las expresiones del valor esperado E[X], valor esperado generalizado E[g(X)], momento kth E[Xk] y función característica ψ(ω) para los casos discretos y continuos:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; 28. Varianza ― La varianza de una variable aleatoria, a menudo señalada como Var(X) o σ2, es una medida de la dispersión de su función de distribución. Se determina de la siguiente manera:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; 29. Desviación estándar ― La desviación estándar de una variable aleatoria a menudo señalada como σ, es una medida de la dispersión de su función de distribución que es compatible con las unidades de la variable aleatoria real. Se determina de la siguiente manera:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; 30. Transformación de variables aleatorias ― Sean las variables X e Y vinculadas por alguna función. Observando fX y fY la función de distribución de X e Y respectivamente, tenemos:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; 31. Regla integral de Leibniz ― Sea g una función de x y potencialmente c, y límites a,b que pueden depender de c. Tenemos:

<br>

**32. Probability Distributions**

&#10230; 32. Distribuciones de Probabilidad

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; 33. Desigualdad de Chebyshev ― Sea X una variable aleatoria con un valor μ. Para k,σ>0, tenemos la siguiente desigualdad:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; 34. Principales distribuciones ― Aquí están las principales distribuciones a tener en cuenta:

<br>

**35. [Type, Distribution]**

&#10230; 35. [Tipo, Distribución]

<br>

**36. Jointly Distributed Random Variables**

&#10230; 36. Distribución Conjunta de Variables Aleatorias

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; 37. Densidad marginal y distribución acumulativa ― De la función de probabilidad de densidad conjunta fXY, tenemos

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; 38. [Caso, Densidad marginal, Función acumulada]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; 39. Densidad condicional ― La densidad condicional de X con respecto a Y, a menudo detonada fX|Y, se define de la siguiente manera:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; 40. Independencia - Se dice que dos variables aleatorias X e Y son independientes si tenemos:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; 41. Covarianza ― Definimos la covarianza de dos variables aleatorias X e Y, que denominamos σ2XY o más comúnmente Cov(X,Y), de la siguiente manera:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; 42. Correlación ― Al observar σX,σY las desviaciones estándar de X e Y, definimos la correlación entre las variables aleatorias X e Y, observadas en ρXY, de la siguiente manera:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; 43. Observación 1: observamos que para cualquier variable aleatoria X,Y, tenemos ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; 44. Observación 2: Si X e Y son independientes, entonces ρXY=0.

<br>

**45. Parameter estimation**

&#10230; 45. Estimación de parámetros

<br>

**46. Definitions**

&#10230; 46. Definiciones

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; 47. Muestra aleatoria ― Una muestra aleatoria es una colección de n variables aleatorias X1,...,Xn independientes y se distribuyen de forma idéntica con X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; 48. Estimador ― Un estimador es una función de los datos que se utilizan para inferir el valor de un parámetro desconocido en un modelo estadístico.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; 49. Sesgo ― El sesgo de un estimador ^θ se define como la diferencia entre el valor esperado de la distribución de ^θ y el valor real, i.e.:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; 50. Observación: se dice que un estimador es insesgado cuando tenemos E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; 51. Estimación de la media

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; 52. Media de la muestra ― La media de la muestra de una muestra aleatoria se utiliza para estimar la media real μ de una distribución, a menudo se denomina ¯¯¯¯¯X y se define de la siguiente manera:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; 53. Observación: La media de la muestra es insesgada, i.e E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; 54. Teorema del Límite Central ― Tengamos una muestra aleatoria X1,...,Xn siguiendo una distribución dada con la media μ y la varianza μ, entonces tenemos:

<br>

**55. Estimating the variance**

&#10230; 55. Estimación de la varianza

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; 56. Variación de la muestra ― La variación de la muestra de una muestra aleatoria se utiliza para estimar la verdadera varianza σ de una distribución, a menudo se indica s2 o ^σ2 y se define de la siguiente manera:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; 57. Observación: la varianza de la muestra no está sesgada, i.e E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; 58. Relación Chi-Cuadrado con la varianza de la muestra ― Sea s2 la varianza de la muestra de una muestra aleatoria. Tenemos:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; 59. [Introducción, Espacio muestral, Evento, Permutación]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; 60. [Probabilidad Condicionada, Regla de Bayes, Independencia]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; 61. [Variables aleatorias, Definiciones, Esperanza, Varianza]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; 62. [Distribuciones de probabilidad, Desigualdad de Chebyshev, Principales distribuciones]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; 63. [Distribución conjunta de variables aleatorias, Densidad, Covarianza, Correlación]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; 64. [Estimación de parámetros, Media, Varianza]
