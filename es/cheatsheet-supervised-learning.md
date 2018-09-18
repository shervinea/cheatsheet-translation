**1. Supervised Learning cheatsheet**

&#10230; Hoja de referencia de Aprendizaje supervisado

<br>

**2. Introduction to Supervised Learning**

&#10230; Introducción al aprendizaje supervisado

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Dado un conjunto de puntos {x(1),...,x(m)} asociado a un conjunto de etiquetas {y(1),...,y(m)}, queremos construir un clasificador que aprenda cómo predecir y dado x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Tipo de predicción ― Los diferentes tipos de modelos de predicción se resumen en la siguiente tabla:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regresión, Clasificador, Etiqueta, Ejemplos]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Continuo, Clases, Regresión lineal, Regresión logística, SVM, Naive Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Tipo de modelo ― Los diferentes tipos de modelos se resumen en la siguiente tabla:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Modelo discriminatorio, Modelo generativo, Objetivo, Qué se aprende?, Ilustración, Ejemplos]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Estima P(y|x), Estima P(x|y) para luego deducir P(y|x), Límite de decisión,  	Distribución probabilistica de los datos, Regresiones, SVMs, GDA, Naive Bayes]

<br>

**10. Notations and general concepts**

&#10230; Notación y conceptos generales

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hipótesis ― La hipótesos se representa con h0 y es el modelo que elegimos. Para un dato de entrada x(i), la predicción dada por el modelo se representa como h0(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Función de pérdida ― Una función de pérdida es una función L:(z,y)∈R×Y⟼L(z,y)∈R que toma como entrada el valor z predecido y el valor real esperado y da como resultado qué tan diferentes son ambos. Las funciones de pérdida más comunes son detalladas en la siguiente tabla:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Mínimo error cuadrático, Logistic loss, Hinge loss, Cross-entropy]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Regresión lineal, Regresión logística, SVM, Red neuronal]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Función de costo ― La función de costo J es comunmente utilizada para evaluar el rendimiento de un modelo y se define utilizando la función de pérdida L de la siguiente forma:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Descenso por gradiente ― Siendo α∈R la tasa de aprendizaje, la regla de actualización de descenso por gradiente se expresa junto a la tasa de aprendizaje y la función de costo J de la siguiente manejra:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Observación: El descenso por gradiente estocástico (SGD, por sus siglas en inglés) actualiza el parámetro basandose en cada ejemplo de entenamiento mientras que el descenso por lotes realiza la actualización del parámetro basandose en un conjunto (un lote) de ejemplos de entrenamiento.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Probabilidad ― La probabilidad de un modelo L(θ) dados los parámetros θ es utilizada para hallar los valores óptimos de  θ a través de la probabilidad. En la práctica se utiliza la log-probabilidad ℓ(θ)=log(L(θ)) la cual es fácil de optimizar. Tenemos:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Algoritmo de Newton ― El algoritmo de Newtow es un método numérico para hallar θ tal que ℓ′(θ)=0. Su regla de actualización es:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Observación: la generalización multidimensional, también conocida como método de Newton-Raphson, tiene la siguiente regla de actualización:

<br>

**21. Linear models**

&#10230; Modelos lineales

<br>

**22. Linear regression**

&#10230; Regresión lineal

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Asumimos que y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Ecuaciones normales ― Sea X la matriz de diseño, el valor de θ que minimiza la función de costo es una solución en forma cerrada tal que:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Algoritmo LMS ― Sea α la tasa de aprendizaje, la regla de actualización del algoritmo LMS (del inglés, Least Mean Squares) para el entrenamiendo de m puntos, conocida también como tasa de aprendizaje de Widrow-Hoff, se define como:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Observación: la regla de actualización es un caso particular del ascenso por gradiente.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR ― Regreción local ponderada, conocida por sus siglas en inglés LWR (Locally Weighted Regression), es una variante de la regresión lineal que pondera cada ejemplo de entrenamoento en su función de costo utilizando w(i)(x), la cual se define con el parámetro τ∈R as:

<br>

**28. Classification and logistic regression**

&#10230; Clasificación y regresión logística

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Función sigmoide ― La función sigmoide g, también conocida como la función logística, se define de la siguiente forma:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Regresión logística ― Asumiendo que y|x;θ∼Bernoulli(ϕ), tenemos la siguiente forma:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Observación: no existe solición en forma cerrada para los casos de regresiones logísticas.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; egresión Softmax ― La regresión softmax, también llamada regresión logística multiclase, es utilizada para generalizar regresiones logísticas cuando hay más de dos clases resultantes. Por convención, se define θK=0, lo que hace al parámetro de Bernoulli ϕi de cada clase i igual a:

<br>

**33. Generalized Linear Models**

&#10230; Modelos lineales generalizados

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Familia exponencial - Se dice que una clase de distribuciónes está en una familia exponencial si es posible escribirla en terminos de un parámetro natural, también llamado parámetro canonico o función de enlace, η, un estadístico suficiente T(y) y una función de log-partición (_log-partition function_) a(η) de la siguiente manera:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Observación: comunmente se tiene T(y)=y. Además, exp(−a(η)) puede ser visto como un parámetro de normalización que asegura que las probabilidades sumen uno.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; La siguiente tabla presenta un resumen de las distribuciones exponenciales más comunes:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Distribución, Bernoulli, Gaussiana, Poisson, Geométrica]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Supuestos de los modelos GLM ― Los modelos lineales generalizados (en inglés, Generalized Linear Models) (GLM) tienen como objetivo la predicción de una variable aleatoria y como una función de x∈Rn+1 bajo los siguientes tres supuestos:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Observación: los métodos de mínimos cuadrados ordinarios y regresión logística son casos particulares de los modelos lineales generalizados.

<br>

**40. Support Vector Machines**

&#10230; Máquinas de vectores de soportes

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; El objetivo de las máquinas de vectores de soportes (en inglés, Support Vector Machines) es hallar la línea que maximiza la mínima disancia a la línea.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Clasificador de margen óptio - El clasificador de márgen óptimo h se define de la siguiente manera:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; donde (w,b)∈Rn×R es la solución del siguiente problema de optimización:

<br>

**44. such that**

&#10230; tal que

<br>

**45. support vectors**

&#10230; vectores de soporte

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Observación: la línea se define como wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Función de pérdida de tipo bisagra - La función de pérdida de tipo bisagra es utilizada en la configuración de SVMs y se define de la siguiente manera:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Núcleo - Dado un mapeo de características ϕ, se define el núcleo K como:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; En la práctica, el núcleo K definido por K(x,z)=exp(−||x−z||22σ2) es conocido como núcleo Gaussiano y es comunmente utilizado.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Separabilidad no lineal, Uso de un mapeo de núcleo, Límite de decisión en el espacio original]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Observación: decimos que utilizamos el "truco del núcleo" (en inglés, kernel trick) para calcular la función de costo porque en realidad no necesitamos saber explícitamente el mapeo ϕ que generalmente es muy complicado. En cambio, solo se necesitan los valores K(x,z).

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangiana - Se define la Lagrangiana L(w,b) de la siguiente manera:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Observación: los coeficientes βi son llamados multiplicadores de Lagrange.

<br>

**54. Generative Learning**

&#10230; Aprendizaje generativo

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Un modelo generativo primero trata de aprender como se generan los datos estimando P(x|y), lo que luego podemos utilizar para estimar P(y|x) utilizando el Teorema de Bayes.
<br>

**56. Gaussian Discriminant Analysis**

&#10230; Análisis discriminante Gaussiano

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Marco - El Análisis discriminante Gaussiano asume que y, x|y=0 y x|y=1 son de la siguiente forma:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Estimación - La siguiente tabla resume las estimaciones encontradas al maximizar la probabilidad:

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Supuestos - El modelo Naive Bayes supone que las características de cada punto de los dato son todas independientes:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Soluciones - Maximizar la log-probabilidad da las siguientes soluciones, con k∈{0,1},l∈[[1,L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Observación: Naive Bayes es comunmente utilizado para la clasificación de texto y la detección de correo no deseado (spam).

<br>

**63. Tree-based and ensemble methods**

&#10230; Métodos basados en árboles y conjuntos

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Estos métodos pueden ser utilizados tanto en problemas de regresión como clasificación.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - Árboles de clasificación y regresión (en inglés, Classification and Regression Trees) (CART), comunmente conocidos como árboles de desición, pueden ser representados como árboles binarios. Presentan la ventaja de ser muy interpretables.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Bosques aleatórios (en inglés Random forest) - Es una téctica basada en árboles que utiliza una gran cantidad de árboles de desición cionstruidos a partir de conjuntos de características seleccionadas al azar. A diferencia del árbol de desición simple, la solición del método de bosques aleatórios es dificilmente inrerpretable aunque por su frecuente buen rendimiento es un algoritmo muy popular.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Observación: el método de bosques aleatorios es un típo de método de conjuntos.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Potenciación (o Boosting) - La idea de la potenciación es combinar varios métodos de aprendizaje débiles para conformar uno más fuerte. La siguiente tabla resume los principales tipos de potenciamiento:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Potenciamiento adaptativo, Potenciamiento del gradiente]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; Se pondera fuertemente en los errores para mejorar en el siguiente paso del potenciamiento

<br>

**71. Weak learners trained on remaining errors**

&#10230; Los métodos de aprendizaje débiles entrenan sobre los errores restantes

<br>

**72. Other non-parametric approaches**

&#10230; Otros métodos no paramétrico

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k vecinos más cercanos = El algorítmo de k vecinos más cercanos (en inglés, k-nearest neighbors algorithm), comunmente conocido como k-NN, es un método no parametrico en el que la respuesta a un punto de los datos está determinada por la naturaleza de sus k vecinos del conjunto de datos. El método puede ser utilizado tanto en clasificaciones como regresiones.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Observación: Cuanto mayor es el parámetro k, mayor es el sesgo, y cuanto menor es el parámetro k, mayor la varianza.

<br>

**75. Learning Theory**

&#10230; Teoría del aprendizaje

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Desigualdad de Boole - Sean A1,...,Ak k eventos, Tenemos que:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Desigualdad de Hoeffding - Sean Z1,..,Zm m variables iid () extraídas de una distribución de Bernoulli de parámetro ϕ. Sea ˆϕ su media emprírica y γ>0 fija. Tenemos que:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Observación: esta desigualdad se conoce también como el límite de Chernoff.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Error de entrenamiento - Para un clasificador dado h, se define el error de entrenamiento ˆϵ(h), también conocido como riesgo empírico o error empírico, de la siguiente forma:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; Aprendizaje correcto probablemente aproximado (en inglés, probably approximately correct learning) (PAC) - PAC es un marco bajo el cual se probaron numerosos resultados en teoría de aprendizaje, y presenta los siguientes supuestos:

<br>

**81: the training and testing sets follow the same distribution **

&#10230; los conjuntos de entrenamiento y de prueba siguien la misma distribución

<br>

**82. the training examples are drawn independently**

&#10230; los ejemplos de entrenamiento son sorteados de forma independiente

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering - Dado un conjunto S={x(1),...,x(d)}, y un conjunto de clasificadores H, decimos que H destroza (shatters) S si para cualquier conjunto de etiquietas {y(1),...,y(d)}, tenemos que:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Teorema de la frontera superior - Sea H una clase de hipótesis finita tal que |H|=k y sea δ y el tamaño de la muestra m fijo. Entonces, con probabilidad de al menos 1−δ, tenemos:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; Dimensión VC - La dimensión de Vapnik-Chervonenkis (VC) de una clase de hipótesis finita H, denotada como VC(H), es el tamaño del conjunto más trande destrozado (shattered) por H.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Observación: la dimensión VC de H={conjunto de clasificadores lineales en dos dimensiones} es 3.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Teorema (Vapnik) - Dado H, con VC(H)=d y m el número de ejemplos de entrenamiento. Con probabilidad de al menos 1−δ, tenemos que:
