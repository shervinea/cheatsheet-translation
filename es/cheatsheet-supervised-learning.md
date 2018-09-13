**1. Aprendizaje supervisado**

&#10230;

<br>

**2. Introducción al aprendizaje supervisado**

&#10230;

<br>

**3. Dado un conjunto de puntos {x(1),...,x(m)} asociado a un conjunto de etiquetas {y(1),...,y(m)}, queremos construir un clasificador que aprenda cómo predecir y dado x.**

&#10230;

<br>

**4. Tipo de predicción ― Los diferentes tipos de modelos de predicción se resumen en la siguiente tabla:**

&#10230;

<br>

**5. [Regresión, Clasificador, Etiqueta, Ejemplos]**

&#10230;

<br>

**6. [Continuo, Clases, Regresión lineal, Regresión logística, SVM, Naive Bayes]**

&#10230;

<br>

**7. Tipo de modelo ― Los diferentes tipos de modelos se resumen en la siguiente tabla:**

&#10230;

<br>

**8. [Modelo discriminatorio, Modelo generativo, Objetivo, Qué se aprende?, Ilustración, Ejemplos]**

&#10230;

<br>

**9. [Estima P(y|x), Estima P(x|y) para luego deducir P(y|x), Límite de decisión,  	Distribución probabilistica de los datos, Regresiones, SVMs, GDA, Naive Bayes]**

&#10230;

<br>

**10. Notación y conceptos generales**

&#10230;

<br>

**11. Hipótesis ― La hipótesos se representa con h0 y es el modelo que elegimos. Para un dato de entrada x(i), la predicción dada por el modelo se representa como h0(x(i)).**

&#10230;

<br>

**12. Función de pérdida ― Una función de pérdida es una función L:(z,y)∈R×Y⟼L(z,y)∈R que toma como entrada el valor z predecido y el valor real esperado y da como resultado qué tan diferentes son ambos. Las funciones de pérdida más comunes son detalladas en la siguiente tabla:**

&#10230;

<br>

**13. [Mínimo error cuadrático, Logistic loss, Hinge loss, Cross-entropy]**

&#10230;

<br>

**14. [Regresión lineal, Regresión logística, SVM, Red neuronal]**

&#10230;

<br>

**15. Función de costo ― La función de costo J es comunmente utilizada para evaluar el rendimiento de un modelo y se define utilizando la función de pérdida L de la siguiente forma:**

&#10230;

<br>

**16. Descenso por gradiente ― Siendo α∈R la tasa de aprendizaje, la regla de actualización de descenso por gradiente se expresa junto a la tasa de aprendizaje y la función de costo J de la siguiente manejra:**

&#10230;

<br>

**17. Observación: El descenso por gradiente estocástico (SGD, por sus siglas en inglés) actualiza el parámetro basandose en cada ejemplo de entenamiento mientras que el descenso por lotes realiza la actualización del parámetro basandose en un conjunto (un lote) de ejemplos de entrenamiento.**

&#10230;

<br>

**18. Probabilidad ― La probabilidad de un modelo L(θ) dados los parámetros θ es utilizada para hallar los valores óptimos de  θ a través de la probabilidad. En la práctica se utiliza la log-probabilidad ℓ(θ)=log(L(θ)) la cual es fácil de optimizar. Tenemos:**

&#10230;

<br>

**19. Algoritmo de Newton ― El algoritmo de Newtow es un método numérico para hallar θ tal que ℓ′(θ)=0. Su regla de actualización es:**

&#10230;

<br>

**20. Observación: la generalización multidimensional, también conocida como método de Newton-Raphson, tiene la siguiente regla de actualización:**

&#10230;

<br>

**21. Modelos lineales**

&#10230;

<br>

**22. Regresión lineal**

&#10230;

<br>

**23. ASumimos que y|x;θ∼N(μ,σ2)**

&#10230;

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230;

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230;

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230;

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230;

<br>

**28. Classification and logistic regression**

&#10230;

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230;

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230;

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230;

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230;

<br>

**33. Generalized Linear Models**

&#10230;

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230;

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230;

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230;

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230;

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230;

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230;

<br>

**40. Support Vector Machines**

&#10230;

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230;

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230;

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230;

<br>

**44. such that**

&#10230;

<br>

**45. support vectors**

&#10230;

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230;

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230;

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230;

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230;

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230;

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230;

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230;

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230;

<br>

**54. Generative Learning**

&#10230;

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230;

<br>

**56. Gaussian Discriminant Analysis**

&#10230;

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230;

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230;

<br>

**59. Naive Bayes**

&#10230;

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230;

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230;

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230;

<br>

**63. Tree-based and ensemble methods**

&#10230;

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230;

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230;

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230;

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230;

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230;

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230;

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230;

<br>

**71. Weak learners trained on remaining errors**

&#10230;

<br>

**72. Other non-parametric approaches**

&#10230;

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230;

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230;

<br>

**75. Learning Theory**

&#10230;

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230;

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230;

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230;

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230;

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230;

<br>

**81: the training and testing sets follow the same distribution **

&#10230;

<br>

**82. the training examples are drawn independently**

&#10230;

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230;

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230;

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230;

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230;

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230;
