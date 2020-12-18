**Supervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning)

<br>

**1. Supervised Learning cheatsheet**

&#10230; Шпаргалка по обучению с учителем

<br>

**2. Introduction to Supervised Learning**

&#10230; Введение в обучение с учителем

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Задан набор точек данных {x(1),...,x(m)} связанный с набором результатов {y(1),...,y(m)}, мы хотим создать классификатор, который научится предсказывать y по x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Тип предсказания ― Различные типы прогнозных моделей перечислены в таблице ниже:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Регрессия, Классификатор, Результат, Примеры]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Непрерывный, Класс, Линейная регрессия, Логистическая регрессия, SVM, Наивный Байес]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Тип модели ― Различные модели перечислены в таблице ниже:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Дискриминационная модель, Генеративная модель, Цель, Что изучено, Иллюстрация, Примеры]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Прямо оценить P(y|x), Оценить P(x|y) затем вывести P(y|x), Граница решения,  	Распределения вероятностей данных, Регрессии, SVM, GDA, Наивный Байес]

<br>

**10. Notations and general concepts**

&#10230; Обозначения и основные понятия

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Гипотеза ― Гипотеза обозначена hθ и является выбранной нами моделью. Для заданных входных данных x(i) предсказанный результат модели обозначен hθ(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Функция потерь ― это функция L:(z,y)∈R×Y⟼L(z,y)∈R которая принимает в качестве входных данных прогнозируемое значение z, соответствующее значению реальных данных y, и выводит, насколько они различны. Общие функции потерь приведены в таблице ниже:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Метод наименьших квадратов (LSE), Логистическая функция потерь, Hinge loss, Перекрёстная энтропия]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Линейная регрессия, Логистическая регрессия, SVM, Нейронная сеть]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Функция стоимости ― функция стоимости J обычно используется для оценки производительности модели и определяется функцией потерь L следующим образом:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Градиентный спуск ― правило обновления для градиентного спуска выражается скоростью обучения α∈R и функцией стоимости J следующим образом:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Примечание: Стохастический градиентный спуск (SGD) обновляет параметры на основе случайного обучающего примера, а пакетный градиентный спуск обновляет параметры используя пакет обучающих примеров.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Likelihood ― Правдоподобие модели L(θ) при заданных параметрах θ используются для нахождения оптимальных параметров θ посредством максимизации правдоподобия. На практике мы используем логарифмическую вероятность ℓ(θ)=log(L(θ)), которую легче оптимизировать. У нас есть:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Алгоритм Ньютона ― это численный метод, который находит такое θ, что ℓ′(θ)=0. Его правила обновления следующие:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Примечание: многомерное обобщение, также известное как метод Ньютона-Рафсона, имеет следующее правило обновления:

<br>

**21. Linear models**

&#10230; Линейные модели

<br>

**22. Linear regression**

&#10230; Линейная регрессия

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Мы предполагаем здесь что y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Нормальные уравнения ― Обозначим X как матрицу данных (объекты-признаки), значение θ, которое минимизирует функцию стоимости, является решением в замкнутой форме, так что:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Алгоритм LMS ― обозначим α скорость обучения, правило обновления алгоритма наименьших средних квадратов (LMS) для обучающего набора из m точек данных, которое также известно как правило обучения Уидроу-Хоффа, выглядит следующим образом:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Примечание: правило обновления - это частный случай градиентного подъема.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR ― Locally Weighted Regression ― Локально взвешенная регрессия, представляет собой вариант линейной регрессии, который взвешивает каждый обучающий пример в его функции стоимости по w(i)(x), которая определяется параметром τ∈R как:

<br>

**28. Classification and logistic regression**

&#10230; Классификация и логистическая регрессия

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Сигмоидальная функция ― сигмовидная функция g, также известная как логистическая функция, определяется следующим образом:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Примечание: there is no closed form solution for the case of logistic regressions.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:

<br>

**33. Generalized Linear Models**

&#10230; Generalized Linear Models

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Примечание: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Here are the most common exponential distributions summed up in the following table:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Distribution, Bernoulli, Gaussian, Poisson, Geometric]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Примечание: ordinary least squares and logistic regression are special cases of generalized linear models.

<br>

**40. Support Vector Machines**

&#10230; Support Vector Machines

<br>

**41. The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; The goal of support vector machines is to find the line that maximizes the minimum distance to the line.

<br>

**42. Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Optimal margin classifier ― The optimal margin classifier h is such that:

<br>

**43. where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; where (w,b)∈Rn×R is the solution of the following optimization problem:

<br>

**44. such that**

&#10230; such that

<br>

**45. support vectors**

&#10230; support vectors

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Примечание: the line is defined as wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Примечание: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangian ― We define the Lagrangian L(w,b) as follows:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Примечание: the coefficients βi are called the Lagrange multipliers.

<br>

**54. Generative Learning**

&#10230; Generative Learning

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Gaussian Discriminant Analysis

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Примечание: Naive Bayes is widely used for text classification and spam detection.

<br>

**63. Tree-based and ensemble methods**

&#10230; Tree-based and ensemble methods

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; These methods can be used for both regression and classification problems.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Примечание: random forests are a type of ensemble methods.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Adaptive boosting, Gradient boosting]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; High weights are put on errors to improve at the next boosting step

<br>

**71. Weak learners trained on remaining errors**

&#10230; Weak learners trained on remaining errors

<br>

**72. Other non-parametric approaches**

&#10230; Other non-parametric approaches

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Примечание: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.

<br>

**75. Learning Theory**

&#10230; Learning Theory

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Union bound ― Let A1,...,Ak be k events. We have:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Примечание: this inequality is also known as the Chernoff bound.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:**

&#10230; Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:

<br>

**81: the training and testing sets follow the same distribution**

&#10230; the training and testing sets follow the same distribution

<br>

**82. the training examples are drawn independently**

&#10230; the training examples are drawn independently

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Примечание: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [Introduction, Type of prediction, Type of model]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [Notations and general concepts, loss function, gradient descent, likelihood]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [Linear models, linear regression, logistic regression, generalized linear models]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [Trees and ensemble methods, CART, Random forest, Boosting]

<br>

**94. [Other methods, k-NN]**

&#10230; [Other methods, k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [Learning theory, Hoeffding inequality, PAC, VC dimension]
