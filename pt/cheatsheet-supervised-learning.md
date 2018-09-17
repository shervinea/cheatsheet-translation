**1. Supervised Learning cheatsheet**

&#10230; Dicas de Aprendizado Supervisionado

<br>

**2. Introduction to Supervised Learning**

&#10230; Introdução ao Aprendizado Supervisionado

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Dado um conjunto de dados {x(1),...,x(m) associados a um conjunto de resultados {y(1),...,y(m)}, nós queremos construir um classificador que aprende como predizer y baseado em x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Tipos de predição - Os diferentes tipos de modelo de predição estão resumidos na tabela abaixo:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regressão, Classificador, Resultado, Exemplos]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Contínuo, Classe, Regressão Linear, Regressão Logística, Máquina de Vetores de Suporte, Naive Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Tipos de modelo - Os diferentes modelos estão resumidos na tabela abaixo:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Modelo discriminativo, Modelo Generativo, Objetivo, O que é aprendido, Ilustração, Exemplos]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Estimar diretamente P(y|x), Estimar P(x|y) para daí deduzir P(x|y), Fronteira de decisão, Probabilidade da distribuição dos dados, Regressões, MVSs, GDA, Naive Bayes] 

<br>

**10. Notations and general concepts**

&#10230; Notações e conceitos gerais

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hipótese - A hipótese é denominada hθ e é o modelo que escolhemos. Para um determinado dado de entrada x(i) o resultado do modelo de predição é hθ(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Função de perda - A função de perda é definida como L:(z,y)∈R×Y⟼L(z,y)∈R que recebe de entrada o valor z resultado da prefição e o real valor de y e retorna quão diferentes eles são. As funções de valor mais comuns estão resumidas na tabela abaixo:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Erro quadrático mínimo, Perda logística, Hinge loss, Entropia cruzada]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Regressão linear, Regressão Logística, MVS, Rede Neural]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Função de custo - A função de custo J é normalmente usada para avaliar a performance de um modelo e é definida usando a função de perda L como:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Gradiente descendente - Definindo α∈R como a taxa de aprendizado, a regra de atualização para o gradiente descendente é expressa usando a taxa de aprendizado e a função de custo J como: 

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Observação: O gradiente descendente estocástico (GDE) atualiza o parâmetro baseado em cada exemplo de treinamento e o gradiente descendente em lote em um conjunto de exemplos de treinamento.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Probabilidade (Likelihood) - A probabilidade de um modelo L(θ) dado os parâmetros θ é usada para encontrar os parâmetros ótimos θ pela maximização da probabilidade. Na prática, é usado o logaritimo da probabilidade (log-likelihood) ℓ(θ)=log(L(θ)) que é mais simples para se otimizar. Tem-se:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Algoritimo de Newton - O algoritmo de Newton é um método numérico que encontra θ tal que ℓ′(θ)=0. Sua regra de atualização é:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Observação: a generalização multidimensional, também conhecida como o método de Newton-Raphson, tem a seguinte regra de atualização:

<br>

**21. Linear models**

&#10230; Modelos lineares

<br>

**22. Linear regression**

&#10230; Regressão linear

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Assume-se que y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Equações normais - Definindo X como o desenho da matriz, o valor θ que minimiza a função de custo em uma solução de forma fechada é dado por: 

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Algoritimo LMS - Definindo α como a taxa de aprendizado, a regra de atualização do algoritmo de Média de Mínimos Quadrados para um conjunto de treinamento de m pontos, também conhecida como a regra de atualização de Widrow-Hoff, é dada por:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Observação: a regra de atualização é umc aso particular do gradiente ascendente.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR - Regressão Ponderada Localmente (Locally Weighted Regression), também conhecida como LWR, é uma variação da regressão linear que sempre pondera cada exemplo de treinamento em sua função de custo por w(i)(x), que é definida com o parâmetro τ∈R como:

<br>

**28. Classification and logistic regression**

&#10230; Classificação e regressão logística

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Função sigmóide - A função sigmóide g, também conhecida como função logística, é definida como:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Regressão logística - Se assume que y|x;θ∼Bernoulli(ϕ). Tem-se a seguinte fórmula:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Observação: não existe uma fórmula de solução fechada para o caso de regressão logística.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Regressão softmax - A regressão softmax, também chamada de regressão logística multiclasse, é usada para generalizar a regressão logística quando existem mais de 2 classes. Por convenção, definimos θK=0, que faz com que o parâmetro de Bernoulli ϕi de cada classe i seja igual a:

<br>

**33. Generalized Linear Models**

&#10230; Modelos Lineares Generalizados

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Família exponencial - Uma classe de distribuições é chamada de família exponencial se ela puder ser escrita em termos de um parâmetro natural, também chamado de parâmetro canônico ou função de link η, uma estatítica suficiente T(y) e de uma função de partição de log a(η) e é dada por:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Lembrte: em geral tem-se T(y)=y. Também, exp(−a(η)) pode ser definido como o parâmetro de normalização que garantirá que as probabilidade somem um.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Na tabela a seguir estão resumidas as distribuições exponenciais mais comuns:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Distribuição, Bernoulli, Gaussiana, Poisson, Geométrica]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Suposições de GLMs - Modelos Lineares Generalizados (GLM) visa predizer uma variável aleatória y através da função x∈Rn+1 e conta com as 3 seguintes premissas:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Observação: mínimos quadrados ordinários e regressão logística são casos especiais de modelos lineares generalizados.

<br>

**40. Support Vector Machines**

&#10230; Máquinas de Vetores de Suporte

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; O objetivo das máquinas de vetores de suporte (support vector machines) é encontrar a linha que maximiza a distância mínima até a linha.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Classificador de margem ideal - O classificador de margem ideal h é definido por:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; onde (w,b)∈Rn×R é a solução para o seguinte problema de otimização:

<br>

**44. such that**

&#10230; tal como

<br>

**45. support vectors**

&#10230; vetores de suporte

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Observação: a linha é definida como wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Perda de articulação (Hinge loss) - A perda de articulação é usada na configuração das máquinas de vetores de suporte (SVMs) e é definida como:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Kernel - Dado um mapeamento de parâmetro ϕ, o kernel K é definido como:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; Na prática, o kernel K definido por K(x,z)=exp(−||x−z||22σ2) é chamado de kernel Gaussiano e é comumente usado.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Separabilidade não-linear, Uso de mapeamento de kernel, Limite de decisão no espaço original]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Observação: é dito que é usado o "truque de kernel" (kernel trick) para calcular a função de custo usando o kernel porque na verdade não precisamos saber o mapeamento explítico de ϕ, que é muito complicado. Ao invés, apenas os valores K(x,z) são necessários.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangiano - O Lagrangiano L(w,b) é definido por:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Observação: os coeficientes βi são chamados de multiplicadores Lagrangeanos.

<br>

**54. Generative Learning**

&#10230; Aprendizado Generativo

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Um modelo generativo primeiro tenta aprender como o dado é gerado estimando P(x|y), o que pode ser usado para estimar P(y|x) usando a regra de Bayes.

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

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230;

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230;

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230;

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230;

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230;

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230;

<br>

**94. [Other methods, k-NN]**

&#10230;

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230;
