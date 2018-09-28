1. **Supervised Learning cheatsheet**

&#10230; 监督学习备忘录

<br>

2. **Introduction to Supervised Learning**

&#10230; 监督学习简介

<br>

3. **Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; 给定一组数据点 {x(1),...,x(m)} 和与其对应的输出 {y(1),...,y(m)} ， 我们想要建立一个分类器，学习如何从 x 预测 y。

<br>

4. **Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230;  预测类型 - 不同类型的预测模型总结如下表：

<br>

5. **[Regression, Classifier, Outcome, Examples]**

&#10230;  [回归，分类，输出，例子]

<br>

6. **[Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [连续，类，线性回归，Logistic回归，SVM，朴素贝叶斯]

<br>

7. **Type of model ― The different models are summed up in the table below:**

&#10230; 型号类型 - 不同型号总结如下表：

<br>

8. **[Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [判别模型，生成模型，目标，所学内容，例图，示例]

<br>

9. **[Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [直接估计P(y|x)，估计P(x|y) 然后推导 P(y|x)，决策边界，数据的概率分布，回归，SVMs，GDA，朴素贝叶斯]

<br>

10. **Notations and general concepts**

&#10230; 符号和一般概念

<br>

11. **Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; 假设 - 假设我们选择的模型是hθ 。 对于给定的输入数据 x(i)，模型预测输出是 hθ(x(i))。

<br>

12. **Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; 损失函数 - 损失函数是一个 L:(z,y)∈R×Y⟼L(z,y)∈R 的函数，其将真实数据值 y 和其预测值 z 作为输入，输出它们的不同程度。 常见的损失函数总结如下表：

<br>

13. **[Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [最小二乘误差，Logistic损失，铰链损失，交叉熵]

<br>

14. **[Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [线性回归，Logistic回归，SVM，神经网络]

<br>

15. **Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; 成本函数 - 成本函数 J 通常用于评估模型的性能，使用损失函数 L 定义如下：

<br>

16. **Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; 梯度下降 - 记学习率为 α∈R，梯度下降的更新规则使用学习率和成本函数 J 表示如下：

<br>

17. **Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; 备注：随机梯度下降（SGD）是根据每个训练样本进行参数更新，而批量梯度下降是在一批训练样本上进行更新。

<br>

18. **Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; 似然 - 给定参数 θ 的模型 L（θ）的似然性用于通过最大化似然性来找到最佳参数θ。 在实践中，我们使用更容易优化的对数似然 ℓ(θ)=log(L(θ)) 。我们有

<br>

19. **Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; 牛顿算法 - 牛顿算法是一种数值方法，目的是找到一个 θ 使得 ℓ′(θ)=0. 其更新规则如下：

<br>

20. **Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; 备注：多维泛化，也称为 Newton-Raphson 方法，具有以下更新规则：

<br>

21. **Linear models**

&#10230; 线性模型

<br>

22. **Linear regression**

&#10230; 线性回归

<br>

23. **We assume here that y|x;θ∼N(μ,σ2)**

&#10230; 我们假设 y|x;θ∼N(μ,σ2)

<br>

24. **Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; 正规方程 - 通过设计 X 矩阵，使得最小化成本函数时 θ 有闭式解：

<br>

25. **LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; LMS算法 - 通过 α 学习率，训练集中 m 个数据的最小均方（LMS）算法的更新规则也称为Widrow-Hoff学习规则，如下

<br>

26. **Remark: the update rule is a particular case of the gradient ascent.**

&#10230; 备注：更新规则是梯度上升的特定情况。

<br>

27. **LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR - 局部加权回归，也称为LWR，是线性回归的变体，通过 w(i)(x) 对其成本函数中的每个训练样本进行加权，其中参数 τ∈R 定义为

<br>

28. **Classification and logistic regression**

&#10230; 分类和逻辑回归

<br>

29. **Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Sigmoid函数 - sigmoid 函数 g，也称为逻辑函数，定义如下：

<br>

30. **Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; 逻辑回归 - 我们假设 y|x;θ∼Bernoulli(ϕ) 。 我们有以下形式：

<br>

31. **Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; 备注：对于逻辑回归的情况，没有闭式解。

<br>

32. **Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Softmax回归 - 当存在超过2个结果类时，使用softmax回归（也称为多类逻辑回归）来推广逻辑回归。 按照惯例，我们设置 θK=0，使得每个类 i 的伯努利参数 ϕi 等于：

<br>

33. **Generalized Linear Models**

&#10230; 广义线性模型

<br>

34. **Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; 指数分布族 - 如果可以用自然参数 η，也称为规范参数或链接函数，充分统计量 T(y) 和对数分割函数a（η）来表示，则称一类分布在指数分布族中， 函数如下：

<br>

35. **Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; 备注：我们经常会有 T(y)=y。 此外，exp(−a(η)) 可以看作是归一化参数，确保概率总和为1

<br>

36. **Here are the most common exponential distributions summed up in the following table:**

&#10230; 下表中是总结的最常见的指数分布：

<br>

37. **[Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [分布，伯努利，高斯，泊松，几何]

<br>

38. **Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; GLM的假设 - 广义线性模型（GLM）是旨在将随机变量 y 预测为 x∈Rn+1 的函数，并依赖于以下3个假设：

<br>

39. **Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; 备注：普通最小二乘法和逻辑回归是广义线性模型的特例

<br>

40. **Support Vector Machines**

&#10230; 支持向量机

<br>

41. **The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; 支持向量机的目标是找到使决策界和训练样本之间最大化最小距离的线。

<br>

42. **Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; 最优间隔分类器 - 最优间隔分类器 h 是这样的：

<br>

43. **where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; 其中 (w,b)∈Rn×R 是以下优化问题的解决方案：

<br>

44. **such that**

&#10230; 使得

<br>

45. **support vectors**

&#10230; 支持向量

<br>

46. **Remark: the line is defined as wTx−b=0.**

&#10230; 备注：该线定义为 wTx−b=0。

<br>

47. **Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; 合页损失 - 合页损失用于SVM，定义如下：

<br>

48. **Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; 核 -  给定特征映射 ϕ，我们定义核 K 为：

<br>

49. **In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; 在实践中，由 K(x,z)=exp(−||x−z||22σ2) 定义的核 K 被称为高斯核，并且经常使用这种核。

<br>

50. **[Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [非线性可分性，核映射的使用，原始空间中的决策边界]

<br>

51. **Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; 备注：我们说我们使用“核技巧”来计算使用核的成本函数，因为我们实际上不需要知道显式映射φ，通常，这非常复杂。 相反，只需要 K(x,z) 的值。

<br>

52. **Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; 拉格朗日 - 我们将拉格朗日 L(w,b)  定义如下：

<br>

53. **Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; 备注：系数 βi 称为拉格朗日乘子。

<br>

54. **Generative Learning**

&#10230; 生成学习

<br>

55. **A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; 生成模型首先尝试通过估计 P(x|y) 来模仿如何生成数据，然后我们可以使用贝叶斯法则来估计 P(y|x) 

<br>

56. **Gaussian Discriminant Analysis**

&#10230; 高斯判别分析

<br>

57. **Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; 设置 - 高斯判别分析假设 y 和 x|y=0 且 x|y=1 如下：

<br>

58. **Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; 估计 - 下表总结了我们在最大化似然时的估计值：

<br>

59. **Naive Bayes**

&#10230; 朴素贝叶斯

<br>

60. **Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; 假设 - 朴素贝叶斯模型假设每个数据点的特征都是独立的：

<br>

61. **Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; 解决方案 - 最大化对数似然给出以下解，k∈{0,1}，l∈[[1,L]]

<br>

62. **Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; 备注：朴素贝叶斯广泛用于文本分类和垃圾邮件检测。

<br>

63. **Tree-based and ensemble methods**

&#10230; 基于树的方法和集成方法

<br>

64. **These methods can be used for both regression and classification problems.**

&#10230; 这些方法可用于回归和分类问题。

<br>

65. **CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - 分类和回归树（CART），通常称为决策树，可以表示为二叉树。它们具有可解释性的优点。

<br>

66. **Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; 随机森林 - 这是一种基于树模型的技术，它使用大量的由随机选择的特征集构建的决策树。 与简单的决策树相反，它是高度无法解释的，但其普遍良好的表现使其成为一种流行的算法。

<br>

67. **Remark: random forests are a type of ensemble methods.**

&#10230; 备注：随机森林是一种集成方法。

<br>

68. **Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; 提升 - 提升方法的思想是将一些弱学习器结合起来形成一个更强大的学习器。 主要内容总结在下表中：

<br>

69. **[Adaptive boosting, Gradient boosting]**

&#10230; [自适应增强， 梯度提升] 

<br>

70. **High weights are put on errors to improve at the next boosting step**

&#10230; 在下一轮提升步骤中，错误的会被置于高权重

<br>

71. **Weak learners trained on remaining errors**

&#10230; 弱学习器训练剩余的错误

<br>

72. **Other non-parametric approaches**

&#10230; 其他非参数方法

<br>

73. **k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-最近邻 - k-最近邻算法，通常称为k-NN，是一种非参数方法，其中数据点的判决由来自训练集中与其相邻的k个数据的性质确定。 它可以用于分类和回归。

<br>

74. **Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; 备注：参数 k 越高，偏差越大，参数 k 越低，方差越大。

<br>

75. **Learning Theory**

&#10230; 学习理论

<br>

76. **Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; 联盟 - 让A1，…，Ak 成为 k 个事件。 我们有：

<br>

77. **Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Hoeffding不等式 - 设Z1，...，Zm是从参数 φ 的伯努利分布中提取的 m iid 变量。 设 φ 为其样本均值，固定 γ> 0。 我们有：

<br>

78. **Remark: this inequality is also known as the Chernoff bound.**

&#10230; 备注：这种不平等也被称为 Chernoff 界限。

<br>

79. **Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; 训练误差 - 对于给定的分类器 h，我们定义训练误差 ϵ(h)，也称为经验风险或经验误差，如下：

<br>

80. **Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; 可能近似正确 (PAC) - PAC是一个框架，在该框架下证明了许多学习理论的结果，并具有以下假设：

<br>

81. **the training and testing sets follow the same distribution **

&#10230; 训练和测试集遵循相同的分布

<br>

82. **the training examples are drawn independently**

&#10230; 训练样本是相互独立的

<br>

83. **Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; 打散 - 给定一个集合 S={x(1),...,x(d)} 和一组分类器 H，如果对于任意一组标签 {y(1),...,y(d)} 都能对分，我们称 H 打散 S ，我们有：

<br>

84. **Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; 上限定理 - 设 H 是有限假设类，使得 |H|=k 并且使 δ 和样本大小 m 固定。 然后，在概率至少为 1-δ 的情况下，我们得到：

<br>

85. **VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC维 - 给定无限假设类 H 的 Vapnik-Chervonenkis(VC) 维，注意 VC(H) 是由 H 打散的最大集合的大小。

<br>

86. **Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; 备注：H = {2维线性分类器集} 的 VC 维数为3。

<br>

87. **Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; 定理 (Vapnik)  - 设H，VC(H)=d ，m 为训练样本数。 概率至少为 1-δ，我们有：

<br>

88. **[Introduction, Type of prediction, Type of model]**

&#10230; [简介，预测类型，模型类型]

<br>

89. **[Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230;  [符号和一般概念，损失函数，梯度下降，似然]

<br>

90. **[Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [线性模型，线性回归，逻辑回归，广义线性模型]

<br>

91. **[Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [支持向量机，最优间隔分类器，合页损失，核]

<br>

92. **[Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [生成学习，高斯判别分析，朴素贝叶斯]

<br>

93. **[Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; 树和集成方法，CART，随机森林，提升]

<br>

94. **[Other methods, k-NN]**

&#10230; [其他方法，k-NN]

<br>

95. **[Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [学习理论，Hoeffding不等式，PAC，VC维]
