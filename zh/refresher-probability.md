1. **Probabilities and Statistics refresher**

&#10230; 概率和统计回顾

<br>

2. **Introduction to Probability and Combinatorics**

&#10230; 概率和组合导引

<br>

3. **Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; 样本空间 - 一个实验的所有可能结果的集合称为实验的样本空间，记作 S。

<br>

4. **Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; 事件 - 样本空间的任何子集 E 被称为一个事件。即，一个事件是一个包含可能结果的集合。如果该实验的结果包含在 E 内，那么我们称 E 发生。

<br>

5. **Axioms of probability - For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; 概率论公理 - 对每个事件 E，我们记 P(E) 为事件 E 出现的概率。

<br>

6. **Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; 公理 1 - 每个概率是在 0 到 1 之间的，包含端点，即：

<br>

7. **Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; 公理 2 - 在整个样本空间中至少一个原子事件会出现的概率是 1，即：

<br>

8. **Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; 公理 3 - 对任何互斥事件 E1,...,En 序列，我们有：

<br>

9. **Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; 置换 - 一个置换是从 n 个对象的池子中按照给定次序安置 r 个对象。这样的安置的数目由 P(n,r) 表示，定义为：

<br>

10. **Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; 组合 - 一个组合是从 n 个对象的池子中无序安置 r 个对象。这样的安置的数目由 C(n,r) 表示，定义为：

<br>

11. **Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; 注：对 0⩽r⩽n，我们有 P(n,r)⩾C(n,r)

<br>

12. **Conditional Probability**

&#10230; 条件概率

<br>

13. **Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; 贝叶斯规则 - 对事件 A 和 B 满足 P(B)>0，我们有：

<br>

14. **Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; 注：我们有 P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

15. **Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; 分划 - 令 {Ai,i∈[[1,n]]} 对所有 i，Ai≠∅。我们称 {Ai} 为一个分划，当有：

<br>

16. **Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; 注：对任意在样本空间中的事件 B 我们有 P(B)=n∑i=1P(B|Ai)P(Ai)。

<br>

17. **Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; 贝叶斯规则的扩展形式 - 令 {Ai,i∈[[1,n]]} 为样本空间的一个分划，我们有：

<br>

18. **Independence ― Two events A and B are independent if and only if we have:**

&#10230; 独立 - 两个事件 A 和 B 是独立的当且仅当我们有：

<br>

19. **Random Variables**

&#10230; 随机变量

<br>

20. **Definitions**

&#10230; 定义

<br>

21. **Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; 随机变量 - 一个随机变量，通常记作 X，是一个将在一个样本空间中的每个元素映射到一个实值的函数。

<br>

22. **Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; 累积分布函数（CDF） - 累积分布函数 F，是单调不减的，其 limx→−∞F(x)=0 且 limx→+∞F(x)=1，定义为：

<br>

23. **Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; 注：我们有 P(a<X⩽B)=F(b)−F(a)。

<br>

24. **Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; 概率密度函数（PDF）- 概率密度函数 f 是 X 取值在两个相邻随机变量的实现间的概率。

<br>

25. **Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; PDF 和 CDF 的关系 - 这里是离散和连续场景下的重要性质。

<br>

26. **[Case, CDF F, PDF f, Properties of PDF]**

&#10230; [类型，CDF F，PDF f，PDF 的性质]

<br>

27. **Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; 分布的期望和矩 - 这里是期望值 E[X] 、一般期望值 E[g(X)]、第 k 阶矩 E[Xk] 和特征函数 ψ(ω) 在离散和连续场景下的表达式：

<br>

28. **Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; 方差 - 随机变量的方差通常记作 Var(X) 或者 σ2，是分布函数的扩散性的一个度量函数。定义如下：

<br>

29. **Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; 标准差 - 随机变量的标准差，通常记作 σ，是分布函数扩散性的一个和实际随机变量值单位相当的度量函数。定义如下：

<br>

30. **Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; 随机变量的变换 - 令变量 X 和 Y 由某个函数联系在一起。记 fX 和 fY 分别为 X 和 Y 的分布函数，我们有：

<br>

31. **Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; 莱布尼兹积分法则 - 令 g 为 x 和 c 的函数，a,b 是可能依赖于 c 的边界。我们有：

<br>

32. **Probability Distributions**

&#10230; 概率分布

<br>

33. **Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; 切比雪夫不等式 - 令 X 为随机变量期望值为 μ。对 k,σ>0，我们有下列不等式：
 
<br>

34. **Main distributions ― Here are the main distributions to have in mind:**

&#10230; 主要的分布 - 这里是主要需要记住的分布：

<br>

35. **[Type, Distribution]**

&#10230; [类型，分布]

<br>

36. **Jointly Distributed Random Variables**

&#10230; 联合分布随机变量

<br>

37. **Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; 边缘密度和累积分布 - 从联合密度概率函数 fXY，我们有：

<br>

38. **[Case, Marginal density, Cumulative function]**

&#10230; [类型，边缘密度函数，累积函数]

<br>

39. **Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; 条件密度 - X 关于 Y 的条件密度通常记作 fX|Y，定义如下：

<br>

40. **Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; 独立性 - 两个随机变量 X 和 Y 被称为独立的当我们有：

<br>

41. **Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; 协方差 - 我们定义两个随机变量 X 和 Y 的协方差，记作 σ2XY 或者更常见的 Cov(X,Y)，如下：

<br>

42. **Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; 相关性 - 记 σX,σY 为 X 和 Y 的标准差，我们定义随机变量 X 和 Y 的相关性，记作 ρXY，如下：

<br>

43. **Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; 注 1：对任何随机变量 X,Y，我们有 ρXY∈[−1,1]。

<br>

44. **Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; 注 2：当 X 和 Y 是独立的时，有 ρXY=0。

<br>

45. **Parameter estimation**

&#10230; 参数估计

<br>

46. **Definitions**

&#10230; 定义

<br>

47. **Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; 随机采样 - 一个随机采样是 n 个和 X 独立同分布的随机变量 X1,...,Xn 的集。

<br>

48. **Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; 估计器 - 估计器是一个用来推断一个统计模型中未知参数值的关于数据的函数。

<br>

49. **Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**
 
&#10230; 偏差 - 估计器 ^θ 的偏差定义为 ^θ 分布的期望值和真实值间的差距，即：

<br>

50. **Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; 注：估计器被称为无偏的当我们有 E[^θ]=θ。

<br>

51. **Estimating the mean**

&#10230; 估计均值

<br>

52. **Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯X and is defined as follows:**

&#10230; 样本均值 - 随机采样的样本均值用来估计一个分布的真实的均值，常记作 ¯X，定义如下：

<br>

53. **Remark: the sample mean is unbiased, i.e E[¯X]=μ.**

&#10230; 注：样本均值无偏的，即 E[¯X]=μ。

<br>

54. **Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; 中央极限定理 - 令一个随机采样 X1,...,Xn 满足一个给定分布均值 μ 方差 σ2，我们有：

<br>

55. **Estimating the variance**

&#10230; 估计方差

<br>

56. **Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; 样本方差 - 样本方差用来估计一个分布的真实方差 σ2，常记作 s2 或者 ^σ2，定义如下：

<br>

57. **Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; 注：样本方差是无偏的，即 E[s2]=σ2。

<br>

58. **Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Chi-方

<br>

59. **[Introduction, Sample space, Event, Permutation]**

&#10230; [导引，样本空间，事件，置换]

<br>

60. **[Conditional probability, Bayes' rule, Independence]**

&#10230; [条件概率，贝叶斯规则，独立性]

<br>

61. **[Random variables, Definitions, Expectation, Variance]**

&#10230; [随机变量，定义，期望，方差]

<br>

62. **[Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [概率分布，切比雪夫不等式，主要的分布]

<br>

63. **[Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [联合分布随机变量，密度函数，协方差，相关性]

<br>

64. **[Parameter estimation, Mean, Variance]**

&#10230; [参数估计，均值，方差]
