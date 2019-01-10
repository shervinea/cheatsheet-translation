1. **Probabilities and Statistics refresher**

&#10230;
機率和統計回顧
<br>

2. **Introduction to Probability and Combinatorics**

&#10230;
幾率與組合數學介紹
<br>

3. **Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230;
樣本空間 - 一個實驗的所有可能結果的集合稱之為這個實驗的樣本空間，記做 S
<br>

4. **Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230;
事件 - 樣本空間的任何子集合 E 被稱之為一個事件。也就是說，一個事件是實驗的可能結果的集合。如果該實驗的結果包含 E，我們稱我們稱 E 發生
<br>

5. **Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230;
機率公理。對於每個事件 E，我們用 P(E) 表示事件 E 發生的機率
<br>

6. **Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230;
公理 1 - 每一個機率值介於 0 到 1 之間，包含兩端點。即：
<br>

7. **Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230;
公理 2 - 至少一個基本事件出現在整個樣本空間中的機率是 1。即：
<br>

8. **Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230;
公理 3 - 對於任何互斥的事件 E1,...,En，我們定義如下：
<br>

9. **Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230;
排列 - 排列指的是從 n 個相異的物件中，取出 r 個物件按照固定順序重新安排，這樣安排的數量用 P(n,r) 來表示，定義為：
<br>

10. **Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230;
組合 - 組合指的是從 n 個物件中，取出 r 個物件，但不考慮他的順序。這樣組合要考慮的數量用 C(n,r) 來表示，定義為：
<br>

11. **Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230;
注意：對於 0⩽r⩽n，我們會有 P(n,r)⩾C(n,r)
<br>

12. **Conditional Probability**

&#10230;
條件機率
<br>

13. **Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230;
貝氏定理 - 對於事件 A 和 B 滿足 P(B)>0 時，我們定義如下：
<br>

14. **Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230;
注意：P(A∩B)=P(A)P(B|A)=P(A|B)P(B)
<br>

15. **Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230;
分割 - 令 {Ai,i∈[[1,n]]} 對所有的 i，Ai≠∅，我們說 {Ai} 是一個分割，當底下成立時：
<br>

16. **Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230;
注意：對於任何在樣本空間的事件 B 來說，P(B)=n∑i=1P(B|Ai)P(Ai)
<br>

17. **Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230;
貝氏定理的擴展 - 令 {Ai,i∈[[1,n]]} 為樣本空間的一個分割，我們定義：
<br>

18. **Independence ― Two events A and B are independent if and only if we have:**

&#10230;
獨立 - 當以下條件滿足時，兩個事件 A 和 B 為獨立事件：
<br>

19. **Random Variables**

&#10230;
隨機變數
<br>

20. **Definitions**

&#10230;
定義
<br>

21. **Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230;
隨機變數 - 一個隨機變數 X，它是一個將樣本空間中的每個元素映射到實數域的函數
<br>

22. **Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230;
累積分佈函數 (CDF) - 累積分佈函數 F 是單調遞增的函數，其 limx→−∞F(x)=0 且 limx→+∞F(x)=1，定義如下：
<br>

23. **Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230;
注意：P(a<X⩽B)=F(b)−F(a)
<br>

24. **Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230;
機率密度函數 - 機率密度函數 f 是隨機變數 X 在兩個相鄰的實數值附近取值的機率
<br>

25. **Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230;
機率密度函數和累積分佈函數的關係 - 底下是一些關於離散 (D) 和連續 (C) 的情況下的重要屬性
<br>

26. **[Case, CDF F, PDF f, Properties of PDF]**

&#10230;
[情況, 累積分佈函數 F, 機率密度函數 f, 機率密度函數的屬性]
<br>

27. **Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230;
分佈的期望值和動差 - 底下是期望值 E[X]、一般期望值  E[g(X)]、第 k 個動差和特徵函數 ψ(ω) 在離散和連續的情況下的表示式：
<br>

28. **Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230;
變異數 - 隨機變數的變異數通常表示為 Var(X) 或 σ2，用來衡量一個分佈離散程度的指標。其表示如下：
<br>

29. **Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230;
標準差 - 一個隨機變數的標準差通常表示為 σ，用來衡量一個分佈離散程度的指標，其單位和實際的隨機變數相容，表示如下：
<br>

30. **Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230;
隨機變數的轉換 - 令變數 X 和 Y 由某個函式連結在一起。我們定義 fX 和 fY 是 X 和 Y 的分佈函式，可以得到：
<br>

31. **Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230;
萊布尼茲積分法則 - 令 g 為 x 和 c 的函數，a 和 b 是依賴於 c 的的邊界，我們得到：
<br>

32. **Probability Distributions**

&#10230;
機率分佈
<br>

33. **Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230;
柴比雪夫不等式 - 令 X 是一隨機變數，期望值為 μ。對於 k, σ>0，我們有以下不等式：
<br>

34. **Main distributions ― Here are the main distributions to have in mind:**

&#10230;
主要的分佈 - 底下是我們需要熟悉的幾個主要的不等式：
<br>

35. **[Type, Distribution]**

&#10230;
[種類, 分佈]
<br>

36. **Jointly Distributed Random Variables**

&#10230;
聯合分佈隨機變數
<br>

37. **Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230;
邊緣密度和累積分佈 - 從聯合密度機率函數 fXY 中我們可以得到：
<br>

38. **[Case, Marginal density, Cumulative function]**

&#10230;
[種類, 邊緣密度函數, 累積函數]
<br>

39. **Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230;
條件密度 - X 對於 Y 的條件密度，通常用 fX|Y 表示如下：
<br>

40. **Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230;
獨立 - 當滿足以下條件時，我們稱隨機變數 X 和 Y 互相獨立：
<br>

41. **Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230;
共變異數 - 我們定義隨機變數 X 和 Y 的共變異數為 σ2XY 或 Cov(X,Y) 如下：
<br>

42. **Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230;
相關性 - 我們定義 σX、σY 為 X 和 Y 的標準差，而 X 和 Y 的相關係數 ρXY 定義如下：
<br>

43. **Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;
注意一：對於任何隨機變數 X 和 Y 來說，ρXY∈[−1,1] 成立
<br>

44. **Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;
注意二：當 X 和 Y 獨立時，ρXY=0
<br>

45. **Parameter estimation**

&#10230;
參數估計
<br>

46. **Definitions**

&#10230;
定義
<br>

47. **Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;
隨機抽樣 - 隨機抽樣指的是 n 個隨機變數 X1,...,Xn 和 X 獨立且同分佈的集合
<br>

48. **Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;
估計量 - 估計量是一個資料的函數，用來推斷在統計模型中未知參數的值
<br>

49. **Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;
偏差 - 一個估計量的偏差 ^θ 定義為 ^θ 分佈期望值和真實值之間的差距：
<br>

50. **Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;
注意：當 E[^θ]=θ 時，我們稱為不偏估計量
<br>

51. **Estimating the mean**

&#10230;
預估平均數
<br>

52. **Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯X and is defined as follows:**

&#10230;
樣本平均 - 一個隨機樣本的樣本平均是用來預估一個分佈的真實平均 μ，通常我們用 ¯X 來表示，定義如下：
<br>

53. **Remark: the sample mean is unbiased, i.e E[¯X]=μ.**

&#10230;
注意：當 E[¯X]=μ 時，則為不偏樣本平均
<br>

54. **Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;
中央極限定理 - 當我們有一個隨機樣本 X1,...,Xn 滿足一個給定的分佈，其平均數為 μ，變異數為 σ2，我們有：
<br>

55. **Estimating the variance**

&#10230;
估計變異數
<br>

56. **Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;
樣本變異數 - 一個隨機樣本的樣本變異數是用來估計一個分佈的真實變異數 σ2，通常使用 s2 或 ^σ2 來表示，定義如下：
<br>

57. **Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;
注意：當 E[s2]=σ2 時，稱之為不偏樣本變異數
<br>

58. **Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;
與樣本變異數的卡方關聯 - 令 s2 是一個隨機樣本的樣本變異數，我們可以得到：
<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230;
[介紹, 樣本空間, 事件, 排列]
<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230;
[條件機率, 貝氏定理, 獨立性]
<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230;
[隨機變數, 定義, 期望值, 變異數]
<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230;
[機率分佈, 柴比雪夫不等式, 主要分佈]
<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230;
[聯合分佈隨機變數, 密度, 共變異數, 相關]
<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230;
[參數估計, 平均數, 變異數]