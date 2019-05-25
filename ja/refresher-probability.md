**1. Probabilities and Statistics refresher**

&#10230;確率と統計

<br>

**2. Introduction to Probability and Combinatorics**

&#10230;確率と組合せの紹介

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230;標本空間 - 試行可能なすべての結果の集合は標本空間として知られ、Sと表します。

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230;事象 - 標本空間のすべての部分集合のEを事象と言います。つまり事象は試行可能な結果で構成された集合です。試行結果がEに含まれるなら、Eが発生した言います。

<br>

**5. Axioms of probability ― For each event E, we denote P(E) as the probability of event E occuring.**

&#10230;確率の公理 - 各事象Eに対して、事象Eが起こる確率をP(E)と書きます。

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230;公理1 - すべての確立は0と1の間に含まれ次のようになります：

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230;公理2 - 全体の標本空間で少なくとも一つの根元事象が起こる確率は1で次のようになります：

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230;公理3 - 相互に排他的なとある連続した事象E1,...Enに対し、次のようになります：

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230;順列(Permutation) - 順列はn個の中からr個を順番を考慮して並べられた配列です。このような配列の数はP(n, r)と表し、次のように定義します:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230;組合せ(Combination) - 組合せはn個の中からr個の順番を勘案しない配列です。このような配列の数はC(n,r)と表し、次のように定義します:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230;注釈: 0⩽r⩽nに対し、P(n,r)⩾C(n,r)となります。

<br>

**12. Conditional Probability**

&#10230;条件付き確率

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230;ベイズの定理 - P(B)>0のような事象A, Bに対して次となります:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230;注釈: P(A∩B)=P(A)P(B|A)=P(A|B)P(B)となります。

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230;分割(Partition) - {Ai,i∈[[1,n]]}はすべてのiに対してAi≠∅としましょう。{Ai}が次のような場合、分割と言います:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230;注釈: 標本空間で任意の事象Bに対して、P(B)=n∑i=1P(B|Ai)P(Ai)となります。

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230;ベイズの定理の応用 - {Ai,i∈[[1,n]]}を標本空間の分割としましょう。次のようになります:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230;独立性 - 次の場合のみ事象AとBは独立であるといいます:

<br>

**19. Random Variables**

&#10230;確率変数

<br>

**20. Definitions**

&#10230;定義

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230;確率変数 - 確率変数は主にXと表記し標本空間のすべての要素に実線で対応する関数です。

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230;累積分布関数(CDF) - 単調非減少の累積分布関数Fはlimx→−∞F(x)=0 and limx→+∞F(x)=1となり次のように定義します:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230;注釈: P(a<X⩽B)=F(b)−F(a)となります。

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230;確率密度関数(PDF) - 確率密度関数Fは隣接する二つの確率変数の間に置かれる確率です。

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230;PDFとCDFとの関係性 - 離散(D)と連続(C)の例から知るべき重要な特性があります。

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230;[例、CDF F、PDF f、PDFの特性]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230;分布の期待値と積率 - 離散または連続の場合、期待値E[X]、一般化した期待値E[g(X)]、k次の積率E[Xk]と特性関数ψ(ω):

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230;分散(Variance) - 確率変数の分散は主にVar(X)またはσ2と表記し、分布関数の散布度を測定したものです。次のように決まります。

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230;標準偏差(Standard deviation) - 確率変数の標準偏差は主にσと表記し実確率変数の単位をしようする分布関数の散布度を測定したものです。次のように決まります。

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230;確率変数の変換 - 変数XとYは任意の関数に繋がってるとします。fXとfYに各々XとYの分布関数を表記すると次のようになります:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230;ライプニッツ積分法 - gをxの関数とし、暫定的にcとしましょう。そしてcに従属的な境界a,bに対して次のようになります。

<br>

**32. Probability Distributions**

&#10230;確率分布

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230;チェビシェフの不等式 - Xを期待値μをの確率変数とします。kに対して、σ>0なら次のような不等式を持ちます。

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230;主な分布 - 覚えておくべき主な分布があります:

<br>

**35. [Type, Distribution]**

&#10230;[タイプ、分布]

<br>

**36. Jointly Distributed Random Variables**

&#10230;結合確率変数

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230;周辺密度と累積分布 - 結合密度確率関数fXYから次のようになります。

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230;[例,、周辺密度、累積関数]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230;条件部密度(Conditional density) - Yに対するXの条件部密度は主にfx|Yと表記され、次のように定義されます:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230;独立性(Independence) - 二つの確率変数XとYは次の場合、独立的と言います。

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230;共分散(Covariance) - 次のようにふたつの確率変数X,Yの共分散をσ2XYまたはさらに一般的にはCov(X,Y)で定義します。

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230;相関関係(Correlation) - X, Yの標準変数をσX,σYで表記し、確率変数X,Yの相関関係をρXYで表記し、次のように定義します。

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;注釈 1: 任意の確率変数X,Yに対してρXY∈[−1,1]となります。

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;注釈 2: XとYが独立ならρXY=0です。

<br>

**45. Parameter estimation**

&#10230;母数推定

<br>

**46. Definitions**

&#10230;定義

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;確率標本(Random sample) - 確率標本はXと独立で同一に分布するn個の確率変数X1,...,Xnの集まりです。

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;推定量(Estimator) - 推定量は統計モデルで未知のパラメータの値を推定するために使用されるデータの関数です。

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;偏り(Bias) - 推定量^θの偏りは^θの期待値と実際の値との差で定義されます。

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;注釈: 推定量はE[^θ]=θの場合、不偏といいます。

<br>

**51. Estimating the mean**

&#10230;平均の推定

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230;標本平均(Sample mean) - 確率標本の標本平均は実の平均μを推定するのに用いられ、主に¯¯¯¯¯Xと表記され次のように定義されます。

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230;注釈: 標本平均は不偏です。すなわちE[¯¯¯¯¯X]=μとなります。

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;中心極限定理 - 平均μと分散σ2を持つ分布を従う確率標本X1,...,Xnがある。その場合、次のようになります。

<br>

**55. Estimating the variance**

&#10230;分散推定

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;標本分散 - 確率標本の標本分散は実の分散σ2を推定するのに用いられ、主にs2または^σ2と表記し次のように定義されます。

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;注釈: 標本分散は不偏です。つまりE[s2]=σ2になります。

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;標本分散とカイ二乗の関係 - 確率標本の標本分散をs2としよう。次のようになります。

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230;[紹介、標本空間、事象、順列]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230;[条件部確率、ベイズの定理、独立]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230;[確率変数、定義、期待値、分散]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230;[確率分布、チェビシェフの不等式、主な分布]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230;[結合分布の確率変数、密度、共分散、相関関係]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230;[母数推定、平均、分散]
