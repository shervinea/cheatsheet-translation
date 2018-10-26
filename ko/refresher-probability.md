
**1. Probabilities and Statistics refresher**

&#10230;확률과 통계

<br>

**2. Introduction to Probability and Combinatorics**

&#10230;확률과 조합론 소개

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230;표본 공간 ― 시행의 가능한 모든 결과 집합은 시행의 표본 공간으로 알려져 있으며 S로 표기합니다.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230;사건 ― 표본 공간의 모든 부분 집합 E를 사건이라고 합니다. 즉, 사건은 시행 가능한 결과로 구성된 집합입니다. 시행 결과가 E에 포함된다면, E가 발생했다고 이야기합니다.

<br>

**5. Axioms of probability ― For each event E, we denote P(E) as the probability of event E occuring.**

&#10230;확률의 공리 ― 각 사건 E에 대하여, 우리는 사건 E가 발생할 확률을 P(E)로 나타냅니다.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230;공리 1 ― 모든 확률은 0과 1사이에 포함됩니다, 즉:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230;공리 2 ― 전체 표본 공간에서 적어도 하나의 근원 사건이 발생할 확률은 1입니다. 즉:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230;공리 3 ― 서로 배반인 어떤 연속적인 사건 E1,...,En 에 대하여, 우리는 다음을 가집니다:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230;순열(Permutation) ― 순열은 n개의 객체들로부터 r개의 객체들의 순서를 고려한 배열입니다. 그러한 배열의 수는 P (n, r)에 의해 주어지며, 다음과 같이 정의됩니다:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230;조합(Combination) ― 조합은 n개의 객체들로부터 r개의 객체들의 순서를 고려하지 않은 배열입니다. 그러한 배열의 수는 다음과 같이 정의되는 C(n, r)에 의해 주어집니다:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230;비고 :우리는 for 0⩽r⩽n에 대해, P(n,r)⩾C(n,r)를 가집니다.

<br>

**12. Conditional Probability**

&#10230;조건부 확률

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230;베이즈 규칙 ― P(B)>0인 사건 A, B에 대해, 우리는 다음을 가집니다:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230;비고 :우리는 P(A∩B)=P(A)P(B|A)=P(A|B)P(B)를 가집니다.

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230;파티션(Partition)― {Ai, i∈ [[1, n]]}은 모든 i에 대해 Ai ≠ ∅이라고 해봅시다. 우리는 {Ai}가 다음과 같은 경우 파티션이라고 말합니다.

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230;비고 : 표본 공간에서 어떤 사건 B에 대해서 우리는 P(B) = nΣi = 1P (B | Ai) P (Ai)를 가집니다.

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230;베이즈 규칙의 확장된 형태 ― {Ai,i∈[[1,n]]}를 표본 공간의 파티션이라고 합시다. 우리는 다음을 가집니다.: 

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230;독립성 ― 다음의 경우에만 두 사건 A, B가 독립적입니다:

<br>

**19. Random Variables**

&#10230;확률 변수

<br>

**20. Definitions**

&#10230;정의

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230;확률 변수 ― 주로 X라고 표기된 확률 변수는 표본 공간의 모든 요소를 ​​실선에 대응시키는 함수입니다.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230;누적 분포 함수 (CDF) ― 단조 감소하지 않고 limx → -∞F (x) = 0 이고, limx → + ∞F (x) = 1 인 누적 분포 함수 F는 다음과 같이 정의됩니다:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230;비고 : 우리는 P(a<X⩽B)=F(b)−F(a)를 가집니다.

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230;확률 밀도 함수 (PDF) ― 확률 밀도 함수 f는 인접한 두 확률 변수의 사이에 X가 포함될 확률입니다.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230;PDF와 CDF의 관계 ― 이산 (D)과 연속 (C) 예시에서 알아야 할 중요한 특성이 있습니다.

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230;[예시, CDF F, PDF f, PDF의 특성]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230;분포의 기대값과 적률 ― 이산 혹은 연속일 때, 기대값 E[X], 일반화된 기대값 E[g(X)], k번째 적률 E[Xk] 및 특성 함수 ψ(ω) :

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230;분산 (Variance) ― 주로 Var(X) 또는 σ2이라고 표기된 확률 변수의 분산은 분포 함수의 산포(Spread)를 측정한 값입니다. 이는 다음과 같이 결정됩니다:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230;표준 편차(Standard Deviation) ― 표준 편차는 실제 확률 변수의 단위를 사용할 수 있는 분포 함수의 산포(Spread)를 측정하는 측도입니다. 이는 다음과 같이 결정됩니다:
<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230;확률 변수의 변환 ― 변수 X와 Y를 어떤 함수로 연결되도록 해봅시다. fX와 fY에 각각 X와 Y의 분포 함수를 표기하면 다음과 같습니다:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230;라이프니츠 적분 규칙 ― g를 x의 함수로, 잠재적으로 c라고 해봅시다. 그리고 c에 종속적인 경계 a, b에 대해 우리는 다음을 가집니다:

<br>

**32. Probability Distributions**

&#10230;확률 분포

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230;체비쇼프 부등식 ― X를 기대값 μ의 확률 변수라고 해봅시다. k에 대하여, σ>0이면 다음과 같은 부등식을 가집니다:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230;주요 분포들― 기억해야 할 주요 분포들이 여기 있습니다:

<br>

**35. [Type, Distribution]**

&#10230;[타입(Type), 분포]

<br>

**36. Jointly Distributed Random Variables**

&#10230;결합 분포 확률 변수

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230;주변 밀도와 누적 분포 ― 결합 밀도 확률 함수 fXY로부터 우리는 다음을 가집니다

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230;[예시, 주변 밀도, 누적 함수]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230;조건부 밀도 ― 주로 fX|Y로 표기되는 Y에 대한 X의 조건부 밀도는 다음과 같이 정의됩니다:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230;독립성 ― 두 확률 변수 X와 Y는 다음과 같은 경우에 독립적이라고 합니다:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230;공분산 ― 다음과 같이 두 확률 변수 X와 Y의 공분산을 σ2XY 혹은 더 일반적으로는 Cov(X,Y)로 정의합니다:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230;상관관계 ― σX, σY로 X와 Y의 표준 편차를 표기함으로써 ρXY로 표기된 임의의 변수 X와 Y 사이의 상관관계를 다음과 같이 정의합니다:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;비고 1 : 우리는 임의의 확률 변수 X, Y에 대해 ρXY∈ [-1,1]를 가진다고 말합니다. 

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;비고 2 : X와 Y가 독립이라면 ρXY=0입니다.

<br>

**45. Parameter estimation**

&#10230;모수 추정

<br>

**46. Definitions**

&#10230;정의

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;확률 표본 ― 확률 표본은 X와 독립적으로 동일하게 분포하는 n개의 확률 변수 X1, ..., Xn의 모음입니다.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;추정량 ―  추정량은 통계 모델에서 알 수 없는 모수의 값을 추론하는 데 사용되는 데이터의 함수입니다.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;편향 ― 추정량 ^θ의 편향은 ^θ 분포의 기대값과 실제값 사이의 차이로 정의됩니다. 즉,:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;비고 : 추정량은 E [^ θ]=θ 일 때, 비 편향적이라고 말합니다.

<br>

**51. Estimating the mean**

&#10230;평균 추정

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230;표본 평균 ― 랜덤 표본의 표본 평균은 분포의 실제 평균 μ를 추정하는 데 사용되며 종종 다음과 같이 정의됩니다:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230;비고 : 표본 평균은 비 편향적입니다, 즉i.e E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;중심 극한 정리 ― 평균 μ와 분산 σ2를 갖는 주어진 분포를 따르는 랜덤 표본 X1, ..., Xn을 가정해 봅시다 그러면 우리는 다음을 가집니다:

<br>

**55. Estimating the variance**

&#10230;분산 추정

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;표본 분산 ― 랜덤 표본의 표본 분산은 분포의 실제 분산 σ2를 추정하는 데 사용되며 종종 s2 또는 σ2로 표기되며 다음과 같이 정의됩니다:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;비고 : 표본 분산은 비 편향적입니다, 즉 E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;표본 분산과 카이 제곱의 관계 ― s2를 랜덤 표본의 표분 분산이라고 합시다. 우리는 다음을 가집니다:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230;[소개, 표본 공간, 사건, 순열]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230;[조건부 확률, 베이즈 규칙, 독립]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230;[확률 변수, 정의, 기대값, 분산]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230;[확률 분포, 체비쇼프 부등식, 주요 분포]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230;[결합 분포의 확률 변수, 밀도, 공분산, 상관관계]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230;[모수 추정, 평균, 분산]
