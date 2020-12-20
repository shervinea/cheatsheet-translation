**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**

&#10230; Probabilities and Statistics refresher

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Introduction to Probability and Combinatorics

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Axiom 1 ― Every probability is between 0 and 1 included, i.e:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, у нас есть:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Примечание: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Conditional Probability

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Bayes' rule ― For events A and B such that P(B)>0, у нас есть:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Примечание: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if у нас есть:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Примечание: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. У нас есть:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Independence ― Two events A and B are independent if and only if у нас есть:

<br>

**19. Random Variables**

&#10230; Random Variables

<br>

**20. Definitions**

&#10230; Definitions

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Примечание: we have P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Case, CDF F, PDF f, Properties of PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Transformation of random variables ― Let the variables X and Y be linked by some function. Обозначим fX and fY the distribution function of X and Y respectively, у нас есть:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. У нас есть:

<br>

**32. Probability Distributions**

&#10230; Probability Distributions

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Main distributions ― Here are the main distributions to have in mind:

<br>

**35. [Type, Distribution]**

&#10230; [Type, Distribution]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Jointly Distributed Random Variables

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Marginal density and cumulative distribution ― From the joint density probability function fXY , we have

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Case, Marginal density, Cumulative function]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Independence ― Two random variables X and Y are said to be independent if у нас есть:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Correlation ― Обозначим σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; Remark 2: If X and Y are independent, then ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Parameter estimation

<br>

**46. Definitions**

&#10230; Definitions

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Примечание: an estimator is said to be unbiased when we have E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Estimating the mean

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Примечание: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then у нас есть:

<br>

**55. Estimating the variance**

&#10230; Estimating the variance

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Примечание: the sample variance is unbiased, i.e E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. У нас есть:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; [Introduction, Sample space, Event, Permutation]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; [Conditional probability, Bayes' rule, Independence]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Random variables, Definitions, Expectation, Variance]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Probability distributions, Chebyshev's inequality, Main distributions]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Jointly distributed random variables, Density, Covariance, Correlation]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Parameter estimation, Mean, Variance]
