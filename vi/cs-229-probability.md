**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**

&#10230; Xác suất và Thống kê cơ bản

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Giới thiệu về Xác suất và Tổ hợp

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Không gian mẫu - Một tập hợp các kết cục có thể xảy ra của một phép thử được gọi là không gian mẫu của phép thử và được kí hiệu là S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Sự kiện (hay còn gọi là biến cố) - Bất kỳ một tập hợp con E nào của không gian mẫu đều được gọi là một sự kiện. Một sự kiện là một tập các kết cục có thể xảy ra của phép thử. Nếu kết quả của phép thử chứa trong E, chúng ta nói sự kiện E đã xảy ra.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Tiên đề của xác suất Với mỗi sự kiện E, chúng ta kí hiệu P(E) là xác suất sự kiện E xảy ra.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Tiên đề 1 - Mọi xác suất bất kì đều nằm trong khoảng 0 đến 1.

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Tiên đề 2 - Xác suất xảy ra của ít nhất một phần tử trong toàn bộ không gian mẫu là 1. 

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Tiên đề 3 - Với một chuỗi các biến cố xung khắc E1,...,En, ta có:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Hoán vị - Hoán vị là một cách sắp xếp r phần tử từ một nhóm n phần tử, theo một thứ tự nhất định. Số lượng cách sắp xếp như vậy là P(n,r), được định nghĩa như sau:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Tổ hợp - Một tổ hợp là một cách sắp xếp r phần tử từ n phần tử, không quan trọng thứ tự. Số lượng cách sắp xếp như vậy là C(n,r), được định nghĩa như sau:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Ghi chú: Chúng ta lưu ý rằng với 0⩽r⩽n, ta có P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Xác suất có điều kiện

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Định lí Bayes - Với các sự kiện A và B sao cho P(B)>0, ta có:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Ghi chú: ta có P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Phân vùng ― Cho {Ai,i∈[[1,n]]} sao cho với mỗi i, Ai≠∅. Chúng ta nói rằng {Ai} là một phân vùng nếu có:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Ghi chú: với bất cứ sự kiện B nào trong không gian mẫu, ta có P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Định lý Bayes mở rộng - Cho {Ai,i∈[[1,n]]} là một phân vùng của không gian mẫu. Ta có:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Sự kiện độc lập - Hai sự kiện A và B được coi là độc lập khi và chỉ khi ta có:

<br>

**19. Random Variables**

&#10230; Biến ngẫu nhiên

<br>

**20. Definitions**

&#10230; Định nghĩa

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Biến ngẫu nhiên - Một biến ngẫu nhiên, thường được kí hiệu là X, là một hàm nối mỗi phần tử trong một không gian mẫu thành một số thực

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Hàm phân phối tích lũy (CDF) ― Hàm phân phối tích lũy F, là một hàm đơn điệu không giảm, sao cho limx→−∞F(x)=0 và limx→+∞F(x)=1, được định nghĩa là:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Ghi chú: chúng ta có P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Hàm mật độ xác suất (PDF) - Hàm mật độ xác suất f là xác suất mà X nhận các giá trị giữa hai giá trị thực liền kề của biến ngẫu nhiên.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Mối quan hệ liên quan giữa PDF và CDF - Dưới đây là các thuộc tính quan trọng cần biết trong trường hợp rời rạc (D) và liên tục (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Trường hợp, CDF F, PDF f, Thuộc tính của PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Kỳ vọng và moment của phân phối - Dưới đây là các biểu thức của giá trị kì vọng E[X], giá trị kì vọng ​​tổng quát E[g(X)], moment bậc k E[Xk] và hàm đặc trưng ψ(ω) cho các trường hợp rời rạc và liên tục:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Phương sai - Phương sai của một biến ngẫu nhiên, thường được kí hiệu là Var (X) hoặc σ2, là một độ đo mức độ phân tán của hàm phân phối. Nó được xác định như sau:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Độ lệch chuẩn - Độ lệch chuẩn của một biến ngẫu nhiên, thường được kí hiệu σ, là thước đo mức độ phân tán của hàm phân phối của nó so với các đơn vị của biến ngẫu nhiên thực tế. Nó được xác định như sau:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Biến đổi các biến ngẫu nhiên - Đặt các biến X và Y được liên kết với nhau bởi một hàm. Kí hiệu fX và fY lần lượt là các phân phối của X và Y, ta có:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Quy tắc tích phân Leibniz - Gọi g là hàm của x và có khả năng c, và a, b là các ranh giới có thể phụ thuộc vào c. Chúng ta có:

<br>

**32. Probability Distributions**

&#10230; Phân bố xác suất

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Bất đẳng thức Chebyshev - Gọi X là biến ngẫu nhiên có giá trị kỳ vọng μ. Với k,σ>0, chúng ta có bất đẳng thức sau:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Các phân phối chính - Dưới là các phân phối chính cần ghi nhớ:

<br>

**35. [Type, Distribution]**

&#10230; [Loại, Phân phối]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Phân phối đồng thời biến ngẫu nhiên

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Mật độ biên và phân phối tích lũy - Từ hàm phân phối mật độ đồng thời fXY, ta có

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Trường hợp, Mật độ biên, Hàm tích lũy]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Mật độ có điều kiện - Mật độ có điều kiện của X với Y, thường được kí hiệu là fX|Y, được định nghĩa như sau:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Tính chất độc lập - Hai biến ngẫu nhiên X và Y độc lập nếu ta có:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Hiệp phương sai - Chúng ta xác định hiệp phương sai của hai biến ngẫu nhiên X và Y, thường được kí hiệu σ2XY hay Cov(X,Y), như sau:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Hệ số tương quan ― Kí hiệu σX,σY là độ lệch chuẩn của X và Y, chúng ta xác định hệ số tương quan giữa X và Y, kí hiệu ρXY, như sau:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; Ghi chú 1: chúng ta lưu ý rằng với bất cứ biến ngẫu nhiên X,Y nào, ta luôn có ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; Ghi chú 2: Nếu X và Y độc lập với nhau thì ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Ước lượng tham số

<br>

**46. Definitions**

&#10230; Định nghĩa

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Mẫu ngẫu nhiên - Mẫu ngẫu nhiên là tập hợp của n biến ngẫu nhiên X1,...,Xn độc lập và được phân phối giống hệt với X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Công cụ ước tính (Estimator) - Công cụ ước tính (Estimator) là một hàm của dữ liệu được sử dụng để suy ra giá trị của một tham số chưa biết trong mô hình thống kê.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Thiên vị (Bias) - Thiên vị (Bias) của Estimator ^θ được định nghĩa là chênh lệch giữa giá trị kì vọng ​​của phân phối ^θ và giá trị thực, tức là

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Ghi chú: một công cụ ước tính (estimator) được cho là không thiên vị (unbias) khi chúng ta có E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Ước lượng trung bình

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Giá trị trung bình mẫu - Giá trị trung bình mẫu của mẫu ngẫu nhiên được sử dụng để ước tính giá trị trung bình thực μ của phân phối, thường được kí hiệu ¯¯¯¯¯X và được định nghĩa như sau:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Ghi chú: Trung bình mẫu là không thiên vị (unbias), nghĩa là E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Định lý giới hạn trung tâm - Giả sử chúng ta có một mẫu ngẫu nhiên X1,...,Xn theo một phân phối nhất định với trung bình μ và phương sai σ2, sau đó chúng ta có:

<br>

**55. Estimating the variance**

&#10230; Ước lượng phương sai

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Phương sai mẫu - Phương sai mẫu của mẫu ngẫu nhiên được sử dụng để ước lượng phương sai thực sự σ2 của phân phối, thường được kí hiệu là s2 hoặc ^σ2 và được định nghĩa như sau:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Ghi chú: phương sai mẫu không thiên vị (unbias), nghĩa là E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Quan hệ Chi-Squared với phương sai mẫu - Với s2 là phương sai mẫu của một mẫu ngẫu nhiên, ta có:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; [Giới thiệu, Không gian mẫu, Sự kiện, Hoán vị]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; [Xác suất có điều kiện, Định lý Bayes, Sự độc lập]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Biến ngẫu nhiên, Định nghĩa, Kì vọng, Phương sai]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Phân bố xác suất, Bất đẳng thức Chebyshev, Xác suất chính]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Các biến ngẫu nhiên đồng thời, Mật độ, Hiệp phương sai, Hệ số tương quan]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Ước lượng tham số, Trung bình, Phương sai]
