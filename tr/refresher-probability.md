**1. Probabilities and Statistics refresher**

&#10230; 1. Olasılık ve İstatistik hatırlatma

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; 2. Olasılık ve Kombinasyonlara Giriş

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; 3. Örnek alanı - Bir deneyin olası tüm sonuçlarının kümesidir, deneyin örnek alanı olarak bilinir ve S ile gösterilir.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; 4. Olay - Örnek alanın herhangi bir E alt kümesi, olay olarak bilinir. Yani bir olay, deneyin olası sonuçlarından oluşan bir kümedir. Deneyin sonucu E'de varsa, E'nin gerçekleştiğini söyleriz.

<br>

**5. Axioms of probability: For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; 5. Olasılık aksiyomları: Her olay E için, E olayının meydana gelme olasılığı olarak P (E) anlamına gelir.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; 6. Aksiyom 1 - Her olasılık dahil 0 ile 1 arasındadır, yani:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; 7. Aksiyom 2 - Tüm örnek uzayındaki temel olaylardan en az birinin ortaya çıkma olasılığı 1'dir, yani:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; 8.  Aksiyom 3 - Karşılıklı özel olayların herhangi bir dizisi için, E1, ..., En,

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; 9. Permütasyon - Permütasyon, n nesneler havuzundan r nesnelerinin belirli bir sıra ile düzenlenmesidir. Bu tür düzenlemelerin sayısı P (n, r) tarafından aşağıdaki gibi tanımlanır:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; 10. Kombinasyon - Bir kombinasyon, sıranın önemli olmadığı n nesneler havuzundan r nesnelerinin bir düzenlemesidir. Bu tür düzenlemelerin sayısı C (n, r) tarafından aşağıdaki gibi tanımlanır:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; 11. Not: 0⩽r⩽n için P (n, r) ⩾C (n, r) değerine sahibiz.

<br>

**12. Conditional Probability**

&#10230; 12. Koşullu Olasılık

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230;13. Bayes kuralı - A ve B olayları için P (B)> 0 olacak şekilde:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; 14. Not: P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; 15. Parça - {Ai,i∈[[1,n]]} olsun; {Ai}'nın bir parçası olduğunu söyleriz:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; 16. Not: Örnek uzaydaki herhangi bir B olayı için P(B)=n∑i=1P(B|Ai)P(Ai)'ye sahibiz.

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; 17. Genişletilmiş Bayes kuralı formu - {Ai,i∈[[1,n]]} örneklemenin bir bölümü olsun. Elde edilen:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; 18. Bağımsızlık - İki olay A ve B birbirinden bağımısz ise, elde edilen: 

<br>

**19. Random Variables**

&#10230; 19. Rastgele Değişkenler

<br>

**20. Definitions**

&#10230; 20. Tanımlamalar

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; 21. 21. Rastgele değişken - Genellikle X işaretli rastgele bir değişken, bir örnek uzayındaki her öğeyi gerçek bir çizgiye eşleyen bir fonksiyondur (işlevdir).

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; 22. Kümülatif dağılım fonksiyonu (KDF/CDF) - Monotonik olarak azalmayan ve limx→−∞F(x)=0 ve limx→+∞F(x)=1 olacak şekilde kümülatif dağılım fonksiyonu F:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; 23. Not: P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; 24. Olasılık yoğunluğu fonksiyonu (OYF/PDF) - Olasılık yoğunluğu fonksiyonu f, X'in rastgele değişkenin iki bitişik gerçekleşmesi arasındaki değerleri alması ihtimalidir.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; 25. PDF ve CDF'yi içeren ilişkiler - Ayrık (D) ve sürekli (C) olaylarında bilmeniz gereken önemli özellikler.

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; 26. [Olay, CDF F, PDF f, PDF Özellikleri]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; 27. Beklenti ve Dağılım Momentleri - Burada, ayrık ve sürekli durumlar için beklenen değer E[X], genelleştirilmiş beklenen değer E[g(X)], k. Moment E[Xk] ve karakteristik fonksiyon ψ(ω) ifadeleri verilmiştir :

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; 28. Varyans - Genellikle Var(X) veya σ2 olarak not edilen rastgele değişkenin varyansı, dağılım fonksiyonunun yayılmasının bir ölçüsüdür. Aşağıdaki şekilde belirlenir:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; 29. Standart sapma - Genellikle σ olarak not edilen rastgele bir değişkenin standart sapması, gerçek rastgele değişkenin birimleriyle uyumlu olan dağılım fonksiyonunun yayılmasının bir ölçüsüdür. Aşağıdaki şekilde belirlenir:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230;30. Rastgele değişkenlerin dönüşümü - X ve Y değişkenlerinin bazı fonksiyonlarla bağlanır. FX ve fY'ye sırasıyla X ve Y'nin dağılım fonksiyonu şöyledir:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; 31. Leibniz integral kuralı - g, x'e ve potansiyel olarak c'nin, c'ye bağlı olabilecek potansiyel c ve a, b sınırlarının bir fonksiyonu olsun. Elde edilen:

<br>

**32. Probability Distributions**

&#10230; 32. Olasılık Dağılımları

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; 33. Chebyshev'in eşitsizliği - X'in beklenen değeri value olan rastgele bir değişken olmasına izin verin. K, σ>0 için aşağıdaki eşitsizliği elde edilir:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; 34. Ana dağıtımlar - İşte akılda tutulması gereken ana dağıtımlar:

<br>

**35. [Type, Distribution]**

&#10230;

<br>

**36. Jointly Distributed Random Variables**

&#10230;

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230;

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230;

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230;

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230;

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230;

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230;

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;

<br>

**45. Parameter estimation**

&#10230;

<br>

**46. Definitions**

&#10230;

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;

<br>

**51. Estimating the mean**

&#10230;

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230;

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230;

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;

<br>

**55. Estimating the variance**

&#10230;

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230;

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230;

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230;

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230;

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230;

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230;
