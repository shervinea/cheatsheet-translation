**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**

&#10230; 1. Odświeżenie prawdopodobieństwa i statystyki

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; 2. Wprowadzenie do prawdopodobieństwa i kombinatoryki

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; 3. Przestrzeń próbki - zbiór wszystkich możliwych wyników eksperymentu jest znany jako przestrzeń próby i jest oznaczony przez S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; 4. Zdarzenie - dowolny podzbiór E przestrzeni próbki jest znany jako zdarzenie. To znaczy, zdarzenie jest zbiorem składającym się z możliwych wyników eksperymentu. Jeśli wynik eksperymentu jest zawarty w E, to mówimy, że E wystąpiło.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; 5. Aksjomaty prawdopodobieństwa dla każdego zdarzenia E oznaczamy P(E), jako prawdopodobieństwo wystąpienia zdarzenia E.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; 6. Aksjomat 1 - Każde prawdopodobieństwo zawiera się między 0 a 1, tj.:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; 7. Aksjomat 2 - Prawdopodobieństwo wystąpienia co najmniej jednego zdarzenia elementarnego w całej przestrzeni próbki wynosi 1, tj.:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; 8. Aksjomat 3 - Dla dowolnej sekwencji wzajemnie wykluczających się zdarzeń E1,...,En, mamy:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; 9. Permutacja - Permutacja to układ r obiektów z puli n obiektów w podanej kolejności. Liczba takich układów jest określona przez P(n,r), zdefiniowane jako:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; 10. Kombinacja - Kombinacja to układ r obiektów z puli n obiektów, w którym kolejność nie ma znaczenia. Liczba takich układów jest określona przez C(n,r), zdefiniowane jako:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; 11. Uwaga: zauważmy, że dla 0⩽r⩽n, mamy P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; 12. Prawdopodobieństwo warunkowe 

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; 13. Twierdzenie Bayesa - w przypadku zdarzeń A i B takich, że P(B)>0, mamy:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; 14. Uwaga: mamy P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; 15. Partycja - niech {Ai,i∈[[1,n]]} będzie takie, że dla wszystkich i, Ai≠∅. Mówimy, że {Ai} to partycja, jeśli mamy:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; 16. Uwaga: dla każdego zdarzenia B w przestrzeni próbki, mamy P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; 17. Rozszerzona postać twierdzenia Bayesa - niech {Ai,i∈[[1,n]]} będzie partycją przestrzeni próbki. Mamy:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; 18. Niezależność - dwa zdarzenia A i B są niezależne tylko wtedy, gdy mamy:

<br>

**19. Random Variables**

&#10230; 19. Zmienne losowe

<br>

**20. Definitions**

&#10230; 20. Definicje

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; 21. Zmienna losowa - zmienna losowa, często oznaczana jako X, jest funkcją, która odwzorowuje każdy element w przestrzeni próbki na rzeczywistą linię.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; 22. Dystrybuanta (CDF) - dystrybuanta F, jest funkcją która nie zmniejsza się monotonicznie i jest taka, że limx→−∞F(x)=0 oraz limx→+∞F(x)=1, jest zdefiniowana jako:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; 23. Uwaga: mamy P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; 24. Funkcja gęstości prawdopodobieństwa (PDF) - Funkcja gęstości prawdopodobieństwa f to prawdopodobieństwo, że X przyjmuje wartości między dwiema sąsiednimi realizacjami zmiennej losowej.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; 25. Relacje dotyczące funkcji gęstości prawdopodobieństwa i dystrybuanty - Oto ważne właściwości, które należy znać w przypadkach dyskretnych (D) i ciągłych (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; 26. [Przypadek, Dystrybuanta F, Funkcja gęstości prawdopodobieństwa f, Właściwości funkcji gęstości prawdopodobieństwa]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; 27. Oczekiwanie i momenty rozkładu - oto wyrażenia wartości oczekiwanej E[X], uogólnionej wartości oczekiwanej E[g(X)], k-tego momentu E[Xk] i funkcji charakterystycznej ψ(ω) dla przypadków dyskretnych i ciągłych:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; 28. Wariancja - wariancja zmiennej losowej, często oznaczana Var(X) lub σ2, jest miarą rozproszenia jej funkcji rozkładu. Jest określana w następujący sposób:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; 29. Odchylenie standardowe - odchylenie standardowe zmiennej losowej, często oznaczane σ, jest miarą rozproszenia funkcji rozkładu, która jest zgodna z jednostkami rzeczywistej zmiennej losowej. Jest określane w następujący sposób:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; 30. Transformacja zmiennych losowych - niech zmienne X i Y zostaną połączone jakąś funkcją. Biorąc pod uwagę fX i fY, i odpowiednio funkcję rozkładu X i Y, mamy:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; 31. Twierdzenie Leibniza (o różniczkowaniu pod znakiem całki) - niech g będzie funkcją x i potencjalnie granic c oraz a, b, które mogą zależeć od c. Mamy:

<br>

**32. Probability Distributions**

&#10230; 32. Rozkłady prawdopodobieństwa

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; 33. Nierówność Czebyszewa - Niech X będzie zmienną losową o oczekiwanej wartości μ. Dla k,σ>0 mamy następującą nierówność:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; 34. Główne rozkłady - oto główne rozkłady, o których należy pamiętać:

<br>

**35. [Type, Distribution]**

&#10230; 35. [Rodzaj, rozkład]

<br>

**36. Jointly Distributed Random Variables**

&#10230; 36. Wspólny rozkład zmiennych losowych

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; 37. Gęstość brzegowa i rozkład skumulowany - Ze złączenia funkcji gęstości prawdopodobieństwa fXY, mamy

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; 38. [Przypadek, gęstość krańcowa, funkcja skumulowana]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; 39. Gęstość warunkowa - Gęstość warunkowa X w odniesieniu do Y, często oznaczana fX|Y, jest zdefiniowana w następujący sposób:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; 40. Niezależność - mówi się, że dwie zmienne losowe X i Y są niezależne, jeśli mamy:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; 41. Kowariancja - definiujemy kowariancję dwóch zmiennych losowych X i Y, które odnotowujemy σ2XY lub częściej Cov(X,Y), w następujący sposób:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; 42. Korelacja - biorąc σX,σY odchylenia standardowe X i Y, definiujemy korelację między zmiennymi losowymi X i Y, oznaczonymi ρXY, w następujący sposób:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; 43. Uwaga 1: zauważmy, że dla dowolnych zmiennych losowych X,Y, mamy ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; 44. Uwaga 2: Jeśli X i Y są niezależne, to ρXY=0.

<br>

**45. Parameter estimation**

&#10230; 45. Oszacowanie parametrów

<br>

**46. Definitions**

&#10230; 46. Definicje

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; 47. Losowa próbka - losowa próbka to zbiór n losowych zmiennych X1,...,Xn, które są niezależne i identycznie rozmieszczone z X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; 48. Estymator - estymator to funkcja danych służąca do wnioskowania o wartości nieznanego parametru w modelu statystycznym.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; 49. Odchylenie - odchylenie estymatora ^θ jest zdefiniowane, jako różnica między wartością oczekiwaną rozkładu ^θ, a wartością rzeczywistą, tj .:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; 50. Uwaga: mówi się, że estymator jest bezstronny, kiedy mamy E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; 51. Oszacowanie średniej

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; 52. Próbka średnia ― próbka średnia z próby losowej służy do oszacowania rzeczywistej średniej μ rozkładu, często jest oznaczana ¯¯¯¯¯X i jest zdefiniowana następująco:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; 53. Uwaga: próbka średnia jest bezstronna, tj. E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; 54. Centralne twierdzenie graniczne - uzyskajmy losową próbkę X1,...,Xn po danym rozkładzie ze średnią μ i wariancją σ2, a następnie mamy:

<br>

**55. Estimating the variance**

&#10230; 55. Oszacowanie wariancji

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; 56. Wariancja próbki - wariancja próbki losowej jest używana do oszacowania prawdziwej wariancji σ2 rozkładu, jest często oznaczana jako s2 lub ^σ2 i jest zdefiniowana następująco:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; 57. Uwaga: wariancja próbki jest bezstronna, tj. E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; 58. Relacja chi-kwadrat z wariancją próbki - niech s2 będzie wariancją próbki losowej. Mamy:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; 59. [Wprowadzenie, przestrzeń próbki, zdarzenie, permutacja]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; 60. [Prawdopodobieństwo warunkowe, twierdzenie Bayesa, niezależność]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; 61. [Zmienne losowe, Definicje, Oczekiwanie, Wariancja]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; 62. [Rozkłady prawdopodobieństwa, nierówność Czebyszewa, rozkłady główne]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; 63. [Wspólnie rozłożone zmienne losowe, Gęstość, Kowariancja, Korelacja]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; 64. [Szacowanie parametrów, Średnia, Wariancja]
