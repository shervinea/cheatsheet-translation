**1. Probabilities and Statistics refresher**

&#10230; Швидке повторення з теорії ймовірностей та комбінаторики.

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Вступ до теорії ймовірностей та комбінаторики.

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Простір елементарних подій ― Множина всіх можливих результатiв експерименту називається простором елементарних подій і позначається літерою S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Випадкова подія - будь-яка підмножина E, що належить до певного простору елементарних подій, називається подією. Таким чином, подія це множина, що містить можливі результати експерименту. Якщо результати експерименту містяться в Е, тоді ми говоримо що Е відбулася.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Аксіоми теорії ймовірностей. Для кожної події Е, P(E) є ймовірністю події Е.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Аксіома 1 - Всі ймовірності існують між 0 та 1 включно.

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Аксіома 2 - Ймовірність що як мінімум одна подія з простору елементарних подій відбудеться дорівнює 1.

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Аксіома 3 - Для будь-якої послідовності взаємновиключних подій E1,...,En, ми маємо:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Підстановка - підстановка це спосіб вибору r об'єктів з набору n об'єктів в певному порядку. Кількість таких способів вибору задається через P(n,r):

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Комбiнацiя - комбiнацiя це спосіб вибору r об'єктів з набору n об'єктів, де порядок не має значення. Кількість таких способів вибору задається через C(n,r):

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Примітка: ми зауважуємо що для 0⩽r⩽n, ми маємо P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Умовна ймовірність

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Теорема Баєса - Для подій А і В таких що P(B)>0, маємо:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Примітка: P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Поділ множини - Нехай {Ai,i∈[[1,n]]} буде таким для всіх i, Ai≠∅. Ми називаємо {Ai} поділом множини якщо:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Примітка: для будь-якої події В в просторі елементарних подій, маємо P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Розгорнута форма теореми Баєса - Нехай {Ai,i∈[[1,n]]} буде поділом множини простору елементарних подій. Маємо:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Незалежність - Дві події А і В є незалежними якщо і тільки якщо ми маємо:

<br>

**19. Random Variables**

&#10230; Випадкові змінні

<br>

**20. Definitions**

&#10230; Означення

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Випадкова змінна - Випадкова змінна, часто означена X, є функцією що проектує кожну подію в просторі елементарних подій на реальну лінію.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Функція розподілу ймовірностей (CDF) - Функція розподілу ймовірностей F, що є монотонно зростаючою і є такою, що limx→−∞F(x)=0 та limx→+∞F(x)=1 і задається як:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Примітка: маємо P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Функція густини імовірності (PDF) - Функція густини імовірності F є імовірністю що X набирає значень між двома сусідніми випадковими величинами.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Залежність між PDF та CDF - Ось деякі важливі характеристики в одиночних i тривалих випадках:

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Випадок, CDF F, PDF f, характеристики PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Математичне сподівання і моменти випадкового значення - Ось вирази очікуваного значення E[X], узагальненого очікуваного значення E[g(X)], k-го моменту E[Xk] та характеристичною функцією ψ(ω) дискретного або неперервного значення величини:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Дисперсія випадкової змiнної - Дисперсія випадкової змiнної, що позначається Var(X) або σ2 є мірою величини розподілення значень Функції. Вона визначаєтья:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Стандартне відхилення - Стандартне відхилення випадкової величини, що позначається σ, є мірою величини розподілення значень функції, сумісною з одиницями випадкової величини. Вона визначаєтья:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Перетворення випадкових величин - Нехай змінні X та Y будуть поєднані певною функцією. Називаючи fX та fY розподілом відповідно функцій X та Y, маємо:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Інтегральне правило Лейбніца - Нехай g буде функцією x і потенційно c, і a,b будуть кордонами що можуть залежати від с. Маємо :

<br>

**32. Probability Distributions**

&#10230; Розподіл ймовірностей

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Нерівність Чебишова ― Нехай X буде випадковою змінною з очікуваною велечиною μ. Для k,σ>0, маємо наступну нерівність :

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Головні розподіли - Ось кілька найважливіших розподілів які варто знати:

<br>

**35. [Type, Distribution]**

&#10230; [Тип, Розподіл]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Спільно розподілені випадкові величини

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Відособлена густина та розподіл ймовірностей - Виходячи з формули спільної густини ймовірностей fXY, маємо :

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Випадок, Відособлена густина, Розподіл ймовірностей]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Умовна густина  ― Умовна густина X відносно Y, означена fX|Y, визначаєтья:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Незалежність - Дві події А і В є незалежними якщо і тільки якщо ми маємо:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Коваріація ― Коваріація двох випадкових змінних X та Y, що означена як σ2XY або частіше як Cov(X,Y), визначаєтья :

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Кореляція ― Означивши σX,σY станартним відхиленням X та Y, ми визначаємо кореляцію X та Y, означену ρXY, в наступний спосіб :

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; Примітка 1: ми зазначаємо що для будь-яких випадкових змінних X, Y, маємо ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; Примітка 2 : Якщо X та Y є незалежними, тоді ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Оцінювання параметрів

<br>

**46. Definitions**

&#10230; Визначення

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Випадкова вибірка ― Випадкова вибірка це набір випадкових змінних X1,...,Xn які є незалежними і ідентично розподіленими в X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Статистична оцінка - Статистична оцінка це функція даних що використовується щоб визначити невідомий параметр статистичної моделі.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Систематична похибка ― Систематична похибка статистичної оцінки ^θ визначаєтья як різниця очікуваної величини розподілу ^θ і фактичної величини, тобіж:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Примітка: оцінка немає похибки якщо E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Оцінка середнього значення

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Середнє значення вибірки ― Середнє значення вибірки ¯¯¯¯¯X вказує середнє μ розподілу і визначаєтья:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Примітка : середнє значення не має похибки, тобто E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Центральна гранична теорема ― Маючи випадкову вибірку X1,...,Xn слідуючи даному розподілу з середнім значенням σ2, маємо :

<br>

**55. Estimating the variance**

&#10230; Розрахунок дисперсії

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Дисперсія вибірки ― Дисперсія випадкової вибірки - s2 або ^σ2, використовується щоб визначити справжню дисперсію σ2 вибірки, і визначаєтья:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Примітка: дисперсія вибірки не має похибки, тобто E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Розподіл хі-квадрат та дисперсія вибірки ― Нехай s2 буде дисперсією випадкової вибірка. Маємо:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230 [Вступ, Простір елементарних подій, Подія, Підстановка];

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230 [Умовна ймовірність, Теорема Баєса, Незалежність];

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Випадкові змінні, Означення, Очікування, Дисперсія]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Розподіли ймовірності, Нерівність Чебишова, Головні розподіли]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Спільно розподілені випадкові величини, Щільність, Коваріація, Кореляція]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Оцінювання параметрів, Середнє значення, Дисперсія]
