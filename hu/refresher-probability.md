**1. Probabilities and Statistics refresher**

&#10230; Valószínűségszámítás és statisztika felfrissítés

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Bevezetés a valószínűségszámításba és kombinatorikába

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Eseménytér ― Egy kísérlet összes lehetséges kimenetelének halmazára azt mondjuk, hogy a kísérlet eseménytere és S-sel jelöljök.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Az eseménytér bármely E részhalmazáról azt mondjuk, hogy esemény. Azaz egy esemény olyan halmaz, mely a kísérlet lehetséges kimeneteleit tartalmazza. Ha kísérlet egy kimenetele E-nek eleme, akkor azt mondjuk, hogy E esemény bekövetkezett.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Valószínűségi axiómák ― Egy E esemény esetén jelölje P(E) az E esemény bekövetkezésének valószínűségét.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; 1. axióma ― a valószínűség 0 és 1 közötti valós szám (a határokat is beleértve), azaz:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; 2. axióma ― Biztos esemény valószínűsége 1, azaz:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Bármely egymást kizáró E1,...,En, eseményekre:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Ismétlés nélküli variáció ― n elem közül r-nek a lehetséges kiválasztása az r darab elem ismétlés nélküli variációjának hívjuk (jel.: P(n,r)) és így definiáljuk:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Ismétlés nélküli kombináció ― ha a kiválasztás sorrendje nem számít, akkor n elem közül r-nek a lehetétséges kiválasztását ismétlés nélküli kombinációnak hívjuk (jel.: C(n,r)), és így definiáljuk:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Megjegyzés: ha 0⩽r⩽n, akkor P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Feltételes valószínűség

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Bayes-tétel ― Legyenek A és B események és P(B)>0. Ekkor

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Megjegyzés: P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Teljes eseményrendszer ― Legyen {Ai,i∈[[1,n]]} olyan, hogy minden i-re Ai≠∅. Ekkor azt mondjuk, hogy {Ai} teljes eseményrendszer, ha

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Megjegyzés: bármely B eseményre fennáll, hogy P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Általánosított Bayes-tétel ― Legyen {Ai,i∈[[1,n]]} teljes eseményrendszer. Ekkor

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Függetlenség ― A és B események függetlenek pontosan akkor, ha

<br>

**19. Random Variables**

&#10230; Valószínűségi változók

<br>

**20. Definitions**

&#10230; Definíciók

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Valószínűségi változó ― Az X valószínűségi változó olyan függvény, mely az eseménytér minden elemét a valós számegyenesre képezi.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Eloszlásfüggvény ― Az X val. változó F eloszlásfüggvényét, mely (a) monoton növő, (b) balról folytonos és (c) igaz rá, hogy limx→−∞F(x)=0 és limx→+∞F(x)=1, a következőképpen definiáljuk:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Megjegyzés: tetszőleges X val. változó esetén fennáll, hogy P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Sűrűségfüggvény ― Az X val. változó abszolút folytonos, ha létezik olyan f nemnegatív függvény, melyre F'(x) = f(x). Ekkor f-et az X sűrűségfüggvényének mondjuk.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Összefüggések az eloszlásfüggvény és sűrűségfüggvény között ― Alább található néhány fontos tulajdonság a diszkrét (D) és folytonos (C) esetre vonatkozóan.

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Eset, Eloszlásfüggvény F, Sűrűségfüggvény f, Sűrűrségfüggvény tulajdonságai]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Várható érték és momentum ― Alább találhatók a várható érték (E[X]), általánosított várható érték (E[g(X)]), k-adik momentum (E[Xk]) és karakterisztikus függvény (ψ(ω)) formulái a diszkrét és folytonos esetben:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Variancia (szórásnégyzet) ― Az X val. változó szórásnégyzete (jel.: Var(X) vagy σ2) a várható értéktől vett átlagos négyzetes eltérés. A következőképpen határozható meg:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Szórás ― Az X val. változó szórása (jel.: σ) a szórásnégyzet gyöke. A következőképpen határozható meg:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Val. változók transzformációja ― Legyenek X és Y val. változók úgy, hogy az egyikből a másikat valamilyen szigorúan monoton növő, folytonosan differenciálható függvénnyel kapjuk. Jelöljük X, ill. Y sűrűségfüggvényét fX-szel, ill. fY-nal, ekkor:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Leibniz-féle integrálszabály ― Legyen g az x és esetleg c függvénye, és a,b intervallumhatárok, melyek függhetnek c-től. Ekkor:

<br>

**32. Probability Distributions**

&#10230; Eloszlások

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Csebisev-egyenlőtlenség ― Legyen X val. változó μ várható értékkel. Ha k,σ>0, akkor igaz az alábbi egyenlőtlenség:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Nevezetes eloszlások

<br>

**35. [Type, Distribution]**

&#10230; [Típus, Eloszlás]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Val. változók együttes eloszlása

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Együttes és peremeloszlás és -sűrűségfüggvények ― Ha fXY az X és Y val. változók együttes sűrűségfüggvénye, akkor:

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Eset, Perem-sűrűségfüggvény, Együttes eloszlásfüggvény]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Feltételes sűrűségfüggvény ― Az X val. változó feltételes sűrűségfüggvényét Y-ra nézve (jel.: fX|Y) így definiáljuk:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Függetlenség ― X és Y val. változókat függetlennek hívjuk, ha sűrűségfüggvényeikre teljesül:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Kovariancia ― X és Y val. változók kovarianciáját (jel.: σ2XY vagy Cov(X,Y)) így definiáljuk:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Korreláció ― Az X és Y val. változók korrelációját így definiáljuk (ahol X és Y szórását rendre σX,σY-nal jelöljük):

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; 1. megj.: bármely véges szórású X,Y val. változókra igaz, hogy ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; 2. megj.: Ha X és Y függetlenek, akkor korrelálatlanok, azaz ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Paraméterbecslés

<br>

**46. Definitions**

&#10230; Definíciók

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Minta ― A minta n darab független, azonos eloszlású (i.i.d.) valószínűségi változóból álló sorozat. 

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Becslés ― A becslés a minta olyan függvénye, mely a minta eloszlásának ismeretlen paraméterét közelíti a statisztikai modellben.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Torzítás (bias) ― A ^θ becslés torzítását a ^θ eloszlásának várható értéke és a valódi érték különbségeként definiáljuk, azaz: 

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Megjegyzés: a becslést torzítatlannak mondjuk, ha E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Középértékbecslés

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Mintaátlag ― Valamely minta mintaátlagát (jel.: ¯¯¯¯¯X) az eloszlás valódi átlagának becslésére használjuk, és így definiáljuk:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Megjegyzés: a mintaátlag torzítatlan, azaz E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Centrális határeloszléstétel ― Legyen X1,...,Xn minta μ várható értékkel és σ2 szórásnégyzettel. Ekkor

<br>

**55. Estimating the variance**

&#10230; Szórásnégyzetbecslés

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Korrigált tapasztalati szórásnégyzet ― A minta korrigált tapasztalati szórásnégyzetét (jel.: s2 vagy ^σ2) az eloszlás valódi szórásnégyzetének (σ2-nek) becslésére használjuk, és így jelöljük:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Megjegyzés: a korrigált tapasztalati szórásnégyzet torzítatlan, azaz E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Khí-négyzet eloszlás és korrigált tapasztalati szórásnégyzet közti kapcsolat ― Legyen s2 a minta korrigált tapasztalati szórásnégyzete. Ekkor:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; [Bevezetés, Eseménytér, Esemény, Variáció]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; [Feltételes valószínűség, Bayes-tétel, Függetlenség]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Valószínűségi változók, Definíciók, Várható érték, Szórásnégyzet]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Eloszlások, Csebisev-egyenlőtlenség, Nevezetes eloszlások]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Együttes eloszlás, Sűrűségfüggvény, Kovariancia, Korreláció]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Paraméterbecslés, Átlag, Szórásnégyzet]
