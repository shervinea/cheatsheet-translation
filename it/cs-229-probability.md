**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**

&#10230; Ripasso di Probabilità e Statistica

<br>

**2. Introduction to Probability and Combinatorics**

&#10230; Introduzione di Probabilità e Calcolo Combinatorio

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230; Spazio campionario ― L'insieme di tutti i risultati possibili di un esperimento è noto come spazio campionario dell'esperimento ed è chiamato S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230; Evento ― Ogni sottinsieme E dello spazio campionario è chiamato evento. Un evento è quindi un insieme di possibili risultati dell'esperimento. Se il risultato dell'esperimento è contenuto in E, diciamo che E è accaduto.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230; Assiomi della probabilità Per ogni evento E, chiamiamo P(E) la probabilità che E accada.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230; Assioma 1 ― Ogni probabilità ha valore tra 0 e 1 inclusi, quindi:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230; Assioma 2 ― La probabilità che almeno uno degli eventi elementari dell'intero spazio campionario avvenga è 1, quindi:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230; Assioma 3 ― Per ogni sequenza di eventi mutualmente esclusivi E1, ..., En, abbiamo:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230; Permutazione ― Una permutazione è una raggruppamento di r oggetti fra n disponibili in un ordine dato. Il numero di tali raggruppamenti è dato da P(n,r) definito come:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230; Combinazione ― Una combinazione è un raggruppamento di r oggetti fra n disponibili dove l'ordine non importa. Il numero di tali raggruppamenti è dato da C(n,r) definito come:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230; Osservazione: notiamo che per 0⩽r⩽n abbiamo che P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230; Probabilità Condizionata

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230; Teorema di Bayes ― Dati due eventi A e B tali che P(B)>0, abbiamo che:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230; Osservazione: abbiamo che P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230; Partizione ― Sia {Ai,i∈[[1,n]]} tale che for ogni i, Ai≠∅. Diciamo che {Ai} è una partizione se abbiamo che:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230; Osservazione: per ogni evento B nello spazio campionario, abbiamo che P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230; Forma estesa del teorema di Bayes ― Sia {Ai,i∈[[1,n]]} una partizione dello spazio campionario. Abbiamo che:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230; Indipendenza ― Due eventi A e B sono indipendenti se e solo se abbiamo che:

<br>

**19. Random Variables**

&#10230; Variabili Aleatorie

<br>

**20. Definitions**

&#10230; Definizioni

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230; Variabile aleatoria ― Una variabile aleatoria, spesso chiamata X, è una funzione dagli elementi dello spazio campionario a un reale.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230; Funzione di ripartizione (cumulativa) ― La funzione di ripartizione F, che è monotona non-decrescente e tale che limx→−∞F(x)=0 e limx→+∞F(x)=1, è definita come:

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230; Osservazione: abbiamo che P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230; Funzione di densità ― La funzione di densità f è la probabilità che X assuma un valore tra due realizzazioni consecutive della variabile aleatoria.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230; Relazioni tra funzione di densità e di ripartizione ― Sono riportate le proprietà importanti da sapere nel caso discreto (D) e continuo (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230; [Caso, funzione di ripartizione F, funzione di densità f, Proprietà della funzione di densità]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230; Valore atteso e Momenti della Distribuzione ― Sono riportate le espressioni del valore atteso E[X], valore atteso generalizzato E[g(X)], momento k-esimo E[Xk] e funzione caratteristica ψ(ω) per il caso discreto e continuo:

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230; Varianza ― La varianza di una variable aleatoria, spesso denotata da Var(X) o σ2, è una misura della variabilità della funzione di distribuzione. È determinata nel modo seguente:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230; Deviazione standard ― La deviazione standard di una variabile aleatoria, spesso denotata da σ, è una misura della variabilità della funzione di distribuzione che è compatibile con l'unità di misura della variabile aleatoria. È determinata nel modo seguente:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230; Trasformazione di una variabile aleatoria ― Siano X e Y variabili collegate da qualche funzione. Siano fX e fY le funzioni di distribuzione di X e Y, rispettivamente. Abbiamo che:

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230; Regola di integrazione di Leibniz ― Sia g una funzione di x e potenzialmente c, e siano a e b contorni che possono dipendere da c. Abbiamo che:

<br>

**32. Probability Distributions**

&#10230; Distribuzioni di Probabilità

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230; Disuguaglianza di Chebyshev ― Sia X una variabile aleatoria con valore atteso μ. Per k,σ>0 abbiamo la seguente disuaglianza:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230; Distribuzioni principali ― Sono presentati le distribuzioni principali da tenere a mente:

<br>

**35. [Type, Distribution]**

&#10230; [Tipologia, Distrubuzione]

<br>

**36. Jointly Distributed Random Variables**

&#10230; Distribuzione congiunta di variabili aleatorie

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230; Densità marginale e distribuzione cumulativa ― Dalla funzione di densità congiunta fXY abbiamo che:

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230; [Caso, Densità marginale, Funzione cumulativa]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230; Densità condizionata ― La densità condizionata di X rispetto a Y, spesso denotata come fX|Y, è definita come:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230; Indipendenza ― Due variabili aleatorie X e Y si dicono indipendenti se:

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230; Covarianza ― Si definisce la covarianza di due variabili aleatorie X e Y, denotata da σ2XY o più comunemente Cov(X,Y), come segue:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230; Correlazione ― Date σX,σY le deviazioni standard di X e Y, definiamo la correlazione tra le variabili aleatorie X e Y, denotata da ρXY, come segue:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230; Osservazione 1: notiamo che per ogni variabile aleatoria X,Y, abbiamo che ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230; Osservazione 2: Se X e Y sono indipendenti, allora ρXY=0.

<br>

**45. Parameter estimation**

&#10230; Stima dei parametri

<br>

**46. Definitions**

&#10230; Definizioni

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230; Campione casuale ― Un campione casuale è un gruppo di n variabili aleatorie X1,...,Xn distribuite in modo indipendente e indentico con X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230; Stimatore ― Uno stimatore è una funzione dei dati usata per dedurre il valore di un parametro sconosciuto in un  modello statistico.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230; Distorsione ― La distorsione di uno stimatore ^θ è definita come la differenza tra il valore atteso della distribuzione di ^θ e il vero valore, quindi:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230; Osservazione: uno stimatore si dice non distorto quando abbiamo E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230; Stima della media

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230; Media campionaria ― La media campionaria di un campione casuale è usata per stimare la vera media μ di una distribuzione, è spesso denotata da ¯¯¯¯¯X ed è definita come segue:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230; Osservazione: la media campionaria non è distorta, quindi E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230; Teorema del Limite Centrale ― Sia X1,...,Xn un campione casuale che segue una data distribuzione di media μ e varianza σ2, di conseguenza abbiamo che:

<br>

**55. Estimating the variance**

&#10230; Stima della varianza

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230; Varianza campionaria ― La varianza campionaria di un campione casuale è usata per stimare il vero valore della varianza σ2 di una distribuzione, è spesso denotata da s2 o ^σ2 ed è definita come segue:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230; Osservazione: la varianza campionaria non è distorta, quindi E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230; Relazione tra Chi-Quadro e la varianza campionaria ― Sia s2 la varianza campionaria di un campione casuale. Abbiamo che:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230; [Introduzione, Spazio campionaria, Evento, Permutazione]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230; [Probabilità condizionata, Teorema di Bayes, Indipendenza]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230; [Variabile aleatoria, Definizioni, Valore atteso, Varianza]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230; [Distribuzioni, Disuguaglianza di Chebyshev, Distribuzioni principali]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230; [Distribuzione congiunta di variabili aleatorie, Densità, Covarianza, Correlazione]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230; [Stima del parametro, Media, Varianza]
