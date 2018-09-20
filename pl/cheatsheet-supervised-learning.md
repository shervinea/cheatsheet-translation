**1. Supervised Learning cheatsheet**

&#10230; Uczenie nadzorowane - ściąga

<br>

**2. Introduction to Supervised Learning**

&#10230; Wprowadzenie do Uczenia nadzorowanego

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Mając zbiór danych {x(1),...,x(m)} i powiązany z nimi zbiór wyników {y(1),...,y(m)}, chcemy zbudować klasyfikator, który nauczy się predykcji y na podstawie x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Rodzaje predykcji ― Różne rodzaje predykcji opisane są w tabelce poniżej:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regresja, Klasyfikacja, Wynik, Przykład]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Ciągłość, Klasa, Regresja liniowa, Regresja logistyczna, SVM, Naive Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Rodzaj modelu ― Różne rodzaje modeli opisane są w tabelce poniżej:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Model dyskryminacyjny, Model generatywny, Cel, Co jest uczone?, Obrazek, Przykład]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Bezpośrednia estymata P(y|x), Estymata P(x|y) aby wydedukować P(y|x), Rozgraniczenie decyzyjne, Rozkład prawdopodobieństwa danych, Regresja, SVM, GDA, Naive Bayes]

<br>

**10. Notations and general concepts**

&#10230; Zapis i stwierdzenia ogólne

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hipoteza ― Hipoteze zapisujemy jako h0 i jest wybranym przez nas modelem. Dla danych danech wejściowych x(i) model tworzy predykcje wyniku h0(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Funkcja straty - Funkcja straty jest funkcją L:(z,y)∈R×Y⟼L(z,y)∈R która bierze za wejście predykowany wynik modelu oraz odpowiadający mu wynik rzeczywisty y i wyraża jak różne są od siebie. Częśto stosowane funkcje straty przedstawione są w tabelce poniżej:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Błąd najmniejszych kwadratów, Strata logistyczny, Strata Hinge-a, Strata logarytmiczny (Cross-entropy)]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Regresja liniowa, Regresja logistyczna, SVM, Sieć neuronowa]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Funkcja kosztu - Funkcja kosztu J jest często używana w celu określenia efektywności modelu, definiuje sie ją za pomocą funkcji straty L w następujący sposób:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Schodzenie gradientu (Gradient descent) ― Przyjmując, że współczynnik uczenia to α∈R, zasadę aktualizacji przy schodzeniu gradientu można wyrazić za pomocą współczynnika uczenia i funkcji kosztu J w następujący sposób:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Przypomnienie: Stochastyczne schodzenie gradientu (Stochastic Gradient Descent, SGD) aktualizuje współczynniki funkcji (wagi) w oparciu o każdy przykład z danych treningowych z osobna, a pakietowe schodzenie gradientu (batch gradient descent) aktualizuje je na podstawie całego pakietu (podzbioru) przykładów z danych treningowych.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Prawdopodobieństwo ― Prawdopodobieństwo modelu L(θ) przy parametrze θ jest wykorzystywane do znalezienia optymalnego parametru θ poprzez maksymalizacje prawdopodobieństwa. W praktyce, używamy prawdopodobieństwa logarytmicznego ℓ(θ)=log(L(θ)) które łatwiej zoptymalizować (logspace). Mamy więc:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Algorytm Newtona ― Algorytm Newtona to numeryczna metoda znalezienia takiego parametru θ, dla którego ℓ′(θ)=0. Zasada jego aktualizacji:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Przypomnienie: wielowymiarowa generalizacja, znana także jako metoda Newtona-Raphsona, ma następującą zasadę aktualizacji:

<br>

**21. Linear models**

&#10230; Modele liniowy

<br>

**22. Linear regression**

&#10230; Regresja liniowa

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Zakładając że y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Równania normalnej - Przyjmując za X macierz, wartość θ minimalizująca funkcje kosztu ma zamknięte rozwiązanie:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Algorytm aproksymacji średniokwadratowej - Przyjmując, że α to współczynnik uczenia, zasada aktualizacji aproksymacji średniokwadratowej (Least Mean Square, LMS) z wykorzystaniem m przykładów z danych treningowych (zwana także algorytmem Widrow-Hoffa) wygląda następująco:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Przypomnienie: zasada aktualizacji to szczególny przypadek wchodzenia gradientu.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR ― Regresja ważona lokalnie, jest odmianą regresji liniowej, w której waży się każdy przykład ze zbioru treningowego funkcją kosztu w(i)(x), która jest zdefiniowana z wykorzystaniem parametru t∈R w sposób następujący:

<br>

**28. Classification and logistic regression**

&#10230; Klasyfikacja i regresja logistyczna

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Funkcja sigmoidalna - Funkcja sigmoidalna g, anana także jako funkcja logistyczna, jest zdefiniowana w następujący sposób:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Regresja logistyczna ―  Zakładając, że y|x;θ∼Bernoulli(ϕ). Mamy następującą formułę:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Przypomnienie: nie istnieje zamknięte rozwiązanie przypadku regresji logistycznej.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Regresja softmax ―  Regresja softmax, zwana także wieloklasową regresją logistyczną, używana jest jako uogólnienie regresji logistycznej w przypadku, gdy mamy więcej niż 2 klasy wynikowe. Konwencją jest, że θK=0, czyni to parametr Bernoulliego ϕi każdej klasy i równy:

<br>

**33. Generalized Linear Models**

&#10230; Generalne modeli liniowych

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Rodzina wykładnicza ― O klasie rozkładu mówi się, że należy do rodziny wykładniczej jeśli można ją zapisać z wykorzystaniem parametrów naturalnych, zwanych także kanonicznymi parametrami η, wystarczającej statystyki T(y) i podzału logarytmicznego funkcji a(η) w nastepujący sposób:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Przypomnienie: często zdarzy się, że T(y)=y. Więc exp(-a(η)) może być rozumiany jako parametr normalizujący, który zapewni, że suma prawdopodobieństw będzie wynosiła 1.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; W tabeli przedstawione są najczęściej spotykane rozkłady wykładnicze:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Rozkład, Bernoulli, Gaussian, Poisson, Geometric]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Założenia generalnych modeli liniowych ― generalne modele liniowe mają za zadanie przewidzieć losową zmienną y jako funkcje x∈Rn+1 i opieraja się na 3 założeniach:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Przypomnienie: zwykła metoda najmniejszych kwadratów i regresja logistyczna to przypadki szczególne generalnych modeli liniowych.

<br>

**40. Support Vector Machines**

&#10230; Maszyny wektorów nośnych (Support Vector Machines)

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; Celem maszyn wektorów nośnych jest znalezienie hiperpłaszczyzny, która maksymalizuje margines pomiędzy przykładami oddzielnych klas.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Klasyfikator optymalnego marginesu ― Klasyfikator optymalnego marginesu h jest opisany następująco:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; gdzie (w,b)∈Rn×R jest rozwiązaniem następującego problemu optymalizacyjnego:

<br>

**44. such that**

&#10230; takich, że

<br>

**45. support vectors**

&#10230; wektory nośne 

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Przypomnienie: linia zdefiniowana jest jako wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Strata Hinge'a ― Strata Hinge'a jest wykorzystywana w maszynach wektorów nośnych, definiowana jest następująco:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Jądro ― Mając mapowanie ϕ, definiujemy jądro K jako:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; W praktyce, jądro K zdefiniowane jako K(x,z)=exp(−||x−z||22σ2) nazywane jest Jądrem Gaussa i jest powszechnie używane.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Nieliniowa separowalność, Użycie mapowania jądra, Rozgraniczenie decyzyjne w oryginalnej przestrzeni]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Przypomnienie: mówimy, że używamy "kernel trick" do opliczenia funkcji kosztu wykorzystując jądro, ponieważ w rzeczywistości nie potrzebujemy znać mapowania ϕ, które często bywa skomplikowane. W zamian, jedynie wartości K(x,z) są potrzebne.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangian ― Definiujemy Lagrangian L(w,b) następująco:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Przypomnienie: współczynniki βi nazywane są mnożnikami Legrange'a

<br>

**54. Generative Learning**

&#10230; Uczenie generatywne

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Model generatywny po pierwsze stara się nauczyć jak dane są generowane poprzez estymacje P(x|y), które możemy użyć do estymacji P(y|x) korzystając z reguły Bayesa.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Analiza dykryminanty Gaussa

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Założenia ―  Analiza dyskryminanty gaussa zakłada że y i x|y=0 i x|y=1 i jest taka, że:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Estymacja ― Następująca tabela przedstawia estymaty, które widać przy maksymalizacji prawdopodobieństwa:

<br>

**59. Naive Bayes**

&#10230; Naiwny klasyfikator bayesowski

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Założenie ― Naiwny klasyfikator bayesowski zakłada, że cechy (features) każdego przykładu są niezależne.

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Rozwiązanie ― Maksymalizując logarytmiczne prawdopodobieństwo otrzymujemy następująze rozwiązanie z k∈{0,1},l∈[[1,L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Przypomnienie: Naiwny klasyfikator bayesowski jest powszechnie używany do klasyfikacji tekstu i detekcji spamu.

<br>

**63. Tree-based and ensemble methods**

&#10230; Metody oparte o drzewa i "ensembling"

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Te metody mogą być używane zarówno do problemów regresyjnych jak i klasyfikacyjnych.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART ― Drzewa klasyfikacyjne i regresyjne (Classification and Regression Trees), zwane także drzewami decyzyjnymi, mogą być reprezentowane jako drzewa binarne. Zaletą tych metod jest ich wysoka interpretowalność.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Las losowy ― Jest to metoda oparta na drzewach, która wykorzystuje dużą ilość drzew decyzyjnych, opeartych na losowo dobieranych cechach. W przeciwieństwie do prostego drzewa decyzyjnego, jest on wysoce nieinterpretowalny, jednak dobra efektywność czyni go popularnym algorytmem.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Przypomnienie: las losowy jest rodzajem algorytmu opartego na ensemblingu.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Boostowanie ― pomysł polega na połączeniu kilku słabszych modeli w celu otworzenia jednego silniejszego. Poniżej przedstawione są główne rodzaje:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Boostowanie adaptacyjne, Boostowanie gradientowe]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; Duża waga kładziona jest na błędy w celu polepszenia wyniku w następnym kroku boostującym 

<br>

**71. Weak learners trained on remaining errors**

&#10230; Słabe modele trenowane są na pozostałych błędach

<br>

**72. Other non-parametric approaches**

&#10230; Inne nie sparametryzowane podejścia

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-najbliżsi sąsiedzi ― Algorytm k-najbliższych sąsiadów, znany powszechnie jako k-NN, jest nie sparametryzowanym podejściem, gdzie przynależność danego przykładu do danej klasy zależy od przynależności k-najbliższych punktów. Może być wykorzystywane zarówno przy klasyfikacji, jak i regresji.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Przypomnienie: Im wyższy parametr k, tym niższe dopasowanie, im mniejszy parametr k, tym wyższe dopasowanie.

<br>

**75. Learning Theory**

&#10230; Teoria uczenia

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Nierównośc Boole'a (Boole's inequality, union bound) ― Przyjmując, że A1,...,Ak są k zdarzeniami. Mamy:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Nierówność Hoeffding'a ― Przyjmując, że Z1,...,Zm są m zmiennymi pobranymi z rozkładu Bernoulli'ego parametru ϕ. Przyjmując, ze ˆϕ jest średnią próbki i y>0, mamy:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Przypomnienie: nierówność ta zwana jest także "Chernoff bound".

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Błąd uczenia ― Dla danego klasyfikatora h, definiujemy błąd treningowy ˆϵ(h), znana także jako błąd empiryczny lub ryzyko empiryczne. Wygląda następująco:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:**

&#10230; 

<br>

**81: the training and testing sets follow the same distribution**

&#10230;

<br>

**82. the training examples are drawn independently**

&#10230;

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230;

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230;

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230;

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230;

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230;
