**1. Supervised Learning cheatsheet**

&#10230; Felügyelt tanulás segédanyag

<br>

**2. Introduction to Supervised Learning**

&#10230; Bevezetés a felügyelt tanulásba

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Adott az {x(1),...,x(m)} mintapontokból mint bemenetből álló halmaz és a hozzájuk tartozó {y(1),...,y(m)} kimenethalmaz. Célunk olyan leképezés megtanulása, mely meg tudja jósolni (másképp: előre tudja jelezni) y-t x-ből és megfelelő általánosítási képességekkel rendelkezik.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Az előrejelzés típusai ― Különböző előrejelző modelleket az alábbi táblázat foglalja össze:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regresszió, Klasszifikáció (osztályozás), Kimenet, Példák]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Folytonos, Osztály, Lineáris regresszió, Logisztikus regresszió, SVM, Naív Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Modelltípusok ― Két különböző modelltípust mutatunk be a következő táblázatban:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Diszkriminatív modell, Generatív modell, Cél, Mit tanul meg, Illusztráció, Példák]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [P(y|x) közvetlen becslése, P(x|y) becslése által P(y|x)-re következtetés, Döntési határ, Az adatok valószínűségi eloszlása, Regressziók, SVM-ek, Gauss-féle diszkriminanciaanalízis (GDA), Naív Bayes]

<br>

**10. Notations and general concepts**

&#10230; Jelölések és általános fogalmak

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hipotézisfüggvény ― A hipotézisfüggvény (jel.: hθ) a megtanulandó leképezés. Adott x(i) bemeneti adatok esetén a modell által előrejelzett kimeneteket hθ(x(i))-vel jelöljük, ahol θ jelzi a modell paramétereit.

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Veszteségfüggvény ― A költségfüggvény az az L:(z,y)∈R×Y⟼L(z,y)∈R leképezés, mely bemenetként az előrejelzett z értékeket és az adott y-értékeket várja, és kimenetként megadja az ezek közti eltérés nagyságát. A leggyakrabban használt költségfüggvényeket az alábbi táblázat tartalmazza:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Legkisebb négyzetek, Logisztikus hiba, Zsanérveszteség (Hinge loss), Kereszt-entrópia]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Lineáris regresszió, Logisztikus regresszió, Tartóvektorgép (SVM), Neurális hálózat]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Költségfüggvény ― A J költségfüggvényt gyakran használjuk a modell teljesítményének méréséhez, és az L veszteségfüggvény segítségével az alábbi módon definiáljuk:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; A gradiensmódszer ― Legyen α∈R a tanulási ráta és J a költségfüggvény, ekkor a gradiensmódszer iteratív képletét az alábbi módon fejezhetjük ki:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Megjegyzés: A sztochasztikus gradiensmódszerben (SGD) a paraméter értékét minden egyes tanítóadat alapján frissítjük, míg a kötegelt gradiensmódszerben a tanítóadatok egy részhalmaza (kötege) alapján.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Likelihood ― Adott θ paraméterek esetén a modell likelihoodját (jel.: L(θ)) úgy számítjuk, hogy megkeressük az optimális θ paramétereket a maximum likelihood becslés segítségével. A gyakorlatban az ℓ(θ)=log(L(θ)) log-likelihoodot használjuk, ugyanis könnyebb optimalizálni. Ekkor: 

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Newton-módszer ― A Newton-módszer (más néven Newton–Raphson-módszer) olyan numerikus gyökkereső módszer, mely megkeresi θ paramétert, melyre ℓ′(θ)=0. Az iteratív képlete:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Megjegyzés: a többdimenziós általánosítás esetén a képlet:

<br>

**21. Linear models**

&#10230; Lineáris modellek

<br>

**22. Linear regression**

&#10230; Lineáris regresszió

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Feltesszük, hogy y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Gauß-féle normálegyenletek ― Legyen X a modellmátrix. Ekkor a költségfüggvényt minimalizáló θ-érték kielégíti az alábbi egyenlőséget:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; LMS-algortimus ― Legyen α a tanulási ráta. Ekkor az LMS-algoritmus formulája (más néven Widrow―Hoff-féle tanulási szabály) m darab tanító adatpont esetén:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Megjegyzés: a formula egy speciális esete a gradiensmódszernek.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; Helyileg súlyozott regresszió (LWR) ― A helyileg súlyozott regresszió a lineáris regresszió fajtája, mely minden tanító adatot w(i)(x)-szel súlyoz, melyet így definiálunk (τ∈R paraméter esetén):

<br>

**28. Classification and logistic regression**

&#10230; Klasszifikáció és logisztikus regresszió

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Szigmoidfüggvény ― Az ún. g szigmoidfüggvényt (más néven logisztikus függvényt) így definiáljuk:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Logisztikus regresszió ― Feltesszük, hogy y|x;θ∼Bernoulli(ϕ). Ekkor fennáll az alábbi formula:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Megjegyzés: nem létezik zárt alak a logisztikus regresszió megoldására.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Szoftmax regresszió ― A szoftmax regresszió (más néven többosztályú logisztikus regresszió) a logisztikus regresszió  általánosítása, amikor több mint két kimeneti osztály adott. Megállapodás szerint legyen θK=0, ami alapján az i-edik osztály ϕi indikátor paraméterére fennáll:

<br>

**33. Generalized Linear Models**

&#10230; Általánosított lineáris modellek (GLM-ek)

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Exponenciális család ― Eloszlások egy osztálya az exponenciális családba tartozik, ha felírható egy η természetes paraméter (más néven kanonikus paraméter vagy kapcsolati függvény), T(y) elégséges statisztika és a(η) log-partíció függvény segítségével az alábbi módon:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Megjegyzés: gyakran T(y)=y. Továbbá tekinthetünk úgy az exp(−a(η)) paraméterre mint a normalizációs konstansra, amely garantálja, hogy a p(y;η) eloszlás y feletti integrálja 1.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Az alábbi táblázat tartalmazza a leggyakoribb exponenciális családbeli eloszlásokat:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Eloszlás, Indikátor, Normális, Poisson, Geometriai]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function of x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; A GLM-ek feltételei ― A GLM-ek célja, hogy előrejelezze az y val. változót x∈Rn+1 függvényében. Ehhez az alábbi három feltétel kell, hogy teljesüljön:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Megjegyzés: a legkisebb négyzetek módszere (OLS) és a logisztikus regresszió is speciális esete a GLM-eknek.

<br>

**40. Support Vector Machines**

&#10230; Tartóvektorgépek (SVM-ek)

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; Az SVM-ek célja olyan hipersík megtalálása, mely maximalizálja az adatpontoknak a hipersíktól vett minimális távolságát.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Optimális margójú klasszifikátor ― A h-val jelölt optimális margójú klasszifikátorra igaz, hogy:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; ahol (w,b)∈Rn×R az alábbi optimalizációs probléma megoldása:

<br>

**44. such that**

&#10230; úgy, hogy 

<br>

**45. support vectors**

&#10230; tartóvektorok

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Megjegyzés: a hipersíkot az következő alakban írhatjuk fel:

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Zsanérveszteség ― A zsanérveszteség-függvényt (hinge loss) az SVM-ek kontextusában használjuk, és így definiáljuk:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Magfüggvény (kernel) ― Adott ϕ tulajdonságleképezés esetén a K magfüggvényt így definiáljuk:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; A gyakorlatban a K(x,z)=exp(−||x−z||22σ2) egyenlőséggel definiált magfüggvényt Gauß-féle magfüggvénynek hívjuk.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Nemlineáris szeparábilitás, Magfüggvény alkalmazása, Döntési határ az eredeti térben]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Megjegyzés: azt mondjuk, hogy az ún. kerneltrükköt alkalmazzuk a költségfüggvény kiszámolására, ugyanis igazából nem szükséges ismernünk az ϕ leképezést (ami sokszor bonyolult). Ehelyett elég ismernünk a K(x,z) értékeket.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrange-függvény ― Az L(w,b) ún. Lagrange-függvényt így definiáljuk:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Megjegyzés: a βi együtthatókat Lagrange-multiplikátoroknak nevezzük.

<br>

**54. Generative Learning**

&#10230; Generatív tanulás

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; A generatív modellek előbb megpróbálják megbecsülni P(x|y) valószínűséget, amit aztán felhasználhatunk P(y|x) kiszámítására a Bayes-tétel alapján.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Diszkriminanciaanalízis

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Felállás ― A diszkriminanciaanalízisban feltesszük, hogy az alábbiak fennállnak:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Becslés ― Az alábbi táblázat foglalja össze azokat a becsléseket, melyeket a likelihood maximalizálásával kapunk:

<br>

**59. Naive Bayes**

&#10230; Naív Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Feltevés ― A Naív Bayes-modellben feltesszük, hogy az adatpontok tulajdonságai függetlenek:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Megoldások ― A log-likelihoodot maximalizálva a következő megoldásokat kapjuk (ahol k∈{0,1},l∈[[1,L]]):

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Megjegyzés: A Naív Bayest széles körben alkalmazzák a szövegklasszifikáció és spamfelismerés területén. 

<br>

**63. Tree-based and ensemble methods**

&#10230; Faalapú és együttes (ensemble) módszerek 

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Ezek a módszerek regressziós és klasszifikációs problémák esetén egyaránt alkalmazhatók. 

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; Klasszifikációs és Regressziós Fák (CART), ismertebb nevükön döntési fák): bináris fáként reprezentálhatóak. Előnyük, hogy könnyen értelmezhetőek.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Véletlen erdők ― Olyan faalapú modell, mely nagy számú döntési fát épít véletlenszerűen választott tulajdonsághalmazból. Az egyszerű döntési fával ellentétben kevésbe értelmezhetőek, de nagyrészt jó teljesítményük miatt eléggé elterjedtek.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Megjegyzés: a véletlen erdők az együttes módszerek egy típuát alkotják.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Gyorsítás (boosting) ― A gyorsító modellek ötlete, hogy néhány, gyengébb alapklasszifikátort kombinálva egy erősebbet kapunk. A leggyakoribbakat az alábbi táblázatban foglaltuk össze:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Adaptív gyorsítás, Gradiensalapú gyorsítás]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; A hibák nagy súlyokat kapnak, hogy a következő gyorsító lépésben javuljon a tanulás.

<br>

**71. Weak learners trained on remaining errors**

&#10230; Az alapklasszifikátorok a maradék hibán tanulnak.

<br>

**72. Other non-parametric approaches**

&#10230; Egyéb, nemparaméteres megközelítések

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k Legközelebbi Szomszéd ― A k Legközelebbi Szomszéd algortimus (jel.: k-NN) olyan nemparaméteres megközelítés, ahol egy adatpont címkéjét a k darab legközelebbi tanulóadat címkéje határozza meg. Alkalmazható klasszifikációs és regressziós feladatokra is.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Megjegyzés: Minél nagyobb a k paraméter értéke, annál nagyobb a torzítás (bias), illetve minél kisebb a k paraméter, annál nagyobb a variancia (variance).

<br>

**75. Learning Theory**

&#10230; Tanuláselmélet

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; σ-szubadditivitás (más néven Boole-egyenlőtlenség) ― Legyenek A1,...,Ak események. Ekkor:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Hoeffding-egyenlőtlenség ― Legyen Z1,..,Zm iid val. változó ϕ paraméterű indikátor eloszlásból. Legyen ˆϕ a mintaátlaguk és γ>0 rögzített. Ekkor:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Megjegyzés: ezt az egyenlőtlenséget Chernoff-határként is nevezik.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Tanulási hiba ― Adott h klasszifikátor esetén a tanulási hibát (jel.: ˆϵ(h)), más néven empirikus hiba) így definiáljuk:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; Valószínűleg közelítőleg helyes (PAC) ― A PAC olyan elv, melynek segítéségével sok tanuláselméletbeli eredmény bizonyítható. Az alábbi feltevések tartoznek ide:

<br>

**81: the training and testing sets follow the same distribution **

&#10230; a tanító és teszthalmazok ugyanolyan eloszlást követnek

<br>

**82. the training examples are drawn independently**

&#10230; a tanító adatok egymástól függetlenek

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Szétzúzás ― Adott S={x(1),...,x(d)} halmaz és H klasszifikátorok halmaza. Ekkor azt mondjuk, hogy H szétzúzza S-et, ha bármely {y(1),...,y(d)} címkehalmazra fennáll:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Felső korlát tétel ― Legyen H egy véges hipotéziscsalád, melyre |H|=k, valamint legyen δ és m (a mintaméret) rögzítettek. Ekkor legalább 1−δ valószínűséggel fennáll, hogy:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC-dimenzió ― Egy adott H végtelen hipotézisosztálynak a Vapnik―Cservonenkis (VC)-dimenziója (jel.: VC(H)) annak a legnagyobb halmaznak a mérete, melyet H szétzúz.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Megjegyzés: ha H={2-dimenziós lineáris klasszifikátorok halmaza}, akkor VC(H)=3.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Tétel (Vapnik) ― Legyen H adott, melyre VC(H)=d, és legyen m a tanító adatok száma. Ekkor legalább 1−δ valószínűséggel fennáll, hogy:

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [Bevezetés, Előrejelzls típusai, Modelltípusok]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [Jelölések és általános fogalmak, veszteségfüggvény, gradiensmódszer, likelihood]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [Lineáris modellek, lineáris regresszió, logisztikus regresszió, általánosított lineáris modellek]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [Tartóvektorgépek (SVM), Optimális margójú klasszifikátor, Zsanérveszteség (hinge loss), Magfüggvény (kernel)]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [Generatív tanulás, Diszkriminanciaanalízis, Naív Bayes]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [Faalapú és együttes (ensemble) módszerek, CART, Véletlen erdők, Gyorsítás (boosting)]

<br>

**94. [Other methods, k-NN]**

&#10230; [Egyéb módszerek, k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [Tanuláselmélet, Hoeffding-egyenlőtlenség, PAC, VC-dimenzió]
