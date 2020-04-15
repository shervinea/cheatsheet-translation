**Supervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning)

<br>

**1. Supervised Learning cheatsheet**

&#10230; Formulario sull'apprendimento con supervisione

<br>

**2. Introduction to Supervised Learning**

&#10230; Introduzione all'apprendimento con supervisione

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Dato un insieme di punti {x (1), ..., x (m)} associati ad un insieme di risultati {y (1), ..., y (m)}, vogliamo costruire un classificatore che impari a prevedere y da x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Tipi di previsione: i diversi tipi di modelli predittivi sono riassunti nella tabella seguente: 
<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regressione, classificatore, risultato, esempi]
<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Continuo, Classe, regressione lineare, regressione logistica, SVM, Naive Bayes]
<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Tipo di modello - I diversi modelli sono riassunti nella tabella seguente:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Modello discriminante, Modello generativo, Obiettivo, Ente Appreso, Illustrazione, Esempi]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Stima direttamente P (y | x), Stima P (x | y) per poi dedurre P (y | x), Limite decisionale, Distribuzioni di probabilità dei dati, Regressioni, SVM, GDA, Naive Bayes]

<br>

**10. Notations and general concepts**

&#10230; Notazione e concetti generali

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Ipotesi - L'ipotesi viene rappresentata come hθ ed è il modello che selezioniamo. Per ogni dato input x(i) il risultato di previsione del modello è hθ(x (i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Funzione di costo - Una funzione di costo è una funzione L: (z, y) ∈R × Y⟼L (z, y) ∈R che accetta come input il valore previsto z corrispondente al valore reale dei dati y e produce quanto diverso loro sono. Le funzioni di costo comuni sono riassunte nella tabella seguente:
<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Errore al minimo quadrato, costo logistica, costo della cerniera, entropia incrociata]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [regressione lineare, regressione logistica, SVM, rete neurale]
<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Funzione di costo: la funzione di costo J viene comunemente usata per valutare le prestazioni di un modello e viene definita tramite la funzione di costo L come segue:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Discesa del gradiente - Notando che α, il tasso di apprendimento,e' tale che α∈R , la regola di aggiornamento per la discesa del gradiente viene espressa attraverso il tasso di apprendimento e la funzione di costo J come di seguito:
<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Osservazione: la discesa del gradiente stocastica (SGD) sta aggiornando il parametro in base a ciascun esempio di allenamento e la discesa del gradiente batch è eseguita su un lotto di istanze di allenamento.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Probabilità - La probabilità di un modello L (θ) dati parametri θ viene utilizzata per trovare i parametri ottimali θ massimizzando le probabilità. In pratica, usiamo la probabilità logaritmica ℓ (θ) = log (L (θ)) che è più facile da ottimizzare. Quindi abbiamo:
<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Algoritmo di Newton - L'algoritmo di Newton è un metodo numerico che trova θ tale che ℓ′(θ) = 0. La regola di aggiornamento è la seguente:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Osservazione: la generalizzazione multidimensionale, nota anche come metodo Newton-Raphson, ha la seguente regola di aggiornamento:

<br>

**21. Linear models**

&#10230; Modelli lineari

<br>

**22. Linear regression**

&#10230; regressione lineare
<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Qui assumiamo che y | x; θ∼N (μ, σ2)
<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Equazioni normali - Notando la struttura della matrice X, il valore di θ che minimizza la funzione di costo è un' equazione a forma chiusa tale che:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Algoritmo LMS - Notando il tasso di apprendimento α, la regola di aggiornamento dell'algoritmo Least Mean Squares (LMS) per un set di addestramento di m punti dati, nota anche come regola di apprendimento Widrow-Hoff, è la seguente:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Osservazione: la regola di aggiornamento è un caso particolare dell'ascesa del gradiente.
<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR - Regressione ponderata localmente, nota anche come LWR (Locally Weighted Regression), è una variante della regressione lineare che pondera ogni esempio di allenamento nella sua funzione di costo attraverso w (i) (x), che è definito con parametro τ∈R come segue:

<br>

**28. Classification and logistic regression**

&#10230; Classificazione e regressione logistica
<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Funzione sigmoidale - La funzione sigmoidale g, nota anche come funzione logistica, è definita come segue:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Regressione logistica - Assumendo che y | x; θ∼Bernoulli (ϕ). Otteniamo la seguente forma:
<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Osservazione: non esiste una soluzione a forma chiusa nel caso di regressioni logistiche.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Regressione softmax - Una regressione softmax, chiamata anche regressione logistica multiclasse, viene utilizzata per generalizzare la regressione logistica quando vi sono più di 2 classi di risultati. Per convenzione, impostiamo θK = 0, il che rende il parametro Bernoulli ϕi di ogni classe i uguale a:

<br>

**33. Generalized Linear Models**

&#10230; Modelli lineari generalizzati

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Famiglia esponenziale - Una classe di distribuzioni rientra nella famiglia esponenziale se può essere scritta in termini di un parametro naturale, chiamato anche parametro canonico o funzione di collegamento, η, una sufficiente statistica T(y) e una funzione logaritmico-partizionata a (η) come segue:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Osservazione: spesso avremo T (y) = y. Inoltre, exp (−a (η)) può essere visto come un parametro di normalizzazione che garantisce che la somma delle probabilità sia uno.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Ecco le distribuzioni esponenziali più comuni riassunte nella seguente tabella:
<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Distribuzione, Bernoulli, Gaussiana, Poisson, Geometrica]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Ipotesi di GLM - I Modelli Generali Generalizzati (GLM) mirano a prevedere la variabile casuale y come una funzione di x∈Rn + 1 e si basa sulle 3 seguenti assunzioni:


<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Osservazione: Ordinary Least Sqares (LMS) e la regressione logistica sono casi speciali di modelli lineari generalizzati.

<br>

**40. Support Vector Machines**

&#10230; Machine a Vettori di Supporto

<br>

**41. The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; L'obiettivo delle macchine a vettori di supporto è trovare la linea che massimizzi la distanza minima dalla linea.

<br>

**42. Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Classificatore di margine ottimale - Il classificatore di margine ottimale h è tale che:
<br>

**43. where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; dove (w, b) ∈Rn × R è la soluzione al seguente problema di ottimizzazione:

<br>

**44. such that**

&#10230; tale che

<br>

**45. support vectors**

&#10230; vettori di supporto

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Osservazione: la linea è definita come wTx − b = 0.
<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Hinge loss - La costo della cerniera viene utilizzata nell'impostazione degli SVM ed è definita come segue: 

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Kernel - Data una mappatura delle caratteristiche ϕ, definiamo il kernel K  come:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; In pratica, il kernel K definito da K (x, z) = exp (- || x − z || 22σ2) è chiamato kernel gaussiano ed è usato comunemente.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Separabilità non lineare, uso della mappatura del kernel ϕ, confine decisionale nello spazio originale]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Osservazione: Usiamo il "trucco del kernel" per calcolare la funzione di costo usando il kernel perché in realtà non abbiamo bisogno di conoscere la mappatura esplicita ϕ, che spesso è molto complicata. Invece, sono necessari solo i valori K (x, z).
<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangiano - Definiamo la Lagrangiana L (w, b) come segue:
<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Osservazione: i coefficienti βi sono chiamati moltiplicatori di Lagrange.
<br>

**54. Generative Learning**

&#10230; Apprendimento generativo

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Un modello generativo prima cerca di imparare come vengono generati i dati stimando P (x | y), che possiamo usare per stimare P (y | x) usando la regola di Bayes.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Analisi gaussiana del discriminante
<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Impostazione - L'analisi gaussiana del discriminante assume che y e x | y = 0 e x | y = 1 siano tali che:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Stima - La seguente tabella riassume le stime che troviamo quando massimizziamo le probabilità:

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Assunzione - Il modello Naive Bayes suppone che le caratteristiche di tutti i punti dati siano tutte indipendenti:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Soluzioni - Massimizzare la probabilità logaritmica restituisce le seguenti soluzioni, con k∈ {0,1}, l∈ [[1, L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Osservazione: Naive Bayes è ampiamente utilizzato per la classificazione di testi e il rilevamento dello spam.

<br>

**63. Tree-based and ensemble methods**

&#10230; Metodi ad albero decisionale ed ensemble

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Questi metodi possono essere utilizzati sia per problemi di regressione che di classificazione.
<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - Gli alberi di classificazione e regressione (CART), comunemente noti come alberi decisionali, possono essere rappresentati come alberi binari. Hanno il vantaggio di essere molto interpretabili.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Foresta casuale: si tratta di una tecnica basata su alberi che utilizza un numero elevato di alberi decisionali basati su insiemi di attributi selezionati casualmente. Contrariamente al semplice albero decisionale, è difficilmente interpretabile ma le generalmente buone prestazioni lo rendono un algoritmo popolare.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Osservazione: le foreste casuali sono un tipo di metodo ensemble.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Potenziamento - L'idea di potenziare i metodi consiste nel combinare diversi modelli di approndimento deboli per formarne uno più potente. I principali sono riassunti nella tabella seguente:
<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Potenziamento adattivo, potenziamento del gradiente]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; Sono assegnati costi elevati agli errori per migliorare nella fase di potenziamento successiva

<br>

**71. Weak learners trained on remaining errors**

&#10230; Modelli deboli addestrati sugli errori rimanenti

<br>

**72. Other non-parametric approaches**

&#10230; Altri approcci non parametrici

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-neighbors neighbors: l'algoritmo k-neighbors neighbors, comunemente noto come k-NN, è un approccio non parametrico la cui risposta ad un punto dato è determinata dalla natura dei suoi vicini k, provenienti dall'insieme di addestramento. Può essere utilizzato sia per compiti di classificazione che per compiti di regressione.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Osservazione: Maggiore il parametro k, maggiore il bias; minore il parametro k, maggiore è la varianza.
<br>

**75. Learning Theory**

&#10230; Teoria dell'apprendimento

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Unione vincolata - Siano A1, ..., Ak k eventi. Abbiamo:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Disuguaglianza di Hoeffding - Siano Z1, .., Zm m variabili identificate tra le variabili tratte da una distribuzione di Bernoulli con parametro ϕ. Sia ˆϕ la media di variabili tratte casualmente come campioni e assumiamo γ> 0 fisso. Abbiamo quindi:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Osservazione: questa disuguaglianza è nota anche come limite di Chernoff.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Errore di allenamento - Per un dato classificatore h, definiamo l'errore di allenamento ˆϵ (h), noto anche come rischio empirico o errore empirico, come segue:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:**

&#10230; Probabilmente approssimativamente corretto (PAC) - PAC è un metodo in base al quale sono stati dimostrati numerosi risultati sulla teoria dell'apprendimento e presenta le seguenti ipotesi:

<br>

**81: the training and testing sets follow the same distribution**

&#10230; i set di addestramento e test seguono la stessa distribuzione

<br>

**82. the training examples are drawn independently**

&#10230; gli esempi di addestramento sono selezionati in modo indipendente
<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering - Dato un set S = {x (1), ..., x (d)} ed un set di classificatori H, diciamo che H schiaccia S se per qualsiasi set di etichette {y (1), ..., y (d)}, abbiamo:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Teorema del limite superiore - Sia H una classe di ipotesi finita tale che | H | = k e siano δ e la dimensione del campione m siano fissi. Allora, con probabilità di almeno 1 − δ, abbiamo:
<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; Dimensione VC - La dimensione Vapnik-Chervonenkis (VC) di una data ipotesi infinita di classe H, nota VC (H) è la dimensione dell'insieme più grande che viene schiacciato da H.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Osservazione: la dimensione VC di H = {set di classificatori lineari in 2 dimensioni} è 3.
<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Teorema (Vapnik) - Dati H, con VC (H) = d e m il numero di esempi di addestramento. Con probabilità almeno 1 − δ, abbiamo:

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [Introduzione, Tipo di previsione, Tipo di modello]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [Notazioni e concetti generali, funzione di costo, discesa del gradiente, probabilità]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [Modelli lineari, regressione lineare, regressione logistica, modelli lineari generalizzati]
<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [Macchine a Vettori di Supporto, Classificatore di Margine Ottimale, Hinge Loss, kernel]
<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [Apprendimento generativo, Analisi gaussiana del discriminante, Naive Bayes]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [Alberi e metodi di ensemble, CART, Foresta casuale, Potenziamento]

<br>

**94. [Other methods, k-NN]**

&#10230; [Altri metodi, k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [Teoria dell'apprendimento, disuguaglianza di Hoeffding, PAC, dimensione VC]

