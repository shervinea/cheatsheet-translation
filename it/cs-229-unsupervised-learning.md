**Unsupervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning)

<br>

**1. Unsupervised Learning cheatsheet**

&#10230; Formulario sull'apprendimento senza supervisione

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Introduzione all'apprendimento senza supervisione

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivazione - L'obiettivo dell'apprendimento senza supervisione è quello di trovare modelli nascosti in dati senza etichetta {x (1), ..., x (m)}. **

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Disuguaglianza di Jensen - Sia f una funzione convessa e X una variabile casuale. Abbiamo la seguente disuguaglianza:

<br>

**5. Clustering** 

&#10230; Clustering

<br>

**6. Expectation-Maximization**

&#10230; Expectation-Maximization

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Variabili latenti: le variabili latenti sono variabili nascoste / non osservate che rendono difficili i problemi di stima e sono spesso indicate da z. Seguono le impostazioni più comuni in cui sono presenti variabili latenti

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Impostazioni, variabile latente z, commenti] 

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Miscela di k gaussiani, analisi fattoriale]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algoritmo - L'algoritmo Expectation-Maximization (EM) fornisce un metodo efficiente per stimare il parametro θ attraverso la stima della massima verosimiglianza costruendo ripetutamente un limite inferiore sulla verosimiglianza (E-step) e ottimizzando quel limite inferiore (M-step) come segue :

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step: valutare la probabilità a posteriori Qi(z(i)) che ciascun punto x(i) provenga da un particolare cluster z(i) come segue: 

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; Fase M: utilizzare la probabilità a posteriori Qi(z(i)) come peso specifico del cluster sui dati x(i) per rivalutare separatamente ciascun modello di cluster come segue: 

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Inizializzazione gaussiana, fase di aspettativa, fase di massimizzazione, convergenza] 

<br>

**14. k-means clustering** 

&#10230; clustering k-signifivo

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Notiamo che c(i) e' il cluster dei dati i e μj il centro del cluster j. 

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritmo - Dopo l'inizializzazione casuale dei centroidi del cluster μ1, μ2, ..., μk∈Rn, l'algoritmo k-medie ripete il passaggio seguente fino alla convergenza: 

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Inizializzazione dei mezzi, assegnazione del cluster, aggiornamento delle meadie, convergenza] 

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Funzione di distorsione - Per vedere se l'algoritmo converge, osserviamo la funzione di distorsione definita come segue: 

<br>

**19. Hierarchical clustering**

&#10230; Cluster gerarchico 

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algoritmo: è un algoritmo di clustering con un approccio gerarchico agglomerato che crea cluster nidificati in modo ripetitivo. 

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Esistono diversi tipi di algoritmi di clustering gerarchici che mirano ad ottimizzare diverse funzioni oggettive, che è riassunto nella tabella seguente: 

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ward linkage, Average linkage, Complete linkage]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Riduci al minimo la distanza del cluster, Riduci al minimo la distanza media tra le coppie del cluster, Riduci al minimo la distanza massima tra le coppie del cluster] 

<br>

**24. Clustering assessment metrics**

&#10230; Metriche di valutazione del cluster

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; In un'impostazione di apprendimento senza supervisione, è spesso difficile valutare le prestazioni di un modello poiché non abbiamo le etichette di verità di base come nel caso dell'apprendimento con supervisione. 

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Coefficiente di sagoma - Notando aeb la distanza media tra un campione e tutti gli altri punti della stessa classe, e tra un campione e tutti gli altri punti nel successivo cluster più vicino, il coefficiente di sagoma s per un singolo campione è definito come segue: 

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Indice di Calinski-Harabaz - Notando k il numero di cluster, Bk e Wk le matrici di dispersione tra e all'interno del cluster rispettivamente definite come 

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; l'indice s (k) di Calinski-Harabaz indica quanto bene un modello di clustering definisce i suoi cluster, in modo tale che più alto è il punteggio, più densi e ben separati sono i cluster. È definito come segue: 

<br>

**29. Dimension reduction**

&#10230; Riduzione dimensionale 

<br>

**30. Principal component analysis**

&#10230; Analisi dei componenti principali
 
<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; È una tecnica di riduzione dimensionale che trova la varianza massimizzando le direzioni su cui proiettare i dati. 

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalore, autovettore - Data una matrice A∈Rn × n, si dice che λ è un autovalore di A se esiste un vettore z∈Rn ∖ {0}, chiamato autovettore, tale che abbiamo: 

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema spettrale - Sia A∈Rn × n. Se A è simmetrico, allora A è diagonale con una matrice ortogonale reale U∈Rn × n. Notando Λ = diag (λ1, ..., λn), abbiamo: 

<br>

**34. diagonal**

&#10230; diagonale

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Nota: l'autovettore associato al più grande autovalore è chiamato autovettore principale della matrice A. 

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritmo - La procedura di analisi dei componenti principali (PCA) è una tecnica di riduzione dimensionale che proietta i dati su k
dimensioni massimizzando la varianza dei dati come segue: 

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Passaggio 1: normalizzare i dati per avere una media di 0 e una deviazione standard di 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Passaggio 2: Calcola Σ = 1mm∑i = 1x (i) x (i) T∈Rn × n, che è simmetrico con autovalori reali. 

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Passaggio 3: calcolare u1, ..., uk∈Rn i k autovettori principali ortogonali di Σ, vale a dire gli autovettori ortogonali dei k autovalori più grandi.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230;Passaggio 4: proiettare i dati su spanR (u1, ..., uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Questa procedura massimizza la varianza tra tutti gli spazi k-dimensionali. 

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Dati nello spazio delle funzioni, Trova componenti principali, Dati nello spazio dei componenti principali] 

<br>

**43. Independent component analysis**

&#10230; Analisi dei componenti indipendenti 

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; È una tecnica intesa a trovare le fonti di generazione sottostanti. 

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Presupposti - Partiamo dal presupposto che i nostri dati x siano stati generati dal vettore sorgente n-dimensionale s = (s1, ..., sn), dove si sono variabili casuali indipendenti, tramite una matrice di miscelazione e non singolare A come segue: 

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; L'obiettivo è trovare la matrice non mescolante W = A − 1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Algoritmo ICA di Bell e Sejnowski: questo algoritmo trova la matrice non mescolante W seguendo i passaggi seguenti: 
 
<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Scrivi la probabilità di x = As = W − 1s come: 

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Scrivi la verosimiglianza dei dati dati i nostri dati di addestramento {x (i), i∈ [[1, m]]} e notando g la funzione sigmoid come: 

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Pertanto, la regola di apprendimento dell'ascensione gradiente stocastica è tale che per ogni esempio di allenamento x (i), aggiorniamo W come segue:

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; I formulari di Machine Learning sono ora disponibili in italiano.

<br> 

**52. Original authors**

&#10230; Autori originali 

<br>

**53. Translated by X, Y and Z**

&#10230; Tradotto da Gian Maria Troiani

<br>

**54. Reviewed by X, Y and Z**

&#10230; Revisionati da X

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Introduzione, motivazione, disuguaglianza di Jensen] 

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Clustering, Massimizzazione delle aspettative, k-medie, Cluster gerarchico, Metriche] 

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Riduzione dimensionale, PCA, ICA]
