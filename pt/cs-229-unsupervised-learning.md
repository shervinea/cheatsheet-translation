**1. Unsupervised Learning cheatsheet**

&#10230; Dicas de aprendizado não supervisionado

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Introdução ao aprendizado não supervisionado

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivação ― O objetivo do aprendizado não supervisionado é encontrar padrões em dados sem rótulo {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Desigualdade  de Jensen's ― Seja f uma função convexa e X uma variável aleatória. Temos a seguinte desigualdade:

<br>

**5. Clustering**

&#10230; Agrupamento

<br>

**6. Expectation-Maximization**

&#10230; Maximização de expectativa

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Variáveis latentes ― Variáveis latentes são variáveis escondidas/não observadas que dificultam problemas de estimativa, e são geralmente indicadas por z. Aqui estão as mais comuns configurações onde há variáveis latentes:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Configuração, Variável latente z, Comentários]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Mistura de Gaussianos k, Análise de fator]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algoritmo ― O algoritmo de maximização de expectativa fornece um método eficiente para estimar o parâmetro θ através da probabilidade máxima estimada ao construir repetidamente uma fronteira inferior na probabilidade (E-step) e otimizar essa fronteira inferior (M-step) como a seguir:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step: Avalia a probabilidade posterior Qi(z(i)) na qual cada ponto de dado x(i) veio de um grupo particular z(i) como a seguir:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step: Usa as probabilidades posteriores Qi(z(i)) como grupo específico de pesos nos pontos de dado x(i) para separadamente estimar cada modelo do grupo como a seguir:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Inicialização Gaussiana, Expectativa de passo, Maximização de passo, Convergência]

<br>

**14. k-means clustering**

&#10230; agrupamento k-means

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Nós indicamos c(i) o grupo de pontos de dados i e μj o centro do grupo j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritmo ― Após aleatoriamente inicializar os centróides do grupo μ1,μ2,...,μk∈Rn, o algoritmo k-means repete os seguintes passos até a convergência:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Inicialização de meio, Atribuição de grupo, Atualização de meio, Convergência]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Função de distorção ― A fim de ver se o algoritmo converge, nós olhamos para a função de distorção definida como se segue:

<br>

**19. Hierarchical clustering**

&#10230; Agrupamento hierárquico

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algoritmo ― É um algoritmo de agrupamento com uma abordagem hierárquica aglometariva que constrói grupos aninhados de uma maneira sucessiva.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Tipos ― Existem diferentes tipos de algoritmos de agrupamento hierárquico que objetivam a otimizar funções objetivas diferentes, os quais estão resumidos na tabela abaixo:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ligação de vigia, Ligação média, Ligação completa]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Minimizar distância dentro do grupo, Minimizar a distância média entre pares de grupos, Minimizar a distância máxima entre pares de grupos]

<br>

**24. Clustering assessment metrics**

&#10230; Métricas de atribuição de agrupamento

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; Em uma configuração de aprendizado não supervisionado, é geralmente difícil acessar o desempenho de um modelo desde que não temos rótulos de verdade como era o caso na configuração de aprendizado supervisionado.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Coeficiente de silhueta ― Ao indicar a e b a distância média entre uma amostra e todos os outros pontos na mesma classe, e entre uma amostra e todos os outros pontos no grupo mais próximo, o coeficiente de silhueta s para uma única amostra é definida como se segue:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Índice Calinski-Harabaz ― Indicando por k o número de grupos, Bk e Wk as matrizes de disperção entre e dentro do agrupamento respectivamente definidos como

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; o índice Calinski-Harabaz s(k) indica quão bem um modelo de agrupamento define o seu grupo, tal que maior a pontuação, mais denso e bem separado os grupos estão. Ele é definido como a seguir:

<br>

**29. Dimension reduction**

&#10230; Redução de dimensão

<br>

**30. Principal component analysis**

&#10230; Análise de componente principal

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; É uma técnica de  redução de dimensão que encontra direções de maximização de variância em que projetam os dados.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalor, autovetor ― Dada uma matriz A∈Rn×n, λ é dito ser um autovalor de A se existe um vetor z∈Rn∖{0}, chamado autovetor, tal que temos:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema espectral - Seja A∈Rn×n. Se A é simétrica, então A é diagonizável por uma matriz ortogonal U∈Rn×n. Denotando Λ=diag(λ1,...,λn), temos:

<br>

**34. diagonal**

&#10230; diagonal

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Observação: o autovetor associado com o maior autovalor é chamado de autovetor principal da matriz A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritmo ― O processo de Análise de Componente Principal (PCA) é uma técnica de redução de dimensão que projeta os dados em dimensões k ao maximizar a variância dos dados como se segue:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Etapa 1: Normalizar os dados para ter uma média de 0 e um desvio padrão de 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Etapa 2: Computar Σ=1mm∑i=1x(i)x(i)T∈Rn×n, a qual é simétrica com autovalores reais.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Etapa 3: Computar u1,...,uk∈Rn os k principais autovetores ortogonais de Σ, i.e. os autovetores ortogonais dos k maiores autovalores

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Etapa 4: Projetar os dados em spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Esse processo maximiza a variância entre todos os espaços dimensionais k.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Dados em espaço característico, Encontrar componentes principais, Dados no espaço de componentes principais]

<br>

**43. Independent component analysis**

&#10230; Análise de componente independente

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; É uma técnica que pretende encontrar as fontes de geração subjacente.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Suposições ― Nós assumimos que nosso dado x foi gerado por um vetor fonte dimensional n s=(s1,...,sn), onde si são variáveis aleatórias independentes, através de uma matriz A misturada e não singular como se segue:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; O objetivo é encontrar a matriz W=A−1. não misturada.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Algoritmo Bell e Sejnowski ICA ― Esse algoritmo encontra a matriz W não misturada pelas seguintes etapas abaixo:

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Escreva a probabilidade de x=As=W−1s como:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Escreva o logaritmo da probabilidade dado o nosso dado treinado {x(i),i∈[[1,m]]} e indicando g a função sigmoide como:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Portanto, a regra de aprendizagem do gradiente ascendente estocástico é tal que para cada exemplo de treinamento x(i), nós atualizamos W como a seguir:

<br>

**51. The Machine Learning cheatsheets are now available in Portuguese.**

&#10230; As dicas de Aprendizado de Máquina agora estão disponíveis em Português.

<br>

**52. Original authors**

&#10230; Autores originais

<br>

**53. Translated by X, Y and Z**

&#10230; Traduzido por X, Y e Z

<br>

**54. Reviewed by X, Y and Z**

&#10230; Revisado por X, Y e Z

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Introdução, Motivação, Desigualdade de Jensen]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Agrupamento, Maximização de expectativa, k-means, Agrupamento hierárquico, Métricas]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Redução de dimensão, PCA, ICA]
