**1. Unsupervised Learning cheatsheet**

&#10230; Hoja de Referencia de Aprendizaje no Supervisado

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Introducción al Aprendizaje no Supervisado

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivación - El objetivo del aprendizaje no supervisado es encontrar patrones ocultos en datos no etiquetados {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Desigualdad de Jensen - Sea f una función convexa y X una variable aleatoria. Tenemos la siguiente desigualdad:

<br>

**5. Clustering**

&#10230; Agrupamiento

<br>

**6. Expectation-Maximization**

&#10230; Expectativa-Maximización

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Variables Latentes - Las variables latentes son variables ocultas / no observadas que dificultan los problemas de estimación y a menudo son denotadas como z. Estos son los ajustes más comunes en los que hay variables latentes:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Ajustes, Variable latente z, Comentarios]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Mezcla de k Gaussianos, Análisis factorial]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algoritmo - El algoritmo Expectativa-Maximización (EM) proporciona un método eficiente para estimar el parámetro θ a través de la estimación por máxima verosimilitud construyendo repetidamente un límite inferior en la probabilidad (E-step) y optimizando ese límite inferior (M-step) de la siguiente manera:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; Paso E: Evalúa la probabilidad posterior Qi(z(i)) de que cada punto de datos x(i) provenga de un determinado clúster z(i) de la siguiente manera:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step: Usa las probabilidades posteriores Qi(z(i)) como pesos específicos del clúster en los puntos de datos x(i) para re-estimar por separado cada modelo de clúster de la siguiente manera:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Inicialización Gaussiana, Etapa de Expectativa, Etapa de Maximización, Convergencia]

<br>

**14. k-means clustering**

&#10230; Agrupamiento k-means

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Denotamos c(i) al clúster de puntos de datos i, y μj al centro del clúster j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritmo - Después de haber iniciado aleatoriamente los centroides del clúster μ1,μ2,...,μk∈Rn, el algoritmo k-means repite el siguiente paso hasta la convergencia:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Inicialización de medias, Asignación de Clúster, Actualización de medias, Convergencia]

<br> 

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Función de Distorsión - Para ver si el algoritmo converge, observamos la función de distorsión definida de la siguiente manera:

<br>

**19. Hierarchical clustering**

&#10230; Agrupación Jerárquica

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algoritmo - Es un algoritmo de agrupamiento con un enfoque de aglomeramiento jerárquico que construye clústeres anidados de forma sucesiva.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Tipos - Hay diferentes tipos de algoritmos de agrupamiento jerárquico que tienen por objetivo optimizar diferentes funciones objetivo, que se resumen en la tabla a continuación:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Enlace de Ward, Enlace promedio, Enlace completo]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Minimizar dentro de la distancia del clúster, Minimizar la distancia promedio entre pares de clúster, Minimizar la distancia máxima entre pares de clúster]

<br>

**24. Clustering assessment metrics**

&#10230; Métricas de evaluación de agrupamiento

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; En un entorno de aprendizaje no supervisado, a menudo es difícil evaluar el rendimiento de un modelo ya que no contamos con las etiquetas verdaderas, como en el caso del aprendizaje supervisado.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Coeficiente de silueta - Sea a y b la distancia media entre una muestra y todos los demás puntos en la misma clase, y entre una muestra y todos los demás puntos en el siguiente grupo más cercano, el coeficiente de silueta s para una muestra individual se define de la siguiente manera:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Índice de Calinski-Harabaz - Sea k el número de conglomerados, Bk y Wk las matrices de dispersión entre y dentro de la agrupación, respectivamente definidas como:

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; El índice de Calinski-Harabaz s(k) indica qué tan bien un modelo de agrupamiento define sus grupos, de tal manera que cuanto mayor sea la puntuación, más denso y bien separados estarán los conglomerados. Se define de la siguiente manera:

<br>

**29. Dimension reduction**

&#10230; Reducción de dimensión

<br>

**30. Principal component analysis**

&#10230; Análisis de componentes principales

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; Es una técnica de reducción de dimensión que encuentra la varianza maximizando las direcciones sobre las cuales se proyectan los datos.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalor (Eigenvalue), Autovector (Eigenvector) ― Dada una matriz A∈Rn×n, se dice que λ es un autovalor de A si existe un vector z∈Rn∖{0}, llamado autovector,  de tal manera que tenemos:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema espectral - Sea A∈Rn×n. Si A es simétrica, entonces A es diagonalizable a través de una matriz ortogonal real U∈Rn×n. Al observar Λ=diag(λ1,...,λn), tenemos:

<br>

**34. diagonal**

&#10230; diagonal

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Observación: el autovector asociado con el autovalor más grande se denomina autovector principal de la matriz A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritmo - El procedimiento de Análisis de Componentes Principales (ACP) es una técnica de reducción de dimensión que proyecta los datos en k dimensiones maximizando la varianza de los datos de la siguiente manera:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Paso 1: Normalizar los datos para obtener una media de 0 y una desviación estándar de 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Paso 2: Calcular Σ=1mm∑i=1x(i)x(i)T∈Rn×n, que es simétrico con autovalores reales.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Paso 3: Calcular u1,...,uk∈Rn los k autovectores ortogonales principales de Σ, es decir, los autovectores ortogonales de los k mayores autovalores. 

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Paso 4: Proyectar los datos en spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Este procedimiento maximiza la varianza entre todos los espacios k-dimensionales.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Datos en el espacio de funciones, Buscar componentes principales, Datos en el espacio de componentes principales]

<br>

**43. Independent component analysis**

&#10230; Análisis de componentes independientes

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; Es una técnica destinada a encontrar las fuentes generadoras subyacentes.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Suposiciones - Suponemos que nuestros datos x han sido generados por el vector fuente n-dimensional s=(s1,...,sn), donde si son variables aleatorias independientes; a través de una matriz A de mezcla y no singular, de la siguiente manera:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; El objetivo es encontrar la matriz separadora W=A−1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Algoritmo ICA de Bell y Sejnowski - Este algoritmo encuentra la matriz separadora W siguiendo los siguientes pasos: 

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Escribir la probabilidad de x=As=W−1s como:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Escriba la probabilidad dado nuestros datos de entrenamiento {x(i),i∈[[1,m]]} y denotando g, la función sigmoide, como:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Por lo tanto, la regla de aprendizaje de ascenso de gradiente estocástica es tal que para cada ejemplo de entrenamiento x(i), actualizamos W de la siguiente manera:

<br>

**51. The Machine Learning cheatsheets are now available in Spanish.**

&#10230; Las hojas de referencia de Machine Learning ahora están disponibles en Español.

<br>

**52. Original authors**

&#10230; Autores Originales

<br>

**53. Translated by X, Y and Z**

&#10230; Traducido por X, Y y Z

<br>

**54. Reviewed by X, Y and Z**

&#10230; Revisado por X, Y y Z

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Introducción, Motivación, Desigualdad de Jensen]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Agrupamiento, Expectativa-Maximización, k-means, Agrupación jerárquica, Métricas]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Reducción de dimensión, ACP, ICA]
