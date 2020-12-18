**Unsupervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning)

<br>

**1. Unsupervised Learning cheatsheet**

&#10230; Unsupervised Learning cheatsheet

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Introduction to Unsupervised Learning

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:

<br>

**5. Clustering**

&#10230; Clustering

<br>

**6. Expectation-Maximization**

&#10230; Expectation-Maximization

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Setting, Latent variable z, Comments]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Mixture of k Gaussians, Factor analysis]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Gaussians initialization, Expectation step, Maximization step, Convergence]

<br>

**14. k-means clustering**

&#10230; k-means clustering

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; We note c(i) the cluster of data point i and μj the center of cluster j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Means initialization, Cluster assignment, Means update, Convergence]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:

<br>

**19. Hierarchical clustering**

&#10230; Hierarchical clustering

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ward linkage, Average linkage, Complete linkage]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]

<br>

**24. Clustering assessment metrics**

&#10230; Clustering assessment metrics

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:

<br>

**29. Dimension reduction**

&#10230; Dimension reduction

<br>

**30. Principal component analysis**

&#10230; Principal component analysis

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:

<br>

**34. diagonal**

&#10230; diagonal

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230;

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Step 4: Project the data on spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; This procedure maximizes the variance among all k-dimensional spaces.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Data in feature space, Find principal components, Data in principal components space]

<br>

**43. Independent component analysis**

&#10230; Independent component analysis

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; It is a technique meant to find the underlying generating sources.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; The goal is to find the unmixing matrix W=A−1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Write the probability of x=As=W−1s as:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; The Machine Learning cheatsheets are now available in [target language].

<br>

**52. Original authors**

&#10230; Original authors

<br>

**53. Translated by X, Y and Z**

&#10230; Translated by X, Y and Z

<br>

**54. Reviewed by X, Y and Z**

&#10230; Reviewed by X, Y and Z

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Introduction, Motivation, Jensen's inequality]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Dimension reduction, PCA, ICA]
