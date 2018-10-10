1. **Unsupervised Learning cheatsheet**

&#10230; 无监督学习简明手册

<br>

2. **Introduction to Unsupervised Learning**

&#10230; 无监督学习导引

<br>

3. **Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; 动机 - 无监督学习的目标是找到在未标记数据 {x(1),...,x(m)} 中的隐含模式。

<br>

4. **Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Jensen 不等式 - 令 f 为一个凸函数而 X 为一个随机变量。我们有下列不等式：

<br>

5. **Clustering**

&#10230; 聚类

<br>

6. **Expectation-Maximization**

&#10230; E-M 算法

<br>

7. **Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; 隐变量 - 隐变量是隐含/不可观测的变量，使得估计问题变得困难，通常被表示成 z。这里是包含隐变量的常见设定：

<br>

8. **[Setting, Latent variable z, Comments]**

&#10230; [设定，隐变量 z，评论]

<br>

9. **[Mixture of k Gaussians, Factor analysis]**

&#10230; [k 元混合高斯分布，因子分析]

<br>

10. **Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; 算法 - E-M 算法给出了通过重复构建似然函数的下界（E-步）和最优化下界（M-步）进行极大似然估计给出参数 θ 的高效估计方法：

<br>

11. **E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-步：计算后验概率 Qi(z(i))，其中每个数据点 x(i) 来自特定的簇 z(i) ，过程如下：

<br>

12. **M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-步：使用后验概率 Qi(z(i)) 作为簇在数据点 x(i) 上的特定权重来分别重新估计每个簇模型，过程如下：

<br>

13. **[Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [高斯初始化，E-步，M-步，收敛]

<br>

14. **k-means clustering**

&#10230; k-均值聚类

<br>

15. **We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; 我们记 c(i) 为数据点 i 的簇，μj 是簇 j 的中心。

<br>

16. **Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; 算法 - 在随机初始化簇中心 μ1,μ2,...,μk∈Rn 后，k-均值算法重复下列步骤直至收敛：

<br>

17. **[Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [均值初始化，簇赋值，均值更新，收敛]

<br>

18. **Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; 失真函数 - 为了看到算法是否收敛，我们看看如下定义的失真函数：

<br>

19. **Hierarchical clustering**

&#10230; 层次化聚类

<br>

20. **Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; 算法 - 结合聚合层次化观点的聚类算法，按照逐次构建嵌套簇的方式进行。

<br>

21. **Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; 类型 - 存在不同的层次化聚类算法，解决不同的目标函数优化问题，在下表中总结列出：

<br>

22. **[Ward linkage, Average linkage, Complete linkage]**

&#10230; [内链，均链，全链]

<br>

23. **[Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [最小化簇内距离，最小化簇对平均距离，最小化簇对的最大距离]

<br>

24. **Clustering assessment metrics**

&#10230; 聚类评测度量

<br>

25. **In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; 在一个无监督学习设定中，通常难以评测一个模型的性能，因为我们没有像监督学习设定中那样的原始真实的类标。

<br>

26. **Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Silhouette 系数 - 通过记 a 和 b 为一个样本和在同一簇中的其他所有点之间的平均距离和一个样本和在下一个最近簇中的所有其他点的平均距离，针对一个样本的 Silhouette 系数 s 定义如下：

<br>

27. **Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Calinski-Harabaz 指标 - 通过记 k 为簇的数目，Bk 和 Wk 分别为簇间和簇内弥散矩阵，定义为：

<br>

28. **the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; Calinski-Harabaz 指标 s(k) 表示一个聚类模型定义簇的好坏，分数越高，簇就越稠密和良好分隔。其定义如下：

<br>

29. **Dimension reduction**

&#10230; 降维

<br>

30. **Principal component analysis**

&#10230; 主成分分析

<br>

31. **It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; 这是一种维度降低的技巧，找到投影数据到能够最大化方差的方向。

<br>

32. **Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; 特征值，特征向量 - 给定矩阵 A∈Rn×n，λ 被称为 A 的一个特征值当存在一个称为特征向量的向量 z∈Rn∖{0}，使得：

<br>

33. **Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; 谱定理 - 令 A∈Rn×n。如果 A 是对称阵，那么 A 可以被一个实正交矩阵 U∈Rn×n 对角化。通过记 Λ=diag(λ1,...,λn) 我们有：

<br>

34. **diagonal** 

&#10230; 为对角阵

<br>

35. **Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; 注：关联于最大的特征值的特征向量被称为矩阵 A 的主特征向量。

<br>

36. **Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; 算法 - 主成分分析（PCA）过程就是一个降维技巧，通过最大化数据的方差而将数据投影到 k 维上：

<br>

37. **Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; 步骤 1：规范化数据使其均值为 0 方差为 1。

<br>

38. **Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; 步骤 2：计算 Σ=1mm∑i=1x(i)x(i)T∈Rn×n，其为有实特征值的对称阵

<br>

39. **Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; 步骤 3：计算 Σ 的 k 个正交的主特征向量 u1,...,uk∈Rn，即对应 k 个最大特征值的正交特征向量。

<br>

40. **Step 4: Project the data on spanR(u1,...,uk).**

&#10230; 步骤 4：投影数据到 spanR(u1,...,uk) 上。

<br>

41. **This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; 这个过程最大化所有 k 维空间的方差

<br>

42. **[Data in feature space, Find principal components, Data in principal components space]**

&#10230; [在特征空间中的数据，找到主成分，在主成分空间中的数据]

<br>

43. **Independent component analysis**

&#10230; 独立成分分析

<br>

44. **It is a technique meant to find the underlying generating sources.**

&#10230; 这是旨在找到背后生成源的技术。

<br>

45. **Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; 假设 - 我们假设数据 x 已经由 n-维源向量 s=(s1,...,sn) 生成出来，其中 si 是独立的随机变量，通过一个混合和非奇异矩阵 A 如下方式产生：

<br>

46. **The goal is to find the unmixing matrix W=A−1.**

&#10230; 目标是要找到去混合矩阵 W=A−1。

<br>

47. **Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Bell-Sejnowski ICA 算法 - 该算法找出去混合矩阵 W ，通过下列步骤：

<br>

48. **Write the probability of x=As=W−1s as:**

&#10230; 记概率 x=As=W−1s 如下：

<br>

49. **Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; 记给定训练数据 {x(i),i∈[[1,m]]} 对数似然函数其中 g 为 sigmoid 函数如下：

<br>

50. **Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; 因此，随机梯度下降学习规则是，对每个训练样本 x(i)，我们如下更新 W：

<br>

51. **The Machine Learning cheatsheets are now available in Mandarin.**

&#10230; 机器学习简明指南中文版至此完成。

<br>

52. **Original authors**

&#10230; 原作者

<br>

53. **Translated by X, Y and Z**

&#10230; 由 朱小虎 译成

<br>

54. **Reviewed by X, Y and Z**

&#10230; 由 X, Y and Z 审阅

<br>

55. **[Introduction, Motivation, Jensen's inequality]**

&#10230; [导引，动机，Jensen 不等式]

<br>

56. **[Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [聚类，E-M 算法，k-均值，层次化聚类，度量]

<br>

57. **[Dimension reduction, PCA, ICA]**

&#10230; [降维，主成分分析，独立成分分析]
