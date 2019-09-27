**1. Unsupervised Learning cheatsheet**

&#10230;教師なし学習チートシート

<br>

**2. Introduction to Unsupervised Learning**

&#10230;教師なし学習の概要

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230;モチベーション - 教師なし学習の目的はラベルのないデータ{x(1),...,x(m)}に隠されたパターンを探すことです。

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230;イェンセンの不等式 - fを凸関数、Xを確率変数とすると、次の不等式が成り立ちます:

<br>

**5. Clustering**

&#10230;クラスタリング

<br>

**6. Expectation-Maximization**

&#10230;期待値最大化

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230;潜在変数 - 潜在変数は推定問題を困難にする隠れた/観測されていない変数であり、多くの場合zで示されます。潜在変数がある最も一般的な設定は次のとおりです:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230;[設定, 潜在変数z, コメント]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230;[k個のガウス分布の混, 因子分析]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230;アルゴリズム - EMアルゴリズムは次のように尤度の下限の構築(E-ステップ)と、その下限の最適化(M-ステップ)を繰り返し行うことによる最尤推定によりパラメーターθを推定する効率的な方法を提供します:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230;E-ステップ: 各データポイントx(i)が特定クラスターz(i)に由来する事後確率Qi(z(i))を次のように評価します:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230;M-ステップ: 事後確率Qi(z(i))をデータポイントx(i)のクラスター固有の重みとして使い、次のように各クラスターモデルを個別に再推定します:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230;[ガウス分布初期化, 期待値ステップ, 最大化ステップ, 収束]

<br>

**14. k-means clustering**

&#10230;k平均法

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230;データポイントiのクラスタをc(i)、クラスタjの中心をμjと表記します。

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230;クラスターの重心μ1,μ2,...,μk∈Rnをランダムに初期化後、k-meansアルゴリズムが収束するまで次のようなステップを繰り返します:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [平均の初期化, クラスター割り当て,平均の更新, 収束]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230;ひずみ関数 - アルゴリズムが収束するかどうかを確認するため、次のように定義されたひずみ関数を参照します:

<br>

**19. Hierarchical clustering**

&#10230; 階層的クラスタリング

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230;アルゴリズム - これは入れ子になったクラスタを逐次的に構築する凝集階層アプローチによるクラスタリングアルゴリズムです。

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; 種類 ― 様々な目的関数を最適化するための様々な種類の階層クラスタリングアルゴリズムが以下の表にまとめられています。

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ward linkage, Average linkage, Complete linkage]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [クラスター内の距離最小化、クラスターペア間の平均距離の最小化、クラスターペア間の最大距離の最小化]

<br>

**24. Clustering assessment metrics**

&#10230; クラスタリング評価指標

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; 教師なし学習では、教師あり学習の場合のような正解ラベルがないため、モデルの性能を評価することが難しい場合が多いです。

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; シルエット係数 ― サンプルと同じクラスタ内のその他全ての点との平均距離をa、最も近いクラスタ内の全ての点との平均距離をbと表記すると、サンプルのシルエット係数sは次のように定義されます:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Calinski-Harabazインデックス ― クラスタの数をkと表記すると、クラスタ間およびクラスタ内の分散行列であるBkおよびWkはそれぞれ以下のように定義されます。

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; Calinski-Harabazインデックスs(k)はクラスタリングモデルが各クラスタをどの程度適切に定義しているかを示します。スコアが高いほど、各クラスタはより密で、十分に分離されています。 それは次のように定義されます:

<br>

**29. Dimension reduction**

&#10230; 次元削減

<br>

**30. Principal component analysis**

&#10230; 主成分分析

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; これはデータを投影する方向で、分散を最大にする方向を見つける次元削減手法です。

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; 固有値、固有ベクトル - 行列 A∈Rn×nが与えられたとき、次の式で固有ベクトルと呼ばれるベクトルz∈Rn∖{0}が存在した場合に、λはAの固有値と呼ばれる。

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; スペクトル定理 - A∈Rn×nとする。Aが対称のとき、Aは実直交行列U∈Rn×nを用いて対角化可能である。Λ=diag(λ1,...,λn)と表記することで、次の式を得る。

<br>

**34. diagonal**

&#10230; 対角

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; 注釈: 最大固有値に対応する固有ベクトルは行列Aの第1固有ベクトルと呼ばれる。

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; アルゴリズム ― 主成分分析 (PCA)の過程は、次のようにデータの分散を最大化することによりデータをk次元に射影する次元削減の技術である。

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; ステップ1：平均が0で標準偏差が1となるようにデータを正規化します。

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; ステップ2：実固有値に関して対称であるΣ=1mm∑i=1x(i)x(i)T∈Rn×nを計算します。

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; ステップ3：k個のΣの対角主値固有ベクトルu1,...,uk∈Rn、すなわちk個の最大の固有値の対角固有ベクトルを計算します。

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; ステップ4：データをspanR(u1,...,uk)に射影します。

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; この過程は全てのk次元空間の間の分散を最大化します。

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [特徴空間内のデータ, 主成分を見つける, 主成分空間内のデータ]

<br>

**43. Independent component analysis**

&#10230; 独立成分分析

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; 隠れた生成源を見つけることを意図した技術です。

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; 仮定 ― 混合かつ非特異行列Aを通じて、データxはn次元の元となるベクトルs=(s1,...,sn)から次のように生成されると仮定します。ただしsiは独立でランダムな変数です：

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; 非混合行列W=A−1を見つけることが目的です。

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; ベルとシノスキーのICAアルゴリズム ― このアルゴリズムは非混合行列Wを次のステップによって見つけます：

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; x=As=W−1sの確率を次のように表します：

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; 学習データを{x(i),i∈[[1,m]]}、シグモイド関数をgとし、対数尤度を次のように表します：

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; そのため、確率的勾配上昇法の学習規則は、学習サンプルx(i)に対して次のようにwを更新するものです：

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; 機械学習チートシートは日本語で読めます。

<br>

**52. Original authors**

&#10230; 原著者

<br>

**53. Translated by X, Y and Z**

&#10230; X, Y, Zによる翻訳

<br>

**54. Reviewed by X, Y and Z**

&#10230; X, Y, Zによるレビュー

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [導入, 動機, イェンセンの不等式]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230;[クラスタリング, EM, k-means, 階層クラスタリング, 指標]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [次元削減, PCA, ICA]
