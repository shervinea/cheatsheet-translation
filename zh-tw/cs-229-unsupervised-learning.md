1. **Unsupervised Learning cheatsheet**

&#10230;
非監督式學習參考手冊
<br>

2. **Introduction to Unsupervised Learning**

&#10230;
非監督式學習介紹
<br>

3. **Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230;
動機 - 非監督式學習的目的是要找出未標籤資料 {x(1),...,x(m)} 之間的隱藏模式
<br>

4. **Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230;
Jensen's 不等式 - 令 f 為一個凸函數、X 為一個隨機變數，我們可以得到底下這個不等式：
<br>

5. **Clustering**

&#10230;
分群
<br>

6. **Expectation-Maximization**

&#10230;
最大期望值
<br>

7. **Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230;
潛在變數 (Latent variables) - 潛在變數指的是隱藏/沒有觀察到的變數，這會讓問題的估計變得困難，我們通常使用 z 來代表它。底下是潛在變數的常見設定：
<br>

8. **[Setting, Latent variable z, Comments]**

&#10230;
[設定, 潛在變數 z, 評論]
<br>

9. **[Mixture of k Gaussians, Factor analysis]**

&#10230;
[k 元高斯模型, 因素分析]
<br>

10. **Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230;
演算法 - 最大期望演算法 (EM Algorithm) 透過重複建構一個概似函數的下界 (E-step) 和最佳化下界 (M-step) 來進行最大概似估計給出參數 θ 的高效率估計方法：
<br>

11. **E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230;
E-step: 評估後驗機率 Qi(z(i))，其中每個資料點 x(i) 來自於一個特定的群集 z(i)，如下：
<br>

12. **M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230;
M-step: 使用後驗機率 Qi(z(i)) 作為資料點 x(i) 在群集中特定的權重，用來分別重新估計每個群集，如下：
<br>

13. **[Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230;
[高斯分佈初始化, E-Step, M-Step, 收斂]
<br>

14. **k-means clustering**

&#10230;
k-means 分群法
<br>

15. **We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230;
我們使用 c(i) 表示資料 i 屬於某群，而 μj 則是群 j 的中心
<br>

16. **Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230;
演算法 - 在隨機初始化群集中心點 μ1,μ2,...,μk∈Rn 後，k-means 演算法重複以下步驟直到收斂：
<br>

17. **[Means initialization, Cluster assignment, Means update, Convergence]**

&#10230;
[中心點初始化, 指定群集, 更新中心點, 收斂]
<br>

18. **Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230;
畸變函數 - 為了確認演算法是否收斂，我們定義以下的畸變函數：
<br>

19. **Hierarchical clustering**

&#10230;
階層式分群法
<br>

20. **Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230;
演算法 - 階層式分群法是透過一種階層架構的方式，將資料建立為一種連續層狀結構的形式。
<br>

21. **Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230;
類型 - 底下是幾種不同類型的階層式分群法，差別在於要最佳化的目標函式的不同，請參考底下：
<br>

22. **[Ward linkage, Average linkage, Complete linkage]**

&#10230;
[Ward 鏈結距離, 平均鏈結距離, 完整鏈結距離]
<br>

23. **[Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230;
[最小化群內距離, 最小化各群彼此的平均距離, 最小化各群彼此的最大距離]
<br>

24. **Clustering assessment metrics**

&#10230;
分群衡量指標
<br>

25. **In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230;
在非監督式學習中，通常很難去評估一個模型的好壞，因為我們沒有擁有像在監督式學習任務中正確答案的標籤
<br>

26. **Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230;
輪廓係數 (Silhouette coefficient) - 我們指定 a 為一個樣本點和相同群集中其他資料點的平均距離、b 為一個樣本點和下一個最接近群集其他資料點的平均距離，輪廓係數 s 對於此一樣本點的定義為：
<br>

27. **Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230;
Calinski-Harabaz 指標 - 定義 k 是群集的數量，Bk 和 Wk 分別是群內和群集之間的離差矩陣 (dispersion matrices)：
<br>

28. **the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230;
Calinski-Harabaz 指標 s(k) 指出分群模型的好壞，此指標的值越高，代表分群模型的表現越好。定義如下：
<br>

29. **Dimension reduction**

&#10230;
維度縮減
<br>

30. **Principal component analysis**

&#10230;
主成份分析
<br>

31. **It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230;
這是一個維度縮減的技巧，在於找到投影資料的最大方差
<br>

32. **Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230;
特徵值、特徵向量 - 給定一個矩陣 A∈Rn×n，我們說 λ 是 A 的特徵值，當存在一個特徵向量 z∈Rn∖{0}，使得：
<br>

33. **Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230;
譜定理 - 令 A∈Rn×n，如果 A 是對稱的，則 A 可以可以透過正交矩陣 U∈Rn×n 對角化。當 Λ=diag(λ1,...,λn)，我們得到：
<br>

34. **diagonal**

&#10230;
對角線
<br>

35. **Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230;
注意：與特徵值所關聯的特徵向量就是 A 矩陣的主特徵向量
<br>

36. **Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230;
演算法 - 主成份分析 (PCA) 是一種維度縮減的技巧，它會透過尋找資料最大變異的方式，將資料投影在 k 維空間上：
<br>

37. **Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230;
第一步：正規化資料，讓資料平均為 0，變異數為 1
<br>

38. **Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230;
第二步：計算 Σ=1mm∑i=1x(i)x(i)T∈Rn×n，即對稱實際特徵值
<br>

39. **Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230;
第三步：計算 u1,...,uk∈Rn，k 個正交主特徵向量的總和 Σ，即是 k 個最大特徵值的正交特徵向量
<br>

40. **Step 4: Project the data on spanR(u1,...,uk).**

&#10230;
第四部：將資料投影到 spanR(u1,...,uk)
<br>

41. **This procedure maximizes the variance among all k-dimensional spaces.**

&#10230;
這個步驟會最大化所有 k 維空間的變異數
<br>

42. **[Data in feature space, Find principal components, Data in principal components space]**

&#10230;
[資料在特徵空間, 尋找主成分, 資料在主成分空間]
<br>

43. **Independent component analysis**

&#10230;
獨立成分分析
<br>

44. **It is a technique meant to find the underlying generating sources.**

&#10230;
這是用來尋找潛在生成來源的技巧
<br>

45. **Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230;
假設 - 我們假設資料 x 是從 n 維的來源向量 s=(s1,...,sn) 產生，si 為獨立變數，透過一個混合與非奇異矩陣 A 產生如下：
<br>

46. **The goal is to find the unmixing matrix W=A−1.**

&#10230;
目的在於找到一個 unmixing 矩陣 W=A−1
<br>

47. **Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230;
Bell 和 Sejnowski 獨立成份分析演算法 - 此演算法透過以下步驟來找到 unmixing 矩陣：
<br>

48. **Write the probability of x=As=W−1s as:**

&#10230;
紀錄 x=As=W−1s 的機率如下：
<br>

49. **Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230;
在給定訓練資料 {x(i),i∈[[1,m]]} 的情況下，其對數概似估計函數與定義 g  為 sigmoid 函數如下：
<br>

50. **Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230;
因此，梯度隨機下降學習規則對每個訓練樣本 x(i) 來說，我們透過以下方法來更新 W：
