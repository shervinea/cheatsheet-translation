1. **Supervised Learning cheatsheet**

&#10230; 監督式學習參考手冊

2. **Introduction to Supervised Learning**

&#10230; 監督式學習介紹

3. **Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; 給定一組資料點 {x(1),...,x(m)}，以及對應的一組輸出 {y(1),...,y(m)}，我們希望建立一個分類器，用來學習如何從 x 來預測 y

4. **Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; 預測的種類 - 根據預測的種類不同，我們將預測模型分為底下幾種：

5. **[Regression, Classifier, Outcome, Examples]**

&#10230; [迴歸, 分類器, 結果, 範例]

6. **[Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [連續, 類別, 線性迴歸, 邏輯迴歸, 支援向量機 (SVM) , 單純貝式分類器]

7. **Type of model ― The different models are summed up in the table below:**

&#10230; 模型種類 - 不同種類的模型歸納如下表：

8. **[Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [判別模型, 生成模型, 目標, 學到什麼, 示意圖, 範例]

9. **[Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [直接估計 P(y|x), 先估計 P(x|y)，然後推論出 P(y|x), 決策分界線, 資料的機率分佈, 迴歸, 支援向量機 (SVM), 高斯判別分析 (GDA), 單純貝氏 (Naive Bayes)]

10. **Notations and general concepts**

&#10230; 符號及一般概念

11. **Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; 假設 - 我們使用 hθ 來代表所選擇的模型，對於給定的輸入資料 x(i)，模型預測的輸出是 hθ(x(i))

12. **Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; 損失函數 - 損失函數是一個函數 L:(z,y)∈R×Y⟼L(z,y)∈R，
目的在於計算預測值 z 和實際值 y 之間的差距。底下是一些常見的損失函數：

13. **[Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [最小平方法, Logistic 損失函數, Hinge 損失函數, 交叉熵]

14. **[Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [線性迴歸, 邏輯迴歸, 支援向量機 (SVM), 神經網路]

15. **Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; 代價函數 - 代價函數 J 通常用來評估一個模型的表現，它可以透過損失函數 L 來定義：

16. **Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; 梯度下降 - 使用 α∈R 表示學習速率，我們透過學習速率和代價函數來使用梯度下降的方法找出網路參數更新的方法可以表示為：

17. **Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; 注意：隨機梯度下降法 (SGD) 使用每一個訓練資料來更新參數。而批次梯度下降法則是透過一個批次的訓練資料來更新參數。

18. **Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; 概似估計 - 在給定參數 θ 的條件下，一個模型 L(θ) 的概似估計的目的是透過最大概似估計法來找到最佳的參數。實務上，我們會使用對數概似估計函數 (log-likelihood) ℓ(θ)=log(L(θ))，會比較容易最佳化。如下：

19. **Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; 牛頓演算法 - 牛頓演算法是一個數值方法，目的在於找到一個 θ，讓 ℓ′(θ)=0。其更新的規則為：

20. **Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; 注意：多維度正規化的方法，或又被稱之為牛頓-拉弗森 (Newton-Raphson) 演算法，是透過以下的規則更新：

21. **Linear models**

&#10230; 線性模型

22. **Linear regression**

&#10230; 線性迴歸

23. **We assume here that y|x;θ∼N(μ,σ2)**

&#10230; 我們假設 y|x;θ∼N(μ,σ2)

24. **Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; 正規方程法 - 我們使用 X 代表矩陣，讓代價函數最小的 θ 值有一個封閉解，如下：

25. **LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; 最小均方演算法 (LMS) - 我們使用 α 表示學習速率，針對 m 個訓練資料，透過最小均方演算法的更新規則，或是叫做 Widrow-Hoff 學習法如下：

26. **Remark: the update rule is a particular case of the gradient ascent.**

&#10230; 注意：這個更新的規則是梯度上升的一種特例

27. **LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; 局部加權迴歸 ，又稱為 LWR，是線性洄歸的變形，通過w(i)(x) 對其成本函數中的每個訓練樣本進行加權，其中參數 τ∈R 定義為：

28. **Classification and logistic regression**

&#10230; 分類與邏輯迴歸

29. **Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Sigmoid 函數 - Sigmoid 函數 g，也可以稱為邏輯函數定義如下：

30. **Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; 邏輯迴歸 - 我們假設 y|x;θ∼Bernoulli(ϕ)，請參考以下：

31. **Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; 注意：對於這種情況的邏輯迴歸，並沒有一個封閉解

32. **Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Softmax 迴歸 - Softmax 迴歸又稱做多分類邏輯迴歸，目的是用在超過兩個以上的分類時的迴歸使用。按照慣例，我們設定 θK=0，讓每一個類別的 Bernoulli 參數 ϕi 等同於：

33. **Generalized Linear Models**

&#10230; 廣義線性模型

34. **Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; 指數族分佈 - 一個分佈如果可以透過自然參數 (或稱之為正準參數或連結函數) η、充分統計量 T(y) 和對數區分函數 (log-partition function) a(η) 來表示時，我們就稱這個分佈是屬於指數族分佈。該分佈可以表示如下：

35. **Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; 注意：我們經常讓 T(y)=y，同時，exp(−a(η)) 可以看成是一個正規化的參數，目的在於讓機率總和為一。

36. **Here are the most common exponential distributions summed up in the following table:**

&#10230; 底下是最常見的指數分佈：

37. **[Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [分佈, 白努利 (Bernoulli), 高斯 (Gaussian), 卜瓦松 (Poisson), 幾何 (Geometric)]

38. **Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; 廣義線性模型的假設 - 廣義線性模型 (GLM) 的目的在於，給定 x∈Rn+1，要預測隨機變數 y，同時它依賴底下三個假設：

39. **Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; 注意：最小平方法和邏輯迴歸是廣義線性模型的一種特例

40. **Support Vector Machines**

&#10230; 支援向量機

41. **The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; 支援向量機的目的在於找到一條決策邊界和資料樣本之間最大化最小距離的線

42. **Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; 最佳的邊界分類器 - 最佳的邊界分類器可以表示為：

43. **where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; 其中，(w,b)∈Rn×R 是底下最佳化問題的答案：

44. **such that**

&#10230; 使得

45. **support vectors**

&#10230; 支援向量

46. **Remark: the line is defined as wTx−b=0.**

&#10230; 注意：該條直線定義為 wTx−b=0

47. **Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Hinge 損失函數 - Hinge 損失函數用在支援向量機上，定義如下：

48. **Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; 核(函數) - 給定特徵轉換 ϕ，我們定義核(函數) K 為：

49. **In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; 實務上，K(x,z)=exp(−||x−z||22σ2) 定義的核(函數) K，一般稱作高斯核(函數)。這種核(函數)經常被使用

50. **[Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [非線性可分, 使用核(函數)進行映射, 原始空間中的決策邊界]

51. **Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; 注意：我們使用 "核(函數)技巧" 來計算代價函數時，不需要真正的知道映射函數 ϕ，這個函數非常複雜。相反的，我們只需要知道 K(x,z) 的值即可。

52. **Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangian - 我們將 Lagrangian L(w,b) 定義如下：

53. **Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; 注意：係數 βi 稱為 Lagrange 乘數

54. **Generative Learning**

&#10230; 生成學習

55. **A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; 生成模型嘗試透過預估 P(x|y) 來學習資料如何生成，而我們可以透過貝氏定理來預估 P(y|x)

56. **Gaussian Discriminant Analysis**

&#10230; 高斯判別分析

57. **Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; 設定 - 高斯判別分析針對 y、x|y=0 和 x|y=1 進行以下假設：

58. **Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; 估計 - 底下的表格總結了我們在最大概似估計時的估計值：

59. **Naive Bayes**

&#10230; 單純貝氏

60. **Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; 假設 - 單純貝氏模型會假設每個資料點的特徵都是獨立的。

61. **Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; 解決方法 - 最大化對數概似估計來給出以下解答，k∈{0,1},l∈[[1,L]]

62. **Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; 注意：單純貝氏廣泛應用在文字分類和垃圾信件偵測上

63. **Tree-based and ensemble methods**

&#10230; 基於樹狀結構的學習和整體學習

64. **These methods can be used for both regression and classification problems.**

&#10230; 這些方法可以應用在迴歸或分類問題上

65. **CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - 分類與迴歸樹 (CART)，通常稱之為決策數，可以被表示為二元樹。它的優點是具有可解釋性。

66. **Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; 隨機森林 - 這是一個基於樹狀結構的方法，它使用大量經由隨機挑選的特徵所建構的決策樹。與單純的決策樹不同，它通常具有高度不可解釋性，但它的效能通常很好，所以是一個相當流行的演算法。

67. **Remark: random forests are a type of ensemble methods.**

&#10230; 注意：隨機森林是一種整體學習方法

68. **Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; 增強學習 (Boosting) - 增強學習方法的概念是結合數個弱學習模型來變成強學習模型。主要的分類如下：

69. **[Adaptive boosting, Gradient boosting]**

&#10230; [自適應增強, 梯度增強]

70. **High weights are put on errors to improve at the next boosting step**

&#10230; 在下一輪的提升步驟中，錯誤的部分會被賦予較高的權重

71. **Weak learners trained on remaining errors**

&#10230; 弱學習器會負責訓練剩下的錯誤

72. **Other non-parametric approaches**

&#10230; 其他非參數方法

73. **k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-最近鄰 - k-最近鄰演算法，又稱之為 k-NN，是一個非參數的方法，其中資料點的決定是透過訓練集中最近的 k 個鄰居而決定。它可以用在分類和迴歸問題上。

74. **Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; 注意：參數 k 的值越大，偏差越大。k 的值越小，變異越大。

75. **Learning Theory**

&#10230; 學習理論

76. **Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; 聯集上界 - 令 A1,...,Ak 為 k 個事件，我們有：

77. **Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; 霍夫丁不等式 - 令 Z1,..,Zm 為 m 個從參數 ϕ 的白努利分佈中抽出的獨立同分佈 (iid) 的變數。令 ˆϕ 為其樣本平均、固定 γ>0，我們可以得到：

78. **Remark: this inequality is also known as the Chernoff bound.**

&#10230; 注意：這個不等式也被稱之為 Chernoff 界線

79. **Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; 訓練誤差 - 對於一個分類器 h，我們定義訓練誤差為 ˆϵ(h)，也可以稱為經驗風險或經驗誤差。定義如下：

80. **Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; 可能近似正確 (PAC) - PAC 是一個框架，有許多學習理論都證明其有效性。它包含以下假設：

81: **the training and testing sets follow the same distribution**

&#10230; 訓練和測試資料集具有相同的分佈

82. **the training examples are drawn independently**

&#10230; 訓練資料集之間彼此獨立

83. **Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; 打散 (Shattering) - 給定一個集合 S={x(1),...,x(d)} 以及一組分類器的集合 H，如果對於任何一組標籤 {y(1),...,y(d)}，H 都能打散 S，定義如下：

84. **Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; 上限定理 - 令 H 是一個有限假設類別，使 |H|=k 且令 δ 和樣本大小 m 固定，結著，在機率至少為 1−δ 的情況下，我們得到：

85. **VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC 維度 - 一個有限假設類別的 Vapnik-Chervonenkis (VC) 維度 VC(H) 指的是 H 最多能夠打散的數量

86. **Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; 注意：H={2 維的線性分類器} 的 VC 維度為 3

87. **Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; 理論 (Vapnik) - 令 H 已給定，VC(H)=d 且 m 是訓練資料級的數量，在機率至少為 1−δ 的情況下，我們得到：

88. **Known as Adaboost**

&#10230; 被稱為 Adaboost
