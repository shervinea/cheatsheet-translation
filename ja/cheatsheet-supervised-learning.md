**1. Supervised Learning cheatsheet**

&#10230;教師あり学習チートシート

<br>

**2. Introduction to Supervised Learning**

&#10230;教師あり学習入門

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230;入力が{x(1),...,x(m)}, 出力が{y(1),...,y(m)}であるとき, xからyを予測する分類器を構築したい。

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230;予測の種類 ― 様々な種類の予測モデルは下表に集約される：

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230;回帰, 分類, 出力, 例

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230;連続値, クラス, 線形回帰, ロジスティック回帰, SVM, ナイーブベイズ

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230;モデルの種類 ― 様々な種類のモデルは下表に集約される：

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230;判別モデル, 生成モデル, 目的, 学習対象, イメージ図, 例

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230;P(y|x)を直接推定する, P(y|x)を推測するためにP(x|y)を推定する, 決定境界, データの確率分布, 回帰, SVM, GDA, ナイーブベイズ

<br>

**10. Notations and general concepts**

&#10230;記法と概念

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230;仮説 ― 仮説はhθと表され、選択されたモデルのことである。与えられた入力x(i)に対して、モデルの予測結果はhθ(x(i))である。

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230;損失関数 ― 損失関数とは(z,y)∈R×Y⟼L(z,y)∈Rを満たす関数Lで、予測値zとそれに対応する正解データ値yを入力とし、その誤差を出力するものである。一般的な損失関数は次表に集約される：

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230;最小2乗誤差, ロジスティック損失, ヒンジ損失, クロスエントロピー

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230;線形回帰, ロジスティック回帰, SVM, ニューラルネットワーク

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230;コスト関数 ― コスト関数Jは一般的にモデルの性能を評価するために用いられ、損失関数をLとして次のように定義される：

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230;勾配降下法 ― 学習率をα∈Rとし、勾配降下法におけるパラメータの更新は学習率とコスト関数Jを用いて次のように行われる：

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230;備考：確率的勾配降下法(SGD)は学習標本全体を用いてパラメータを更新し、バッチ勾配降下法は学習標本の各バッチ毎に更新する。

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230;尤度 ― パラメータをθとすると、あるモデルの尤度L(θ)を最大にすることにより最適なパラメータを求められる。実際には、最適化しやすい対数尤度ℓ(θ)=log(L(θ))を用いる。すなわち：

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230;ニュートン法 ― ニュートン法とはℓ′(θ)=0となるθを求める数値計算アルゴリズムである。そのパラメータ更新は次のように行われる：

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230;備考：高次元正則化またはニュートン-ラフソン法ではパラメータ更新は次のように行われる：

<br>

**21. Linear models**

&#10230;線形モデル

<br>

**22. Linear regression**

&#10230;線形回帰

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230;ここでy|x;θ∼N(μ,σ2)であるとする。

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230;正規方程式 ― Xを行列とすると、コスト関数を最小化するθの値は次のような閉形式の解である：

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230;最小2乗法 ― 学習率をαとすると、m個のデータ点からなる学習データに対する最小2乗法（LMSアルゴリズム）によるパラメータ更新は次のように行われ、これはウィドロウ-ホフの学習規則としても知られている：

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230;備考：この更新は勾配上昇法の特殊な例である。

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230;局所重み付き回帰 ― 局所重み付き回帰は、LWRとも呼ばれ、線形回帰の派生形である。パラメータをτ∈Rとして次のように定義されるw(i)(x)により、個々の学習標本をそのコスト関数において重み付けする：

<br>

**28. Classification and logistic regression**

&#10230;分類とロジスティック回帰

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230;シグモイド関数 ― シグモイド関数gは、ロジスティック関数とも呼ばれ、次のように定義される：

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230;ロジスティック回帰 ― ここでy|x;θ∼Bernoulli(ϕ)であるとすると、次の形式を得る：

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230;備考：ロジスティック回帰については閉形式の解は存在しない。

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230;ソフトマックス回帰 ― ソフトマックス回帰は、多クラス分類ロジスティック回帰とも呼ばれ、3個以上の結果クラスがある場合にロジスティック回帰を一般化するためのものである。慣習的に、θK=0とすると、各クラスiのベルヌーイ分布のパラメータϕiは次と等しくなる：

<br>

**33. Generalized Linear Models**

&#10230;一般化線形モデル

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230;指数分布族 ― 正準パラメータまたはリンク関数とも呼ばれる自然パラメータη、十分統計量T(y)及び対数分配関数a(η)を用いて、次のように表すことのできる一群の分布は指数分布族と呼ばれる：

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230;備考：T(y)=yとすることが多い。また、exp(−a(η))は確率の合計が１になることを担保する正規化定数だと見なせる。

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230;最も一般的な指数分布族は下表に集約される：

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230;分布, ベルヌーイ, ガウス, ポワソン, 幾何

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function of x∈Rn+1 and rely on the following 3 assumptions:**

&#10230;GLMの仮定 ― 一般化線形モデル(GLM)はランダムな変数yをx∈Rn+1の関数として予測することを目的とし、次の3つの仮定に依拠する：

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230;備考：最小2乗回帰とロジスティック回帰は一般化線形モデルの特殊な例である。

<br>

**40. Support Vector Machines**

&#10230;サポートベクターマシン

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230;サポートベクターマシンの目的は、データ点からの最短距離が最大となる境界線を求めることである。

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230;最適マージン分類器 ― 最適マージン分類器hは次のようなものである：

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230;ここで、(w,b)∈Rn×Rは次の最適化問題の解である：

<br>

**44. such that**

&#10230;ただし

<br>

**45. support vectors**

&#10230;サポートベクター

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230;備考：直線はwTx−b=0と定義する。

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230;ヒンジ損失 ― ヒンジ損失はSVMの設定に用いられ、次のように定義される：

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230;カーネル ― 特徴写像をϕとすると、カーネルKは次のように定義される：

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230;実際には、K(x,z)=exp(−||x−z||22σ2)と定義され、ガウシアンカーネルと呼ばれるカーネルKがよく使われる。

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230;非線形分離問題, カーネル写像の適用, 元の空間における決定境界

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230;備考：カーネルを用いてコスト関数を計算する「カーネルトリック」を用いる。なぜなら、明示的な写像ϕを実際には知る必要はないし、それはしばしば非常に複雑になってしまうからである。代わりに、K(x,z)の値のみが必要である。

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230;ラグランジアン ― ラグランジアンL(w,b)を次のように定義する：

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230;備考：係数βiはラグランジュ乗数と呼ばれる。

<br>

**54. Generative Learning**

&#10230;生成学習

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230;生成モデルは、P(x|y)を推定することによりデータがどのように生成されるのかを学習しようとする。それはその後ベイズの定理を用いてP(y|x)を推定することに使える。

<br>

**56. Gaussian Discriminant Analysis**

&#10230;ガウシアン判別分析

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230;前提 ― ガウシアン判別分析はyとx|y=0とx|y=1は次のようであることを前提とする：

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230;推定 ― 尤度を最大にすると得られる推定量は下表に集約される：

<br>

**59. Naive Bayes**

&#10230;ナイーブベイズ

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230;仮定 ― ナイーブベイズモデルは、個々のデータ点の特徴量が全て独立であると仮定する：

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230;解 ― 対数尤度を最大にすると次の解を得る。ただし、k∈{0,1},l∈[[1,L]]とする。

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230;備考：ナイーブベイズはテキスト分類やスパム検知に幅広く使われている。

<br>

**63. Tree-based and ensemble methods**

&#10230;ツリーとアンサンブル学習

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230;これらの方法は回帰と分類問題の両方に使える。

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230;CART ― 分類・回帰ツリー (CART)は、一般には決定木として知られ、二分木として表される。非常に解釈しやすいという利点がある。

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230;ランダムフォレスト ― これはツリーをベースにしたもので、ランダムに選択された特徴量の集合から構築された多数の決定木を用いる。単純な決定木と異なり、非常に解釈しにくいが、一般的に良い性能が出るのでよく使われるアルゴリズムである。

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230;備考：ランダムフォレストはアンサンブル学習の1種である。

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230;ブースティング ― ブースティングの考え方は、複数の弱い学習器を束ねることで1つのより強い学習器を作るというものである。主なものは次の表に集約される：

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230;[適応的ブースティング, 勾配ブースティング]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230;次のブースティングステップにて改善すべき誤分類に大きい重みが課される。

<br>

**71. Weak learners trained on remaining errors**

&#10230;残っている誤分類を弱い学習器が学習する。

<br>

**72. Other non-parametric approaches**

&#10230;他のノン・パラメトリックな手法

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230;k近傍法 ― k近傍法は、一般的にk-NNとして知られ、あるデータ点の応答はそのk個の最近傍点の性質によって決まるノン・パラメトリックな手法である。分類と回帰の両方に用いることができる。

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230;備考：パラメータkが大きくなるほど、バイアスが大きくなる。パラメータkが小さくなるほど、分散が大きくなる。

<br>

**75. Learning Theory**

&#10230;学習理論

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230;和集合上界 ― A1,...,Akというk個の事象があるとき、次が成り立つ：

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230;ヘフディング不等式 ― パラメータϕのベルヌーイ分布から得られるm個の独立同分布変数をZ1,..,Zmとする。その標本平均をˆϕとし、γは正の定数であるとすると、次が成り立つ：

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230;備考：この不等式はチェルノフ上界としても知られる。

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230;学習誤差 ― ある分類器hに対して、学習誤差、あるいは経験損失か経験誤差としても知られる、ˆϵ(h)を次のように定義する：

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230;確率的に近似的に正しい (PAC) ― PACとは、その下で学習理論に関する様々な業績が証明されてきたフレームワークであり、次の前提がある：

<br>

**81: the training and testing sets follow the same distribution **

&#10230;学習データと検証データは同じ分布に従う。

<br>

**82. the training examples are drawn independently**

&#10230;学習標本は独立に取得される。

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230;細分化 ― 集合S={x(1),...,x(d)}と分類器の集合Hがあるとき、もし任意のラベル{y(1),...,y(d)}の集合に対して次が成り立つとき、HはSを細分化する：

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230;上界定理 ― Hを|H|=kで有限の仮説集合とし、δとサンプルサイズmは定数とする。そのとき、少なくとも1-δ の確率で次が成り立つ：

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230;VC次元 ― ある仮説集合Hのヴァプニク・チェルヴォーネンキス次元 (VC)は、VC(H)と表記され、それはHによって細分化される最大の集合のサイズである。

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230;備考：2次元の線形分類器の集合であるHのVC次元は3である。

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230;定理（ヴァプニク） ― あるHについてVC(H)=dであり、mを学習標本の数とする。少なくとも1−δの確率で次が成り立つ：

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230;[導入, 予測の種類, モデルの種類]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230;[記法と全般的な概念, 損失関数, 勾配降下, 尤度]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230;

<br>[線形モデル, 線形回帰, ロジスティック回帰, 一般化線形モデル]

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230;

<br>[サポートベクターマシン, 最適マージン分類器, ヒンジ損失, カーネル]

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230;

<br>[生成学習, ガウシアン判別分析, ナイーブベイズ]

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230;[ツリーとアンサンブル学習, CART, ランダムフォレスト, ブースティング]

<br>

**94. [Other methods, k-NN]**

&#10230;[他の手法, k近傍法]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230;[学習理論, ヘフディング不等式, PAC, VC次元]
