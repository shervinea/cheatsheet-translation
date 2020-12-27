**1. Supervised Learning cheatsheet**

&#10230; 지도 학습 치트시트

<br>

**2. Introduction to Supervised Learning**

&#10230; 지도 학습 소개

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; 일련의 데이터 포인트 {x(1),...,x(m)}와 연관된 출력 {y(1),...,y(m)}이 주어졌을 때 분류기가 x로부터 y를 예측하는 방법을 학습한다.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; 예측의 종류 - 예측 모델의 종류가 아래 표에 정리되어 있다:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [회귀(Regression), 분류(Classification), 출력, 샘플]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [연속, 클래스(Class), 선형 회귀(Linear regression), 로지스틱 회귀(Logistic regression), SVM, 나이브 베이즈(Naive Bayes)]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; 모델의 종류 - 모델의 종류가 아래 표에 정리되어 있습니다:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [판별 모델, 생성 모델, 목표, 학습하는 것, 그림, 예]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [직접 P(y|x)를 추정, P(x|y)를 추정하여 P(y|x)를 추론하기 , 결정 경계, 데이터의 확률 분포, 회귀, SVM, GDA, 나이브 베이즈]

<br>

**10. Notations and general concepts**

&#10230; 표기법과 일반 개념

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; 가설 - 가설은 hθ로 표시하며 선택한 하나의 모델입니다. 입력 데이터 x(i)에 대한 모델의 예측 출력은 hθ(x(i))가 된다. 

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; 손실 함수 - 손실 함수 L은 L:(z,y)∈R×Y⟼L(z,y)∈R 으로 표현되며, 이 함수는 실제 데이터 y에 상응하는 예측값 z를 입력값으로 받고, 그 두 값이 얼마나 다른지를 출력한다. 일반적인 손실 함수는 아래 테이블에 정리되어 있다:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [최소 제곱 오차, 로지스틱 손실, 힌지 손실, 크로스-엔트로피]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [선형 회귀, 로지스틱 회귀, 서포트 벡터 머신, 인공신경망]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; 비용 함수 - 비용 함수 J는 일반적으로 모델의 퍼포먼스를 평가하기 위해 사용되며, 다음의 손실함수 L과 함께 정의된다:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; 경사 하강법 - 학습률(learning rate)을 α∈R 라고 표기한다면, 경사하강법의 업데이트 규칙은 다음과 같이 학습률과 비용 함수와 함께 정의된다:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; 비고: 확률적 경사 하강법 (SGD)은 각각의 훈련 데이터로 파라미터를 업데이트하며, 배치 경사 하강법은 훈련 데이터의 한 묶음으로 파라미터를 업데이트한다.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; 가능도 - 파라미터 θ가 주어졌을 때의 모델 L(θ)의 가능도 혹은 우도는 우도를 최대화 시키면서 최적의 파라미터 θ를 찾기 위해 사용된다. 보통은 관례적으로 더 최적화하기 쉬운 로그-가능도 ℓ(θ)=log(L(θ))를 사용한다. 이때: 

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; 뉴턴 알고리즘 - 뉴턴 알고리즘은 ℓ′(θ)=0 를 만족하는 θ를 찾는 수치적 방법이다. 업데이트 방법은 다음과 같다:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; 비고: 뉴턴-랩슨법이라고도 불리는 다차원 일반화는 다음과 같은 업데이트 방식을 갖는다:

<br>

**21. Linear models**

&#10230; 선형 모델 

<br>

**22. Linear regression**

&#10230; 선형 회귀 

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; 이때 우리는 y|x;θ∼N(μ,σ2) 라고 가정한다.

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; 정규방정식 (Normal equations) - 행렬 X에 대하여, 비용 함수를 최소화 시키는 값 θ는 다음과 같은 닫힌 해를 갖는다:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; 최소 평균 제곱법 - α를 학습률 이라고 하면, m개의 학습 데이터에 대한 Widrow-Hoff 학습법이라고도 알려진 최소 평균 제곱법(Least Mean Squares)의 업데이트 방식은 다음과 같다:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; 비고: 이 업데이트 방식은 경사 상승법의 특수한 케이스입니다.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR 이라고 불리는 국소가중회귀 (Locally Weighted Regression) 는 비용 함수 안에서 각각의 훈련 데이터를 w(i)(x)와 곱하 선형회귀의 변형으로서,는 파라미터 τ∈R 와 함께 다음과 같이 정의된다:

<br>

**28. Classification and logistic regression**

&#10230; 분류와 로지스틱 회귀

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; 시그모이드 함수 - 로지스틱 함수 라고도 불리는 시그모이드 함수 g는, 다음과 같이 정의된다:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; 로지스틱 회귀 - y|x;θ∼Bernoulli(ϕ) 라고 가정할 때 우리는 다음과 같은 식을 얻는다: 

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; 비고: 로지스틱 회귀는 닫힌 형식의 해는 없다.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; 소프트맥스 회귀 - 다중 로지스틱 회귀 라고 불리는 소프트맥스 회귀는 출력 클래스가 2개보다 많을 때에 로지스틱 회귀를 일반화 하기 위해 사용된다. 관례적으로, 우리는 θK=0 로 설정하고 이는 각 클래스 i에 해당하는 브르누이 파라미터 ϕi를 다음과 같게 만는다: 

<br>

**33. Generalized Linear Models**

&#10230; 일반적 선형 모델 

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; 지수족 (Exponential family) - 어떠한 분포가 canonical paramter 또는 link function 이라고도 불리는 자연 모수 (natural parameter)에 관하여 정의될 수 있다면 그 분포는 지수족 안에 포함된다고 말할 수 있으며, canonical parameter (η), sufficient statistic (T(y)) 그리고 log-partition function (a(η)) 을 사용하여 다음과 같이 정의된다: 

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; 비고: 대부분의 경우 T(y)=y 이다. 또한, exp(−a(η))는 전체 확률의 합을 1로 만드는 정규화 매개변수로 볼 수 있다. 

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; 다음의 표는 가장 흔한 지수 분포에 대한 내용이다: 

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [분포, 브르누이, 가우시안, 푸아송, 기하학적]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; GLM의 가정 - 일반화 선형 모델 (GLM)은 확률변수(random variable) y를 x∈Rn+1 의 함수로서 예측하는 것에 목표를 두며 다음 3가지 가정에 의존한다: 

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; 최소자승법 (ordinary least squares)과 로지스틱 회귀는 일반화 선형 모델의 특수한 경우이다. 

<br>

**40. Support Vector Machines**

&#10230; 서포트 벡터 머신

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; 서포트 벡터 머신의 목적은 선으로의 최단거리를 최대화 시키는 그 선을 찾는 것이다. 

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; 최적 마진 분류기(Optimal margin classifier) - 최적 마진 분류기  h는: 

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; (w,b)∈Rn×R 는 다음의 최적화 문제의 해답이다.

<br>

**44. such that**

&#10230; 다음과 같이

<br>

**45. support vectors**

&#10230; 서포트 벡터

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; 비고: 선은 wTx−b=0 로 정의된다.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; 힌지 손실 (Hinge loss) - 힌지 손실은 SVM을 설정 할때 사용되며 다음과 같이 정의된다: 

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; 커널 - 피처를 매핑하는 함수 ϕ가 주어졌을 때, 커널 K를 다음과 같이 정의한다: 

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; 보통은 커널 K는 K(x,z)=exp(−||x−z||22σ2)로 정의되며 이를 가우시안 커널이라고 한다. 

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [비선형 구분가능, 커널 매핑의 사용, 원래의 공간에서의 결정경계] 

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; 비고: 우리는 보통 매우 복잡한 명시적 매핑 ϕ를 알 필요가 없기 때문에 커널을 사용하여 비용 함수를 계산하기 할 때 "커널 트릭"을 사용한다고 말한다. 이때 K(x,z) 값만 알면 된다.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; 라그랑지안 - 라그랑지안 L(w,b)를 다음과 같이 정의한다:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; 비고: 계수 βi들은 라그랑주 승수(Lagrange multipliers)라고 한다.

<br>

**54. Generative Learning**

&#10230; 생성 학습

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; 생성 모델은 우선 데이터가 어떻게 생성되는지 P(x|y)를 추정하며 배우는데 우리는 이를 베이즈 정리를 사용하여 P(y|x)를 추정하는 데에 사용할 수 있다.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; 가우시안 판별 분석

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; 설정 - 가우시안 판별 분석(Gaussian Discriminant Analysis)은 다음을 충족하는 y 와 x|y=0 와 x|y=1를 가정한다:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; 추정 - 다음 표는 우도/가능도를 최대화 시킬 때의 추정치들의 정리본이다:

<br>

**59. Naive Bayes**

&#10230; 나이브 베이즈

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; 가정 - 나이브 베이즈 모델은 피처의 각각의 데이터들이 모두 서로 독립적이라고 가정한다: 

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; 해답 - 로그-가능도를 최대화하면 k∈{0,1},l∈[[1,L]]를 포함한 해가 나온다

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; 비고: 나이브 베이즈는 텍스트 분류와 스팸 탐지에 널리 사용된다.

<br>

**63. Tree-based and ensemble methods**

&#10230; 트리 기반 방법 그리고 앙상블 방법

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; 이 방법들은 회귀와 분류 문제에 모두 사용될 수 있다.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - 분류와 회귀트리(Classification and Regression Trees)의 줄임말로서 결정트리로 익히 알려져 있으며 이진트리로 표현될 수 있다. 이 모델은 해석이 가능하다는 장점이 있다.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; 랜덤 포레스트 - 무작위로 선택된 피처들로부터 형성된 많은 수의 의사 결정 트리들을 사용하는 트리 기반 기술이다. 단순한 의사 결정 트리와는 달리 해석이 매우 어렵지만 일반적으로 좋은 성능을 내어 널리 사용되는 알고리즘이다. 

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; 비고: 랜덤 포레스트는 앙상블 방법 중의 하나이다.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; 부스팅 - 부스팅 방법은 여러 약한 학습기(weak learners)들을 합쳐서 더 강한 학습기를 만드는 것이다. 주요 목록들이 밑 테이블에 있다:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [아다 부스팅(Adaptive boosting), 그라디언트 부스팅(Gradient boosting)]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; 다음 부스팅 스텝에서 더 나아지기 위해 에러에 높은 가중치가 매겨지게 된다

<br>

**71. Weak learners trained on remaining errors**

&#10230; 남아있는 에러로 학습된 약한 학습기 

<br>

**72. Other non-parametric approaches**

&#10230; 다른 non-parametric 접근법들  

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-최근접 이웃 - 일반적으로 k-NN으로 알려진 k-최근접 이웃 알고리즘은 데이터 포인트의 응답이 훈련 세트에서의 k개의 이웃들의 특성에 의해 결정되는 비모수적 (non-parametric) 접근법이다. 이는 분류와 회귀 모두에서 사용될 수 있다. 

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; 비고: 파라미터 k가 높을 수록 편향이 커지고, 파라미터 k가 낮을 수록 분산이 커진다. 

<br>

**75. Learning Theory**

&#10230; 학습 이론 

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; 합계 상한(Union bound) - A1,...,Ak가 k개의 사건이라 하자. 이때: 

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; 호에프딩 부등식 - Z1,..,Zm를 인자 ϕ의 브루누이 분포로부터 추출된 m개의 독립 동일 분포(independent and identically distributed) 변수라고 하자. 표본 평균 ˆϕ 와 γ>0가 있을 때: 

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; 비고: 위 부등식은 체르노프 유계(Chernoff bound) 라고도 한다.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; 훈련 오차 - 주어진 분류기 h에 대해 경험적 위험 또는 경험적 오차라고도하는 훈련 오차 ˆϵ(h)를 다음과 같이 정의한다: 

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; Probably Approximately Correct (PAC) ― PAC는 학습 이론에 대한 수많은 결과가 입증 된 프레임 워크이며 다음과 같은 가정을 한다: 

<br>

**81: the training and testing sets follow the same distribution **

&#10230; 학습 세트와 테스팅 세트는 모두 같은 분포를 따른다

<br>

**82. the training examples are drawn independently**

&#10230; 학습 데이터들은 모두 독립적으로 추출 되었다

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering - 집합 S={x(1),...,x(d)}와 분류기의 집합 H가 주어졌을 때, 아래와 같은 조건을 만족한다면 레이블 {y(1),...,y(d)}에 대해 H가 S를 shatter 한다고 한다: 

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Upper bound 이론 - H를 |H|=k를 만족하는 유한 가설 클래스로 정하고 δ와 표본 크기 m을 고정한다. 그러면 적어도 1−δ의 확률로 다음을 얻는다: 

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC 차원 ― 주어진 무한 가설 클래스 H의 Vapnik-Chervonenkis (VC) 차원, VC(H)는 H에 의해 shatter 되는 가장 큰 집합의 크기이다. 

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; 비고: H={2차원에서의 선형 분류기의 집합}의 VC 차원은 3이다.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; 이론 (Vapnik) - VC(H)=d 와 함께 H가 주어지고 m이 훈련 예제들의 개수라고 하자. 적어도 1−δ의 확률로 아래의 식을 만족한다: 

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [소개, 예측의 종류, 모델의 종류]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [표기법과 일반적 개념, 손실 함수, 경사 하강법, 가능도]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [선형 모델, 선형 회귀, 로지스틱 회귀, 일반화되 선형 모델]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [서포트 벡터 머신, 최적 마진 분류기, 힌지 손실, 커널]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [생성 학습, 가우시안 판별 분석, 나이브 베이즈]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [트리와 앙상블 방법, CART(분류와 회귀트리), 랜덤 포레스트, 부스팅]

<br>

**94. [Other methods, k-NN]**

&#10230; [다른 방법론, k-NN(k-최근접 이웃)]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [학습 이론, 호에프딩 부등식, PAC, VC 차원]