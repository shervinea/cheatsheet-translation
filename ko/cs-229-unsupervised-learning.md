**1. Unsupervised Learning cheatsheet**

&#10230; 비지도 학습 cheatsheet

<br>

**2. Introduction to Unsupervised Learning**

&#10230; 비지도 학습 소개

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; 동기부여 - 비지도학습의 목표는 {x(1),...,x(m)}와 같이 라벨링이 되어있지 않은 데이터 내의 숨겨진 패턴을 찾는것이다.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; 옌센 부등식 - f를 볼록함수로 하며 X는 확률변수로 두고 아래와 같은 부등식을 따르도록 하자.

<br>

**5. Clustering**

&#10230; 군집화

<br>

**6. Expectation-Maximization**

&#10230; 기댓값 최대화

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; 잠재변수 - 잠재변수들은 숨겨져있거나 관측되지 않는 변수들을 말하며, 이러한 변수들은 추정문제의 어려움을 가져온다. 그리고 잠재변수는 종종 z로 표기되어진다. 일반적인 잠재변수로 구성되어져있는 형태들을 살펴보자 

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; 표기형태, 잠재변수 z, 주석

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; 가우시안 혼합모델, 요인분석

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter  θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; 알고리즘 - 기댓값 최대화 (EM) 알고리즘은 모수 θ를 추정하는 효율적인 방법을 제공해준다. 모수 θ의 추정은 아래와 같이 우도의 아래 경계지점을 구성하는(E-step)과 그 우도의 아래 경계지점을 최적화하는(M-step)들의 반복적인 최대우도측정을 통해 추정된다. 

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step : 각각의 데이터 포인트 x(i)은 특정 클러스터 z(i)로 부터 발생한 후 사후확률Qi(z(i))를 평가한다. 아래의 식 참조

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step : 데이터 포인트 x(i)에 대한 클러스트의 특정 가중치로 사후확률 Qi(z(i))을 사용, 각 클러스트 모델을 개별적으로 재평가한다. 아래의 식 참조

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; Gaussians 초기값, 기대 단계, 최대화 단계, 수렴

<br>

**14. k-means clustering**

&#10230; k-평균 군집화

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; c(i)는 데이터 포인트 i 와 j군집의 중앙인 μj 들의 군집이다.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; 알고리즘 - 군집 중앙에 μ1,μ2,...,μk∈Rn 와 같이 무작위로 초기값을 잡은 후, k-평균 알고리즘이 수렴될때 까지 아래와 같은 단계를 반복한다.

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; 평균 초기값, 군집분할, 평균 재조정, 수렴

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; 왜곡 함수 - 알고리즘이 수렴하는지를 확인하기 위해서는 아래와 같은 왜곡함수를 정의해야 한다.

<br>

**19. Hierarchical clustering**

&#10230; 계층적 군집분석

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; 알고리즘 - 연속적 방식으로 중첩된 클러스트를 구축하는 결합형 계층적 접근방식을 사용하는 군집 알고리즘이다.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; 종류 - 다양한 목적함수의 최적화를 목표로하는 다양한 종류의 계층적 군집분석 알고리즘들이 있으며, 아래 표와 같이 요약되어있다.

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; Ward 연결법, 평균 연결법, 완전 연결법

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; 군집 거리 내에서의 최소화, 한쌍의 군집간 평균거리의 최소화, 한쌍의 군집간 최대거리의 최소화

<br>

**24. Clustering assessment metrics**

&#10230; 군집화 평가 metrics

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; 비지도학습 환경에서는, 지도학습 환경과는 다르게 실측자료에 라벨링이 없기 때문에 종종 모델에 대한 성능평가가 어렵다.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; 실루엣 계수 -  a와 b를 같은 클래스의 다른 모든점과 샘플 사이의 평균거리와 다음 가장 가까운 군집의 다른 모든 점과 샘플사이의 평균거리로 표기하면 단일 샘플에 대한 실루엣 계수 s는 다음과 같이 정의할 수 있다. 

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Calinski-Harabaz 색인 - k개 군집에 Bk와 Wk를 표기하면, 다음과 같이 각각 정의 된 군집간 분산행렬이다.

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; Calinski-Harabaz 색인 s(k)는 군집모델이 군집화를 얼마나 잘 정의하는지를 나타낸다. 가령 높은 점수일수록 군집이 더욱 밀도있으며 잘 분리되는 형태이다. 아래와 같은 정의를 따른다. 

<br>

**29. Dimension reduction**

&#10230; 차원 축소

<br>

**30. Principal component analysis**

&#10230; 주성분 분석

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; 차원축소 기술은 데이터를 반영하는 최대 분산방향을 찾는 기술이다.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; 고유값, 고유벡터 - A∈Rn×n 행렬이 주어질때, λ는 A의 고유값이 되며, 만약 z∈Rn∖{0} 벡터가 있다면 고유함수이다. 

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; 스펙트럼 정리 - A∈Rn×n 이라고 하자 만약 A가 대칭이라면, A는 실수 직교 행렬 U∈Rn×n에 의해 대각행렬로 만들 수 있다.

<br>

**34. diagonal** 

&#10230; 대각선

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; 참조: 가장 큰 고유값과 연관된 고유 벡터를 행렬 A의 주요 고유벡터라고 부른다

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230; 알고리즘 - 주성분 분석(PCA) 절차는 데이터 분산을 최대화하여 k 차원의 데이터를 투영하는 차원 축소 기술로 다음과 같이 따른다.

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; 1단계: 평균을 0으로 표준편차가 1이되도록 데이터를 표준화한다. 

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; 2단계: 실제 고유값과 대칭인 Σ=1mm∑i=1x(i)x(i)T∈Rn×n를 계산합니다. 

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; 3단계: k 직교 고유벡터의 합을 u1,...,uk∈Rn와 같이 계산한다. 다시말하면, 가장 큰 고유값 k의 직교 고유벡터이다. 

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; 4단계: R(u1,...,uk) 범위에 데이터를 투영하자.

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; 해당 절차는 모든 k-차원의 공간들 사이에 분산을 최대화 하는것이다. 

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; 변수공간의 데이터, 주요성분들 찾기, 주요성분공간의 데이터

<br>

**43. Independent component analysis**

&#10230; 독립성분분석

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; 근원적인 생성원을 찾기위한 기술을 의미한다.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; 가정 - 다음과 같이 우리는 데이터 x가 n차원의 소스벡터 s=(s1,...,sn)에서부터 생성되었음을 가정한다. 이때 si는 독립적인 확률변수에서 나왔으며, 혼합 및 비특이 행렬 A를 통해 생성된다고 가정한다. 

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; 비혼합 행렬 W=A−1를 찾는 것을 목표로 한다.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Bell과 Sejnowski 독립성분분석(ICA) 알고리즘 - 다음의 단계들을 따르는 비혼합 행렬 W를 찾는 알고리즘이다.

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; x=As=W−1s의 확률을 다음과 같이 기술한다.

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; 주어진 학습데이터 {x(i),i∈[[1,m]]}에 로그우도를 기술하고 시그모이드 함수 g를 다음과 같이 표기한다.

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; 그러므로, 확률적 경사상승 학습 규칙은 각 학습예제 x(i)에 대해서 다음과 같이 W를 업데이트하는 것과 같다. 

<br>

**51. The Machine Learning cheatsheets are now available in Korean.**

&#10230; 머신러닝 cheatsheets는 현재 한국어로 제공된다.

<br>

**52. Original authors**

&#10230; 원저자

<br>

**53. Translated by X, Y and Z**

&#10230; X,Y,Z에 의해 번역되다. 

<br>

**54. Reviewed by X, Y and Z**

&#10230; X,Y,Z에 의해 검토되다.

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; 소개, 동기부여, 얀센 부등식

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; 군집화, 기댓값-최대화, k-means, 계층적 군집화, 측정지표

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; 차원축소, 주성분분석(PCA), 독립성분분석(ICA) 
