**1. Machine Learning tips and tricks cheatsheet**

&#10230;머신러닝 팁과 트릭 치트시트

<br>

**2. Classification metrics**

&#10230;분류 측정 항목

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230;이진 분류 상황에서 모델의 성능을 평가하기 위해 눈 여겨 봐야하는 주요 측정 항목이 여기에 있습니다.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230;혼동 행렬 ― 혼동 행렬은 모델의 성능을 평가할 때, 보다 큰 그림을 보기위해 사용됩니다. 이는 다음과 같이 정의됩니다.

<br>

**5. [Predicted class, Actual class]**

&#10230;[예측된 클래스, 실제 클래스]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230;주요 측정 항목들 ― 다음 측정 항목들은 주로 분류 모델의 성능을 평가할 때 사용됩니다.

<br>

**7. [Metric, Formula, Interpretation]**

&#10230;[측정 항목, 공식, 해석]

<br>

**8. Overall performance of model**

&#10230;전반적인 모델의 성능

<br>

**9. How accurate the positive predictions are**

&#10230;예측된 양성이 정확한 정도

<br>

**10. Coverage of actual positive sample**

&#10230;실제 양성의 예측 정도

<br>

**11. Coverage of actual negative sample**

&#10230;실제 음성의 예측 정도

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230;불균형 클래스에 유용한 하이브리드 측정 항목

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230;ROC(Receiver Operating Curve) ― ROC 곡선은 임계값의 변화에 따른 TPR 대 FPR의 플롯입니다. 이 측정 항목은 아래 표에 요약되어 있습니다:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230;[측정 항목, 공식, 같은 측도]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230;AUC(Area Under the receiving operating Curve) ― AUC 또는 AUROC라고도 하는 이 측정 항목은 다음 그림과 같이 ROC 곡선 아래의 영역입니다:

<br>

**16. [Actual, Predicted]**

&#10230;[실제값, 예측된 값]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230;기본 측정 항목 ― 회귀 모델 f가 주어졌을때, 다음의 측정 항목들은 모델의 성능을 평가할 때 주로 사용됩니다:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;[총 제곱합, 설명된 제곱합, 잔차 제곱합]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230;결정 계수 ― 종종 R2 또는 r2로 표시되는 결정 계수는 관측된 결과가 모델에 의해 얼마나 잘 재현되는지를 측정하는 측도로서 다음과 같이 정의됩니다:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230;주요 측정 항목들 ― 다음 측정 항목들은 주로 변수의 수를 고려하여 회귀 모델의 성능을 평가할 때 사용됩니다:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230;여기서 L은 가능도이고 ^σ2는 각각의 반응과 관련된 분산의 추정값입니다.

<br>

**22. Model selection**

&#10230;모델 선택

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230;어휘 ― 모델을 선택할 때 우리는 다음과 같이 가지고 있는 데이터를 세 부분으로 구분합니다:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230;[학습 세트, 검증 세트, 테스트 세트]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230;[모델 훈련, 모델 평가, 모델 예측]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230;[주로 데이터 세트의 80%, 주로 데이터 세트의 20%]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230;[홀드아웃 또는 개발 세트라고도하는, 보지 않은 데이터]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230;모델이 선택되면 전체 데이터 세트에 대해 학습을 하고 보지 않은 데이터에서 테스트합니다. 이는 아래 그림에 나타나있습니다.

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230;교차-검증 ― CV라고도하는 교차-검증은 초기의 학습 세트에 지나치게 의존하지 않는 모델을 선택하는데 사용되는 방법입니다. 다양한 유형이 아래 표에 요약되어 있습니다:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230;[k-1 폴드에 대한 학습과 나머지 1폴드에 대한 평가, n-p개 관측치에 대한 학습과 나머지 p개 관측치에 대한 평가]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230;[일반적으로 k=5 또는 10, p=1인 케이스는 leave-one-out]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;가장 일반적으로 사용되는 방법은 k-폴드 교차-검증이라고하며 이는 학습 데이터를 k개의 폴드로 분할하고, 그 중 k-1개의 폴드로 모델을 학습하는 동시에 나머지 1개의 폴드로 모델을 검증합니다. 이 작업을 k번 수행합니다. 오류는 k 폴드에 대해 평균화되고 교차-검증 오류라고 부릅니다. 

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;정규화 ― 정규화 절차는 데이터에 대한 모델의 과적합을 피하고 분산이 커지는 문제를 처리하는 것을 목표로 합니다. 다음의 표는 일반적으로 사용되는 정규화 기법의 여러 유형을 요약한 것입니다:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;[계수를 0으로 축소, 변수 선택에 좋음, 계수를 작게 함, 변수 선택과 작은 계수 간의 트래이드오프]

<br>

**35. Diagnostics**

&#10230;진단

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;편향 ― 모델의 편향은 기대되는 예측과 주어진 데이터 포인트에 대해 예측하려고하는 올바른 모델 간의 차이입니다.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230;분산 ― 모델의 분산은 주어진 데이터 포인트에 대한 모델 예측의 가변성입니다.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230;편향/분산 트래이드오프 ― 모델이 간단할수록 편향이 높아지고 모델이 복잡할수록 분산이 커집니다.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230;[증상, 회귀 일러스트레이션, 분류 일러스트레이션, 딥러닝 일러스트레이션, 가능한 처리방법]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230;[높은 학습 오류, 테스트 오류에 가까운 학습 오류, 높은 편향, 테스트 에러 보다 약간 낮은 학습 오류, 매우 낮은 학습 오류, 테스트 오류보다 훨씬 낮은 학습 오류, 높은 분산]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230;[모델 복잡화, 특징 추가, 학습 증대, 정규화 수행, 추가 데이터 수집]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230;오류 분석 ― 오류 분석은 현재 모델과 완벽한 모델 간의 성능 차이의 근본 원인을 분석합니다.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230;애블러티브 분석 ― 애블러티브 분석은 현재 모델과 베이스라인 모델 간의 성능 차이의 근본 원인을 분석합니다.

<br>

**44. Regression metrics**

&#10230;회귀 측정 항목

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230;[분류 측정 항목, 혼동 행렬, 정확도, 정밀도, 리콜, F1 스코어, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230;[회귀 측정 항목, R 스퀘어, 맬로우의 CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230;[모델 선택, 교차-검증, 정규화]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230;[진단, 편향/분산 트래이드오프, 오류/애블러티브 분석]
