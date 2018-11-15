**1. Machine Learning tips and tricks cheatsheet**

&#10230;　機械学習チップ&トリック　チートシート

<br>

**2. Classification metrics**

&#10230;　分類評価指標

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; 二値分類の文脈では，次のようなモデルの性能を評価するための重要な評価指標があります．

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; 混同行列 - 混同行列はモデルの性能を評価する際に，より完全な描像を得るために使われます．

<br>

**5. [Predicted class, Actual class]**

&#10230; [予測したクラス, 実際のクラス]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; 主要な評価指標 - 次の指標が分類モデルの性能の評価のために一般的に使用されます．

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [評価指標,式,解釈]

<br>

**8. Overall performance of model**

&#10230;　モデルの全体的な性能

<br>

**9. How accurate the positive predictions are**

&#10230;　正と判断された予測の正答率

<br>

**10. Coverage of actual positive sample**

&#10230; 実際には正であるサンプルを正しく正と予測した割合

<br>

**11. Coverage of actual negative sample**

&#10230;　実際には負であるサンプルを正しく負と予測した割合

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230;　不均衡データに対する有用な複合指標

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC曲線 - 受信者動作特性曲線(ROC)は閾値を変えていく際のFPRに対するTPRのグラフです．

<br>

**14. [Metric, Formula, Equivalent]**

&#10230;　[評価指標,式,等価な指標]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - ROC曲線下面積(AUC,AUROC)は次の図のようにROC曲線の下側の面積のことです．

<br>

**16. [Actual, Predicted]**

&#10230; [実際，予測]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230;　[基本的な評価指標] 回帰モデルfが与えられたとき，次のようなよう化指標がモデルの性能を評価するために一般的に用いられます．

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;　[総平方和,説明された平方和,残差平方和]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230;　決定係数 - よくR2やr2と書かれる決定係数は，実際の結果がモデルによってどの程度よく再現されているかを測る評価指標であり，次のように定義される．

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; 主要な評価指標 - 次の評価指標は説明変数の数を考慮して回帰モデルの性能を評価するために，一般的に用いられています．

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; ここでLは尤度であり，ˆσ2は各応答に対する誤差分散の推定値です．

<br>

**22. Model selection**

&#10230; モデル選択

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; 用語 - モデルを選択するときには，次のように，データの種類を異なる３つに区別します．

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [訓練データセット,検証データセット,テストセット]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [モデルが学習される,モデルが評価される,モデルを用いて予測する]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [通常はデータセットの80%,通常はデータセットの20%]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [ホールドアウトセットや，開発セットとも呼ばれる,未知のデータ]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230;　一度モデル選択が行われた場合，学習はデータセットの全体を用いて行われ，またテストは未知のテストセットに対して行われます．これらは次のように表されます．

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; 交差検証 - 交差検証(CV)は，初期の学習データセットに強く依存しないようにモデル選択を行う方法です．いくつかの種類を下にまとめます．

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [k-1群で学習，残りの1群で評価,n-p個で学習，残りのp個で評価]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [k=5か10が一般的,p=1の場合はLeave-one-out cross validation法と呼ばれます．]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;

<br>

**35. Diagnostics**

&#10230;

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230;

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230;

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230;

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230;

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230;

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230;

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230;

<br>

**44. Regression metrics**

&#10230;

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230;

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230;

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230;

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230;
