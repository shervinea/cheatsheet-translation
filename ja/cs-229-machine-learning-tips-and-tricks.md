**1. Machine Learning tips and tricks cheatsheet**

&#10230; 機械学習のアドバイスやコツのチートシート

<br>

**2. Classification metrics**

&#10230; 分類評価指標

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; 二値分類において、モデルの性能を評価する際の主要な指標として次のものがあります。

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; 混同行列 ― 混同行列はモデルの性能を評価する際に、より完全に理解するために用いられます。次のように定義されます：

<br>

**5. [Predicted class, Actual class]**

&#10230; [予測したクラス, 実際のクラス]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; 主要な評価指標 ― 分類モデルの性能を評価するために、一般的に次の指標が用いられます。

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [評価指標,式,解釈]

<br>

**8. Overall performance of model**

&#10230; モデルの全体的な性能

<br>

**9. How accurate the positive predictions are**

&#10230; 陽性判定は、どれくらい正確ですか

<br>

**10. Coverage of actual positive sample**

&#10230; 実際に陽性であるサンプル

<br>

**11. Coverage of actual negative sample**

&#10230; 実際に陰性であるサンプル

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; 不均衡データに対する有用な複合指標

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC曲線 ― 受信者動作特性曲線(ROC)は閾値を変えていく際のFPRに対するTPRのグラフです。これらの指標は下表の通りまとめられます。

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [評価指標,式,等価な指標]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― ROC曲線下面積(AUC,AUROC)は次の図に示される通りROC曲線の下側面積のことです。

<br>

**16. [Actual, Predicted]**

&#10230; [実際，予測]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; 基本的な評価指標 ― 回帰モデルfが与えられたとき，次のようなよう化指標がモデルの性能を評価するために一般的に用いられます。

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [全平方和,回帰平方和,残差平方和]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; 決定係数 ― よくR2やr2と書かれる決定係数は，実際の結果がモデルによってどの程度よく再現されているかを測る評価指標であり，次のように定義される。

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; 主要な評価指標 ― 次の評価指標は説明変数の数を考慮して回帰モデルの性能を評価するために，一般的に用いられています。

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; ここでLは尤度であり，ˆσ2は各応答に対する誤差分散の推定値です。

<br>

**22. Model selection**

&#10230; モデル選択

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; 用語 ― モデルを選択するときには，次のようにデータの種類を異なる３つに区別します。

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [訓練セット,検証セット,テストセット]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [モデルを学習させる,モデルを評価する,モデルが予測する]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [通常はデータセットの80%,通常はデータセットの20%]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [ホールドアウトセットや，開発セットとも呼ばれる,未知のデータ]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; 一度モデル選択が行われた場合,学習にはデータセット全体が用いられ,テストには未知のテストセットが使用されます。これらは次の図ように表されます。

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; 交差検証 ― 交差検証(CV)は，初期の学習データセットに強く依存しないようにモデル選択を行う方法です。２つの方法を下表にまとめました。

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [k-1群で学習，残りの1群で評価,n-p個で学習，残りのp個で評価]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [一般的にはk=5または10,p=1の場合は一個抜き交差検証と呼ばれます]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; 最も一般的に用いられている方法はk交差検証法です．データセットをk群に分けた後，1群を検証に使用し残りのk-1群を学習に使用するという操作を順番にk回繰り返します。求められた検証誤差はk群すべてにわたって平均化されます。この平均された誤差のことを交差検証誤差と呼びます。

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; 正則化 ― 正則化はモデルの過学習状態を回避することが目的であり,したがってハイバリアンス問題(オーバーフィット問題)に対処できます。一般的に使用されるいくつかの正則化法を下表にまとめました。

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [係数を0にする,変数選択に適する,係数を小さくする,変数選択と係数を小さくすることのトレードオフ]

<br>

**35. Diagnostics**

&#10230; 診断方法

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; バイアス ― ある標本値群を予測する際の期待値と正しいモデルの結果との差異のことです。

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; バリアンス ― モデルのバリアンスとは，ある標本値群に対するモデルの予測値のばらつきのことです。

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; バイアス・バリアンストレードオフ ― よりシンプルなモデルではバイアスが高くなり，より複雑なモデルはバリアンスが高くなります。

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [症状,回帰モデルでの図,分類モデルでの図,深層学習での図,可能な解決策]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [高い訓練誤差,訓練誤差がテスト誤差に近い，高いバイアス,訓練誤差がテスト誤差より少しだけ小さい,極端に小さい訓練誤差,訓練誤差がテスト誤差に比べて非常に小さい,高いバリアンス]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [より複雑なモデルを試す,特徴量を増やす，より長く学習する,正則化を導入する,データ数を増やす]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; エラー分析 ― エラー分析は現在のモデルと完璧なモデル間の性能差の主要な要因を分析することです。

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; アブレーション分析 ― アブレーション分析は，ベースライン・モデルと現在されたモデル間で発生したパフォーマンスの差異の原因を分析することです。

<br>

**44. Regression metrics**

&#10230; 回帰評価指標

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [分類評価指標,混同行列,正解率,適合率,再現率,F値,ROC曲線]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [回帰評価指標,R二乗,マローズのCp,AIC,BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [モデルの選択，交差検証，正則化]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [診断方法，バイアス・バリアンストレードオフ，エラー・アブレーション分析]
