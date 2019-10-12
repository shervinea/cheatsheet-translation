1. **Machine Learning tips and tricks cheatsheet**

&#10230;
機器學習秘訣和技巧參考手冊
<br>

2. **Classification metrics**

&#10230;
分類器的評估指標
<br>

3. **In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230;
在二元分類的問題上，底下是主要用來衡量模型表現的指標
<br>

4. **Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230;
混淆矩陣 - 混淆矩陣是用來衡量模型整體表現的指標
<br>

5. **[Predicted class, Actual class]**

&#10230;
[預測類別, 真實類別]
<br>

6. **Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230;
主要的衡量指標 - 底下的指標經常用在評估分類模型的表現
<br>

7. **[Metric, Formula, Interpretation]**

&#10230;
[指標, 公式, 解釋]
<br>

8. **Overall performance of model**

&#10230;
模型的整體表現
<br>

9. **How accurate the positive predictions are**

&#10230;
預測的類別有多精準的比例
<br>

10. **Coverage of actual positive sample**

&#10230;
實際正的樣本的覆蓋率有多少
<br>

11. **Coverage of actual negative sample**

&#10230;
實際負的樣本的覆蓋率
<br>

12. **Hybrid metric useful for unbalanced classes**

&#10230;
對於非平衡類別相當有用的混合指標
<br>

13. **ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230;
ROC - 接收者操作特徵曲線 (ROC Curve)，又被稱為 ROC，是透過改變閥值來表示 TPR 和 FPR 之間關係的圖形。這些指標總結如下：
<br>

14. **[Metric, Formula, Equivalent]**

&#10230;
[衡量指標, 公式, 等同於]
<br>

15. **AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230;
AUC - 在接收者操作特徵曲線 (ROC) 底下的面積，也稱為 AUC 或 AUROC：
<br>

16. **[Actual, Predicted]**

&#10230;
[實際值, 預測值]
<br>

17. **Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230;
基本的指標 - 給定一個迴歸模型 f，底下是經常用來評估此模型的指標：
<br>

18. **[Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;
[總平方和, 被解釋平方和, 殘差平方和]
<br>

19. **Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230;
決定係數 - 決定係數又被稱為 R2 or r2，它提供了模型是否具備復現觀測結果的能力。定義如下：
<br>

20. **Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230;
主要的衡量指標 - 藉由考量變數 n 的數量，我們經常用使用底下的指標來衡量迴歸模型的表現：
<br>

21. **where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230;
當中，L 代表的是概似估計，ˆσ2 則是變異數的估計
<br>

22. **Model selection**

&#10230;
模型選擇
<br>

23. **Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230;
詞彙 - 當進行模型選擇時，我們會針對資料進行以下區分：
<br>

24. **[Training set, Validation set, Testing set]**

&#10230;
[訓練資料集, 驗證資料集, 測試資料集]
<br>

25. **[Model is trained, Model is assessed, Model gives predictions]**

&#10230;
[用來訓練模型, 用來評估模型, 模型用來預測用的資料集]
<br>

26. **[Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230;
[通常是 80% 的資料集, 通常是 20% 的資料集]
<br>

27. **[Also called hold-out or development set, Unseen data]**

&#10230;
[又被稱為 hold-out 資料集或開發資料集, 模型沒看過的資料集]
<br>

28. **Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230;
當模型被選擇後，就會使用整個資料集來做訓練，並且在沒看過的資料集上做測試。你可以參考以下的圖表：
<br>

29. **Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230;
交叉驗證 - 交叉驗證，又稱之為 CV，它是一種不特別依賴初始訓練集來挑選模型的方法。幾種不同的方法如下：
<br>

30. **[Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230;
[把資料分成 k 份，利用 k-1 份資料來訓練，剩下的一份用來評估模型效能, 在 n-p 份資料上進行訓練，剩下的  p 份資料用來評估模型效能]
<br>

31. **[Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230;
[一般來說 k=5 或 10, 當 p=1 時，又稱為 leave-one-out]
<br>

32. **The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;
最常用到的方法叫做 k-fold 交叉驗證。它將訓練資料切成 k 份，在 k-1 份資料上進行訓練，而剩下的一份用來評估模型的效能，這樣的流程會重複 k 次次。最後計算出來的模型損失是 k 次結果的平均，又稱為交叉驗證損失值。
<br>

33. **Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;
正規化 - 正歸化的目的是為了避免模型對於訓練資料過擬合，進而導致高方差。底下的表格整理了常見的正規化技巧：
<br>

34. **[Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;
[將係數縮減為 0, 有利變數的選擇, 將係數變得更小, 在變數的選擇和小係數之間作權衡]
<br>

35. **Diagnostics**

&#10230;
診斷
<br>

36. **Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;
偏差 - 模型的偏差指的是模型預測值與實際值之間的差異
<br>

37. **Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230;
變異 - 變異指的是模型在預測資料時的變異程度
<br>

38. **Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230;
偏差/變異的權衡 - 越簡單的模型，偏差就越大。而越複雜的模型，變異就越大
<br>

39. **[Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230;
[現象, 迴歸圖示, 分類圖示, 深度學習圖示, 可能的解法]
<br>

40. **[High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230;
[訓練錯誤較高, 訓練錯誤和測試錯誤接近, 高偏差, 訓練誤差會稍微比測試誤差低, 訓練誤差很低, 訓練誤差比測試誤差低很多, 高變異]
<br>

41. **[Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230;
[使用較複雜的模型, 增加更多特徵, 訓練更久, 採用正規化化的方法, 取得更多資料]
<br>

42. **Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230;
誤差分析 - 誤差分析指的是分析目前使用的模型和最佳模型之間差距的根本原因
<br>

43. **Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230;
銷蝕分析 (Ablative analysis) - 銷蝕分析指的是分析目前模型和基準模型之間差異的根本原因
<br>
