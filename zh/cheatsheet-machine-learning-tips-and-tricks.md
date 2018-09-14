1. **Machine Learning tips and tricks cheatsheet**

&#10230;机器学习技巧和秘诀速查表

<br>

2. **Classification metrics**

&#10230;分类问题的度量

<br>

3. **In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230;

<br>在二分类问题中，下面这些主要度量标准对于评估模型的性能非常重要。

4. **Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230;混淆矩阵 - 混淆矩阵可以用来评估模型的整体性能情况。它的定义如下：

<br>

5. **[Predicted class, Actual class]**

&#10230;预测类别，实际类别

<br>

6. **Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230;主要度量标准 - 通常用下面的度量标准来评估分类模型的性能：

<br>

7. **[Metric, Formula, Interpretation]**

&#10230;性能度量，公式，说明

<br>

8. **Overall performance of model**

&#10230;模型总体性能

<br>

9. **How accurate the positive predictions are**

&#10230;预测为正样本的准确度

<br>

10. **Coverage of actual positive sample**

&#10230;真正样本的覆盖度

<br>

11. **Coverage of actual negative sample**

&#10230;真负样本的覆盖度

<br>

12. **Hybrid metric useful for unbalanced classes**

&#10230;混合度量，对于不平衡类别非常有效

<br>

13. **ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230;ROC - 受试者工作曲线，又叫做ROC曲线，它使用真正例率和假正例率分别作为纵轴和横轴并且进过调整阈值绘制出来。下表汇总了这些度量标准：

<br>

14. **[Metric, Formula, Equivalent]**

&#10230;性能度量，公式，等价形式

<br>

15. **AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230;AUC - 受试者工作曲线的之下的部分，又叫做AUC或者AUROC，如下图所示ROC曲线下的部分：

<br>

16. **[Actual, Predicted]**

&#10230;真实值，预测值

<br>

17. **Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230;基本性能度量 - 给定一个回归模型f，下面的度量标准通常用来评估模型的性能

<br>

18. **[Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;全部平方和，解释平方和，残差平方和

<br>

19. **Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230;确定性系数 - 确定性系数，记作R2或r2，提供了模型复现观测结果的能力，定义如下：

<br>

20. **Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230;主要性能度量 - 以下性能度量通过考虑变量n的数量，常用于评估回归模型的性能：

<br>

21. **where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230;L代表似然，ˆσ2代表方差

<br>

22. **Model selection**

&#10230;模型选择

<br>

23. **Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230;词汇 - 在选择模型时，我们将数据分为的3个不同部分：

<br>

24. **[Training set, Validation set, Testing set]**

&#10230;训练集，验证集，测试集

<br>

25. **[Model is trained, Model is assessed, Model gives predictions]**

&#10230;模型训练，模型评估，模型预测

<br>

26. **[Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230;一般数据集中的80%，一般数据集中的20%

<br>

27. **[Also called hold-out or development set, Unseen data]**

&#10230;又叫做留出集或者开发集，未知数据

<br>

28. **Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230;一旦选择了模型，就会在整个数据集上进行训练，并在测试集上进行测试。如下图所示：

<br>

29. **Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230;交叉验证 - 交叉验证，记为CV，是一种不必特别依赖于初始训练集的模型选择方法。下表汇总了几种不同的方式：

<br>

30. [**Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230;在k-1个子集上训练，在剩余的一个子集中评估，在n-p个子集上训练，在剩余的p个子集评估模型

<br>

31. **[Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230;通常k=5或10,p=1时又叫做留一法

<br>

32. **The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;最常用的模型选择方法是k折交叉验证，将训练集划分为k个子集，在k-1个子集上训练模型，在剩余的一个子集上评估模型，用这种划分方式重复训练k次。交叉验证损失是k次k折交叉验证的损失均值。

<br>

33. **Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;正则化 - 正则化方法可以解决高方差问题，避免模型对于训练数据产生过拟合。下表展示了常用的正则化方法：

<br>

34. **[Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;将系数收缩为0，有利于变量选择，使系数更小，在变量选择和小系数之间进行权衡

<br>

35. **Diagnostics**

&#10230;诊断

<br>

36. **Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;偏差 - 模型的偏差是模型预测值和真实值之间的差距

<br>

37. **Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230;方差 - 模型的方差是给定数据点的模型预测的可变性

<br>

38. **Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230;偏差/方差权衡 - 模型越简单，偏差越高，模型越复杂，方差越高。

<br>

39. **[Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230;症状，回归图，分类图，深度学习插图，可能的补救措施

<br>

40. **[High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230;高训练误差，训练误差接近测试误差，高偏差，训练误差略低于测试误差，极低训练误差，训练误差远低于测试误差，高方差

<br>

41. **[Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230;模型复杂性，添加更多特征，训练更长时间，实施正则化，获得更多数据

<br>

42. **Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230;错误分析 - 错误分析分析当前模型和完美模型之间性能差异的根本原因

<br>

43. **Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230;烧蚀分析 - 烧蚀分析可以分析当前和基线模型之间性能差异的根本原因

<br>
