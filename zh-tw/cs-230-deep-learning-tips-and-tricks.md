**Deep Learning Tips and Tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; 深度學習秘訣和技巧參考手冊

<br>


**2. CS 230 - Deep Learning**

&#10230; CS230 - 深度學習

<br>


**3. Tips and tricks**

&#10230; 秘訣和技巧

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [數據處理, 數據擴增, 批量標準化]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [訓練類神經網路, 回合, 小批量, 交叉熵損失, 反向傳播法, 梯度下降法, 更新權重, 梯度檢驗]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [參數調整 , Xavier 初始化, 遷移學習, 學習率, 自適應學習率]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [正規化, 丟棄法, 權重正規化, 早停法]**

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [最佳實務, 過度擬合小數量數據, 梯度檢驗]**

<br>


**9. View PDF version on GitHub**

&#10230; 於 GitHub 上閱讀 PDF 版

<br>


**10. Data processing**

&#10230; 數據處理

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; 數據擴增 - 深度學習模型通常需要很多數據來訓練。一般來說利用現有的數據，以數據擴增的技術來得到更多的數據是有用的。主要的技術整理如下。給定一影像，以下為我們可以應用的技術：

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [原始, 翻轉, 旋轉, 隨機剪裁]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [沒有任何修正的影像, 根據軸來翻轉且影像的意義仍存在, 輕微的旋轉, 模擬不正確的水平校準, 隨機聚焦在影像的一部分, 可一次進行數個隨機剪裁]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [色移, 加入雜訊, 資訊損耗, 調整對比]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; %%[改變細微的 RGB 值, 可捕捉曝光時出現的雜訊, 加入雜訊, More tolerance to quality variation of inputs, 部分的影像被忽略, Mimics potential loss of parts of image, 亮度改變, Controls difference in exposition due to time of day]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; 備註：數據擴增通常會與訓練同時進行。

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 批量標準化 - 使用超參數 γ,β 來標準化批量 {xi}。 以 μB (平均數), σ2B (變異數)來表示我們想要進行標準化的方式, 定義如下:

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 通常使用於全連接層或卷積層之後、非線性層之前, 目的於用更高的學習率, 且減少初始化的影響

<br>


**19. Training a neural network**

&#10230; 訓練類神經網路

<br>


**20. Definitions**

&#10230; 定義

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; 回合 - 在訓練模型時, 回合是一術語表示模型將整個訓練資料集看過一輪, 且更新其參數。

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; 小批量梯度下降法 - 在訓練的過程中, 因為計算的複雜度，並不會基於整個訓練資料集來更新權重，也因為會有雜訊, 而不會基於單筆資料。一般會利用小批量為單位來更新權重, 而小批次的數量為我們可調整的超參數。

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; 損失函數 - 為了量化模型的表現, 損失函數L評估模型預測z實際值y的能力。

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 交叉熵損失 - 在利用類神經網路進行二元分類時, 交叉熵損失L(z,y)定義為：

<br>


**25. Finding optimal weights**

&#10230; 尋找最佳權重

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; 反向傳播法 - 反向傳播法是一個利用預測值與理想值來更新類神經網路中權重的方法。權重w的導數可利用鏈鎖律來計算。

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; 使用此方法, 每個權重依照以下規則更新：

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 更新權重 - 在類神經網路之中, 權重依照以下步驟更新：

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [步驟一：取訓練資料集中一批量, 且透過計算正向傳播以取得誤差, 步驟二：反向傳播誤差來得到誤差對於每個權重的梯度, 步驟三：利用梯度來更新網路中的權重]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [正向傳播, 反向傳播, 權重更新]

<br>


**31. Parameter tuning**

&#10230; 參數調整

<br>


**32. Weights initialization**

&#10230; 權重初始化

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Xavier 初始化 - 不同於隨機初始化權重, Xavier初始化考慮到個別架構的特性。

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; 遷移學習 - 訓練深度學習模型仰賴於大量的數據與時間。我們時常善用使用已花費數天或禮拜訓練於大量數據的權重於自己的應用。取決於你有多少數據, 這裡提供幾個不同的方法：

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [訓練資料集大小, 圖示, 解釋]

<br>


**36. [Small, Medium, Large]**

&#10230; [小, 中, 大]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [固定所有層, 於歸一化指數函數訓練權重, 固定大部分層, 於最後一層與歸一化指數函數訓練權重, 用預訓練為初始化來訓練整個網路]

<br>


**38. Optimizing convergence**

&#10230; 優化收斂

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.
**

&#10230; 學習率 - 學習率通常表示為 α 或 η, 表示更新權重的步伐。

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; 自適應學習率 - 讓學習率可以隨著訓練改變，此技巧可減少訓練時間且提升最佳解法。其中，適應性矩估計優化器是最常使用的技巧，而其他的方法也很有用，整理如下：

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [方法, 解釋, 更新權重 w, 更新偏差值 b]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [動量, 阻尼震盪, SGD 的升級版, 有兩個參數需調整]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, 均方根傳遞, 透過控制震盪以加速學習演算法]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [Adam, 適應性矩估計, 最熱門的方法, 有四個參數需調整]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; 備註：其他方法包括 Adadelta, Adagrad 與 SGD。

<br>


**46. Regularization**

&#10230; 正規化

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; 丟棄法 - 丟棄法是一個透過在訓練中根據機率(p>0)丟棄神經元，用於防範過度擬合的技術。此技術強迫模型避免仰賴某些特定的特徵。

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; 備註：大部分的深度學習架構將丟棄法的參數設為「保留」(1−p)。

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; 權重正規化：為了讓權重不要變得太大或模型過度擬合, 正規化技術大多應用在模型的權重上。主要的方法整理於下表：

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230; [套索算法, 脊算法, 彈性網路]

<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; 

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230;

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230;

<br>


**53. Good practices**

&#10230;

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230;

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230;

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230;

<br>


**57. [Formula, Comments]**

&#10230;

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230;

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230;

<br>


**60. The Deep Learning cheatsheets are now available in [target language].

&#10230; 深度學習參考手冊目前已有[目標語言]版。


**61. Original authors**

&#10230; 原始作者

<br>

**62.Translated by X, Y and Z**

&#10230; 由 X, Y 與 Z 翻譯

<br>

**63.Reviewed by X, Y and Z**

&#10230; 由 X, Y 與 Z 檢閱

<br>

**64.View PDF version on GitHub**

&#10230; 於 GitHub 上閱讀 PDF 版

<br>

**65.By X and Y**

&#10230; 由X與Y

<br>
