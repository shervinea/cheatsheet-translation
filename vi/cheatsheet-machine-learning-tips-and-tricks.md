**1. Machine Learning tips and tricks cheatsheet**

&#10230; Các lời khuyên và kinh nghiệm trong Machine Learning (Học máy)

<br>

**2. Classification metrics**

&#10230; Độ đo phân loại

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Trong ngữ cảnh của phân loại nhị phân (binary classification), ở đây là các độ đo chính, chúng khá quan trọng để theo dõi (track), qua đó đánh giá hiệu năng của mô hình (model)

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Ma trận nhầm lẫn (Confusion matrix) - Confusion matrix được sử dụng để có nhiều hơn các kết quả hoàn chỉnh khi đánh giá hiệu năng của model. Nó được định nghĩa như sau:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Lớp dự đoán, lớp thực sự]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Độ đo chính - Các độ đo sau thường được sử dụng để đánh giá hiệu năng của mô hình phân loại:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Độ đo, Công thức, Diễn giải]

<br>

**8. Overall performance of model**

&#10230; Hiệu năng tổng thể của mô hình

<br>

**9. How accurate the positive predictions are**

&#10230; Các dự đoán positive chính xác bao nhiêu

<br>

**10. Coverage of actual positive sample**

&#10230; Bao phủ các ví dụ chính xác (positive) thực sự 

<br>

**11. Coverage of actual negative sample**

&#10230; Bao phủ các ví dụ sai (negative) thực sự 

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Độ đo Hybrid hữu ích cho các lớp không cân bằng (unbalanced classes)

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC - Đường cong hoạt động nhận, được kí hiệu là ROC, là minh hoạ của TPR với FPR bằng việc thay đổi ngưỡng (threshold). Các độ đo này được tổng kết ở bảng bên dưới:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Độ đo, Công thức, Tương đương]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - Khu vực phía dưới đường cong thao tác nhận, còn được gọi tắt là AUC hoặc AUROC, là khu vực phía dưới ROC như hình minh hoạ phía dưới: 

<br>

**16. [Actual, Predicted]**

&#10230; [Thực sự, Dự đoán]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Độ đo cơ bản - Cho trước mô hình hồi quy f, độ đo sau được sử dụng phổ biến để đánh giá hiệu năng của mô hình: 

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Tổng của tổng các bình phương, Mô hình tổng bình phương, Tổng bình phương dư]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Hệ số quyết định - Hệ số quyết định, thường được kí hiệu là R2 hoặc r2, cung cấp độ đo mức độ tốt của kết quả quan sát đầu ra và được nhân rộng bởi mô hình, được định nghĩa như sau:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Độ đo chính - Độ đo sau đây thường được sử dụng để đánh giá hiệu năng của mô hình hồi quy, bằng cách tính số lượng các biến n mà độ đo đó sẽ cân nhắc:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; trong đó L là khả năng và ˆσ2 là giá trị ước tính của phương sai tương ứng với mỗi response (hồi đáp) 

<br>

**22. Model selection**

&#10230; Lựa chọn model (mô hình)

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulary - Khi lựa chọn mô hình, chúng ta phân biệt 3 phần khác nhau của dữ liệu mà ta có như sau:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Tập huấn luyện, Tập xác thực, Tập kiểm tra (testing)]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Mô hình được huấn luyện, mô hình được xác thực, mô hình đưa ra dự đoán]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Thường là 80% tập dữ liệu, Thường là 20% tập dữ liệu]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Cũng được gọi là hold-out hoặc development set, Dữ liệu chưa hề biết]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Khi mô hình đã được chọn, nó được huấn luyện trên tập dữ liệu đầu vào, được test trên tập dữ liệu test hoàn toàn khác. Tất cả được minh hoạ ở hình bên dưới:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Cross-validation - Cross-validation, còn được gọi là CV, một phương thức được sử dụng để chọn ra 1 mô hình không dựa quá nhiều vào tập dữ liệu huấn luyện ban đầu. Các loại khác nhau được tổng kết ở bảng bên dưới:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Huấn luyện trên k-1 phần mà đánh giá trên 1 phần còn lại, Huấn luyện trên n-p phần và đánh giá trên p phần còn lại]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Thường thì k=5 hoặc 10, Trường hợp p=1 được gọi là leave-one-out]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; Phương thức hay được sử dụng được gọi là k-fold cross-validation và chia dữ liệu huấn luyện thành k phần, đánh giá mô hình trên 1 phần trong khi huấn luyện mô hình trên k-1 phần còn lại, tất cả k lần. Lỗi sau đó được tính trung bình trên k phần và được đặt tên là cross-validation error.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Chuẩn hoá - Mục đích của thủ tục chính quy hoá là tránh cho mô hình bị overfit với dữ liệu, do đó gặp phải vấn đề phương sai lớn. Bảng sau đây sẽ tổng kết các loại khác nhau của kĩ thuật chính quy hoá hay được sử dụng:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Giảm hệ số xuống còn 0, Tốt cho việc lựa chọn biến, Làm cho hệ số nhỏ hơn, Thay đổi giữa chọn biến và hệ số nhỏ hơn]

<br>

**35. Diagnostics**

&#10230; Dự đoán (Diagnostics)

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Bias - Bias của mô hình là sai số giữa dự đoán mong đợi và dự đoán của mô hình trên các điểm dữ liệu cho trước.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Phương sai - Phương sai của một mô hình là sự thay đổi dự đoán của mô hình trên các điểm dữ liệu cho trước.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Thay đổi/ Thay thế Bias/phương sai - Mô hình càng đơn giản bias càng lớn, mô hình càng phức tạp phương sai càng cao.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Symptoms, Minh hoạ hồi quy, Minh hoạ phân loại, Minh hoạ deep learning (học sâu), Biện pháp khắc phục có thể dùng]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Lỗi huấn luyện cao, Lỗi huấn luyện tiến gần tới lỗi test, Bias cao, Lỗi huấn luyện thấp hơn một chút so với lỗi test, Lỗi huẩn luyện rất thấp, Lỗi huấn luyện thấp hơn lỗi test rất nhiều, Phương sai cao]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Mô hình phức tạp, Thêm nhiều đặc trưng, Huấn luyện lâu hơn, Thực hiện chuẩn hóa, Lấy nhiều dữ liệu hơn]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Phân tích lỗi - Phân tích lỗi là phân tích nguyên nhân của sự khác biệt trong hiệu năng giữa mô hình hiện tại và mô hình lí tưởng.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Phân tích Ablative - Phân tích Ablative là phân tích nguyên nhân của sự khác biệt giữa hiệu năng của mô hình hiện tại và mô hình cơ sở.

<br>

**44. Regression metrics**

&#10230; Độ đo hồi quy

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Độ đo phân loại, Ma trận confusion, chính xác, dự đoán, recall, Điểm F1, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Độ đo hồi quy, Bình phương R, CP của Mallow, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Lựa chọn mô hình, cross-validation, Chuẩn hoá (regularization)]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Dự đoán, Thay thế Bias/phương sai, Phân tích lỗi/ablative]
