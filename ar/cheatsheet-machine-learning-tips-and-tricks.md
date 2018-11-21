**1. Machine Learning tips and tricks cheatsheet**

&#10230;
نصائح وحيل لتعلم الآلة
<br>


**2. Classification metrics**

&#10230;
مقاييس التصنيف
<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230;
في سياق التصنيف الثنائي، هذه أهم المقاييس التي يجب اتباعها لتقييم أداء النموذج
<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230;
مصفوفة الخلط ― تستعمل مصفوفة الخلط للحصول على نظرة شاملة عند تقييم أداء النموذج. وهي تعرف كالتالي:
<br>

**5. [Predicted class, Actual class]**

&#10230;
[الصنف المتوقع، الصنف الفعلي]
<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230;
المقاييس  ― المقاييس الآتية شائعة الإستعمال لتقييم أداء نماذج التصنيف:
<br>

**7. [Metric, Formula, Interpretation]**

&#10230;
[المقياس، الصيغة، التأويل]
<br>

**8. Overall performance of model**

&#10230;
الأداء الإجمالي للنموذج
<br>

**9. How accurate the positive predictions are**

&#10230;
كم دقة التوقعات الإيجابية
<br>

**10. Coverage of actual positive sample**

&#10230;
تغطية العينات الإيجابية الفعلية
<br>

**11. Coverage of actual negative sample**

&#10230;
تغطية العيانات السلبية الفعلية
<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230;
مقياس هجين مفيج عند الأصناف غير المتوازنة
<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230;
منحنى تشغيل المتلقي، هو رسم لنسبة الموجبات الصحيحة مقابل نسبة الموجبات الخاطئة مع تغيير العتبة. هذه المقاييس ملخصة في الجدول أسفله:
<br>

**14. [Metric, Formula, Equivalent]**

&#10230;
[المقياس، الصيفة، المقابل]
<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230;
المنطقة الواقعة تحت منحنى تشغيل المتلقي ، هي المنطقة الواقعة أسفل منحنى تشغيل المتلقي (المقياس السابق) كما هو موضح في الشكل التالي:
<br>

**16. [Actual, Predicted]**

&#10230;
[الفعلي، المتوقع]
<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230;
مقاييس أساسية ― بالنظر إلى نموذج الإنحدار "إف"، المقاييس التالية شائعة الإستعمال لتقييم أداء النموذج:
<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;
[المجموع الإجمالي للمربعات، المجموع المشروح للمربعات، المجموع المتبقي للمربعات]
<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230;
معامل التحديد ― يعطي معامل التحديد قياسا لجودة النتائج الملاحظة وكيف تم تكرارها من طرف النموذج وتعرف كالآتي:

<br>
**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230;
المقاييس الرئيسية ― المقاييس الآتية شائعة الإستعمال لتقييم أداء نماذج الإنحدار، بالأخذ بعين الإعتبار عدد المتغيرات التي تأخذ بعين الإعتبار

<br>
**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230;
حيث L هي الاحتمال و ˆσ2 هو تقدير للتباين المرتبط بكل استجابة.

<br>
**22. Model selection**

&#10230;
إختيار النموذج

<br>
**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230;
المفردات ― عند اختيار النموذج، نميز بين ثلاث أجزاء مختلفة للبيانات كالتالي:

<br>
**24. [Training set, Validation set, Testing set]**

&#10230;
[مجموعة التدريب، مجموعة التحقق، مجموعة الإختبار]

<br>
**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230;
[النموذج يدرَب، النموذج يقيَم، النموذج يعطي توقعات]

<br>
**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230;
[عادة 80 في المئة من البيانات، عادة 20 في المئة من البيانت]

<br>
**27. [Also called hold-out or development set, Unseen data]**

&#10230;
[المعروف أيضا بمجموعة الإنتظار أو التطوير، البيانات غير المشاهَدة]

<br>
**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230;
بمجرد اختيار النموذج، يتم تدريبه على مجموع البيانات واختباره على مجموعة اختبار لم يشاهدها من قبل. وهي ممثلة في الشكل أسفله:
<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230;
التصديق المتقاطع ― هي طريقة تستعمل لاختيار النموذج الذي لا يعتمد كثيرا على مجموعة التدريب الأولية. الأنواع المختلفة ملخصة في الجدول أسفله:
<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230;
[التدريب على k-1 طية والتقييم على الطيات المتبقية، التدريب على n-p ملاحظة والتقييم على الملاحظات p المتبقية]
<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230;
[عامة k=5 أو 10، الحالة p=1 ]
<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;
الطريقة الأكثر استعمالا تسمى "ك" طية (k-folds) للتصديق المتقاطع وتقسم مجموعة التداريب إلى "ك" طية للتصديق على النموذج في طية واحدة بينما التدريب يتم على الطيات "ك-1" المتبقية، وتكرر العملية "ك" مرة. الخطأ يحتسب كمعدل لكل الطيات "ك" ويسمى خطأ التصديق المتقاطع.
<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;
التنظيم ― التنظيم يهدف إلى تفادي التعلم الزائد للنموذج وبالتالي التعامل مع المشاكل التي قد تنتج قيم مرتفعة للتباين. الجدول التالي يختزل مختلف تقنيات التنظيم شائعة الإستعمال: 
<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;
[معاملات الإنكماش نحو الصفر، جيد لاختيار المتغيرات، يجعل المعاملات أصغر، مفاضلة بين اختيار المتغيرات ومعاملات أصغر]
<br>

**35. Diagnostics**

&#10230;
التشخيصات
<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;
الإنحياز ― إنحياز نموذج هو الفرق بين التنبؤ المرتقب والنموذج الصحيح الذي نحاول التنبؤ به لنقط معينة من البيانات.
<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230;
التباين ― تباين نموذج هو تغير تنبؤ النموذج لنقط معينة من البيانات.
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
