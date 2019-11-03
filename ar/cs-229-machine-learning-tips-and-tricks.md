**Machine Learning tips and tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

<br>

**1. Machine Learning tips and tricks cheatsheet**

<div dir="rtl">
مرجع سريع لنصائح وحيل تعلّم الآلة
</div>
<br>

**2. Classification metrics**

<div dir="rtl">
مقاييس التصنيف
</div>
<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

<div dir="rtl">
في سياق التصنيف الثنائي، هذه المقاييس (metrics) المهمة التي يجدر مراقبتها من أجل تقييم آداء النموذج.
</div>
<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

<div dir="rtl">
مصفوفة الدقّة (confusion matrix) - تستخدم مصفوفة الدقّة لأخذ تصور شامل عند تقييم أداء النموذج. وهي تعرّف كالتالي: 
</div>
<br>

**5. [Predicted class, Actual class]**

<div dir="rtl">
[التصنيف المتوقع، التصنيف الفعلي]
</div>
<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

<div dir="rtl">
المقاييس الأساسية - المقاييس التالية تستخدم في العادة لتقييم أداء نماذج التصنيف:
</div>
<br>

**7. [Metric, Formula, Interpretation]**

<div dir="rtl">
[المقياس، المعادلة، التفسير]
</div>
<br>

**8. Overall performance of model**

<div dir="rtl">
الأداء العام للنموذج
</div>
<br>

**9. How accurate the positive predictions are**

<div dir="rtl">
دقّة التوقعات الإيجابية (positive)
</div>
<br>

**10. Coverage of actual positive sample**

<div dir="rtl">
تغطية عينات التوقعات الإيجابية الفعلية
</div>
<br>

**11. Coverage of actual negative sample**

<div dir="rtl">
تغطية عينات التوقعات السلبية الفعلية
</div>
<br>

**12. Hybrid metric useful for unbalanced classes**

<div dir="rtl">
مقياس هجين مفيد للأصناف غير المتوازنة (unbalanced)
</div>
<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

<div dir="rtl">
منحنى دقّة الأداء (ROC) - منحنى دقّة الآداء، ويطلق عليه ROC، هو رسمة لمعدل التصنيفات الإيجابية الصحيحة (TPR) مقابل معدل التصنيفات الإيجابية الخاطئة (FPR) باستخدام قيم حد (threshold) متغيرة. هذه المقاييس ملخصة في الجدول التالي:
</div>
<br>

**14. [Metric, Formula, Equivalent]**

<div dir="rtl">
[المقياس، المعادلة، مرادف]
</div>
<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

<div dir="rtl">
المساحة تحت منحنى دقة الأداء (المساحة تحت المنحنى) (AUC) - المساحة تحت منحنى دقة الأداء (المساحة تحت المنحنى)، ويطلق عليها  AUC أو AUROC، هي المساحة تحت ROC كما هو موضح في الرسمة التالية:
</div>
<br>

**16. [Actual, Predicted]**

<div dir="rtl">
[الفعلي، المتوقع]
</div>
<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

<div dir="rtl">
المقاييس الأساسية - إذا كان لدينا نموذج الانحدار f، فإن المقاييس التالية غالباً ما تستخدم لتقييم أداء النموذج:
</div>
<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

<div dir="rtl">
[المجموع الكلي للمربعات، مجموع المربعات المُفسَّر، مجموع المربعات المتبقي]
</div>
<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

<div dir="rtl">
مُعامل التحديد (Coefficient of determination) - مُعامل التحديد، وغالباً يرمز له بـ R2 أو r2، يعطي قياس لمدى مطابقة النموذج للنتائج الملحوظة، ويعرف كما يلي:
</div>
<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

<div dir="rtl">
المقاييس الرئيسية - المقاييس التالية تستخدم غالباً لتقييم أداء نماذج الانحدار، وذلك بأن يتم الأخذ في الحسبان عدد المتغيرات n المستخدمة فيها:
</div>
<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

<div dir="rtl">
حيث L هو الأرجحية، و ˆσ2 تقدير التباين الخاص بكل نتيجة.
</div>
<br>

**22. Model selection**

<div dir="rtl">
اختيار النموذج
</div>
<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

<div dir="rtl">
مفردات - عند اختيار النموذج، نفرق بين 3 أجزاء من البيانات التي لدينا كالتالي:
</div>
<br>

**24. [Training set, Validation set, Testing set]**

<div dir="rtl">
[مجموعة تدريب، مجموعة تحقق، مجموعة اختبار]
</div>
<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

<div dir="rtl">
[يتم تدريب النموذج، يتم تقييم النموذج، النموذج يعطي التوقعات]
</div>
<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

<div dir="rtl">
[غالباً 80% من مجموعة البيانات، غالباً 20% من مجموعة البيانات]
</div>
<br>

**27. [Also called hold-out or development set, Unseen data]**

<div dir="rtl">
[يطلق عليها كذلك المجموعة المُجنّبة أو مجموعة التطوير، بيانات لم يسبق رؤيتها من قبل]
</div>
<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

<div dir="rtl">
بمجرد اختيار النموذج، يتم تدريبه على مجموعة البيانات بالكامل ثم يتم اختباره على مجموعة اختبار لم يسبق رؤيتها من قبل. كما هو موضح في الشكل التالي:
</div>
<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

<div dir="rtl">
التحقق المتقاطع (Cross-validation) - التحقق المتقاطع، وكذلك يختصر بـ CV، هو طريقة تستخدم لاختيار نموذج بحيث لا يعتمد بشكل كبير على مجموعة بيانات التدريب المبدأية. أنواع التحقق المتقاطع المختلفة ملخصة في الجدول التالي:
</div>
<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

<div dir="rtl">
[التدريب على k-1 جزء والتقييم باستخدام الجزء الباقي، التدريب على n−p عينة والتقييم باستخدام الـ p عينات المتبقية]
</div>
<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

<div dir="rtl">
[بشكل عام k=5 أو 10، الحالة p=1 يطلق عليها الإبقاء على واحد (leave-one-out)]
</div>
<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

<div dir="rtl">
الطريقة الأكثر استخداماً يطلق عليها التحقق المتقاطع س جزء/أجزاء (k-fold)، ويتم فيها تقسيم البيانات إلى k جزء، بحيث يتم تدريب النموذج باستخدام k−1 والتحقق باستخدام الجزء المتبقي، ويتم تكرار ذلك k مرة. يتم بعد ذلك حساب معدل الأخطاء في الأجزاء k ويسمى خطأ التحقق المتقاطع.
</div>
<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

<div dir="rtl">
ضبط (Regularization) - عمليه الضبط تهدف إلى تفادي فرط التخصيص (overfit) للنموذج، وهو بذلك يتعامل مع مشاكل التباين العالي. الجدول التالي يلخص أنواع وطرق الضبط الأكثر استخداماً:
</div>
<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

<div dir="rtl">
[يقلص المُعاملات إلى 0، جيد لاختيار المتغيرات، يجعل المُعاملات أصغر، المفاضلة بين اختيار المتغيرات والمُعاملات الصغيرة]
</div>
<br>

**35. Diagnostics**

<div dir="rtl">
التشخيصات
</div>
<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

<div dir="rtl">
الانحياز (Bias) - الانحياز للنموذج هو الفرق بين التنبؤ المتوقع والنموذج الحقيقي الذي نحاول تنبؤه للبيانات المعطاة.
</div>
<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

<div dir="rtl">
التباين (Variance) - تباين النموذج هو مقدار التغير في تنبؤ النموذج لنقاط البيانات المعطاة.
</div>
<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

<div dir="rtl">
موازنة الانحياز/التباين (Bias/variance tradeoff) - كلما زادت بساطة النموذج، زاد الانحياز، وكلما زاد تعقيد النموذج، زاد التباين.
</div>
<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

<div dir="rtl">
[الأعراض، توضيح الانحدار، توضيح التصنيف، توضيح التعلم العميق، العلاجات الممكنة]
</div>
<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

<div dir="rtl">
[خطأ التدريب عالي، خطأ التدريب قريب من خطأ الاختبار، انحياز عالي، خطأ التدريب أقل بقليل من خطأ الاختبار، خطأ التدريب منخفض جداً، خطأ التدريب أقل بكثير من خطأ الاختبار، تباين عالي]
</div>
<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

<div dir="rtl">
[زيادة تعقيد النموذج، إضافة المزيد من الخصائص، تدريب لمدة أطول، إجراء الضبط (regularization)، الحصول على المزيد من البيانات]
</div>
<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

<div dir="rtl">
تحليل الخطأ - تحليل الخطأ هو تحليل السبب الرئيسي للفرق في الأداء بين النماذج الحالية والنماذج المثالية.
</div>
<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

<div dir="rtl">
تحليل استئصالي (Ablative analysis) - التحليل الاستئصالي هو تحليل السبب الرئيسي للفرق في الأداء بين النماذج الحالية والنماذج المبدئية (baseline).
</div>
<br>

**44. Regression metrics**

<div dir="rtl">
مقاييس الانحدار
</div>
<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

<div dir="rtl">
[مقاييس التصنيف، مصفوفة الدقّة، الضبط (accuracy)، الدقة (precision)، الاستدعاء (recall)، درجة F1]
</div>
<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

<div dir="rtl">
[مقاييس الانحدار، مربع R، معيار معامل مالوس (Mallow's)، معيار آكياك المعلوماتي (AIC)، معيار المعلومات البايزي (BIC)]
</div>
<br>

**47. [Model selection, cross-validation, regularization]**

<div dir="rtl">
[اختيار النموذج، التحقق المتقاطع، الضبط]
</div>
<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

<div dir="rtl">
[التشخيصات، موازنة الانحياز/التباين، تحليل الخطأ/التحليل الاستئصالي]
</div>
