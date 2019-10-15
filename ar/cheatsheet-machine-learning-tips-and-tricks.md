**1. Machine Learning tips and tricks cheatsheet**

مرجع سريع لنصائح وحيل تعلّم الآلة

<br>

**2. Classification metrics**

مقاييس التصنيف

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

في سياق التصنيف الثنائي، هذه المقاييس (metrics) المهمة التي يجدر مراقبتها من أجل تقييم آداء النموذج.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

مصفوفة الدقّة (confusion matrix) - تستخدم مصفوفة الدقّة لأخذ تصور شامل عند تقييم أداء النموذج. وهي تعرّف كالتالي: 

<br>

**5. [Predicted class, Actual class]**

[التصنيف المتوقع، التصنيف الفعلي]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

المقاييس الأساسية - المقاييس التالية تستخدم في العادة لتقييم أداء نماذج التصنيف:

<br>

**7. [Metric, Formula, Interpretation]**

[المقياس، المعادلة، التفسير]

<br>

**8. Overall performance of model**

الأداء العام للنموذج

<br>

**9. How accurate the positive predictions are**

دقّة التوقعات الإيجابية (positive)

<br>

**10. Coverage of actual positive sample**

تغطية عينات التوقعات الإيجابية الفعلية

<br>

**11. Coverage of actual negative sample**

تغطية عينات التوقعات السلبية الفعلية

<br>

**12. Hybrid metric useful for unbalanced classes**

مقياس هجين مفيد للأصناف غير المتوازنة (unbalanced)

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

منحنى دقّة الأداء (ROC) - منحنى دقّة الآداء، ويطلق عليه ROC، هو رسمة لمعدل التصنيفات الإيجابية الصحيحة (TPR) مقابل معدل التصنيفات الإيجابية الخاطئة (FPR) باستخدام قيم حد (threshold) متغيرة. هذه المقاييس ملخصة في الجدول التالي:
<br>

**14. [Metric, Formula, Equivalent]**

[المقياس، المعادلة، مرادف]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

المساحة تحت منحنى دقة الأداء (المساحة تحت المنحنى) (AUC) - المساحة تحت منحنى دقة الأداء (المساحة تحت المنحنى)، ويطلق عليها  AUC أو AUROC، هي المساحة تحت ROC كما هو موضح في الرسمة التالية:

<br>

**16. [Actual, Predicted]**

[الفعلي، المتوقع]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

المقاييس الأساسية - إذا كان لدينا نموذج الانحدار f، فإن المقاييس التالية غالباً ما تستخدم لتقييم أداء النموذج:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

[المجموع الكلي للمربعات، مجموع المربعات المُفسَّر، مجموع المربعات المتبقي]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

مُعامل التحديد (Coefficient of determination) - مُعامل التحديد، وغالباً يرمز له بـ R2 أو r2، يعطي قياس لمدى مطابقة النموذج للنتائج الملحوظة، ويعرف كما يلي:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

المقاييس الرئيسية - المقاييس التالية تستخدم غالباً لتقييم أداء نماذج الانحدار، وذلك بأن يتم الأخذ في الحسبان عدد المتغيرات n المستخدمة فيها:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

حيث L هو الأرجحية، و ˆσ2 تقدير التباين الخاص بكل نتيجة.

<br>

**22. Model selection**

اختيار النموذج

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

مفردات - عند اختيار النموذج، نفرق بين 3 أجزاء من البيانات التي لدينا كالتالي:

<br>

**24. [Training set, Validation set, Testing set]**

[مجموعة تدريب، مجموعة تحقق، مجموعة اختبار]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

[يتم تدريب النموذج، يتم تقييم النموذج، النموذج يعطي التوقعات]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

[غالباً 80% من مجموعة البيانات، غالباً 20% من مجموعة البيانات]

<br>

**27. [Also called hold-out or development set, Unseen data]**

[يطلق عليها كذلك المجموعة المُجنّبة أو مجموعة التطوير، بيانات لم يسبق رؤيتها من قبل]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

بمجرد اختيار النموذج، يتم تدريبه على مجموعة البيانات بالكامل ثم يتم اختباره على مجموعة اختبار لم يسبق رؤيتها من قبل. كما هو موضح في الشكل التالي:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

التحقق المتقاطع (Cross-validation) - التحقق المتقاطع، وكذلك يختصر بـ CV، هو طريقة تستخدم لاختيار نموذج بحيث لا يعتمد بشكل كبير على مجموعة بيانات التدريب المبدأية. أنواع التحقق المتقاطع المختلفة ملخصة في الجدول التالي:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

[التدريب على k-1 جزء والتقييم باستخدام الجزء الباقي، التدريب على n−p عينة والتقييم باستخدام الـ p عينات المتبقية]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

[بشكل غالب k=5 أو 10، الحالة p=1 يطلق عليها الإبقاء على واحد (leave-one-out)]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

الطريقة الأكثر استخداماً يطلق عليها التحقق المتقاطع س جزء/أجزاء (k-fold)، ويتم فيها تقسيم البيانات إلى k جزء، بحيث يتم تدريب النموذج باستخدام k−1 والتحقق باستخدام الجزء المتبقي، ويتم تكرار ذلك k مرة. يتم بعد ذلك حساب معدل الأخطاء في الأجزاء k ويسمى خطأ التحقق المتقاطع.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

ضبط (Regularization) - عمليه الضبط تهدف إلى تفادي فرط التخصيص (overfit) للنموذج، وهو بذلك يتعامل مع مشاكل التباين العالي. الجدول التالي يلخص أنواع وطرق الضبط الأكثر استخداماً:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

[يقلص المُعاملات إلى 0، جيد لاختيار المتغيرات، يجعل المُعاملات أصغر، المفاضلة بين اختيار المتغيرات والمُعاملات الصغيرة]

<br>

**35. Diagnostics**

التشخيصات

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

الانحياز (Bias) - الانحياز للنموذج هو الفرق بين التنبؤ المتوقع والنموذج الحقيقي الذي نحاول تنبؤه للبيانات المعطاة.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

التباين (Variance) - تباين النموذج هو مقدار التغير في تنبؤ النموذج لنقاط البيانات المعطاة.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

موازنة الانحياز/التباين (Bias/variance tradeoff) - كلما زادت بساطة النموذج، زاد الانحياز، وكلما زاد تعقيد النموذج، زاد التباين.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

[الأعراض، توضيح الانحدار، توضيح التصنيف، توضيح التعلم العميق، العلاجات الممكنة]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

[خطأ التدريب عالي، خطأ التدريب قريب من خطأ الاختبار، انحياز عالي، خطأ التدريب أقل بقليل من خطأ الاختبار، خطأ التدريب منخفض جداً، خطأ التدريب أقل بكثير من خطأ الاختبار، تباين عالي]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

[زيادة تعقيد النموذج، إضافة المزيد من الخصائص، تدريب لمدة أطول، إجراء الضبط (regularization)، الحصول على المزيد من البيانات]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

تحليل الخطأ - تحليل الخطأ هو تحليل السبب الرئيسي للفرق في الأداء بين النماذج الحالية والنماذج المثالية.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

تحليل استئصالي (Ablative analysis) - التحليل الاستئصالي هو تحليل السبب الرئيسي للفرق في الأداء بين النماذج الحالية والنماذج المبدئية (baseline).

<br>

**44. Regression metrics**

مقاييس الانحدار

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

[مقاييس التصنيف، مصفوفة الدقّة، الضبط (accuracy)، الدقة (precision)، الاستدعاء (recall)، درجة F1]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

[مقاييس الانحدار، مربع R، معيار معامل مالوس (Mallow's)، معيار آكياك المعلوماتي (AIC)، معيار المعلومات البايزي (BIC)]

<br>

**47. [Model selection, cross-validation, regularization]**

[اختيار النموذج، التحقق المتقاطع، الضبط]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

[التشخيصات، موازنة الانحياز/التباين، تحليل الخطأ/التحليل الاستئصالي]
