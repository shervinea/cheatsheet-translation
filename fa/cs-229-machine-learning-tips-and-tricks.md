**1. Machine Learning tips and tricks cheatsheet**

<div dir="rtl">
راهنمای کوتاه نکات و ترفندهای یادگیری ماشین
</div>

<br>

**2. Classification metrics**

<div dir="rtl">
معیارهای دسته‌بندی
</div>

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

<div dir="rtl">
معیارهای اساسی و مهم برای پیگیری در زمینه‌ی دسته‌بندی دوتایی و به منظور ارزیابی عملکرد مدل در زیر آمده‌اند.
</div>

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

<div dir="rtl">
ماتریس درهم‌ریختگی - از ماتریس درهم‌ریختگی برای دست یافتن به تصویری جامع‌تر در ارزیابی عملکرد مدل استفاده می‌شود. این ماتریس بصورت زیر تعریف می‌شود:
</div>

<br>

**5. [Predicted class, Actual class]**

<div dir="rtl">
[دسته پیش‌بینی‌شده، دسته واقعی]
</div>

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

<div dir="rtl">
معیارهای اصلی - معیارهای زیر معمولا برای ارزیابی عملکرد مدل‌های دسته‌بندی بکار برده می‌شوند.
</div>

<br>

**7. [Metric, Formula, Interpretation]**

<div dir="rtl">
[معیار، فرمول، تفسیر]
</div>

<br>

**8. Overall performance of model**

<div dir="rtl">
عملکرد کلی مدل
</div>

<br>

**9. How accurate the positive predictions are**

<div dir="rtl">
پیش‌بینی‌های مثبت چقدر دقیق هستند
</div>

<br>

**10. Coverage of actual positive sample**

<div dir="rtl">
پوشش نمونه‌ی مثبت واقعی 
</div>

<br>

**11. Coverage of actual negative sample**

<div dir="rtl">
پوشش نمونه‌ی منفی واقعی 
</div>

<br>

**12. Hybrid metric useful for unbalanced classes**

<div dir="rtl">
معیار ترکیبی مفید برای دسته‌های نامتوازن
</div>

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

<div dir="rtl">
ROC - منحنی عملیاتی گیرنده که تحت عنوان ROC نیز شناخته می‌شود تصویر TPR به ازای FPR و با تغییر مقادیر آستانه است. این معیارها بصورت خلاصه در جدول زیر آورده شده‌اند:
</div>

<br>

**14. [Metric, Formula, Equivalent]**

<div dir="rtl">
[معیار، فرمول، معادل]
</div>

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

<div dir="rtl">
AUC - ناحیه‌ی زیر منحنی عملیاتی گیرنده، که با AUC یا AUROC نیز شناخته می‌شود، مساحت زیر منحنی ROC که در شکل زیر نشان داده شده است: 
</div>

<br>

**16. [Actual, Predicted]**

<div dir="rtl">
[واقعی، پیش‌بینی‌شده]
</div>

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

<div dir="rtl">
معیارهای ابتدایی - با توجه به مدل وایازش f، معیارهای زیر برای ارزیابی عملکرد مدل مورد استفاده قرار می‌گیرند:
</div>

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

<div dir="rtl">
[مجموع کل مربعات، مجموع مربعات توضیح داده شده، باقی‌مانده‌ی مجموع مربعات]
</div>

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

<div dir="rtl">
ضریب تعیین - ضریب تعیین، که با $r^2$ یا $R^2$ هم نمایش داده می‌شود، معیاری برای سنجش این است که مدل به چه اندازه می‌تواند نتایج مشاهده‌شده را تکرار کند، و به صورت زیر تعریف می‌شود:
</div>

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

<div dir="rtl">
معیارهای اصلی – از معیارهای زیر معمولا برای ارزیابی عملکرد مدل‌های وایازش با در نظر گرفتن تعداد متغیرهای n که در نظر می‌گیرند، استفاده می‌شود:
</div>

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

<div dir="rtl">
که $L$ درست‌نمایی و $\hat{\sigma}^2$ تخمینی از واریانس مربوط به هر یک از پاسخ‌ها است.
</div>

<br>

**22. Model selection**

<div dir="rtl">
انتخاب مدل
</div>

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

<div dir="rtl">
واژگان - هنگام انتخاب مدل، سه بخش مختلف از داده‌ها را به صورت زیر مشخص می‌کنیم:
</div>

<br>

**24. [Training set, Validation set, Testing set]**

<div dir="rtl">
[مجموعه آموزش، مجموعه اعتبارسنجی، مجموعه آزمایش]
</div>

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

<div dir="rtl">
[مدل آموزش داده شده است، مدل ارزیابی شده است، مدل پیش‌بینی می‌کند]
</div>

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

<div dir="rtl">
[معمولا ۸۰ درصد از مجموعه داده‌ها، معمولا ۲۰ درصد از مجموعه داده‌ها]
</div>

<br>

**27. [Also called hold-out or development set, Unseen data]**

<div dir="rtl">
[این مجموعه همچنین تحت عنوان مجموعه بیرون نگه‌داشته‌شده یا توسعه نیز شناخته می شود، داده‌های دیده نشده]
</div>

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

<div dir="rtl">
بعد از اینکه مدل انتخاب شد، روی کل مجموعه داده‌ها آموزش داده می‌شود و بر روی مجموعه دادگان دیده نشده آزمایش می‌شود. این مراحل در شکل زیر آمده‌اند:
</div>

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

<div dir="rtl">
اعتبارسنج متقاطع – اعتبارسنجی متقاطع، که CV نیز نامیده می‌شود، عبارت است از روشی برای انتخاب مدلی که بیش از حد به مجموعه‌ی آموزش اولیه تکیه نمی‌کند. انواع مختلف بصورت خلاصه در جدول زیر ارائه شده‌اند:
</div>

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

<div dir="rtl">
[آموزش بر روی $k-1$ بخش دیگر و ارزیابی بر روی بخش باقی‌مانده، آموزش بر روی $n - p$ مشاهده و ارزیابی بر روی $p$ مشاهده‌ی باقی‌مانده]
</div>

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

<div dir="rtl">
[معمولا $k=5$ یا $k=10$، مورد $p=1$ تحت عنوان حذف تک‌مورد گفته می‌شود]
</div>

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

<div dir="rtl">
رایج‌ترین روش مورد استفاده، اعتبار سنجی متقاطع $k$-بخشی نامیده می‌شود که داده‌های آموزشی را به $k$ بخش تقسیم می‌کند تا مدل روی یک بخش ارزیابی شود و در عین حال مدل را روی $k-1$ بخش دیگر آموزش دهد، و این عمل را $k$ بار تکرار می‌کند. سپس میانگین خطا بر روی $k$ بخش محاسبه می‌شود که خطای اعتبارسنجی متقاطع نامیده میشود.
</div>

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

<div dir="rtl">
نظام‌بخشی – هدف از رویه‌ی نظام‌بخشی جلوگیری از بیش‌برازش به داده‌ها توسط مدل است و در نتیجه با مشکل واریانس بالا طرف است. جدول زیر خلاصه‌ای از انواع روش‌های متداول نظام‌بخشی را ارائه می‌دهد:
</div>

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

<div dir="rtl">
[ضرایب را تا ۰ کاهش می‌دهد، برای انتخاب متغیر مناسب است، ضرایب را کوچکتر می‌کند، بین انتخاب متغیر و ضرایب کوچک مصالحه می‌کند]
</div>

<br>

**35. Diagnostics**

<div dir="rtl">
عیب‌شناسی
</div>

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

<div dir="rtl">
پیش‌قدر – پیش‌قدر مدل اختلاف بین پیش‌بینی مورد انتظار و مدل صحیح است که تلاش می‌کنیم برای نمونه داده‌های داده‌شده پیش‌بینی کنیم.
</div>

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

<div dir="rtl">
واریانس - واریانس یک مدل تنوع پیش‌بینی مدل برای نمونه داده‌های داده‌شده است.
</div>

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

<div dir="rtl">
تعادل پیش‌قدر/واریانس – هر چقدر مدل ساده‌تر باشد، پیش‌قدر بیشتر خواهد بود، و هر چه مدل پیچیده‌تر باشد واریانس بیشتر خواهد شد.
</div>

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

<div dir="rtl">
[علائم، نمایش وایازش، نمایش دسته‌بندی، نمایش یادگیری عمیق، اصلاحات احتمالی]
</div>

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

<div dir="rtl">
[خطای بالای آموزش، خطای آموزش نزدیک به خطای آزمایش، پیش‌قدر زیاد، خطای آموزش کمی کمتر از خطای آزمایش، خطای آموزش بسیار کم، خطای آموزش بسیار کم‌تر از خطای آزمایش، واریانس بالا]
</div>

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

<div dir="rtl">
[مدل را پیچیده‌تر کنید، ویژگی‌های بیشتری  اضافه کنید، مدت طولانی‌تری آموزش دهید، نظام‌بخشی انجام دهید، داده‌های بیشتری گردآوری کنید]
</div>

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

<div dir="rtl">
تحلیل خطا - تحلیل خطا به بررسی علت اصلی اختلاف در عملکرد بین مدل‌های کنونی و مدل‌های صحیح می‌پردازد.
</div>

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

<div dir="rtl">
تحلیل تقطیعی - تحلیل تقطیعی به بررسی علت اصلی اختلاف بین مدل‌های کنونی و مدل‌های پایه می‌پردازد.
</div>

<br>

**44. Regression metrics**

<div dir="rtl">
معیارهای وایازش
</div>

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

<div dir="rtl">
[معیارهای دسته‌بندی، ماتریس درهم‌ریختگی، صحت، دقت، فراخوانی، امتیاز F1، ROC]
</div>

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

<div dir="rtl">
[معیارهای وایازش، $R^2$، عدد CP از Mallow، AIC، BIC]
</div>

<br>

**47. [Model selection, cross-validation, regularization]**

<div dir="rtl">
[انتخاب مدل، اعتبارسنجی متقاطع، نظام‌بخشی]
</div>

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

<div dir="rtl">
[عیب‌شناسی، تقابل تعادل پیش‌قدر/واریانس، تحلیل خطا/تقطیعی]
</div>
