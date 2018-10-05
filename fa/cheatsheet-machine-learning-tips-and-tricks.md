**1. Machine Learning tips and tricks cheatsheet**

<div dir="rtl">
راهنمای کوتاه ترفندهای یادگیری ماشینی
</div>
<br>

**2. Classification metrics**

<div dir="rtl">
طبقه‌بندی معیارها
</div>

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

<div dir="rtl">
در زمینه طبقه‌بندی دوتایی، این معیارهای اصلی هستند که برای ارزیابی عملکرد مدل مهم هستند.
</div>

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**
<div dir="rtl">
ماتریس درهم - ماتریس درهم‌ریختگی برای داشتن یک تصویر کامل‌تر در هنگام ارزیابی عملکرد یک مدل مورد استفاده قرار می‌گیرد. این تعریف به شرح زیر تعریف می‌شود:
</div>
<br>

**5. [Predicted class, Actual class]**

<div dir="rtl">
[کلاس پیش‌بینی‌شده، کلاس موجود]
</div>

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

<div dir="rtl">
معیارهای اصلی - معیارهای زیر برای ارزیابی عملکرد مدل‌های طبقه‌بندی استفاده می‌شوند .
</div>

<br>

**7. [Metric, Formula, Interpretation]**

<div dir="rtl">
[معیار, فرمول, تفسیر]
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
پوشش نمونه مثبت موجود
</div>

<br>

**11. Coverage of actual negative sample**

<div dir="rtl">
پوشش نمونه منفی موجود
</div>

<br>

**12. Hybrid metric useful for unbalanced classes**

<div dir="rtl">
معیار ترکیبی مفید برای کلاس های نامتعادل
</div>

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

<div dir="rtl">
ROC ― منحنی عملیاتی گیرنده، که به ROC نیز اشاره دارد، طرح TPR در برابر fpr با تغییر آستانه است. این معیارها در جدول زیر خلاصه می‌شوند.
</div>

<br>

**14. [Metric, Formula, Equivalent]**

<div dir="rtl">
[معیار, فرمول, تفسیر]
</div>

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

<div dir="rtl">
AUC ― این ناحیه تحت منحنی عملیاتی قرار دارد همچنین AUC یا AUROC، ناحیه‌ای است که در زیر ROC نشان داده می‌شود و در شکل زیر نشان‌داده شده‌است.
</div>

<br>

**16. [Actual, Predicted]**

<div dir="rtl">
[موجود, پیش‌بینی‌شده]
</div>

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**
<div dir="rtl">
معیارهای ابتدایی - با توجه به مدل رگرسیون f، معیارهای زیر برای ارزیابی عملکرد مدل مورد استفاده قرار می‌گیرند.
</div>
<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

<div dir="rtl">
[کل مجموع مربعات، مجموع مربعات توضیح داده شده، باقی مانده مجموع مربعات]
</div>

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

<div dir="rtl">
ضریب تعیین - ضریب تعیین، اغلب به r۲ یا R2 اشاره دارد ، معیاری برای این که چگونه نتایج مشاهده‌شده توسط مدل تکرار می‌شوند و به شرح زیر تعریف می‌شود ، ارایه می‌دهد.
</div>

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

<div dir="rtl">
معیارهای اصلی - معیارهای زیر برای ارزیابی عملکرد مدل‌های رگرسیون با در نظر گرفتن تعداد متغیرهای n که در نظر می‌گیرند به کار می‌رود.
</div>

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

<div dir="rtl">
معیارهای اصلی - معیارهای زیر برای ارزیابی عملکرد مدل‌های رگرسیون با در نظر گرفتن تعداد متغیرهای n که در نظر می‌گیرند به کار می‌رود.
</div>

<br>

**22. Model selection**

<div dir="rtl">
انتخاب مدل
</div>
<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

<div dir="rtl">
واژگان - هنگام انتخاب یک مدل، ما ۳ بخش مختلف اطلاعاتی را که به شکل زیر داریم ، متمایز می‌کنیم:
</div>

<br>

**24. [Training set, Validation set, Testing set]**

<div dir="rtl">
[بسته آموزشی, بسته ارزیابی, بسته تست]
</div>

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

<div dir="rtl">
[مدل آموزش داده می‌شود، مدل ارزیابی می‌شود، مدل پیش‌بینی می‌کند.]
</div>

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**
<div dir="rtl">
[معمولا ۸۰ درصد از مجموعه داده‌ها، معمولا ۲۰ ٪ از مجموعه داده‌ها]
</div>
<br>

**27. [Also called hold-out or development set, Unseen data]**

<div dir="rtl">
[همچنین انبار یا مجموعه توسعه، داده‌های نادیده‌گرفته، نامیده می‌شود]
</div>
<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

<div dir="rtl">
همانطور که در شکل زیر نشان داده‌شده‌است هنگامی که مدل انتخاب شد، روی کل مجموعه داده‌ها آموزش داده می‌شود و بر روی مجموعه تست  می‌شود:
</div>
<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

<div dir="rtl">
اعتبارسنجی متقابل - عبارت اعتبارسنجی  به اختصار CV، روشی است که برای انتخاب مدلی که بیش از حد بر مجموعه آموزشی اولیه تکیه نمی‌کند ، استفاده می‌شود. انواع مختلفش در جدول زیر خلاصه شده‌اند:
</div>

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

<div dir="rtl">
[آموزش روی k - ۱  چین و ارزیابی در مورد بقیه، آموزش بر روی مشاهدات n - p و ارزیابی در موارد باقیمانده p.]
</div>
<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

<div dir="rtl">
[به طور کلی k = ۵ یا ۱۰، مورد p = ۱ بیرون کشیدن یک مورد، نامیده می‌شود]
</div>

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

<div dir="rtl">
رایج‌ترین روش مورد استفاده، اعتبار سنجی متقابل k-چین نامیده می‌شود و داده‌های آموزشی را به K-چین تقسیم می‌کند تا مدل را در یک چین اعتبار دهد در حالی که مدل را روی k - ۱ دیگر و تمام این k بار آموزش می‌دهد. سپس خطا بر روی k چین میانگین گرفته می‌شود و خطای اعتبارسنجی نامیده می‌شود.
</div>

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**
<div dir="rtl">
تنظیم‌کننده - روش تنظیم‌کننده با هدف اجتناب از مدل برای تجزیه و تحلیل داده‌ها و در نتیجه با مسایل واریانس بالا، سر و کار دارد. جدول زیر خلاصهٔ انواع مختلفی از تکنیک‌های متداول که مورد استفاده قرار می‌گیرند، است.
</div>
<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

<div dir="rtl">
کاهش ضریب به ۰، برای انتخاب متغیر خوب است، باعث می‌شود که ضرایب کوچک‌تر شوند، تقابل بین انتخاب متغیر و ضرایب کوچک
</div>
<br>

**35. Diagnostics**

<div dir="rtl">
عیب‌شناسی
</div>

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

<div dir="rtl">
سوگیری - جهت‌گیری یک مدل تفاوت بین پیش‌بینی مورد انتظار و مدل صحیح است که ما سعی می‌کنیم برای نقاط داده معین پیش‌بینی کنیم.
</div>

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

<div dir="rtl">
واریانس - واریانس یک مدل تغییرپذیری پیش‌بینی مدل برای نقاط داده داده شده‌است.
</div>

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

<div dir="rtl">
تقابل سوگیری/ واریانس - مدل ساده‌تر، هرچه گرایش بیشتر باشد ، و مدل پیچیده‌تر می‌شود واریانس بیشتر می‌شود.
</div>

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

<div dir="rtl">
[علایم، تصویر رگرسیون، تصویر دسته‌بندی، تصویر یادگیری عمیق، روش‌های ممکن]
</div>

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

<div dir="rtl">
[خطای آموزش بالا، خطای آموزش نزدیک به خطای تست، سوگیری بالا، خطای آموزش کمی پایین‌تر از خطای تست، خطای آموزش بسیار پایین، خطای آموزش کم‌تر از خطای تست، واریانس بالا.]
</div>

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

<div dir="rtl">
[مدل را پیچیده‌تر کنید، مشخصه‌های بیشتری را اضافه کنید، تنظیم را طولانی‌تر کنید، تنظیمات را انجام دهید، اطلاعات بیشتری کسب کنید.]
</div>

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

<div dir="rtl">
تحلیل خطا - تحلیل خطا در تجزیه و تحلیل علت ریشه‌ای تفاوت در عملکرد بین مدل‌های فعلی و مدل‌های کامل
</div>

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

<div dir="rtl">
آنالیز مفعول - آنالیر مغعول علت اصلی تفاوت در عملکرد بین مدل‌های فعلی و مبنا را بررسی می‌کند.
</div>

<br>

**44. Regression metrics**

<div dir="rtl">
معیارهای رگرسیون
</div>
<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

<div dir="rtl">
[معیارهای دسته‌بندی، ماتریس درهم‌ریختگی، دقت، دقت، به یاد آوردن ، امتیاز F۱، ROC]
</div>

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

<div dir="rtl">
[معیارهای رگرسیون، R squared، Mallow's CP، AIC، BIC]
</div>

<br>

**47. [Model selection, cross-validation, regularization]**

<div dir="rtl">
[انتخاب مدل، اعتبار سنجی متقابل، تنظیمات]
</div>

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

<div dir="rtl">
[تشخیص، سوگیری / واریانس واریانس، تحلیل خطای / تحلیل مفعول]
</div>
