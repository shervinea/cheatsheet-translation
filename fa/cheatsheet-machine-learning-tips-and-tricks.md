**1. Machine Learning tips and tricks cheatsheet**

1. برگه ی یادداشت  نکات و ترفندهای یادگیری ماشین

<br>

**2. Classification metrics**

2. معیارهای دسته بندی

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

3. در اینجا معیارهای اساسی و مهم برای پیگیری در زمینه ی طبقه بندی دودویی و به منظور ارزیابی عملکرد مدل ارائه میشوند.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

4. ماتریس درهم ریختگی – از ماتریس درهم ریختگی برای دست یافتن به تصویری جامعتر در ارزیابی عملکرد مدل استفاده میشود. این ماتریس بصورت زیر تعریف میشود:

<br>

**5. [Predicted class, Actual class]**

5. [کلاس پیش¬بینی¬شده، کلاس واقعی]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

6. معیارهای اصلی - معیارهای زیر معمولا برای ارزیابی عملکرد مدلهای طبقه بندی بکار برده میشوند.

<br>

**7. [Metric, Formula, Interpretation]**

7. [معیار، فرمول، تفسیر]

<br>

**8. Overall performance of model**

8. عملکرد کلی مدل

<br>

**9. How accurate the positive predictions are**

9. میزان دقت پیش بینی های مثبت عبارتست از:

<br>

**10. Coverage of actual positive sample**

10.  پوشش نمونه ی مثبت واقعی 

<br>

**11. Coverage of actual negative sample**

11. پوشش نمونه ی منفی واقعی 

<br>

**12. Hybrid metric useful for unbalanced classes**

12. معیار ترکیبی، موثر در کلاس های نامتوازن

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

13. ROC - منحنی مشخصه عملکرد سیستم که تحت عنوان ROC نیز شناخته میشود تصویر TPR به ازای FPR و با تغییر مقادیر آستانه است. این معیارها بصورت خلاصه در جدول زیر آورده شده اند:

<br>

**14. [Metric, Formula, Equivalent]**

14. [معیار، فرمول، معادل]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

15. AUC – مساحت زير منحني که تحت عنوان AUC یا AUROC نیز شناخته میشود عبارتست از سطح زیر ROC که در شکل زیر نشان داده شده است:

<br>

**16. [Actual, Predicted]**

16. [واقعی، پیش¬بینی¬شده]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

17. معیارهای اساسی - با توجه به مدل رگرسیون f، معمولا از معیارهای زیر برای ارزیابی عملکرد مدل استفاده میشود:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

18. [مجموع کل مربعات، مجموع مربعات توضیح داده شده، مجموع مربعات باقیمانده]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

19. ضریب تشخیص - ضریب تشخیص که اغلب با R2 یا r2 نشان داده میشود معیاری از تکرار مناسب نتایج مشاهده شده توسط مدل را ارائه میدهد که بصورت زیر تعریف میشود:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

20. معیارهای اصلی – از معیارهای زیر معمولا برای ارزیابی عملکرد مدلهای رگرسیون با توجه به تعداد متغیرهای n ی که در نظر میگیرند،استفاده میشود:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

21. که L نشان دهنده درست نمایی و σ2 تخمینی از واریانس مربوط به هر یک از پاسخ ها است.

<br>

**22. Model selection**

22. انتخاب مدل

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

23. واژگان - هنگام انتخاب مدل، سه بخش مختلف از دادهها را به صورت زیر مشخص میکنیم:

<br>

**24. [Training set, Validation set, Testing set]**

24. [مجموعه آموزشی، مجموعه اعتبارسنجی، مجموعه آزمایش]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

25. [مدل آموزش داده شده، مدل ارزیابی شده، مدل پیش بینی ها را ارائه میدهد]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

26. [معمولا 80% از پایگاه داده، معمولا 20% از پایگاه داده ]

<br>

**27. [Also called hold-out or development set, Unseen data]**

27. [این مجموعه همچنین تحت عنوان مجموعه بیرون نگهدار(ارزیابی) یا توسعه نیز شناخته می شود، داده های دیده نشده]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

28. هنگامی که مدل انتخاب شد،آن را بر روی کل مجموعه داده ها آموزش داده و بر روی مجموعه داده های دیده نشده آزمایش میشود. این داده ها در شکل زیر نشان داده شده اند:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

29. اعتبارسنجی متقاطع - اعتبارسنجی متقاطع که بصورت CV نیز بیان میشود عبارتست از روشی برای انتخاب مدلی که بیش از حد به مجموعه آموزش اولیه تکیه نمیکند. انواع مختلف بصورت خلاصه در جدول زیر ارائه شده اند:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

30. [آموزش در K-1 برابر و ارزیابی روی موارد باقیمانده، آموزش در N-P مشاهده و ارزیابی روی p مورد باقیمانده]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

31. [معمولا k=5 یا k=10، مورد p=1 تحت عنوان حذف یک [مورد] شناخته میشود]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

32. متداولترین روش عبارتست از اعتبار سنجی متقاطع k برابر [بخشی] که داده های آموزشی را به k بخش تقسیم میکند تا مدل را روی یک بخش ثابت کند و در عین حال مدل را روی k-1 بخش دیگر آموزش دهد که همه ی آنها k دفعه انجام میشود. سپس میانگین خطا بر روی k بخش بدست آورده شده و خطای اعتبارسنجی متقاطع نامیده میشود.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

33. تنظیم – هدف از روش تنظیم عبارتست از جلوگیری از بیش برارزش شدن مدل توسط داده ها و در نتیجه فائق آمدن با مشکل واریانس بالا.جدول زیر خلاصه ای از انواع روشهای متداول در تنظیم را ارائه میدهد.

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

34. [ضرایب را تا 0 کاهش میدهد، برای انتخاب متغیر مناسب است، ضرایب را کوچکتر میکند، بین انتخاب متغیر و ضرایب کوچک مصالحه میکند].

<br>

**35. Diagnostics**

35. تشخیص ها

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

36. انحراف– انحراف مدل عبارتست از اختلاف بین پیش بینی مورد انتظار و مدل صحیح که تلاش میکنیم برای نقاط داده معینی پیش بینی کنیم.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

37. واریانس - واریانس مدل عبارتست از تغییرپذیری پیش بینی مدل برای نقاط داده معین

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

38. تعادل بین انحراف/واریانس – هر چقدر مدل ساده تر باشد، انحراف بیشتر خواهد بود و هرچقدر مدل پیچیده تر باشد واریانس بیشتر خواهد شد.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

39. [علائم، نمایش رگرسیون، نمایش طبقه بندی، نمایش یادگیری عمیق، اصلاحات احتمالی]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

40. [خطای بالای آموزش، خطای آموزش نزدیک به خطای آزمایش، انحراف زیاد، خطای آموزش کمی کمتر از خطای آزمایش، خطای آموزش بسیار کم، خطای آموزش بسیار پایینتر از خطای آزمایش، واریانس بالا]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

41. [مدل پیچیدگی، اضافه کردن ویژگی های بیشتر، آموزش طولانی تر، انجام تنظیم، جمع آوری داده های بیشتر]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

42. تجزیه و تحلیل خطا - تجزیه و تحلیل خطا به آنالیز علت اصلی اختلاف در عملکرد بین مدلهای کنونی و کامل میپردازد.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

43. تجزیه و تحلیل آبلیتیو - تجزیه و تحلیل آبلیتیو به آنالیز علت اصلی اختلاف بین مدلهای کنونی و پایه میپردازد.

<br>

**44. Regression metrics**

44.   معیارهای رگرسیون
