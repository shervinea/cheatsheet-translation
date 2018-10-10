**1. Supervised Learning cheatsheet**

<div dir="rtl">
راهنمای کوتاه یاگیری با نظارت
</div>

<br>

**2. Introduction to Supervised Learning**

<div dir="rtl">
مبانی یادگیری با نظارت
</div>

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

<div dir="rtl">
با در نظر گرفتن مجموعه‌ای از نمونه‌های داده‌ی $\{x^{(i)}, \dots, x^{(m)} \}$ متناظر با مجموعه‌ی خروجی‌های $\{y^{(i)}, \dots, y^{(m)} \}$، هدف ساخت دسته‌بندی است که پیش‌بینی $y$ از روی $x$ را یاد می‌گیرد.
</div>

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

<div dir="rtl">
انواع پیش‌بینی - انواع مختلف مدل‌های پیش‌بینی کننده در جدول زیر به اختصار آمده‌اند:
</div>

<br>

**5. [Regression, Classifier, Outcome, Examples]**

<div dir="rtl">
[وایازش (رگرسیون)، دسته‌بندی، خروجی، نمونه‌ها]
</div>

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

<div dir="rtl">
[اعداد پیوسته، دسته، وایازش خطی، وایازش لجستیک، ماشین بردار پشتیبان، بیز ساده]
</div>

<br>

**7. Type of model ― The different models are summed up in the table below:**

<div dir="rtl">
نوع مدل ـ انواع مختلف مدل‌ها در جدول زیر به اختصار آمده‌اند.
</div>

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

<div dir="rtl">
[مدل متمایزکننده، مدل مولد، هدف، چیزی که یاد گرفته می‌شود، تصویر، نمونه‌ها]
</div>

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

<div dir="rtl">
[تخمین مستقیم $P(y|x)$، تخمین $P(x|y)$ و سپس نتیجه‌گیریِ $P(y|x)$، مرز تصمیم‌گیری، توزیع احتمال داده‌ها، وایازش‌ها، ماشین‌های بردار پشتیبان، GDA، بِیز ساده]
</div>

<br>

**10. Notations and general concepts**

<div dir="rtl">
نماد‌ها و مفاهیم کلی
</div>

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

<div dir="rtl">
فرضیه - فرضیه که با $h_\theta$ نمایش داده می‌شود، همان مدلی است که ما انتخاب می‌کنیم. به ازای هر نمونه داده ورودی $x^{(i)}$، حاصل پیش‌یینی مدل $h_\theta(x^{(i)})$ می‌باشد.
</div>

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

<div dir="rtl">
تابع خطا - تابع خطا تابعی است به صورت $\mathcal{L}:(z,y) \in \mathbb{R} \times \mathcal{Y} \longmapsto \mathcal{L}(z,y) \in \mathbb{R}$ که به عنوان ورودی مقدار پیش‌بینی‌شده‌ی $z$ متناظر با مقدار داده‌ی حقیقی $y$ را می‌گیرد و اختلاف این دو را خروجی می‌دهد. توابع خطای معمول در جدول زیر آمده‌اند:
</div>

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

<div dir="rtl">
[خطای کمترین مربعات، خطای لجستیک، خطای Hinge، آنتروپی متقاطع]
</div>

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

<div dir="rtl">
[وایازش خطی، وایازش لجستیک، ماشین بردار پشتیبان، شبکه‌ی عصبی]
</div>

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

<div dir="rtl">
تابع هزینه - تابع هزینه‌ی $J$ معمولاً برای ارزیابی عملکرد یک مدل استفاده می‌شود و با توجه به تابع خطای $L$ به صورت زیر تعریف می‌شود:
</div>

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

<div dir="rtl">
گرادیان کاهشی - با نمایش نرخ یادگیری به صورت $\alpha \in \mathbb{R}$، رویه‌ی به‌روزرسانی گرادیان کاهشی که با نرخ‌یادگیری و تابع هزینه‌ی $J$ بیان می‌شود به شرح زیر است:
</div>

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

<div dir="rtl">
نکته: گرادیان کاهشی تصادفی (SGD) عوامل را بر اساس تک‌تک نمونه‌های آموزش به‌روزرسانی می‌کند، در حالی که گرادیان کاهشی دسته‌ای این کار را بر اساس دسته‌ای از نمونه‌های آموزش انجام می‌دهد.
</div>

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

<div dir="rtl">
درست‌نمایی - از مقدار درست‌نمایی یک مدل $L(\theta)$ با پارامتر‌های $\theta$ در پیدا کردن عوامل بهینه $\theta$ ‌از طریق روش بیشینه‌سازی درست‌نمایی مدل استفاده می‌شود. البته در عمل از لگاریتم درست‌نمایی $\ell(\theta) = \log(L(\theta))$ که به‌روزرسانی آن ساده‌تر است استفاده می‌شود. داریم::
</div>

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

<div dir="rtl">
الگوریتم نیوتن ― الگوریتم نیوتن یک روش عددی است که $\theta$ را به گونه‌ای پیدا می‌کند که  $\ell'(\theta)=0$ باشد. رویه‌ی به‌روزرسانی آن به صورت زیر است:
</div>

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

<div dir="rtl">
نکته: تعمیم چندبُعدی این روش، که به روش نیوتون-رافسون معروف است، قانون به‌روزرسانی زیر را دارد:    
</div>

<br>

**21. Linear models**

<div dir="rtl">
مدل‌های خطی
</div>

<br>

**22. Linear regression**

<div dir="rtl">
وایازش خطی
</div>

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

<div dir="rtl">
در این‌جا فرض می‌کنیم $y|x;\theta\sim\mathcal{N}(\mu,\sigma^2)$ 
</div>

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

<div dir="rtl">
معادلات نرمال ― اگر $X$ یک ماتریس باشد، مقداری از $\theta$ که تابع هزینه را کمینه می‌کند یک راه‌حل به فرم بسته دارد به طوری که:
</div>

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

<div dir="rtl">
الگوریتم LMS - با نمایش نرخ یادگیری با $\alpha$، رویه‌ی به‌روزرسانی الگوریتم کمینه‌ی میانگین مربعات (LMS) برای یک مجموعه‌ی آموزش با $m$ نمونه داده، که به رویه‌ی به‌روزرسانی Widrow-Hoff نیز معروف است، به صورت زیر خواهد بود:
</div>

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

<div dir="rtl">
نکته: این رویه‌ی به‌روزرسانی، حالت خاصی از الگوریتم گرادیان کاهشی است.
</div>

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

<div dir="rtl">
LWR ― وایازش محلی‌وزن‌دار یا LWR نوعی دیگر از انواع وایازش‌های خطی است که در محاسبه‌ی تابع هزینه‌ی خود هر کدام از نمونه‌های آموزش را وزن $w^{(i)}(x)$ می‌دهد، که این وزن با عامل $\tau \in \mathbb{R}$ به شکل زیر تعریف می‌شود:
</div>

<br>

**28. Classification and logistic regression**

<div dir="rtl">
دسته‌بندی و وایازش لجستیک
</div>

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

<div dir="rtl">
تابع سیگموئید ― تابع سیگموئید $g$ که به تابع لجستیک هم معروف است به صورت زیر تعریف می‌شود:
</div>

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

<div dir="rtl">
وایازش لجستیک ― فرض می‌کنیم که $y|x; \theta \sim \textrm{Bernoulli}(\phi)$. داریم: 
</div>

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

<div dir="rtl">
نکته: هیچ راه‌حل بسته‌ای برای وایازش لجستیک وجود ندارد. 
</div>

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

<div dir="rtl">
وایازش Softmax ― وایازش Softmax یا وایازش چنددسته‌ای، در مواقعی که بیش از ۲ کلاس خروجی داریم برای تعمیم وایازش لجستیک استفاده می‌شود. طبق قرارداد داریم $\theta_K=0$. در نتیجه عامل برنولی $\psi_i$ برای هر کلاس $i$ به صورت زیر خواهد بود:
</div>

<br>

**33. Generalized Linear Models**

<div dir="rtl">
مدل‌های خطی تعمیم‌یافته
</div>

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

<div dir="rtl">
خانواده‌ی نمایی - به گروهی از توزیع‌ها خانواده‌ی نمایی گوییم اگر بتوان آن‌ها را با استفاده از عامل طبیعی $\eta$، که معمولاً عامل متعارف یا تابع پیوند نیز گفته می‌شود، آماره‌ی کافی $T(y)$، و تابع دیواره‌بندی لگاریتمی $a(\eta)$ به صورت زیر نوشت:
</div>

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

<div dir="rtl">
نکته: معمولاً داریم $T(y)=y$. هم‌چنین می‌توان به $\exp(-a(\eta))$ به عنوان یک عامل نرمال‌کننده نگاه کرد که باعث می‌شود جمع احتمال‌ها حتماً برابر با یک شود.
</div>

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

<div dir="rtl">
رایج‌ترین توزیع‌های نمایی در جدول زیر به اختصار آمده‌اند:
</div>

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

<div dir="rtl">
[توزیع، برنولی، گاوسی، پواسون، هندسی]
</div>

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

<div dir="rtl">
فرضیه‌های مدل‌های خطی تعمیم‌یافته - مدل‌های خطی تعمیم‌یافته به دنبال پیش‌بینی متغیر تصادفی $y$ به عنوان تابعی از $x \in \mathbb{R}^{n + 1}$ هستند و بر سه فرض زیر استوارند:
</div>

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

<div dir="rtl">
نکته: کمینه‌ی مربعات و وایازش لجستیک حالت‌های خاصی از مدل‌های خطی تعمیم‌یافته هستند.
</div>

<br>

**40. Support Vector Machines**

<div dir="rtl">
ماشین‌های بردار پشتیبان
</div>

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

<div dir="rtl">
هدف ماشین‌های بردار پشتیبان پیدا کردن خطی هست که حداقل فاصله تا خط را بیشینه می‌کند. 
</div>

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

<div dir="rtl">
دسته‌بند حاشیه‌ی بهینه - دسته‌بند حاشیه‌ی بهینه‌ی $h$ به گونه‌ای است که:
</div>

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

<div dir="rtl">
که $(w, b) \in \mathbb{R}^n \times \mathbb{R}$ راه‌حلی برای مساله‌ی بهینه‌سازی زیر باشد:
</div>

<br>

**44. such that**

<div dir="rtl">
به طوری که
</div>

<br>

**45. support vectors**

<div dir="rtl">
بردارهای پشتیبان
</div>

<br>

**46. Remark: the line is defined as wTx−b=0.**

<div dir="rtl">
نکته:  در این‌جا خط با $w^Tx-b=0$ تعریف شده است.
</div>

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

<div dir="rtl">
خطای Hinge ― در ماشین‌های بردار پشتیبان از تابع خطای Hinge استفاده می‌شود و تعریف آن به صورت زیر است:
</div>

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

<div dir="rtl">
هسته - برای هر تابع نگاشت ویژگی‌های $\phi$، هسته‌ی $K$ به صورت زیر تعریف می‌شود:
</div>

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

<div dir="rtl">
در عمل، به هسته‌ی $K$ که به صورت $K(x,z)= \exp \left(-\frac{\|x-z\|^2}{2\sigma^2}\right)$ تعریف شده باشد، هسته‌ی گاوسی می‌گوییم. این نوع هسته یکی از هسته‌های پراستفاده محسوب می‌شود.
</div>

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

<div dir="rtl">
[جداپذیری غیر خطی، به کارگیری نگاشت هسته، مرز تصمیم در فضای اصلی ]
</div>

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

<div dir="rtl">
نکته: می‌گوییم برای محاسبه‌ی تابع هزینه از «حقه‌ی هسته» استفاده می‌شود چرا که در واقع برای محاسبه‌ی آن، نیازی به دانستن دقیق نگاشت $\phi$ که بیشتر مواقع هم بسیار پیچیده‌ست، نداریم؛ تنها دانستن مقادیر $K(x,z)$ کافیست.
</div>

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

<div dir="rtl">
لاگرانژی - لاگرانژی $\mathcal{L}(w,b)$ به صورت زیر تعریف می‌کنیم:
</div>

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

<div dir="rtl">
نکته: به ضرایب $\beta_i$ ضرایب لاگرانژ هم می‌گوییم.
</div>

<br>

**54. Generative Learning**

<div dir="rtl">
یادگیری مولِد
</div>

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

<div dir="rtl">
یک مدل مولد ابتدا با تخمین زدن $P(x|y)$ سعی می‌کند یاد بگیرد چگونه می‌توان داده را تولید کرد، سپس با استفاده از $P(x|y)$ و هم‌چنین قضیه‌ی بِیز، $P(y|x)$ را تخمین می‌زند.
</div>

<br>

**56. Gaussian Discriminant Analysis**

<div dir="rtl">
تحلیل متمایزکننده‌ی گاوسی
</div>

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

<div dir="rtl">
فرضیات - در تحلیل متمایزکننده‌ی گاوسی فرض می‌کنیم $y$ و $x|y = 0$ و $x|y = 1$ به طوری که:
</div>

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

<div dir="rtl">
تخمین - جدول زیر تخمین‌هایی که هنگام بیشینه‌کردن تابع درست‌نمایی به آن می‌رسیم را به اختصار آورده‌است:
</div>

<br>

**59. Naive Bayes**

<div dir="rtl">
دسته‌بند بِیز ساده
</div>

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

<div dir="rtl">
فرض - مدل بِیز ساده فرض می‌کند تمام خصوصیات هر نمونه‌ی داده از هم‌دیگر مستقل است. 
</div>

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

<div dir="rtl">
راه‌حل‌ها - بیشنه‌کردن لگاریتم درست‌نمایی به پاسخ‌های زیر می‌رسد، که $k\in\{0,1\},l\in[\![1,L]\!]$
</div>

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

<div dir="rtl">
نکته: دسته‌بند بِیز ساده در مساله‌‌های دسته‌بندی متن و تشخیص هرزنامه به صورت گسترده استفاده می‌شود.
</div>

<br>

**63. Tree-based and ensemble methods**

<div dir="rtl">
روش‌های مبتنی بر درخت و گروه
</div>

<br>

**64. These methods can be used for both regression and classification problems.**

<div dir="rtl">
این روش‌ها هم در مسائل وایازش و هم در مسائل دسته‌بندی می‌توانند استفاده شوند.
</div>

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

<div dir="rtl">
CART - درخت‌های وایازش و دسته‌بندی، عموما با نام درخت‌های تصمیم‌گیری شناخته می‌شوند. می‌توان آن‌ها را به صورت درخت‌هایی دودویی نمایش داد. مزیت ‌آن‌ها قابل تفسیر بودنشان است.
</div>

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

<div dir="rtl">
جنگل تصادفی - یک تکنیک مبتی بر درخت است، که تعداد زیادی درخت تصمیم‌گیری که روی مجموعه‌هایی تصادفی از خصوصیات ساخته‌شده‌اند، را به کار می‌گیرد. روش جنگل تصادفی برخلاف درخت تصمیم‌گیری ساده، بسیار غیر قابل تفسیر است البته عمکرد عموماً خوب آن باعث شده است به الگوریتم محبوبی تبدیل شود.
</div>

<br>

**67. Remark: random forests are a type of ensemble methods.**

<div dir="rtl">
نکته: جنگل تصادفی یکی از انواع «روش‌های گروهی» است.
</div>

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

<div dir="rtl">
ترقی‌دادن - ایده‌ی اصلی روش‌های ترقی‌دادن ترکیب چند مدل ضعیف و ساخت یک مدل قوی از آن‌هاست. انواع اصلی آن به صورت خلاصه در جدول زیر آمده‌اند:
</div>

<br>

**69. [Adaptive boosting, Gradient boosting]**

<div dir="rtl">
[ترقی‌دادن سازگارشونده، ترقی‌دادن گرادیانی]
</div>

<br>

**70. High weights are put on errors to improve at the next boosting step**

<div dir="rtl">
برای خطاها وزن‌ بالایی در نظر می‌گیرد تا در مرحله‌ی بعدیِ ترقی‌دادن، مدل بهبود یابد.
</div>

<br>

**71. Weak learners trained on remaining errors**

<div dir="rtl">
چند مدل ضعیف روی باقی خطاها آموزش می‌یابند.
</div>

<br>

**72. Other non-parametric approaches**

<div dir="rtl">
سایر رویکرد‌های غیر عاملی
</div>

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

<div dir="rtl">
k-همسایه‌ی نزدیک - الگوریتم k-همسایه‌ی نزدیک که عموماً با k-NN نیز شناخته می‌شود، یک الگوریتم غیرعاملی است که پاسخ مدل به هر نمونه داده از روی k همسایه‌ی آن در مجموعه دادگان آموزش تعیین می‌شود. این الگوریتم هم در دسته‌بندی و هم در وایازش استفاده می‌شود.
</div>

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

<div dir="rtl">
نکته: هرچه پارامتر k برزرگ‌تر باشد پیش‌قدر مدل بیشتر خواهد بود، و هر چه کوچکتر باشد واریانس مدل بیشتر خواهد شد.
</div>

<br>

**75. Learning Theory**

<div dir="rtl">
نظریه یادگیری
</div>

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

<div dir="rtl">
کران اجتماع ― اگر $A_1, \dots, A_k$، $k$ عدد رخداد باشد، داریم:
</div>

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

<div dir="rtl">
نامساوی هوفدینگ ― اگر $Z_1, \dots, Z_m$، $m$ عدد متغیر تصادفی مستقل با توزیع یکسان و  نمونه‌برداری‌شده از توزیع برنولی با پارامتر $\phi$ باشند و هم‌چنین $\widehat{\phi}$ میانگین آن‌ها و $\gamma>0$ ثابت باشد، داریم:
</div>

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

<div dir="rtl">
نکته: این نامساوی به کران چرنوف نیز معروف است.
</div>

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

<div dir="rtl">
خطای آموزش ― به ازای هر دسته‌بند $h$، خطای آموزش $\widehat{\epsilon}(h)$ (یا همان خطای تجربی)، به صورت زیر تعریف می‌شود:
</div>

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

<div dir="rtl">
احتمالاً تقریباً درست (PAC)  ― چارچوبی است که در ذیل آن نتایج متعددی در نظریه یادگیری اثبات شده است و فرض‌های زیر را در بر دارد:
</div>

<br>

**81: the training and testing sets follow the same distribution **

<div dir="rtl">
مجموعه‌ی آموزش و مجموعه‌ی آزمایش از یک توزیع هستند. 
</div>

<br>

**82. the training examples are drawn independently**

<div dir="rtl">
نمونه‌های آموزشی مستقل از یکدیگر انتخاب شده‌اند.
</div>

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

<div dir="rtl">
خرد شدن ― برای مجموعه‌ی $S=\{x^{(1)},...,x^{(d)}\}$ و مجموعه‌ای از دسته‌بندهای $\mathcal{H}$ می‌گوییم، $\mathcal{H}$ مجموعه‌ی D $S$ را اصطلاحاً خرد می‌کند اگر به ازای هر مجموعه‌ای از برچسب‌های $\{y^{(1)}, ..., y^{(d)}\}$ داشته باشیم:
</div>

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

<div dir="rtl">
قضیه‌ی کران بالا ― اگر $\mathcal{H}$ یک مجموعه‌ی متناهی از فرضیه ها (دسته‌بندها) باشد به طوری که $|\mathcal{H}|=k$  باشد و $\delta$ و $m$ ثابت باشند، آنگاه با احتمالِ حداقل $1-\delta$ داریم:
</div>

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

<div dir="rtl">
بُعد VC ― بُعد Vapnik-Chervonenkis برای هر مجموعه‌ی نامتناهی از فرضیه‌ها (دسته‌بندها) $\mathcal{H}$ که با $VC(\mathcal{H}) نمایش داده می‌شود، برابر است با اندازه‌ی بزرگ‌ترین مجموعه‌‌ای که می‌توان با استفاده از $\mathcal{H}$ آن‌ را خرد کرد.
</div>

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

<div dir="rtl">
نکته:‌بُعد VC مجموعه‌ی $H=${همه‌ی دسته‌بندهای خطی در ۲ بعد} برابر با ۳ است.
</div>

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

<div dir="rtl">
قضیه (Vapnik) ― به ازای $\mathcal{H}$ به طوری که $VC(\mathcal{H}) = d$ و هم‌چنین $m$ تعداد نمونه‌های آموزشی باشد، با احتمالِ حداقل $1 - \delta$ داریم:
</div>

<br>

**88. [Introduction, Type of prediction, Type of model]**

<div dir="rtl">
[مبانی، نوع پیش‌بینی، نوع مدل]
</div>

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

<div dir="rtl">
[نماد‌ها و مفاهیم کلی، تابع  خطا، گرادیان کاهشی، درست‌نمایی]
</div>

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

<div dir="rtl">
[مدل‌های خطی، وایازش خطی، وایازش لجستیک، مدل‌های خطی تعمیم‌یافته]
</div>

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

<div dir="rtl">
[ماشین‌های بردار پشتیبان، دسته‌بند حاشیه‌ی بهینه، خطای Hinge، هسته]
</div>

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

<div dir="rtl">
[یادگیری مولد، ‌تحلیل متمایزکننده‌ی گاوسی، دسته‌بند بِیز ساده]
</div>

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

<div dir="rtl">
[روش‌های مبتنی بر درخت و گروهی، CART، جنگل تصادفی، ترقی دادن]
</div>

<br>

**94. [Other methods, k-NN]**

<div dir="rtl">
[سایر روش‌ها، k-NN]
</div>

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

<div dir="rtl">
[نظریه‌ی یادگیری، نامساوی هوفدینگ، PAC، بُعد VC]
</div>
