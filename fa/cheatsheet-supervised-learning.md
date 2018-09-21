**1. Supervised Learning cheatsheet**

راهنمای کوتاه یاگیری با ناظر

<br>

**2. Introduction to Supervised Learning**

مبانی یادگیری با ناظر

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

با در نظر گرفتن اینکه مجموعه‌ای از نمومه‌های داده \{x^{(1)}, ..., x^{(m)}\} با مجموعه‌ای از نتایج \{y^{(1)}, ..., y^{(m)}\}  مرتبط شده‌است، می‌خواهیم طبقه‌بندی بسازیم که یاد بگیرد چگونه از روی x، y را پیش‌بینی کند.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

انواع پیش‌بینی - انواع مختلف مدل‌های پیش‌بینی کنند در جدول زیر به اختصار آمده‌اند:

<br>

**5. [Regression, Classifier, Outcome, Examples]**

[رگرسیون، طبقه‌بند، حاصل، مثال]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

[اعداد پیوسته، کلاس، رگرسیون خطی، رگرسیون لجستیک، SVM، طبقه‌بند بیز ساده ]

<br>

**7. Type of model ― The different models are summed up in the table below:**

انواع مدل ـ انواع مختلف مدل‌ها در جدول زیر به اختصار آمده‌اند.

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

[مدل تمیزدهنده، مدل مولد، هدف، چه چیزی یاد می‌گیرد، تصویر، مثال]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

[تخمین مستقیم P(y|x)، تخمین P(x|y) و سپس نتیجه‌گیریِ P(y|x)، مرز تصمیم‌گیری، توزیع احتمال داده، رگرسیون‌ها، SVMها، GDA، بیز ساده ]

<br>

**10. Notations and general concepts**

نماد‌ها و مفاهیم کلی

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

فرضیه - فرضیه که با h_\theta نمایش داده می‌شود، همان مدلی است که ما انتخاب می‌کنیم. به ازای هر ورودی داده x^{(i)} حاصل پیش‌یینی مدل h_\theta(x^{(i)}) می‌باشد.

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

تابع اتلاف - تابع اتلاف، یک تابع L:(z,y)\in\mathbb{R}\times Y\longmapsto L(z,y)\in\mathbb{R} است که مقدار پیش‌بینی شده‌ z‌ و مقدار داده‌ی واقعی متناظر با آن y را می‌گیرد و به عنوان خروجی می‌گوید این‌ دو چه قدر با هم تفاوت دارند. 
تابع‌های خطای رایج در جدول زیر به اختصار آمده‌اند.

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

[خطای مربع کمینه، اتلاف رگرسیون، ی اتلاف هینج، Cross-entopy  ]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

[رگرسیون خطی، رگریسیون لجستیک، SVM، شبکه‌های عصبی]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

تابع هزینه - تابع هزینه‌ی $J$ معمولا برای ارزیابی عملکرد یک مدل استفاده می‌شود و با توجه به تابع اتلاف $L$ به صورت زیر تعریف می‌شود:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

گرادیان کاهشی - با نمایش نرخ یادگیری به صورت $\alpha\in\mathbb{R}$، قانون آپدیت برای گرادیان کاهشی که با نرخ‌یادگیری و تابع هزینه‌ی J بیان می‌شود، به شرح زیر است:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

نکته: گرادیان کاهشی تصادفی(SGD) بر اساس هر کدام از نمونه‌های آموزشی به روزرسانی می‌شود و گرادیان کاهشی دسته‌ای، روی دسته‌ای از نمونه‌های آموزشی به روزرسانی می‌شود.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

درستی - از درستی (یا تابع درست‌نمایی) یک مدل $L(\theta)$ با پارامتر‌های $\theta$، برای پیدا کردن پارامتر‌های بهینه $\theta$ ‌با روش بیشینه‌سازی درستی مدل استفاده می‌گردد. البته در عمل ما از درستی-لوگارتیمی که برابر با $log(L(\theta))$ است استفاده می‌کنیم چرا که بهینه‌سازی درستی-لوگاریتمی از بهینه‌سازی تابع درستی به تنهایی، راحت‌تر است. داریم:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

الگوریتم نیوتن ― الگوریتم نیوتن یک روش عددیست که $\theta$ را به طوری که $\ell'(\theta)=0$ باشد، پیدا می‌کند.

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

نکته: تعمیم چندبعدی که به روش نیوتون-رافسون معروف قانون به روز رسانی زیر را دارد:    

<br>

**21. Linear models**

مدل‌های خطی

<br>

**22. Linear regression**

رگرسیون خطی

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

در این‌جا فرض می‌کنیم $y|x;\theta\sim\mathcal{N}(\mu,\sigma^2)$ 

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

معادلات نرمال ― با نمایش $X$ به عنوان یک ماتریس، مقداری از $\theta$ که تابع هزینه را کمینه می‌کند یک راه‌حل به فرم بسته‌ دارد به طوری که:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

اگوریتم LMS ― اگر $\alpha$ نمایش نرخ یادگیری باشد، روش آپدیت در الگوریتم LMS ( که به روش یادگیری Widrow-Hoff هم شناخته می‌شود) برای مجموعه‌ای از $m$ داده‌ی آموزشی، به شرح زیر است:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

نکته: این روش اپدیت، حالت خاصی از الگوریتم گرادیان کاهشی است.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

LWR ― رگرسیون وزن‌دار محلی یا LWR نوعی دیگر از رگرسیون‌های خطی است که روی هر کدام از نمونه‌های آموزشی در تابع خود هزینه‌ی خود، وزن $w^{(i)}(x)$ را می‌گذارد این وزن با پرامتر $\tau\in\mathbb{R}$ به شکل زیر تعریف می‌شود:

<br>

**28. Classification and logistic regression**

طبقه‌بندی و رگرسیون‌های لجستیک

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

تابع سیگموئید ― تابع سیگموئید $g$ که به تابع لجستیک هم معروف است به صورت زیر تعریف می‌شود:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

رگرسیون لجستیک ― فرض می‌کنیم $y|x;\theta\sim\textrm{Bernoulli}(\phi)$. داریم: 

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

نکته: هیچ راه حل بسته‌ای برای حالت رگرسیون لجستیک وجود ندارد. 

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

رگرسیون سافت‌مکس ― رگرسیون سافت‌مکس که رگرسیون چند‌کلاسه هم نامیده‌ می‌شود، در مواقعی که بیش‌از ۲ کلاس خروجی داریم برای تعمیم رگرسیون لجستیک استفاده می‌شود. ما برای راحتی $\theta_K=0$ قرار می‌دهیم، در نتیجه پارامتر برنولی برای هر کلاس به صورت زیر خواهد بود:

<br>

**33. Generalized Linear Models**

مدل‌های خطی تعمیم‌یافته

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

خانواده‌ی نمایی ― به گروهی از توزیع‌ها، خانواده‌ی نمایی گوییم اگر بتوان آن را با استفاده از پارامتر طبیعی $\eta$، آماره‌ی $T(y)$ و یک تابع log-partition با نمایش $a(\eta)$ به صورت زیر نوشت:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

نکته: البته معمولا $T(y)=y$ خواهد بود. هم‌چنین می‌توان به $\exp(-a(\eta))$ به چشم یک عامل نرمال‌کننده نگاه کرد که باعث می‌شود جمع احتمال‌ها حتما برابر با یک شود.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

رایج‌ترین توزیع‌های نمایی در زیر به اختصار آمده‌اند.

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

[توزیع، برنولی، گاوسی، پواسون، هندسی]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

فرض‌های GLMها ― مدل‌های خطی تعمیم‌یافته (GLM) به دنبال پیش‌بینی یک متغیر رندم $y$ به عنوان تابعی از $x\in\mathbb{R}^{n+1}$ هستند. آن‌ها بر سه فرض استوارند:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

نکته: مربع کمینه و رگرسیون لجستیک حالت‌ خاصی از GLM ها هستند.

<br>

**40. Support Vector Machines**

SVMها

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

هدف SVMها پیدا کردن خطیست که کمتر فاصله از آن خط را بیشینه می‌کند.  

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

طبقه‌بند فاصله‌ بهینه ― طبقه‌بند فاصله بهینه $h$ به صورت زیر تعریف می‌شود:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

که $(w, b)\in\mathbb{R}^n\times\mathbb{R}$ راحل برای مساله‌ی بهینه‌سازی زیر باشد:

<br>

**44. such that**

به طوری که

<br>

**45. support vectors**

بردار‌های پشتبیانی

<br>

**46. Remark: the line is defined as wTx−b=0.**

نکته: خطر به صورت $\boxed{w^Tx-b=0}$ تعریف شده است.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

در SVMها از تابع اتلاف هینج استفاده می‌شود و تعریف آن به صورت زیر است:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

کرنل ― برای نگاشت خصوصیات $\phi$، کرنل $K$ به صورت زیر تعریف می‌شود:


<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

در عمل، به کرنل $K$ که به صورت ‍$K(x,z)=\exp\left(-\frac{||x-z||^2}{2\sigma^2}\right)$ تعریف شده باشد، کرنل گاوسی می‌گوییم. این نوع کرنل یکی از کرنل‌های پراستفاده محسوب می‌شود.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

[جداپذیری غیر خطی،  به کارگرفتن یک نگاشت خطی $phi$، مرز تصمیم در فضای اصلی ]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

نکته: ما برای محاسبه‌ی تابع از هزینه «حقه‌ی کرنل» استفاده می‌کنیم چرا که در واقع برای محاسبه‌ی آن، نیازی به دانستن دقیق نگاشت $\phi$ که بیشتر مواقع هم بسیار پیچیده‌ست، نداریم؛ دانستن مقادیر $K(x,z)$ برای ما کافیست.

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

لاگرانژین ― لاگرانژین $\mathcal{L}(w,b)$ به صورت زیر تعریف می‌کنیم:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

نکته به ضرایب $\beta_i$ ضرایب لاگرانژ هم می‌گوییم.

<br>

**54. Generative Learning**

یادگیری مدل‌های مولد

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

یک مدل مولد ابتدا با تخمین زدن $P(x|y)$ سعی می‌کند یاد بگیرد چگونه می‌توان دیتا را تولید کرد، سپس با استفاده از $P(x|y)$ و هم‌چنین قضیه‌ی بیز، $P(y|x)$ را تخمین می‌زند.

<br>

**56. Gaussian Discriminant Analysis**

تحلیل تشخیصی گاوسی

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

فرضیات ― در تحلیل تشخیصی گاوسی فرض می‌کنیم $y$ و $x|y = 0$ و $x|y = 1$ به طوری که:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

تخمین ― جدول زیر تخمین‌هایی که هنگام بیشینه‌کردن تابع درستی (یا تابع درست‌نمایی) به آن می‌رسیم را به اختصار آورده‌است.

<br>

**59. Naive Bayes**

طبقه‌بند بیز ساده

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

فرض ― مدل بیز ساده فرض می‌کند تمام خصوصیات نمونه‌های داده از هم‌دیگر مستقل است. 

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

راه‌حل‌ها ― بیشنه‌کردن log-likelihood به پاسخ‌های زیر می‌رسد. $k\in\{0,1\},l\in[\![1,L]\!]$

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

نکته: مدل بیز ساده در مساله‌‌های طبقه‌بندی متن و تشخیص اسپم به صورت گسترده استفاده می‌شود.

<br>

**63. Tree-based and ensemble methods**

روش‌های ترکیبی و مبتنی بر درخت

<br>

**64. These methods can be used for both regression and classification problems.**

این روش‌ها هم در مسائل رگرسیون و هم در مسائل طبقه‌بندی می‌توانند استفاده شوند.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

CART ― درخت‌های رگرسیون و طبقه‌بندی.عموما با نام درخت‌های تصمیم‌گیری شناخته می‌شوند. می‌توان آن‌ها را به صورت درخت‌هایی دودویی نمایش داد. مزیت ‌آن‌ها قابل تفسیر بودنشان است.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

جنگل تصادفی ― یک تکنیک مبتی بر درخت است. این مدل از تعداد زیادی درخت تصمیم‌گیری که روی مجموعه‌هایی تصادفی از خصوصیات ساخته‌شدهاند استفاده می‌کند. برخلاف یک درخت تصمیم‌گیری ساده بسیار غیر قابل تفسیر است البته عمکرد عموما خوب آن باعث به الگوریتمی محبوب تبدیل شود.

<br>

**67. Remark: random forests are a type of ensemble methods.**

نکته: جنگل تصادفی نوعی از روش ترکیبی است.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

بوستینگ ― ایده‌ی اصلی این متد، ترکیب چند مدل ضعیف و ساخت یک مدل قوی از آن‌هاست. انواع اصلی آن در جدول به صورت خلاصه آمده‌اند:

<br>

**69. [Adaptive boosting, Gradient boosting]**

[بوستینگ سازگار، بوستینگ گرادیان]

<br>

**70. High weights are put on errors to improve at the next boosting step**

برای خطاها وزن‌ بالایی در نظر می‌گیرد تا در مرحله‌ی بعد بوستینگ بهبود یابد.

<br>

**71. Weak learners trained on remaining errors**

چند مدل ضعیف روی باقی خطاها آموزش می‌یابند.

<br>

**72. Other non-parametric approaches**

سایر رویکرد‌های غیر پارامتری

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

$k$-nearest neighbors ― عموما با نام $k$-NN شناخته می‌شود. این اگوریتم پاسخ هر نمونه داده برا اساس $k$ تا از نزدیک‌ترین همسایه‌های آن تعیین می‌کند و هم در طبقه‌بندی و هم در رگرسیون استفاده می‌شود.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

نکته: هرچه پارامتر $k$ برزرگ‌تر باشد بایاس مدل بیشتر خواهد بود و هر چه کوچکتر باشد واریانس مدل بیشتر خواهد شد.

<br>

**75. Learning Theory**

نظریه یادگیری

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

کران اجتماع ― اگر $A_1, ..., A_k$، $k$ رخداد باشد، داریم:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

نامساوی هوفدینگ ― اگر $Z_1, ..., Z_m$، $m$ متغیر تصادفی مستقل با توزیع یکسان و کشیده شده از توزیع برنولی با پارامتر $\phi$ باشند و هم‌چنین $\widehat{\phi}$ میانگین آن‌ها و $\gamma>0$ ثابت باشد، داریم:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

نکته: این نامساوری به کران چرنوف نیز معروف است.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

خطای آموزش ― به ازای هر طبقه‌بند $h$، خطای آموزش $\widehat{\epsilon}(h)$ (یا همان خطای تجربی)، به صورت زیر تعریف می‌شود:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

Probably Approximately Correct (PAC)  ― چارچوبی است در ذیل آن نتایج متعددی از نظریه یادگیری اثبات شده است و فرض‌های زیر را در بر دارد:
<br>

**81: the training and testing sets follow the same distribution **

مجموعه‌ی آموزش و مجموعه‌ی تست از یک توزیع هستند. 

<br>

**82. the training examples are drawn independently**

نمومه‌های آموزشی مستقل از یک‌ دیگر انتخاب شده‌اند.

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

انفجار ― برای مجموعه $S=\{x^{(1)},...,x^{(d)}\}$ و مجموعه‌ای از طبقه‌بند‌ها $\mathcal{H}$ می‌گوییم $\mathcal{H}$ مجموعه $S$ را متلاشی کرده است اگر به ازای هر مجموعه‌ای از برچسب‌ها $\{y^{(1)}, ..., y^{(d)}\}$ داشته باشیم:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

قضیه‌ی کران بالا ― اگر $\mathcal{H}$ یک مجموعه متناهی از فرضیه ها(طبقه‌بند‌‌ها ) به طوری که $|\mathcal{H}|=k$  باشد و $\delta$ و $m$ ثابت باشند، آنگاه با احتمال حداقل $1-\deta$ داریم:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

بُعد VC ― بعد VC یک مجموعه نامتناهی از فرضیه‌ها(طبقه‌بند‌ها) $\mathcal{H}$ که با $VC(\mathcal{H}) نمایش داده می‌شود اندازه‌ی بزرگ‌ترین مجموعه‌است می‌توان با $\mathcal{H}$ متلاشی کرد.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

نکته: بُعد VC مجموعه‌ی ${\small\mathcal{H}=\{\textrm{همه‌ی طبقه‌بند‌های خطی در ۲ بعد }\}}$ برابر با ۳ است.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

قضیه (Vapnik) ― به ازای $\mathcal{H}$ به طوری‌که $VC(\mathcal{H}) = d$ و هم‌چنین $m$ تعداد نمونه‌های آموزشی با احتمال حداقل $\delta - 1$ داریم:

<br>

**88. [Introduction, Type of prediction, Type of model]**

[مبانی، انواع پیش‌بینی، انواع مدل ]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

[نماد‌ها و مفاهیم کلی، تابع اتلاف، گرادیان کاهشی، درستی (تابع درست‌نمایی)]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

[مدل‌های خطی، رگرسیون خطی، رگرسیون لجستیک، مدل‌های خطی تعمیم‌یافته ]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

[SVMها، طبقه‌بند فاصله بهینه، اتلاف هینج، کرنل]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

[یادگیری مدل‌های مولد،‌تحلیل تشخیصی گاوسی، طبقه‌بند بیز ساده]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

[روش‌های ترکیبی و مبتنی بر درخت، CART، جنگل تصادفی، بوستینگ]

<br>

**94. [Other methods, k-NN]**

[سایر روش‌ها، k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

[نظریه‌ی یادگیری، نامساوی هوفدینگ، PAC، بُعد VC ]
