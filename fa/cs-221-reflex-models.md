**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**
<div dir="rtl">
مدل‌های عکس‌العمل-محور با یادگیری ماشین
</div>
<br>

**2. Linear predictors**
<div dir="rtl">
پیش‌بینی‌گر‌های خطی
</div>
<br>

**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**
<div dir="rtl">
در این بخش، مدل‌های عکس‌العمل-محوری را که با تجربه، از طریق بررسی نمونه‌هایی که به صورت جفت‌های ورودی و خروجی هستند، بهبود می‌یابند بررسی می‌کنیم.
</div>
<br>

**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**
<div dir="rtl">
بردار ویژگی - بردار ویژگی ورودی $x$ که با $phi(x)\$ نمایش داده می‌شود و به صورتی است که:
</div>
<br>

**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**
<div dir="rtl">
امتیاز:‌ امتیاز $s(x, w)$ برای نمونه $(\phi(x), y) \in \mathbb{R}^d \times \mathbb{R}$ مرتبط با مدلی خطی با وزن‌های $w \in \mathbb‪{‬R‪}‬^d$ توسط ضرب داخلی به صورت زیر محاسبه می‌شود:
</div>
<br>

**6. Classification**
<div dir="rtl">
دسته‌بندی
</div>
<br>

**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**
<div dir="rtl">
دسته‌بند خطی - با داشتن بردار وزن $w \in \mathbb‪{‬R‪}‬^d$ و بردار ویژگی $\phi(x) \in \mathbb‪{‬R‪}‬^d$ ، دسته‌بند دودویی خطی $f_w$ به صورت زیر است:
</div>
<br>

**8. if**
<div dir="rtl">
اگر
</div>
<br>

**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**
<div dir="rtl">
حاشیه - حاشیه‌ی $m(x, y, w) \in \mathbb{R}$ نمونه‌ی $(\phi(x), y) \in \mathbb{R}^d \times \{-1, +1\}$ مرتبط با مدل خطی با وزن‌های $w \in \mathbb‪{‬R‪}‬^d$ اطمینان پیش‌بینی مدل را اندازه‌گیری می‌کند: مقادیر بزرگ‌تر بهتر هستند. حاشیه به شکل زیر محاسبه می‌شود:
</div>
<br>

**10. Regression**
<div dir="rtl">
وایازش
</div>
<br>

**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**
<div dir="rtl">
وایازش خطی - با داشتن بردار وزن $w \in \mathbb‪{‬R‪}‬^d$ و بردار ویژگی $\phi(x) \in \mathbb‪{‬R‪}‬^d$، خروجی وایازش خطی با وزن های $w$ با $f_w$ نمایش داده می‌شود و به شکل زیر محاسبه می‌شود:
</div>
<br>

**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**
<div dir="rtl">
باقی‌مانده - باقی‌مانده‌ی $res(x, y, w) \in \mathbb{R}$ برابر با مقداری است که $f_w(x)$ مقدار هدف $y$ را اضافه‌تر پیش‌بینی می‌کند.
</div>
<br>

**13. Loss minimization**
<div dir="rtl">
کمینه‌سازی خطا
</div>
<br>

**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**
<div dir="rtl">
تابع خطا - تابع خطای $Loss(x, y, w)$ مقدار ناخشنودی ما را از وزن‌های $w$ برای پیش‌بینی خروجی $y$ از روی ورودی $x$ به شکل کمّی بیان می‌کند. این خطا مقداری است که قصد داریم آن را در طول فرآیند آموزش کمینه کنیم.
</div>
<br>

**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**
<div dir="rtl">
حالت دسته‌بندی - دسته‌بندی نمونه $x$ با برچسب درست $y \in \{-1, +1\}$ با استفاده از مدلی خطی با وزن‌های $w$ می‌تواند از طریق پیش‌بینی گر $f_w(x) \triangleq \text{sign}(s(x, w))$ انجام شود. در این شرایط، حاشیه‌ی $m(x, y, w)$ معیار موردنظری است که کیفیت دسته‌بندی را اندازه‌گیری می‌کند و می‌تواند با توابع خطای زیر استفاده شود:
</div>
<br>

**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**
<div dir="rtl">
[نام، تصویر، خطای صفر-یک، خطای Hinge، خطای لجیستیک]
</div>
<br>

**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**
<div dir="rtl">
حالت وایازش - پیش‌بینی نمونه‌ی $x$ با برچسب درست $y \in \mathbb{R}$ با استفاده از مدلی با وزن‌های $w$ می‌تواند با پیش‌بینی‌گر $f_w(x) \triangleq s(x, w)$ انجام شود. در این شرایط، حاشیه‌‌ی $res(x, y, w)$ معیار مورد نظری است که کیفیت وایازش را اندازه‌گیری می‌کند و می‌تواند با توابع خطای زیر استفاده شود:
</div>
<br>

**18. [Name, Squared loss, Absolute deviation loss, Illustration]**
<div dir="rtl">
[نام، خطای مربعات، خطای انحراف مطلق، تصویر]
</div>
<br>

**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**
<div dir="rtl">
چارچوب کمینه سازی خطا - برای آموزش مدل، ما قصد داریم تابع خطای آموزش را که به شکل زیر تعریف شده است کمینه کنیم:
</div>
<br>

**20. Non-linear predictors**
<div dir="rtl">
پیش‌بینی‌گر‌های غیرخطی:
</div>
<br>

**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**
<div dir="rtl">
 نزردیک‌ترین همسایه‌ها‫-‬$k$: الگوریتم ‪$‬k‪$‬-نزدیک‌ترین همسایه‌ها، که معمولا با ‪$‬k‪$-‬NN شناخته می‌شود، یک روش غیرپارامتری است که در آن پاسخ یک نمونه داده توسط ماهیت ‪$‬k‪$‬ همسایه‌اش در مجموعه‌ی آموزش تعیین می‌شود. این الگوریتم می‌تواند در هر دو حالت دسته‌بندی و وایازش استفاده شود.
</div>
<br>

**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**
<div dir="rtl">
نکته: هر چه پارامتر $k$ بزرگتر باشد،‌ پیش‌قدر بزرگ‌تر است، و هر چه پارامتر $k$ کوچکتر باشد، واریانس بزرگتر است.
</div>
<br>

**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**
<div dir="rtl">
شبکه‌های عصبی - شبکه‌های عصبی نوعی از مدل‌ها هستند که با لایه‌ها ساخته می‌شوند. انواع معمول شبک‌های عصبی شامل شبکه‌های عصبی پیچشی و شبکه‌های عصبی بازگشتی می‌شوند. واژگان مربوط به معماری‌های شبکه‌های عصبی در شکل زیر بیان شده‌اند:
</div>
<br>

**24. [Input layer, Hidden layer, Output layer]**
<div dir="rtl">
[لایه‌ی ورودی، ‌لایه‌ی نهان، لایه‌ی خروجی]
</div>
<br>

**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**
<div dir="rtl">
با نمایش $i$ به عنوان لایه‌ی $i$ام شبکه و $j$ به عنوان $j$امین واحد نهان لایه، داریم:
</div>
<br>

**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**
<div dir="rtl">
که $x$ ،$b$ ،$w$ و $z$ به ترتیب نشان‌دهنده‌ی وزن، پیش‌قدر، ورودی، و خروجی فعال نشده‌ی سلول عصبی هستند.
</div>
<br>

**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**
<div dir="rtl">
برای شرح جزئی‌تر مفاهیم بالا، راهنمای کوتاه یادگیری بانظارت را مطالعه کنید!
</div>
<br>

**28. Stochastic gradient descent**
<div dir="rtl">
گرادیان کاهشی تصادفی
</div>
<br>

**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**
<div dir="rtl">
گرادیان کاهشی - با نمایش نرخ یادگیری به صورت $\eta \in \mathbb{R}$ (که طول گام نیز نامیده می‌شود)، رویه‌ی به‌روزرسانی برای گرادیان کاهشی توسط نرخ یادگیری و تابع خطای $Loss(x, y, w)$ به صورت زیر بیان می‌شود:
</div>
<br>

**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**
<div dir="rtl">
به‌روزرسانی‌های تصادفی - گرادیان کاهشی تصادفی (SGD) عامل‌های مدل را برحسب یک نمونه آموزش ‪$(\phi(x), y) \in D_{train}$‬ در هر زمان به‌روزرسانی می‌کند. این روش منجر به به‌روزرسانی های گاها نادقیق، اما سریع می‌شود.
</div>
<br>

**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**
<div dir="rtl">
به‌روزرسانی‌های دسته‌ای - گرادیان کاهشی دسته‌ای (BGD) عامل‌های مدل را بر حسب دسته‌ای از نمونه‌‌ داده‌ها (برای مثال تمام داده‌های مجموعه آموزش) در هر زمان به‌روزرسانی می‌کند. این روش جهت‌های  به‌روزرسانی پایدار را، با هزینه‌ی محاسباتی بیشتر، محاسبه می‌کند.
</div>
<br>

**32. Fine-tuning models**
<div dir="rtl">
تنظیم دقیق مدل‌ها
</div>
<br>

**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**
<div dir="rtl">
کلاس فرضیه -کلاس فرضیه‌ی $F$  مجموعه‌ی پیش‌بینی‌گر‌های محتمل با $\phi(x)$ ثابت و $w$ متغیر است.
</div>
<br>

**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**
<div dir="rtl">
تابع لجیستیک - تابع لجیستیک $\sigma$، که تابع سیگموید نیز نامیده می‌شود، به صورت زیر تعریف می‌شود:
</div>
<br>

**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**
<div dir="rtl">
نکته:‌داریم $\sigma^\prime(z) = \sigma(z)(1 - \sigma(z))$.
</div>
<br>

**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**
<div dir="rtl">
انتشار معکوس - انتشار مستقیم از طریق $f_i$ انجام می‌شود، که مقدار زیرعبارتی است که از $i$ ریشه می‌گیرد، در حالی که انتشار معکوس از طریق $g_i = \frac{\partial{out}}{\partial{f_i}}$ انجام می‌گیرد و نشان‌دهنده‌ی چگونگی تاثیر $f_i$ روی خروجی است.
</div>
<br>

**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**
<div dir="rtl">
خطای تقریب و تخمین - خطای تقریب $\epsilon_{approx}$ نشان‌دهنده‌ی میزان دوری کلاس فرضیه $F$ از پیش‌بینی‌گر هدف $g^*$ است، در حالی که خطای تخمین $\epsilon_{est}$ خوب بودن $\hat{f}$ نسبت به بهترین پیش‌بینی‌گر $f^*$ از کلاس فرضیه‌ی $F$ را اندازه‌گیری می‌کند.
</div>
<br>

**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**
<div dir="rtl">
نظام‌بخشی - هدف از رویه‌ی نظام‌بخشی جلوگیری از بیش‌برازش مدل به داده‌ها است و در نتیجه با مشکل واریانس بالا طرف است. جدول زیر خلاصه‌ای از انواع روش‌های متداول نظام‌بخشی را ارائه می‌دهد:
</div>
<br>

**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**
<div dir="rtl">
[ضرایب را تا ۰ کاهش می‌دهد، برای انتخاب متغیر مناسب است، ضرایب را کوچکتر می‌کند، بین انتخاب متغیر و ضرایب کوچک مصالحه می‌کند]
</div>
<br>

**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**
<div dir="rtl">
فراعامل‌ها - فراعامل‌ها خصوصیات الگوریتم یادگیری هستند، و شامل ویژگی‌ها، عامل نظام بخشی $lambda\$، تعداد تکرار‌ها $T$، طول گام $\eta$، و غیره می‌شوند.
</div>
<br>

**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**
<div dir="rtl">
واژگان مجموعه‌ها - وقتی مدلی را انتخاب می‌کنیم، ۳ بخش متفاوت از نمونه داده‌هایی که داریم را به شکل زیر مشخص می‌کنیم:
</div>
<br>

**42. [Training set, Validation set, Testing set]**
<div dir="rtl">
[مجموعه آموزش، مجموعه اعتبارسنجی، مجموعه آزمایش]
</div>
<br>

**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**
<div dir="rtl">
[مدل آموزش داده شده است، معمولا ۸۰ درصد از مجموعه داده‌ها، مدل ارزیابی می‌شود، معمولا ۲۰ درصد از مجموعه داده‌ها، این مجموعه همچنین تحت عنوان مجموعه بیرون نگه‌داشته‌شده یا توسعه نیز شناخته می شود، مدل پیش‌بینی می‌کند، داده‌های دیده نشده]
</div>
<br>

**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**
<div dir="rtl">
بعد از اینکه مدل انتخاب شد، روی کل مجموعه داده‌ها آموزش داده می‌شود و بر روی مجموعه دادگان دیده نشده آزمایش می‌شود. این مراحل در شکل زیر آمده‌اند:
</div>
<br>

**45. [Dataset, Unseen data, train, validation, test]**
<div dir="rtl">
[داده، داده‌های دیده نشده، آموزش، اعتبارسنجی، آزمایش]
</div>
<br>

**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**
<div dir="rtl">
برای شرح جزئی‌تر مفاهیم بالا، راهنمای کوتاه نکات و ترفند‌های یادگیری ماشین را مطالعه کنید!
</div>
<br>

**47. Unsupervised Learning**
<div dir="rtl">
یادگیری بدون نظارت
</div>
<br>

**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**
<div dir="rtl">
هدف از یادگیری بدون نظارت کشف ساختار داده‌ها است که ممکن است‌ از ساختار‌های نهان غنی‌ای برخوردار باشد. 
</div>
<br>

**49. k-means**
<div dir="rtl">
میانگین-$k$
</div>
<br>

**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**
<div dir="rtl">
خوشه‌بندی - با فرض داده‌شدن مجموعه‌ی آموزش متشکل از نقاط ورودی $D_{train}$، هدف الگوریتم خوشه‌بندی اختصاص دادن یک خوشه $z_i \in \{1,...,k\}$ به هر نقطه $\phi(x_i)$ است.
</div>
<br>

**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**
<div dir="rtl">
تابع هدف - تابع خطا برای یکی از الگوریتم‌های اصلی خوشه‌بندی، $k$-میانگین، به صورت زیر است:
</div>
<br>

**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**
<div dir="rtl">
الگوریتم - بعد از مقداردهی اولیه‌ی تصادفی مراکز خوشه‌ها $\mu_1, \mu_2, \dots, \mu_k \in \mathbb{R}^n$، الگوریتم $k$-میانگین مراحل زیر را تا هم‌گرایی تکرار می‌کند:
</div>
<br>

**53. and**
<div dir="rtl">
و
</div>
<br>

**54. [Means initialization, Cluster assignment, Means update, Convergence]**
<div dir="rtl">
[مقداردهی اولیه میانگین‌ها، تخصیص خوشه، به‌روزرسانی میانگین‌ها، هم‌گرایی]
</div>
<br>

**55. Principal Component Analysis**
<div dir="rtl">
تحلیل مولفه‌های اصلی
</div>
<br>

**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**
<div dir="rtl">
مقدار ویژه، بردار ویژه - برای ماتریس دلخواه ‪$‬A \in \mathbb‪{‬R‪}^{‬n \times n‪}$‬، ‪$‬\lambda‪$‬  مقدار ویژه‌ی ماتریس $A$ است اگر وجود داشته باشد بردار $z \in \mathbb{R}^n \\ \{0\}$ که
 به آن بردار ویژه می‌گویند، به طوری که:
</div>
<br>

**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**
<div dir="rtl">
قضیه‌ی طیفی - فرض کنید $A \in \mathbb{R}^{n \times n}$ باشد. اگر $A$ متقارن باشد، در این صورت $A$ توسط یک ماتریس حقیقی متعامد $U \in \mathbb{R} ^{n \times n}$ قطری‌پذیر است. با نمایش $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$ داریم:
</div>
<br>

**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**
<div dir="rtl">
نکته: بردار ویژه‌ی متناظر با بزرگ‌ترین مقدار ویژه، بردار ویژه‌ی اصلی ماتریس $A$ نام دارد.
</div>
<br>

**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**
<div dir="rtl">
الگوریتم - رویه‌ی تحلیل مولفه‌های اصلی یک روش کاهش ابعاداست که داده‌ها را در فضای $k$-بعدی با بیشینه کردن واریانس داده‌ها، به صورت زیر تصویر می‌کند:
</div>
<br>

**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**
<div dir="rtl">
مرحله‌ی ۱: داده‌ها به گونه‌ای نرمال‌سازی می‌شوند که میانگین ۰ و انحراف معیار ۱ داشته باشند.
</div>
<br>

**61. [where, and]**
<div dir="rtl">
[و، و]
</div>
<br>

**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**
<div dir="rtl">
مرحله‌ی ۲: مقدار $\Sigma = \frac{1}{m} \sum_{i=1}^m \phi(x(i)) \phi(x(i))^T \in \mathbb{R}^{n \times n}$، که ماتریسی متقارن با مقادیر ویژه‌ی حقیقی است محاسبه می‌شود. مرحله‌ی ۳: بردارهای $u_1, \dots, u_k \in \mathbb{R}^n$ که $k$ بردارهای ویژه‌ی اصلی متعامد $\Sigma$ هستند محاسبه می‌شوند. این بردارهای ویژه متناظر با $k$ مقدار ویژه با بزرگ‌ترین مقدار هستند. مرحله‌ی ۴: داده‌ها بر روی فضای $\text{span}_ {\mathbb{R}} (u_1, \dots, u_k)$ تصویر می‌شوند.
</div>
<br>

**63. This procedure maximizes the variance among all k-dimensional spaces.**
<div dir="rtl">
این رویه واریانس را در میان تمام فضاهای $k$-بعدی بیشینه می‌کند.
</div>
<br>

**64. [Data in feature space, Find principal components, Data in principal components space]**
<div dir="rtl">
[داده‌ها در فضای ویژگی، پیدا‌کردن مؤلفه‌های اصلی، داده‌ها در فضای مؤلفه‌های اصلی]
</div>
<br>

**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**
<div dir="rtl">
برای شرح جزئی‌تر مفاهیم بالا، راهنمای کوتاه یادگیری بدون نظارت را مطالعه کنید!
</div>
<br>

**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin]**
<div dir="rtl">
[پیش‌بینی‌گر‌های خطی، بردار ویژگی، دسته‌بند/وایازش‌گر خطی، حاشیه]
</div>
<br>

**67. [Loss minimization, Loss function, Framework]**
<div dir="rtl">
[کمینه‌سازی خطا،‌تابع خطا، چارچوب]
</div>
<br>

**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**
<div dir="rtl">
[پیش‌بینی‌گر غیرخطی، $k$-نزدیک‌ترین همسایه‌ها، شبکه‌های عصبی]
</div>
<br>

**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**
<div dir="rtl">
[گرادیان کاهشی تصادفی، گرادیان، به‌روزرسانی تصادفی، به‌روزرسانی دسته‌ای]
</div>
<br>

**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**
<div dir="rtl">
[تنظیم دقیق مدل‌ها، کلاس فرضیه، انتشار معکوس، نظام‌بخشی، واژگان مجموعه‌ها]
</div>
<br>

**71. [Unsupervised Learning, k-means, Principal components analysis]**
<div dir="rtl">
[یادگیری بدون نظارت، $k$-میانگین، تحلیل مؤلفه‌های اصلی]
</div>
<br>

**72. View PDF version on GitHub**
<div dir="rtl">
[نسخه‌ی پی‌دی‌اف را در گیت‌هاب ببینید]
</div>
<br>

**73. Original authors**
<div dir="rtl">
متن اصلی از
</div>
<br>

**74. Translated by X, Y and Z**
<div dir="rtl">
ترجمه شده توسط
</div>
<br>

**75. Reviewed by X, Y and Z**
<div dir="rtl">
بازبینی شده توسط
</div>
<br>

**76. By X and Y**
<div dir="rtl">
توسط
</div>
<br>

**77. The Artificial Intelligence cheatsheets are now available in [target language].**
<div dir="rtl">
راهنمای کوتاه هوش مصنوعی ترجمه شده به ‪]‬زبان مقصد‪[‬ هم‌اکنون در دسترس هستند. 
</div>
<br>
