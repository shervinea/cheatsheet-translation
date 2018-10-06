**1. Unsupervised Learning cheatsheet**
<div dir="rtl">
راهنمای کوتاه یادگیری بدون نظارت
</div>
<br>

**2. Introduction to Unsupervised Learning**
<div dir="rtl">
مبانی یادگیری بدون نظارت
</div>
<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**
<div dir="rtl">
انگیزه - هدف از یادگیری بدون نظارت کشف الگوهای پنهان در داده‌های بدون برچسب $\{x_1, \dots, x_m\}$ است.
</div>
<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**
<div dir="rtl">
نابرابری ینسن - فرض کنید $f$ تابعی محدب و $X$ یک متغیر تصادفی باشد. در این صورت نابرابری زیر را داریم:
</div>
<br>

**5. Clustering**
<div dir="rtl">
خوشه‌بندی
</div>
<br>

**6. Expectation-Maximization**
<div dir="rtl">
بیشینه‌سازی امید ریاضی
</div>
<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**
<div dir="rtl">
متغیرهای نهفته - متغیرهای نهفته متغیرهای پنهان یا مشاهده‌نشده‌ای هستند که مسائل تخمین را دشوار می‌کنند، و معمولاً با $z$ نمایش داده می‌شوند. شرایط معمول که در آن‌ها متغیرهای نهفته وجود دارند در زیر آمده‌اند:
</div>
<br>

**8. [Setting, Latent variable z, Comments]**
<div dir="rtl">
[موقعیت، متغیر نهفته‌ی $z$، توضیحات]
</div>
<br>

**9. [Mixture of k Gaussians, Factor analysis]**
<div dir="rtl">
[ترکیب $k$ توزیع گاوسی، تحلیل عامل]
</div>
<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**
<div dir="rtl">
الگوریتم - الگوریتم بیشینه‌سازی امید ریاضی روشی بهینه برای تخمین پارامتر $theta$ از طریق تخمین درستی بشینه در اختیار قرار می‌دهد. این کار با تکرار مرحله‌ی به دست آوردن یک کران پایین برای درستی (مرحله‌ی امید ریاضی) و همچنین بهینه‌سازی آن کران پایین (مرحله‌ی بیشینه‌سازی) طبق توضیح زیر انجام می‌شود:
</div>
<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**
<div dir="rtl">
مرحله‌ی امید ریاضی:‌احتمال پسین $Q_i(z(i))$ که هر نمونه داده $x(i)$ متعلق به خوشه‌ی $z(i)$ باشد به صورت زیر محاسبه می‌شود:
</div>
<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**
<div dir="rtl">
مرحله‌ی بیشینه‌سازی: با استفاده از احتمالات پسین $Q_i(z(i))$ به عنوان وزن‌های وابسته به خوشه‌ها برای نمونه‌های داده‌ی $x(i)$، مدل مربوط به هر کدام از خوشه‌ها، طبق توضیح زیر، دوباره تخمین زده می‌شوند:
</div>
<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**
<div dir="rtl">
[مقداردهی اولیه‌ی توزیع‌های گاوسی، مرحله‌ی امید ریاضی، مرحله‌ی بیشینه‌سازی، هم‌گرایی]
</div>
<br>

**14. k-means clustering**
<div dir="rtl">
خوشه‌بندی $k$-میانگین
</div>
<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**
<div dir="rtl">
توجه کنید که $c(i)$ خوشه‌ی نمونه داده‌ی $i$ و $\mu_j$ مرکز خوشه‌ی $j$ است.
</div>
<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**
<div dir="rtl">
الگوریتم - بعد از مقداردهی اولیه‌ی تصادفی مراکز خوشه‌ها $\mu_1, \mu_2, \dots, \mu_k \in \mathbb{R}^n$، الگوریتم $k$-میانگین مراحل زیر را تا هم‌گرایی تکرار می‌کند:
</div>
<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**
<div dir="rtl">
[مقداردهی اولیه‌ی میانگین‌ها، تخصیص خوشه، به‌روزرسانی میانگین‌ها، هم‌گرایی]
</div>
<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**
<div dir="rtl">
تابع اعوجاج - برای تشخیص اینکه الگوریتم به هم‌گرایی رسیده است، به تابع اعوجاج که به صورت زیر تعریف می‌شود رجوع می‌کنیم:
</div>
<br>

**19. Hierarchical clustering**
<div dir="rtl">
خوشه‌بندی سلسله‌مراتبی
</div>
<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**
<div dir="rtl">
الگوریتم - یک الگوریتم خوشه‌بندی سلسله‌مراتبی تجمعی است که خوشه‌های تودرتو را به صورت پی‌در‌پی ایجاد می‌کند.
</div>
<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**
<div dir="rtl">
انواع - انواع مختلفی الگوریتم خوشه‌بندی سلسله‌مراتبی وجود دارند که هر کدام به دنبال بهینه‌سازی توابع هدف مختلفی هستند، که در جدول زیر به اختصار آمده‌اند:
</div>
<br>

**22. [Ward linkage, Average linkage, Complete linkage]**
<div dir="rtl">
[پیوند بخشی، پیوند میانگین، پیوند کامل]
</div>
<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**
<div dir="rtl">
[کمینه‌کردن فاصله‌ی درونِ خوشه، کمینه‌کردن فاصله‌ی میانگین بین هر دو جفت خوشه، کمینه‌کردن حداکثر فاصله بین هر دو جفت خوشه]
</div>
<br>

**24. Clustering assessment metrics**
<div dir="rtl">
معیارهای ارزیابی خوشه‌بندی
</div>
<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**
<div dir="rtl">
در یک وضعیت یادگیری بدون نظارت، معمولاً ارزیابی یک مدل کار دشواری است، زیرا برخلاف حالت یادگیری نظارتی اطلاعاتی در مورد برچسب‌های حقیقی داده‌ها نداریم.
</div>
<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**
<div dir="rtl">
ضریب نیم‌رخ - با نمایش $a$ به عنوان میانگین فاصله‌ی یک نمونه با همه‌ی نمونه‌های دیگر در همان کلاس، و با نمایش $b$ به عنوان میانگین فاصله‌ی یک نمونه با همه‌ی نمونه‌های دیگر از نزدیک‌ترین خوشه، ضریب نیم‌رخ $s$ به صورت زیر تعریف می‌شود:
</div>
<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**
<div dir="rtl">
شاخص Calinski-Harabasz - با در نظر گرفتن $k$ به عنوان تعداد خوشه‌ها، ماتریس پراکندگی درون خوشه‌ای $B_k$ و ماتریس پراکندگی میان‌خوشه‌ای $W_k$ به صورت زیر تعریف می‌شوند:
</div>
<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**
<div dir="rtl">
شاخص Calinski-Harabasz $s(k)$ بیان می‌کند که یک مدل خوشه‌بندی چگونه خوشه‌های خود را مشخص می‌کند، به گونه‌ای که هر چقدر مقدار این شاخص بیشتر باشد، خوشه‌ها متراکم‌تر و از هم تفکیک‌یافته‌تر خواهند بود. این شاخص به صورت زیر تعریف می‌شود:
</div>
<br>

**29. Dimension reduction**
<div dir="rtl">
کاهش ابعاد
</div>
<br>

**30. Principal component analysis**
<div dir="rtl">
تحلیل مولفه‌های اصلی
</div>
<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**
<div dir="rtl">
روشی برای کاهش ابعاد است که جهت‌هایی را با حداکثر واریانس پیدا می‌کند تا داده‌ها را در آن جهت‌ها تصویر کند.
</div>
<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**
<div dir="rtl">
مقدار ویژه، بردار ویژه - برای ماتریس دلخواه $A \in \mathbb{R}^{n \times n}$، $\lambda$ مقدار ویژه‌ی ماتریس $A$ است اگر وجود داشته باشد بردار $z \in \mathbb{R}^n \\ \{0\}$ که به آن بردار ویژه می‌گویند، به طوری که:
</div>
<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**
<div dir="rtl">
قضیه‌ی طیفی - فرض کنید $A \in \mathbb{R}^{n \times n}$ باشد. اگر $A$ متقارن باشد، در این صورت $A$ توسط یک ماتریس حقیقی متعامد $U \in \mathbb{R} ^{n \times n}$ قطری‌پذیر است. با نمایش $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$ داریم:
</div>
<br>

**34. diagonal**
<div dir="rtl">
قطری
</div>
<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**
<div dir="rtl">
نکته: بردار ویژه‌ی متناظر با بزرگ‌ترین مقدار ویژه، بردار ویژه‌ی اصلی ماتریس $A$ نام دارد.
</div>
<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**
<div dir="rtl">
الگوریتم - رویه‌ی تحلیل مولفه‌های اصلی یک روش کاهش ابعاد است که داده‌ها را در فضای $k$-بعدی با بیشینه کردن واریانس داده‌ها، به صورت زیر تصویر می‌کند:
</div>
<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**
<div dir="rtl">
مرحله‌ی ۱: داده‌ها به گونه‌ای نرمال‌سازی می‌شوند که میانگین ۰ و انحراف معیار ۱ داشته باشند.
</div>
<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**
<div dir="rtl">
مرحله‌ی ۲: مقدار $\Sigma = \frac{1}{m} \sum_{i=1}^m x(i) x(i)^T \in \mathbb{R}^{n \times n}$، که ماتریسی متقارن با مقادیر ویژه‌ی حقیقی است محاسبه می‌شود.
</div>
<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**
<div dir="rtl">
مرحله‌ی ۳: بردارهای $u_1, \dots, u_k \in \mathbb{R}^n$ که $k$ بردارهای ویژه‌ی اصلی متعامد $\Sigma$ هستند محاسبه می‌شوند. این بردارهای ویژه متناظر با $k$ مقدار ویژه با بزرگ‌ترین مقدار هستند.
</div>
<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**
<div dir="rtl">
مرحله‌ی ۴: داده‌ها بر روی فضای $\text{span}_ {\mathbb{R}} (u_1, \dots, u_k)$ تصویر می‌شوند.
</div>
<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**
<div dir="rtl">
این رویه واریانس را در فضای $k$-بعدی به دست آمده بیشینه می‌کند.
</div>
<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**
<div dir="rtl">
[داده‌ها در فضای ویژگی، پیدا کردن مولفه‌های اصلی، داده‌ها در فضای مولفه‌های اصلی]
</div>
<br>

**43. Independent component analysis**
<div dir="rtl">
تحلیل مولفه‌های مستقل
</div>
<br>

**44. It is a technique meant to find the underlying generating sources.**
<div dir="rtl">
روشی است که برای پیدا کردن منابع مولد داده به کار می‌رود.
</div>
<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**
<div dir="rtl">
فرضیه‌ها - فرض می‌کنیم که داده‌ی $x$ توسط بردار $n$-بعدی $s=(s_1, \dots, s_n)$ تولید شده است، که $s_i$ها متغیرهای تصادفی مستقل  هستند، و این تولید داده از طریق بردار منبع به وسیله‌ی یک ماتریس معکوس‌پذیر و ترکیب‌کننده‌ی $A$ به صورت زیر انجام می‌گیرد:
</div>
<br>

**46. The goal is to find the unmixing matrix W=A−1.**
<div dir="rtl">
هدف پیدا کردن ماتریس ضدترکیب $W=A^{-1}$ است.
</div>
<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**
<div dir="rtl">
الگوریتم تحلیل مولفه‌های مستقل Bell و Sejnowski - این الگوریتم ماتریس ضدترکیب $W$ را در مراحل زیر پیدا می‌کند:
</div>
<br>

**48. Write the probability of x=As=W−1s as:**
<div dir="rtl">
احتمال $x = As = W^{-1}s$ به صورت زیر نوشته می‌شود:
</div>
<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**
<div dir="rtl">
با نمایش تابع سیگموئید با $g$، لگاریتم درست‌نمایی با توجه به داده‌های $\{x(i), \in [1, m]\}$ به صورت زیر نوشته می‌شود:
</div>
<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**
<div dir="rtl">
بنابراین، رویه‌ی یادگیری گرادیان تصادفی افزایشی برای هر نمونه از داده‌های آموزش $x(i)$ به گونه‌ای است که برای به‌روزرسانی $W$ داریم:
</div>
<br>

**51. The Machine Learning cheatsheets are now available in Farsi.**
<div dir="rtl">
راهنماهای کوتاه یادگیری ماشین هم‌اکنون به زبان فارسی نیز در دسترس می‌باشند.
</div>
<br>

**52. Original authors**
<div dir="rtl">
نویسنده‌های اصلی
</div>
<br>

**53. Translated by X, Y and Z**
<div dir="rtl">
ترجمه‌شده توسط X، Y، و Z
</div>
<br>

**54. Reviewed by X, Y and Z**
<div dir="rtl">
بازبینی‌شده توسط X، Y، و Z
</div>
<br>

**55. [Introduction, Motivation, Jensen's inequality]**
<div dir="rtl">
[معرفی، انگیزه، نابرابری جنسن]
</div>
<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**
<div dir="rtl">
[خوشه‌بندی، بیشینه‌سازی امید ریاضی، k-میانگین، خوشه‌بندی سلسله‌مراتبی، معیارها]
</div>
<br>

**57. [Dimension reduction, PCA, ICA]**
<div dir="rtl">
[کاهش ابعاد، تحلیل مولفه‌های اصلی، تحلیل مولفه‌های مستقل]
</div>
