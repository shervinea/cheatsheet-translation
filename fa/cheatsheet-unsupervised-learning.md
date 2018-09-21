**1. Unsupervised Learning cheatsheet**

راهنمای کوتاه یادگیری بدون نظارت

<br>

**2. Introduction to Unsupervised Learning**

مبانی یادگیری بدون نظارت

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

انگیزه - هدف از یادگیری بدون نظارت کشف الگوهای پنهان در داده‌های بدون برچسب $\{x_1, \dots, x_m\}$ است.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

نابرابری ینسن - فرض کنید $f$ تابعی محدب و $X$ یک متغیر تصادفی باشد. در این صورت نابرابری زیر را داریم:

<br>

**5. Clustering**

خوشه‌بندی

<br>

**6. Expectation-Maximization**

بیشینه‌سازی امید ریاضی

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

متغیرهای نهفته - متغیرهای نهفته متغیرهای پنهان یا مشاهده‌نشده‌ای هستند که مسائل تخمین را دشوار می‌کنند، و معمولاً با $z$ نمایش داده می‌شوند. شرایط معمول که در آن‌ها متغیرهای نهفته وجود دارند در زیر آمده‌اند:

<br>

**8. [Setting, Latent variable z, Comments]**

[موقعیت، متغیر نهفته‌ی $z$، توضیحات]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

[ترکیب $k$ توزیع گاوسی، تحلیل عامل]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

الگوریتم - الگوریتم بیشینه‌سازی امید ریاضی روشی بهینه برای تخمین پارامتر $
theta$ از طریق تخمین درستی بشینه در اختیار قرار می‌دهد. این کار با تکرار مرحله‌ی به دست آوردن یک کران پایین برای درستی (مرحله‌ی امید ریاضی) و همچنین بهینه‌سازی آن کران پایین (مرحله‌ی بیشینه‌سازی) طبق توضیح زیر انجام می‌شود:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

مرحله‌ی امید ریاضی:‌احتمال پسین $Q_i(z(i))$ که هر نمونه داده $x(i)$ متعلق به خوشه‌ی $z(i)$ باشد به صورت زیر محاسبه می‌شود:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

مرحله‌ی بیشینه‌سازی: با استفاده از احتمالات پسین $Q_i(z(i))$ به عنوان وزن‌های وابسته به خوشه‌ها برای نمونه‌های داده‌ی $x(i)$، مدل مربوط به هر کدام از خوشه‌ها، طبق توضیح زیر، دوباره تخمین زده می‌شوند: 

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

[مقداردهی اولیه‌ی توزیع‌های گاوسی، مرحله‌ی امید ریاضی، مرحله‌ی بیشینه‌سازی، هم‌گرایی]

<br>

**14. k-means clustering**

خوشه‌بندی $k$-میانگین

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

توجه کنید که $c(i)$ خوشه‌ی نمونه داده‌ی $i$ و $\mu_j$ مرکز خوشه‌ی $j$ است.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

الگوریتم - بعد از مقداردهی اولیه‌ی تصادفی مراکز خوشه‌ها $\mu_1, \mu_2, \dots, \mu_k \in \mathbb{R}^n$، الگوریتم $k$-میانگین مراحل زیر را تا هم‌گرایی تکرار می‌کند:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

[مقداردهی اولیه‌ی میانگین‌ها، تخصیص خوشه، به‌روزرسانی میانگین‌ها، هم‌گرایی]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

تابع اعوجاج - برای تشخیص اینکه الگوریتم به هم‌گرایی رسیده است، به تابع اعوجاج که به صورت زیر تعریف می‌شود رجوع می‌کنیم:

<br>

**19. Hierarchical clustering**

خوشه‌بندی سلسله‌مراتبی

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

الگوریتم - یک الگوریتم خوشه‌بندی سلسله‌مراتبی تجمعی است که خوشه‌های تودرتو را به صورت پی‌در‌پی ایجاد می‌کند.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

انواع - انواع مختلفی الگوریتم خوشه‌بندی سلسله‌مراتبی وجود دارند که هر کدام به دنبال بهینه‌سازی توابع هدف مختلفی هستند، که در جدول زیر به اختصار آمده‌اند:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

[پیوند بخشی، پیوند میانگین، پیوند کامل]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

[کمینه‌کردن فاصله‌ی درونِ خوشه، کمینه‌کردن فاصله‌ی میانگین بین هر دو جفت خوشه، کمینه‌کردن حداکثر فاصله بین هر دو جفت خوشه]

<br>

**24. Clustering assessment metrics**

معیارهای ارزیابی خوشه‌بندی

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

در یک وضعیت یادگیری بدون نظارت، معمولاً ارزیابی یک مدل کار دشواری است، زیرا برخلاف حالت یادگیری نظارتی اطلاعاتی در مورد برچسب‌های حقیقی داده‌ها نداریم.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

ضریب نیم‌رخ - با نمایش $a$ به عنوان میانگین فاصله‌ی یک نمونه با همه‌ی نمونه‌های دیگر در همان کلاس، و با نمایش $b$ به عنوان میانگین فاصله‌ی یک نمونه با همه‌ی نمونه‌های دیگر از نزدیک‌ترین خوشه، ضریب نیم‌رخ $s$ به صورت زیر تعریف می‌شود:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

شاخص Calinski-Harabasz - با در نظر گرفتن $k$ به عنوان تعداد خوشه‌ها، ماتریس پراکندگی درون خوشه‌ای $B_k$ و ماتریس پراکندگی میان‌خوشه‌ای $W_k$ به صورت زیر تعریف می‌شوند:

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

شاخص Calinski-Harabasz $s(k)$ بیان می‌کند که یک مدل خوشه‌بندی چگونه خوشه‌های خود را مشخص می‌کند، به گونه‌ای که هر چقدر مقدار این شاخص بیشتر باشد، خوشه‌ها متراکم‌تر و از هم تفکیک‌یافته‌تر خواهند بود. این شاخص به صورت زیر تعریف می‌شود:

<br>

**29. Dimension reduction**

کاهش ابعاد

<br>

**30. Principal component analysis**

تحلیل مولفه‌های اصلی

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

روشی برای کاهش ابعاد است که جهت‌هایی را با حداکثر واریانس پیدا می‌کند تا داده‌ها را در آن جهت‌ها تصویر کند.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

مقدار ویژه، بردار ویژه - برای ماتریس دلخواه $A \in \mathbb{R}^{n \times n}$، $\lambda$ مقدار ویژه‌ی ماتریس $A$ است اگر وجود داشته باشد بردار $z \in \mathbb{R}^n \\ \{0\}$ که به آن بردار ویژه می‌گویند، به طوری که:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

قضیه‌ی طیفی - فرض کنید $A \in \mathbb{R}^{n \times n}$ باشد. اگر $A$ متقارن باشد، در این صورت $A$ توسط یک ماتریس حقیقی متعامد $U \in \mathbb{R} ^{n \times n}$ قطری‌پذیر است. با نمایش $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$ داریم:

<br>

**34. diagonal**

قطری

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

نکته: بردار ویژه‌ی متناظر با بزرگ‌ترین مقدار ویژه، بردار ویژه‌ی اصلی ماتریس $A$ نام دارد.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

الگوریتم - رویه‌ی تحلیل مولفه‌های اصلی یک روش کاهش ابعاد است که داده‌ها را در فضای $k$-بعدی با بیشینه کردن واریانس داده‌ها، به صورت زیر تصویر می‌کند:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

مرحله‌ی ۱: داده‌ها به گونه‌ای نرمال‌سازی می‌شوند که میانگین ۰ و انحراف معیار ۱ داشته باشند.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

مرحله‌ی ۲: مقدار $\Sigma = \frac{1}{m} \sum_{i=1}^m x(i) x(i)^T \in \mathbb{R}^{n \times n}$، که ماتریسی متقارن با مقادیر ویژه‌ی حقیقی است محاسبه می‌شود. 

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

مرحله‌ی ۳: بردارهای $u_1, \dots, u_k \in \mathbb{R}^n$ که $k$ بردارهای ویژه‌ی اصلی متعامد $\Sigma$ هستند محاسبه می‌شوند. این بردارهای ویژه متناظر با $k$ مقدار ویژه با بزرگ‌ترین مقدار هستند.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

مرحله‌ی ۴: داده‌ها بر روی فضای $\text{span}_ {\mathbb{R}} (u_1, \dots, u_k)$ تصویر می‌شوند.

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

این رویه واریانس را در فضای $k$-بعدی به دست آمده بیشینه می‌کند.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

[داده‌ها در فضای ویژگی، پیدا کردن مولفه‌های اصلی، داده‌ها در فضای مولفه‌های اصلی]

<br>

**43. Independent component analysis**

تحلیل مولفه‌های مستقل

<br>

**44. It is a technique meant to find the underlying generating sources.**

روشی است که برای پیدا کردن منابع مولد داده به کار می‌رود.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

فرضیه‌ها - فرض می‌کنیم که داده‌ی $x$ توسط بردار $n$-بعدی $s=(s_1, \dots, s_n)$ تولید شده است، که $s_i$ها متغیرهای تصادفی مستقل  هستند، و این تولید داده از طریق بردار منبع به وسیله‌ی یک ماتریس معکوس‌پذیر و ترکیب‌کننده‌ی $A$ به صورت زیر انجام می‌گیرد:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

هدف پیدا کردن ماتریس ضدترکیب $W=A^{-1}$ است.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

الگوریتم تحلیل مولفه‌های مستقل Bell و Sejnowski - این الگوریتم ماتریس ضدترکیب $W$ را در مراحل زیر پیدا می‌کند:

<br>

**48. Write the probability of x=As=W−1s as:**

احتمال $x = As = W^{-1}s$ به صورت زیر نوشته می‌شود:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

با نمایش تابع سیگموئید با $g$، لگاریتم درست‌نمایی با توجه به داده‌های $\{x(i), \in [1, m]\}$ به صورت زیر نوشته می‌شود:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

بنابراین، رویه‌ی یادگیری گرادیان تصادفی افزایشی برای هر نمونه از داده‌های آموزش $x(i)$ به گونه‌ای است که برای به‌روزرسانی $W$ داریم:

