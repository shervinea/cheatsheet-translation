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

[موقعیت، متغیر نفهته‌ی $z$، توضیح]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

[ترکیب $k$ توزیع گاوسی، تحلیل عامل]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

الگوریتم - الگوریتم بیشینه‌سازی امید ریاضی روشی بهینه برای تخمین پارامترهای $
theta$ از طریق تخمین درستی بشینه در اختیار قرار می‌دهد. این کار با تکرار مرحله‌ی به دست آوردن یک کران پایین برای درستی (مرحله‌ی امید ریاضی) و همچنین بهینه‌سازی آن کران پایین (مرحله‌ی بیشینه‌سازی) طبق توضیح زیر انجام می‌شود:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

مرحله‌ی امید ریاضی:‌احتمال پسین $Q_i(z(i))$ که هر نمونه داده $x(i)$ متعلق به خوشه‌ی $z(i)$ باشد به صورت زیر محاسب می‌شود:

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

تابع اعوجاج - برای تشخیص اینکه الگوریتم به هم‌گرایی رسیده است، با تابع اعوجاج که به صورت زیر تعریف می‌شود رجوع می‌کنیم:

<br>

**19. Hierarchical clustering**

خوشه‌بندی سلسله‌مراتبی

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

الگوریتم - یک الگوریتم خوشه‌بندی سلسله‌مراتبی تجمعی است که خوشه‌های تودرتو را به صورت پی‌در‌پی ایجاد می‌کند.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230;

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230;

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230;

<br>

**24. Clustering assessment metrics**

&#10230;

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230;

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230;

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230;

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230;

<br>

**29. Dimension reduction**

&#10230;

<br>

**30. Principal component analysis**

&#10230;

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230;

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230;

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230;

<br>

**34. diagonal**

&#10230;

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230;

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230;

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230;

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230;

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230;

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230;

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230;

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230;

<br>

**43. Independent component analysis**

&#10230;

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230;

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230;

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230;

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230;

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230;

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230;

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230;

