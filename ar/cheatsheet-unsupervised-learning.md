**1. Unsupervised Learning cheatsheet**

<div dir="rtl">
ورقة مراجعة للتعلم بدون إشراف
</div>

<br>

**2. Introduction to Unsupervised Learning**

<div dir=\"rtl\">
  مقدمة للتعلم بدون إشراف
</div>

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

<div dir=\"rtl\"> 
  {x(1),...,x(m)} الحافز ― الهدف من التعلم بدون إشراف هو إيجاد الأنماط الخفية في البيانات غير الموسومة 
</div> 

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

<div dir="rtl">
متباينة جينسن  ― لتكن f دالة محدبة و X متغير عشوائي. لدينا المتفاوتة التالية
:
</div>

<br>

**5. Clustering**

<div dir="rtl">
  تجميع
</div>
<br>

**6. Expectation-Maximization**

<div dir="rtl">
تحقيق أقصى قدر للتوقع
</div>
<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

<div dir="rtl">
المتغيرات الكامنة ― المتغيرات الكامنة هي متغيرات باطنية/غير معاينة تزيد من صعوبة مشاكل التقدير، غالبا ما ترمز بالحرف z. في مايلي الإعدادات الشائعة التي تحتوي على متغيرات كامنة.</div>
<br>

**8. [Setting, Latent variable z, Comments]**

<div dir="rtl">
إعداد، متغير كامن z، تعاليق</div>
<br>

**9. [Mixture of k Gaussians, Factor analysis]**

<div dir="rtl">
مزيج من k غاوسيات، تحليل العوامل </div>
<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

<div dir="rtl">
خوارزمية ― خوارزمية تحقيق أقصى قدر للتوقع هي عبارة عن طريقة فعالة لتقدير المعامل θ عبر تقدير الاحتمال الأرجح، و يتم ذلك بشكل تكراري حيث يتم إيجاد حد أدنى لدالة الإمكان (الخطوة M) ثم يتم استمثال ذلك الحد الأدنى (الخطوة E) كما يلي:
</div>
<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

<div dir="rtl">
الخطوة E : حساب الاحتمال البعدي Qi(z(i)) بأن تصدر كل نقطة x(i) من التجمع z(i) كما يلي:
</div>
<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

<div dir="rtl">
الخطوة M : يتم استعمال الاحتمالات البعدية Qi(z(i)) كأثقال خاصة لكل تجمع على النقط x(i) ، لكي يتم تقدير نموذج لكل تجمع بشكل منفصل، و ذلك كما يلي: 
</div>
<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

<div dir="rtl">
[ استهلالات غاوسية، خطوة التوقع، خطوة التعظيم، تقارب]
</div>
<br>

**14. k-means clustering**

<div dir="rtl">
تجميع k-متوسطات
</div>
<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

<div dir="rtl">
نرمز تجمع النقط i ب c(i) ، و نرمز ب μj  j مركز التجمع
</div>
<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

<div dir="rtl">
بعد الاستهلال العشوائي لمتوسطات التجمعات μ1,μ2,...,μk∈Rn، خوارزمية تجميع k-متوسطات تكرر الخطوة التالية حتى التقارب
</div>
<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

<div dir="rtl">
[استهلال المتوسطات، تعيين تجمع، تحديث المتوسطات، التقارب]</div>
<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

<div dir="rtl">
  دالة التشويه - لكي نتأكد من أن الخوارزمية تقاربت، ننظر إلى دالة التشويه المعرفة كما يلي:
</div>
<br>

**19. Hierarchical clustering**

<div dir="rtl">
  التجميع الهرمي
</div>
<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

<div dir="rtl">
  خوارزمية - هي عبارة عن خوارزمية تجميع تعتمد على طريقة تجميعية هرمية تبني مجموعات متداخلة بشكل متتال
</div>
<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

<div dir="rtl">
أنواع هنالك عدة أنواع من خوارزميات التجميع الهرمي التي ترمي إلى تحسين دوال هدف مختلفة، هاته الأنواع ملخصة في الجدول أسفله
</div>
<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

<div dir="rtl">
[الربط البَينِي، الربط المتوسط، الربط الكامل]</div>
<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

<div dir="rtl">
[تقليل داخل مسافة التجمع، تقليل متوسط المسافات بين أزواج التجمعات، تقليل المسافة القصوى بين أزواج التجمعات]</div>
<br>

**24. Clustering assessment metrics**

<div dir="rtl">
مقاييس تقدير التجميع
</div>
<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

<div dir="rtl">
في إعداد للتعلم بدون إشراف، من الصعب غالبا تقدير أداء نموذج ما لأننا لا نتوفر على القيم الحقيقية كما كان الحال في إعداد التعلم تحت إشراف 
</div>
<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

<div dir="rtl">
المعامل الظِلِّي - إذا رمزنا  aو b متوسط المسافة بين عينة و كل النقط المنتمية لنفس الصنف، و بين عينة  و كل النقط المنتمية لأقرب صنف، المعامل الظِلِّي s لعينة وحيدة معرف كالتالي:
</div>
<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

<div dir="rtl">
مؤشر كالينسكي هاراباز - إذا رمزنا بk لعدد التجمعات، Bk و Wk مصفوفات التشتت بين التجمعات و داخلها معرفة كالتالي: </div>
<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

<div dir="rtl">
مؤشر كالينسكي هاراباز s(k) يعطي تقييما للتجمعات الناتجة عن نموذج تجميعي، بحيث كلما كان التقييم أعلى كلما دل ذلك على  أن التجمعات أكثر كثافة و أكثر انفصالا. هذا المؤشر معرّف كالتالي</div>
<br>

**29. Dimension reduction**

<div dir="rtl">
تخفيض الأبعاد</div>
<br>

**30. Principal component analysis**

<div dir="rtl">
تحليل المكون الرئيسي
</div>
<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

<div dir="rtl">
إنها تقنية لخفض الأبعاد ترمي إلى إيجاد الاتجاهات المكبرة للتباين و التي تسقط عليها البيانات
</div>
<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

<div dir="rtl">
  قيمة ذاتية، متجه ذاتي - لتكن A∈Rn×n مصفوفة ، نقول أن λ قيمة ذاتية للمصفوفة A إذا وُجِد متجه z∈Rn∖{0} يسمى متجها ذاتيا، بحيث:
</div>
<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

<div dir="rtl">
  نظرية الطّيف لتكن A∈Rn×n. إذا كانت A متماثلة فإنها شبه قطرية بمصفوفة متعامدة U∈Rn×n. إذا رمزنا Λ=diag(λ1,...,λn) ، لدينا:
</div>
<br>

**34. diagonal**

<div dir="rtl">
قطري
</div>
<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

<div dir="rtl">
ملحوظة: المتجه الذاتي المرتبط بأكبر قيمة ذاتية يسمى بالمتجه الذاتي الرئيسي للمصفوفة A</div>
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

<br>

**51. The Machine Learning cheatsheets are now available in Arabic.**

&#10230;

<br>

**52. Original authors**

&#10230;

<br>

**53. Translated by X, Y and Z**

&#10230;

<br>

**54. Reviewed by X, Y and Z**

&#10230;

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230;

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230;

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230;
