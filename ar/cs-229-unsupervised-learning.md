**1. Unsupervised Learning cheatsheet**

<div dir="rtl">
  مرجع سريع للتعلّم غير المُوَجَّه
</div>

<br>

**2. Introduction to Unsupervised Learning**

<div dir="rtl">
  مقدمة للتعلّم غير المُوَجَّه
</div>

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

<div dir=\"rtl\">
  {x(1),...,x(m)} الحافز ― الهدف من التعلّم غير المُوَجَّه هو إيجاد الأنماط الخفية في البيانات غير المٌعلمّة
</div>

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

<div dir="rtl">
متباينة جينسن ― لتكن f دالة محدبة و X متغير عشوائي. لدينا المتباينة التالية:
</div>

<br>

**5. Clustering**

<div dir="rtl">
  التجميع
</div>
<br>

**6. Expectation-Maximization**

<div dir="rtl">
  تعظيم القيمة المتوقعة (Expectation-Maximization)
</div>
<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

<div dir="rtl">
  المتغيرات الكامنة ― المتغيرات الكامنة هي متغيرات مخفية/غير معاينة تزيد من صعوبة مشاكل التقدير، غالباً ما ترمز بالحرف z. في مايلي الإعدادات الشائعة التي تحتوي على متغيرات كامنة:
</div>
<br>

**8. [Setting, Latent variable z, Comments]**

<div dir="rtl">
  [الإعداد، المتغير الكامن z، ملاحظات]
</div>
<br>

**9. [Mixture of k Gaussians, Factor analysis]**

<div dir="rtl">
  [خليط من k توزيع جاوسي، تحليل عاملي]
</div>
<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

<div dir="rtl">
خوارزمية ― تعظيم القيمة المتوقعة (Expectation-Maximization) هي عبارة عن طريقة فعالة لتقدير المُدخل θ عبر تقدير تقدير الأرجحية الأعلى (maximum likelihood estimation)، ويتم ذلك بشكل تكراري حيث يتم إيجاد حد أدنى للأرجحية (الخطوة M)، ثم يتم تحسين (optimizing) ذلك الحد الأدنى (الخطوة E)، كما يلي:
</div>
<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

<div dir="rtl">
الخطوة E : حساب الاحتمال البعدي Qi(z(i)) بأن تصدر كل نقطة x(i) من مجموعة (cluster) z(i) كما يلي:
</div>
<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

<div dir="rtl">
  الخطوة M : يتم استعمال الاحتمالات البعدية Qi(z(i)) كأوزان خاصة لكل مجموعة (cluster) على النقط x(i)، لكي يتم تقدير نموذج لكل مجموعة بشكل منفصل، و ذلك كما يلي:
</div>
<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

<div dir="rtl">
[استهلالات جاوسية، خطوة القيمة المتوقعة، خطوة التعظيم، التقارب]
</div>
<br>

**14. k-means clustering**

<div dir="rtl">
التجميع بالمتوسطات k (k-mean clustering)
</div>
<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

<div dir="rtl">
نرمز لمجموعة النقط i بـ c(i)، ونرمز بـ μj مركز المجموعات j.
</div>
<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

<div dir="rtl">
خوارزمية - بعد الاستهلال العشوائي للنقاط المركزية (centroids) للمجوعات μ1,μ2,...,μk∈Rn، التجميع بالمتوسطات k تكرر الخطوة التالية حتى التقارب:
</div>
<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

<div dir="rtl">
[استهلال المتوسطات، تعيين المجموعات، تحديث المتوسطات، التقارب]
</div>
<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

<div dir="rtl">
دالة التحريف (distortion function) - لكي نتأكد من أن الخوارزمية تقاربت، ننظر إلى دالة التحريف المعرفة كما يلي:
</div>
<br>

**19. Hierarchical clustering**

<div dir="rtl">
  التجميع الهرمي
</div>
<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

<div dir="rtl">
خوارزمية - هي عبارة عن خوارزمية تجميع تعتمد على طريقة تجميع هرمية تبني مجموعات متداخلة بشكل متتال.
</div>
<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

<div dir="rtl">
الأنواع - هنالك عدة أنواع من خوارزميات التجميع الهرمي التي ترمي إلى تحسين دوال هدف (objective functions) مختلفة، هذه الأنواع ملخصة في الجدول التالي:
</div>
<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

<div dir="rtl">
[ربط وارْد (ward linkage)، الربط المتوسط، الربط الكامل]
</div>
<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

<div dir="rtl">
[تصغير المسافة داخل المجموعة، تصغير متوسط المسافة بين أزواج المجموعات، تصغير المسافة العظمى بين أزواج المجموعات]</div>
<br>

**24. Clustering assessment metrics**

<div dir="rtl">
مقاييس تقدير المجموعات
</div>
<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

<div dir="rtl">
في التعلّم غير المُوَجَّه من الصعب غالباً تقدير أداء نموذج ما، لأن القيم الحقيقية تكون غير متوفرة كما هو الحال في التعلًم المُوَجَّه.</div>
<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

<div dir="rtl">
معامل الظّل (silhouette coefficient) - إذا رمزنا a و b لمتوسط المسافة بين عينة وكل النقط المنتمية لنفس الصنف، و بين عينة وكل النقط المنتمية لأقرب مجموعة، المعامل الظِلِّي s لعينة واحدة معرف كالتالي:
</div>
<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

<div dir="rtl">
مؤشر كالينسكي-هارباز (Calinski-Harabaz index) - إذا رمزنا بـ k لعدد المجموعات، فإن Bk و Wk مصفوفتي التشتت بين المجموعات وداخلها تعرف كالتالي:
</div>
<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

<div dir="rtl">
مؤشر كالينسكي-هارباز s(k) يشير إلى جودة نموذج تجميعي في تعريف مجموعاته، بحيث كلما كانت النتيجة أعلى كلما دل ذلك على أن المجموعات أكثر كثافة وأكثر انفصالاً فيما بينها. هذا المؤشر معرّف كالتالي:
</div>
<br>

**29. Dimension reduction**

<div dir="rtl">
تقليص الأبعاد</div>
<br>

**30. Principal component analysis**

<div dir="rtl">
تحليل المكون الرئيس
</div>
<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

<div dir="rtl">
إنها طريقة لتقليص الأبعاد ترمي إلى إيجاد الاتجاهات المعظمة للتباين من أجل إسقاط البيانات عليها.
</div>
<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

<div dir="rtl">
قيمة ذاتية (eigenvalue)، متجه ذاتي (eigenvector) - لتكن A∈Rn×n مصفوفة، نقول أن λ قيمة ذاتية للمصفوفة A إذا وُجِد متجه z∈Rn∖{0} يسمى متجهاً ذاتياً، بحيث:
</div>
<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

<div dir="rtl">
مبرهنة الطّيف (Spectral theorem) - لتكن A∈Rn×n. إذا كانت A متناظرة فإنها يمكن أن تكون شبه قطرية عن طريق مصفوفة متعامدة حقيقية U∈Rn×n. إذا رمزنا Λ=diag(λ1,...,λn) ، لدينا:
</div>
<br>

**34. diagonal**

<div dir="rtl">
قطري
</div>
<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

<div dir="rtl">
ملحوظة: المتجه الذاتي المرتبط بأكبر قيمة ذاتية يسمى بالمتجه الذاتي الرئيسي (principal eigenvector) للمصفوفة A.
</div>
<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

<div dir="rtl">
خوارزمية - تحليل المكون الرئيس (Principal Component Analysis (PCA)) طريقة لخفض الأبعاد تهدف إلى إسقاط البيانات على k بُعد بحيث يتم تعطيم التباين (variance)، خطواتها كالتالي:
</div>
<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

<div dir="rtl">
الخطوة 1: تسوية البيانات بحيث تصبح ذات متوسط يساوي صفر وانحراف معياري يساوي واحد.
</div>
 <br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

<div dir="rtl">
الخطوة 2: حساب Σ=1mm∑i=1x(i)x(i)T∈Rn×n، وهي متناظرة وذات قيم ذاتية حقيقية.
</div>
<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

<div dir="rtl">
الخطوة 3: حساب u1,...,uk∈Rn المتجهات الذاتية الرئيسية المتعامدة لـ Σ وعددها k ، بعبارة أخرى، k من المتجهات الذاتية المتعامدة ذات القيم الذاتية الأكبر.
</div>
<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

<div dir="rtl">
الخطوة 4: إسقاط البيانات على spanR(u1,...,uk).
</div>
<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

<div dir="rtl">
هذا الإجراء يعظم التباين بين كل الفضاءات البُعدية.
</div>
<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

<div dir="rtl">
[بيانات في فضاء الخصائص, أوجد المكونات الرئيسية, بيانات في فضاء المكونات الرئيسية]
</div>
<br>

**43. Independent component analysis**

<div dir="rtl">
تحليل المكونات المستقلة
</div>
<br>

**44. It is a technique meant to find the underlying generating sources.**

<div dir="rtl">
هي طريقة تهدف إلى إيجاد المصادر التوليدية الكامنة.
</div>
<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

<div dir="rtl">
افتراضات - لنفترض أن بياناتنا x تم توليدها عن طريق المتجه المصدر s=(s1,...,sn) ذا n بُعد، حيث si متغيرات عشوائية مستقلة، وذلك عبر مصفوفة خلط غير منفردة (mixing and non-singular) A كالتالي:
</div>
<br>

**46. The goal is to find the unmixing matrix W=A−1.**

<div dir="rtl">
الهدف هو العثور على مصفوفة الفصل W=A−1.
</div>
<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**
<div dir="rtl">
خوارزمية تحليل المكونات المستقلة (ICA) لبيل وسجنوسكي (Bell and Sejnowski) - هذه الخوارزمية تجد مصفوفة الفصل W عن طريق الخطوات التالية:
</div>
<br>

**48. Write the probability of x=As=W−1s as:**

<div dir="rtl">
اكتب الاحتمال لـ x=As=W−1s كالتالي:
</div>
<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

<div dir="rtl">
لتكن {x(i),i∈[[1,m]]} بيانات التمرن و g دالة سيجمويد، اكتب الأرجحية اللوغاريتمية (log likelihood) كالتالي:
</div>
<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

<div dir="rtl">
هكذا، باستخدام الصعود الاشتقاقي العشوائي (stochastic gradient ascent)، لكل عينة تدريب x(i) نقوم بتحديث W كما يلي:
</div>
<br>

**51. The Machine Learning cheatsheets are now available in Arabic.**

<div dir="rtl">
المرجع السريع لتعلم الآلة متوفر الآن باللغة العربية.
</div>
<br>

**52. Original authors**

<div dir="rtl">
المحررون الأصليون
</div>
<br>

**53. Translated by X, Y and Z**

<div dir="rtl">
تمت الترجمة بواسطة X,Y و Z
</div>
<br>

**54. Reviewed by X, Y and Z**

<div dir="rtl">
تمت المراجعة بواسطة X,Y و Z
</div>
<br>

**55. [Introduction, Motivation, Jensen's inequality]**

<div dir="rtl">
[مقدمة، الحافز، متباينة جينسن]
</div>
<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

<div dir="rtl">
[التجميع، تعظيم القيمة المتوقعة، تجميع k-متوسطات، التجميع الهرمي، مقاييس]
</div>
<br>

**57. [Dimension reduction, PCA, ICA]**

<div dir="rtl">
[تقليص الأبعاد، تحليل المكون الرئيس (PCA)، تحليل المكونات المستقلة (ICA)]
</div>
<br>
