**1. Unsupervised Learning cheatsheet**

&#10230; Gözetimsiz Öğrenme El Kitabı

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Gözetimsiz Öğrenmeye Giriş

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Motivasyon ― Gözetimsiz öğrenmenin amacı etiketlenmemiş verilerdeki gizli örüntüleri bulmaktır {x (1), ..., x (m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Jensen eşitsizliği - f bir konveks fonksiyon ve X bir rastgele değişken olsun. Aşağıdaki eşitsizliklerimiz:

<br>

**5. Clustering**

&#10230; Kümeleme

<br>

**6. Expectation-Maximization**

&#10230; Beklenti-Ençoklama (Maksimizasyon)

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Gizli değişkenler - Gizli değişkenler, tahmin problemlerini zorlaştıran ve çoğunlukla z olarak adlandırılan gizli / gözlemlenmemiş değişkenlerdir. Gizli değişkenlerin bulunduğu yerlerdeki en yaygın ayarlar şöyledir:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; Yöntem, Gizli değişken z, Açıklamalar

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [K Gaussianların birleşimi, Faktör analizi]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Algoritma - Beklenti-Ençoklama (Maksimizasyon) (BE) algoritması, θ parametresinin maksimum olabilirlik kestirimiyle tahmin edilmesinde, olasılığa ard arda alt sınırlar oluşturan (E-adımı) ve bu alt sınırın (M-adımı) aşağıdaki gibi optimize edildiği etkin bir yöntem sunar:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-adımı: Her bir veri noktasının x(i)'in belirli bir kümeden z(i) geldiğinin sonsal olasılık değerinin Qi(z(i)) hesaplanması aşağıdaki gibidir:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-adımı: Her bir küme modelini ayrı ayrı yeniden tahmin etmek için x(i) veri noktalarındaki kümeye özgü ağırlıklar olarak Qi(z(i)) sonsal olasılıklarının kullanımı aşağıdaki gibidir:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Gauss ilklendirme, Beklenti adımı, Maksimizasyon adımı, Yakınsaklık]

<br>

**14. k-means clustering**

&#10230; k-ortalamalar (k-means) kümeleme

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; C(i), i veri noktasının bulunduğu küme olmak üzere, μj j kümesinin merkez noktasıdır.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritma - Küme ortalamaları μ1, μ2, ..., μk∈Rn rasgele olarak başlatıldıktan sonra, k-ortalamalar algoritması yakınsayana kadar aşağıdaki adımı tekrar eder:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Başlangıç ortalaması, Küme Tanımlama, Ortalama Güncelleme, Yakınsama]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Bozulma fonksiyonu - Algoritmanın yakınsadığını görmek için aşağıdaki gibi tanımlanan bozulma fonksiyonuna bakarız:

<br>

**19. Hierarchical clustering**

&#10230; Hiyerarşik kümeleme

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Algoritma - Ardışık olarak iç içe geçmiş kümelerden oluşturan hiyerarşik bir yaklaşıma sahip bir kümeleme algoritmasıdır.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Türler - Aşağıdaki tabloda özetlenen farklı amaç fonksiyonlarını optimize etmeyi amaçlayan farklı hiyerarşik kümeleme algoritmaları vardır:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Ward bağlantı, Ortalama bağlantı, Tam bağlantı]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Küme mesafesi içinde minimize edin, Küme çiftleri arasındaki ortalama uzaklığı en aza indirin, Küme çiftleri arasındaki maksimum uzaklığı en aza indirin]

<br>

**24. Clustering assessment metrics**

&#10230; Kümeleme değerlendirme metrikleri

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; Gözetimsiz bir öğrenme ortamında, bir modelin performansını değerlendirmek çoğu zaman zordur, çünkü gözetimli öğrenme ortamında olduğu gibi, gerçek referans etiketlere sahip değiliz.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Siluet katsayısı - Bir örnek ile aynı sınıftaki diğer tüm noktalar arasındaki ortalama mesafeyi ve bir örnek ile bir sonraki en yakın kümedeki diğer tüm noktalar arasındaki ortalama mesafeyi not ederek, tek bir örnek için siluet katsayısı aşağıdaki gibi tanımlanır:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Calinski-Harabaz indeksi - k kümelerin sayısını belirtmek üzere Bk ve Wk sırasıyla, kümeler arası ve küme içi dağılım matrisleri olarak aşağıdaki gibi tanımlanır

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; Calinski-Harabaz indeksi s(k), kümelenme modelinin kümeleri ne kadar iyi tanımladığını gösterir, böylece skor ne kadar yüksek olursa, kümeler daha yoğun ve iyi ayrılır. Aşağıdaki şekilde tanımlanmıştır:

<br>

**29. Dimension reduction**

&#10230; Boyut küçültme

<br>

**30. Principal component analysis**

&#10230; Temel bileşenler analizi

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; Verilerin yansıtılacağı yönleri maksimize eden varyansı bulan bir boyut küçültme tekniğinidir.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Özdeğer, özvektör - Bir matris A∈Rn×n verildiğinde λ'nın, özvektör olarak adlandırılan bir vektör z∈Rn∖{0} varsa, A'nın bir özdeğeri olduğu söylenir:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spektral teorem - A∈Rn×n olsun. Eğer A simetrik ise, o zaman A gerçek ortogonal matris U∈Rn×n n ile diyagonalleştirilebilir. Λ=diag(λ1, ..., λn) yazarak, bizde:

<br>

**34. diagonal**

&#10230; diyagonal

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Not: En büyük özdeğere sahip özvektör, matris A'nın temel özvektörü olarak adlandırılır.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritma - Temel Bileşen Analizi (TBA) yöntemi, verilerin aşağıdaki gibi varyansı en üst düzeye çıkararak veriyi k boyutlarına yansıtan bir boyut azaltma tekniğidir:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Adım 1: Verileri ortalama 0 ve standart sapma 1 olacak şekilde normalleştirin.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Adım 2: Gerçek özdeğerler ile simetrik olan Σ=1mm∑i=1x(i)x(i)T∈Rn×n hesaplayın.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; u1, ...,uk∈Rn olmak üzere Σ ort'nin ortogonal ana özvektörlerini, yani k en büyük özdeğerlerin ortogonal özvektörlerini hesaplayın.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Adım 4: spanR (u1, ..., uk) üzerindeki verileri gösterin.

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Bu yöntem tüm k-boyutlu uzaylar arasındaki varyansı en üst düzeye çıkarır.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Öznitelik uzayındaki veri, Temel bileşenleri bul, Temel bileşenler uzayındaki veri]

<br>

**43. Independent component analysis**

&#10230; Bağımsız bileşen analizi

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; Temel oluşturan kaynakları bulmak için kullanılan bir tekniktir.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Varsayımlar - Verilerin x'in n boyutlu kaynak vektörü s=(s1, ..., sn) tarafından üretildiğini varsayıyoruz, burada si bağımsız rasgele değişkenler, bir karışım ve tekil olmayan bir matris A ile aşağıdaki gibi:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; Amaç, işlem görmemiş matrisini W=A−1 bulmaktır.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Bell ve Sejnowski ICA algoritması - Bu algoritma, aşağıdaki adımları izleyerek işlem görmemiş matrisi W'yi bulur:

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; X=As=W−1s olasılığını aşağıdaki gibi yazınız:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Eğitim verisi {x(i),i∈[[1, m]]} ve g sigmoid fonksiyonunu not ederek log olasılığını yazınız:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Bu nedenle, rassal (stokastik) eğim yükselme öğrenme kuralı, her bir eğitim örneği için x(i), W'yi aşağıdaki gibi güncelleştiririz:

<br>

**51. The Machine Learning cheatsheets are now available in Turkish.**

&#10230; Makine Öğrenmesi El Kitabı artık Türkçe dilinde mevcuttur.

<br>

**52. Original authors**

&#10230; Orjinal yazarlar

<br>

**53. Translated by X, Y and Z**

&#10230; X, Y ve Z ile çevrilmiştir.

<br>

**54. Reviewed by X, Y and Z**

&#10230; X, Y ve Z tarafından yorumlandı

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Giriş, Motivasyon, Jensen'in eşitsizliği]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Kümeleme, Beklenti-Ençoklama (Maksimizasyon), k-ortalamalar, Hiyerarşik kümeleme, Metrikler]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Boyut küçültme, TBA(PCA), BBA(ICA)]
