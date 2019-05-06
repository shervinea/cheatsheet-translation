**1. Unsupervised Learning cheatsheet**

&#10230; **1. Ringkasan Unsupervised Learning**

<br>

**2. Introduction to Unsupervised Learning**

&#10230; **2. Pengenalan Unsupervised Learning**

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; **3. Motivasi ― Tujuan dari unsupervised learning adalah untuk menemukan pola dari data yang tidak memiliki label {x(1),..., x(m)}**

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; **4. Ketidaksamaan Jensen ― Diberikan f yang merupakan fungsi yang konveks dan X adalah random variabel. Kita memiliki ketidaksamaan:**

<br>

**5. Clustering**

&#10230;  **5. Clustering**

<br>

**6. Expectation-Maximization**

&#10230; **6. Expection-Maximization**

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; **7. Latent variables ― Variabel laten adalah variabel yang tersembunyi atau belum dilakukan observasi sehingga membuat estimasi masalah menjadi lebih sulit, dan seringkali didenotasikan dengan z. Berikut merupakan setting dimana latent variables umum digunakan:**

<br>AlgoritmaAlgoritma

**8. [Setting, Latent variable z, Comments]**

&#10230; **8. [Setting, Variabel laten z, Komentar ]**
<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; **9. [Gabungan dari gaussin K, analisis faktor]**
<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230;**10. Algoritma ― Expectation-maximizaiton (EM) memberikan metode yang efisien dalam mengestimasi parameter θ melalui estimasi maximum likelihood dengan mengknonstruksi secara berulang lower-bound dari likelihood (E-step) dan optimasasi lowen-bound (M-Step) sebagai berikut:**

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; **11. E-step: Mengevaluasi probabilitas posterior Qi(z(i)) yang setiap data point x(i) dari kluster khusus z(i) sebagai berikut:**

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; **12. M-step: Menggunakan probabilitas posterior Qi(z(i)) sebagai kluster khusus pada bobot data poin x(i) untuk memisahkan perhitungan estimasi model setiap kluster sebagai berikut:** 

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; **13. [Gaussian initialization, Expectation step, Maximization step, Convergence]**

<br>

**14. k-means clustering**

&#10230; **14. k-means clustering**

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; **15. Catatan: c(i) merupakan cluster data poin i dan μj merupakan pusat kluster j.**

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; **16. Algoritma ― Setelah secara random menginisialisasi kluster centroids μ1,μ2,...,μk∈Rn, algoritma k-means mengulangi langkah-langkah berikut sampai koonvergen:**

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; **17. [Means initialization, Cluster assignment, Means update, Convergence]**

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; **18. Fungsi distorsi ― Untuk melihat jika sebuah algoritma telah konvergen, kita lihat pada fungsi distorsi yang didefiniskan sebagai berikut:**

<br>

**19. Hierarchical clustering**

&#10230; **19. Hierarchical clustering**

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; **20. Algoritma ― merupakan sebuah algoritma clustering dengan pendekatan hirarki agglomeratif yang membangung sarang kluster dalam bentuk yang sempurna** 

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; **21. Jenis ― Terdapat beberapa perbedaan dari algoritma klustering yang bertujuan untuk mengoptimalisasi perbedaan fungsi-fungsi tertentu, yang diringkas dalam tabel berikut:**

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; **22. [Ward linkage, Average linkage, Complete lingkage]**

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; **23. [Minimalisasi jarak antar kluster,Minimilasisasi rata-rata jarak antar pasangan kluster, Minimalisasi jarak maksimum antar pasangan kluster]** 

<br>

**24. Clustering assessment metrics**

&#10230; **24. Matriks penilaian kluster**

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; **25. Dalam setting unspervised learning, seringnya lebih sulit untuk melakukan penilaian performa sebuah model saat kita tidak memiliki label dasar yang benar sebagaimana yang dilakukan dalam setting supervised learning**

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; **26. Koefisien Silhoutte ― Dengan notasi a dan b yang merupakan rerata jarak antara sampel a dan semua point dalam kelas yang sama, dan antara sample dan semua point dalam kluster terdekat, koefisien siloute s untuk sebuah sample tunggal didefinisikan sebagai berikut:**

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; **27. Index Calinski-Harabaz ― Dengan notasi k merupkan jumlah kluster, Bk dan Wk merupakan matriks dispersi di luar dan di dalam kluster, didefinisikan sebagai berikut:**

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; **28. Index Calinski-Harabaz s(k) mengindikasikan bagaimana bagus/tidaknya sebuah model kluster mendefinisikan klusternya, seperti nilai score yang tinggi, semakin padat dan terpisah klusternya. Didefiniskan sebagai berikut:**

<br>

**29. Dimension reduction**

&#10230; **29. Pengurangan Dimensi (Dimension reduction)**

<br>

**30. Principal component analysis**

&#10230; **30. Principal component analysis**

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; **31. Merupakan teknik pengurangan dimensi dengan menemukan varian maksimal arah yang memproyeksikan data.**

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; **32. Eigenvalue, eigenvector ― Diberikan sebuah matriks A∈Rn×n, λ merupakan sebuah eigenvalue dari A dimana terdapat sebuah vektor z∈Rn∖{0}, disebut eigenvector, sehingga kita memiliki:** 
<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; **33. Theorema Spectral ― Diberikan A∈Rn×n. Jika A adalah simetris, maka A bersifat diagonal dengan sebuah orthogonal matriks U∈Rn×n. Dengan notasi Λ=diag(λ1,...,λn), kita memiliki:**

<br>

**34. diagonal**

&#10230; **34. Diagonal**

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; **35. Catatan: Eigenvector diasosiasikan dengan eigenvalue terbesar disebut prinsipal eigenvector dari matriks A.**

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230; **36. Algoritma ― Prosedur PCA adalah teknik pengurangan dimensi yang memproyeksikan data pada dimensi K dengan mamaksimalkan variansi dari data sebagai berikut:** 

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; **37. Langkah 1: Normalisasi data yang memiliki rerata 0 dan standar deviasi 1.**

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; **38. Langkah 2: Hitung Σ=1mm∑i=1x(i)x(i)T∈Rn×n, yang merupakan simetris dengan nilai real eigenvalues.**

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; **39. Langkah 3: Hitung u1,...,uk∈Rn dimana k merupakan prinsipal ortoghonal eigenvector dari Σ, sebagai contohnya nilai eigenvector dari eigenvalues k terbesar.**

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; **40. Langkah 4: Proyeksikan data pada spanR(u1,...,uk).**

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; **41. Prosedur tersebut memaksimalkan variansi nilai diantara semua ruang k-dimensional.**

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; **42. [Data dalam ruang fitur, Temukan prinsipal komponen, Data di ruang prinsipal komponen]**

<br>

**43. Independent component analysis**

&#10230; **43. Analisis komponen independen**

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; **44. Merupakan teknik yang bermaksud untuk menemukan sumber paling dasar.**

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; **45. Asumsi ― Kita mengasumsikan bahwa data kita x telah dibuat dengan sumber n-dimensional vector s=(s1,...,sn), dimana si adalah variabel random independen, melalui mixing dan matriks non-singular sebagai berikut:**

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; **46. Tujuannya adalah untuk menemukan unmixing matrix dari W=A-1**

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; **47. Algoritma Bell dan Sejnowski ― Alogoritma ini bertujuan untuk menemukan unimixing matriks W dengan langkah-langkah sebagai berikut:**

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; **48. Tulis probabilitas dari x=As=W−1s sebagai berikut:**

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; **49. Tulis kecenderungan log dari data latih {x(i),i∈[[1,m]]} dan dengan notasi g yang merupakan fungsi sigmoid sebagai berikut:**

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; **50. Sehingga, aturan learning dari stochastic gradient ascent adalah bahwa setiap contoh data latih x(i), kita memperbarui W sebagai berikut:**

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; **51. Catatan ringkas machine learning ini terdapat dalam versi bahasa Indonesia.**

<br>

**52. Original authors**

&#10230; **52. Penulis Asli: Shervine Amidi**

<br>

**53. Translated by X, Y and Z**

&#10230; **53. Diterjemahkan oleh Sony Wicaksono**

<br>

**54. Reviewed by X, Y and Z**

&#10230; **54. Disunting oleh X, Y, dan Z**

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; **55. [Pengenalan, motivasi, Pertidaksamaan Jensen]**

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; **56. [Clustering, Expectataion-Maximization, k-means, Hierarchical clustering, Metrics]**

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; **57. [Pengurangan Dimensi, PCA, ICA]**
