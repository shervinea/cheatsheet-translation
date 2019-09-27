**Unsupervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning)

<br>

**1. Unsupervised Learning cheatsheet**

&#10230; Cheatsheet học không giám sát

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Giới thiệu về học không giám sát 

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Mục tiêu - Mục đích của học học không giám sát là tìm ra mô hình ẩn trong dữ liệu chưa được gán nhán {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Bất đẳng thức Jensen - Với f là một hàm lồi và X là một biến ngấu nhiên. Ta có bất đẳng thức dưới đây: 

<br>

**5. Clustering**

&#10230; Phân cụm

<br>

**6. Expectation-Maximization**

&#10230; Tối đa hóa kỳ vọng

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Biến tiềm ẩn - Biến tiềm ẩn là những biến bị ẩn/không được nhìn thấy dấn đến các vấn đề khó về đánh giá, và thường được ký hiệu là z. Đây là các thiết lập phổ biến nhất khi có biến tiềm ẩn

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Thiết lập, Biến tiềm ẩn z, Ghi chú]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Kết hợp của k Gaussian, Phân tích nhân tố]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Giải thuật ― Giải thuật "Tối ưu hóa kỳ vọng" (EM) đưa ra một phương pháp hiệu quả cho việc đánh giá tham số θ thông qua việc tối đa hóa việc đánh giá hàm likelihood bằng việc xây dựng lặp đi lặp lại một cận dưới trên hàm likelihood ("bước kỳ vọng" E-step) và tối ưu hóa cận dưới ("bước tối đa hóa" M-step) như sau:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; "Bước kỳ vọng" E-step: Tính xác suất sau Qi(z(i)) với từng điểm dữ liệu x(i) đến từ một cụm thực tế z(i) như sau:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; "Bước tối đa hóa" M-step: Dùng xác suất sau Qi(z(i)) như là trọng số cụm đặc biệt trên điểm dữ liệu x(i) để riêng rẽ đánh giá lại từng mô hình cluster như sau:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230;[Khởi tạo Gaussian, Bước kỳ vọng, Bước tối đa hóa , Hội tụ]

<br>

**14. k-means clustering**

&#10230; Phân cụm K-Means

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Chúng ta chú ý c(i) là cụm của điểm dữ liệu i và μj là chính giữa của j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Giải thuật - Sau khi khởi tạo ngẫu nhiêu các điểm trung tâm của các cụm μ1,μ2,...,μk∈Rn, thuật toán K-Means thực hiện lặp đi các bước dưới đây cho đến khi hội tụ:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Khởi tạo , Gán cụm, Cập nhật, Hội tụ]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Hàm sai số ― Để biết thuật toán đã hội tụ, chúng ta nhìn vào hàm sai số được định nghĩa như sau:

<br>

**19. Hierarchical clustering**

&#10230; Phân cụm phân tầng

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Giải thuật - Đây là giải thuật phân cụm với một phần tầng "đi từ dưới lên" để xây dựng các cụm lồng nhau theo cách liên tiếp

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Loại - Có nhiều loại giải thuật phân cụm phân tầng để nhằm mục đích tối ưu các hàm mục tiêu khác nhau, những hàm được tổng kết dưới bảng sau:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Liên kết nhóm, Liên kết trung bình, Liên kết hoàn chỉnh]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Tối thiểu hóa khoảng cách giữa các cụm, tối thiếu hiểu hóa khoảng cách trung bình giữa các cặp cụm, Tối thiểu hóa khoảng cách tối đa giữa các cặp cụm]

<br>

**24. Clustering assessment metrics**

&#10230; Ma trận đánh giá phân cụm

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; Trong một cài đặt học không giám sát, thường khó để đánh giá hiệu quả của một mô hình bở chúng ta không có tập dữ liệu đã gán nhán chuẩn như trong trường hợp cài đặt học có giám sát. 

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Hệ số hình chiéu - Với a là khoảng cách trung bình giữa một mẫu với tất các điểm khác trong cùng nhóm, và b là khoảng cách trung bình giữa một mẫu với tất cacr các điểm khác trong cụm gần nhất, hệ số hình chiếu s cho một mẫu đơn được định nghĩa như sau: 

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Chỉ mục Calinski-Harabaz - Với k là số cụm, Bk và Wk là ................ được định nghĩa bởi

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; chỉ mục Calinski-Harabaz s(k) cho biết độ tốt của mô hình phân cụm định nghĩa các cụm của nó, như là điểm càng cao, thì các cụm càng dày và càng tách biệt. Nó được định nghĩa như sau: 

<br>

**29. Dimension reduction**

&#10230; Giảm chiều

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

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

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
