**Unsupervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning)

<br>

**1. Unsupervised Learning cheatsheet**

&#10230; Cheatsheet học không giám sát

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Giới thiệu về học không giám sát

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Động lực ― Mục tiêu của học không giám sát là tìm được mẫu ẩn (hidden pattern) trong tập dữ liệu không được gán nhãn {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Bất đẳng thức Jensen - Cho f là một hàm lồi và X là một biến ngẫu nhiên. Chúng ta có bất đẳng thức sau:

<br>

**5. Clustering**

&#10230; Phân cụm

<br>

**6. Expectation-Maximization**

&#10230; Tối đa hoá kì vọng

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Các biến Latent - Các biến Latent là các biến ẩn/ không thấy được khiến cho việc dự đoán trở nên khó khăn, và thường được kí hiệu là z. ĐÂy là các thiết lập phổ biến mà các biến latent thường có:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Thiết lập, Biến Latent z, Các bình luận]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Sự kết hợp của k Gaussians, Phân tích hệ số]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Thuật toán - Thuật toán tối đa hoá kì vọng (EM) mang lại một phương thức có hiệu quả trong việc ước lượng tham số θ thông qua tối đa hoá giá trị ước lượng likelihood bằng cách lặp lại việc tạo nên một cận dưới cho likelihood (E-step) và tối ưu hoá cận dưới (M-step) như sau:   

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-step: Đánh giá xác suất hậu nghiệm Qi(z(i)) cho mỗi điểm dữ liệu x(i) đến từ một cụm z(i) cụ thể như sau:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-step: Sử dụng xác suất hậu nghiệm Qi(z(i)) như các trọng số cụ thể của cụm trên các điểm dữ liệu x(i) để ước lượng lại một cách riêng biệt cho mỗi mô hình cụm như sau:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Khởi tạo Gaussians, Bước kì vọng, Bước tối đa hoá, Hội tụ]

<br>

**14. k-means clustering**

&#10230; Phân cụm k-means

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Chúng ta kí hiệu c(i) là cụm của điểm dữ liệu i và μj là điểm trung tâm của cụm j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Thuật toán - Sau khi khởi tạo ngẫu nhiên các tâm của cụm (centroids) μ1,μ2,...,μk∈Rn, thuật toán k-means lặp lại bước sau cho đến khi hội tụ:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Khởi tạo giá trị trung bình, Gán cụm, Cập nhật giá trị trung bình, Hội tụ]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Hàm Distortion - Để nhận biết khi nào thuật toán hội tụ, chúng ta sẽ xem xét hàm distortion được định nghĩa như sau: 

<br>

**19. Hierarchical clustering**

&#10230; Hierarchical clustering

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Thuật toán - Là một thuật toán phân cụm với cách tiếp cận phân cấp kết tập, cách tiếp cận này sẽ xây dựng các cụm lồng nhau theo một quy tắc nối tiếp.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Các loại - Các loại thuật toán hierarchical clustering khác nhau với mục tiêu là tối ưu hoá các hàm đối tượng khác nhau sẽ được tổng kết trong bảng dưới đây:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Liên kết Ward, Liên kết trung bình, Liên kết hoàn chỉnh]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Tối thiểu hoá trong phạm vi khoảng cách của một cụm, Tối thiểu hoá khoảng cách trung bình giữa các cặp cụm, Tối thiểu hoá khoảng cách tối đa giữa các cặp cụm]

<br>

**24. Clustering assessment metrics**

&#10230; Các số liệu đánh giá phân cụm

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; Trong quá trình thiết lập học không giám sát, khá khó khăn để đánh giá hiệu năng của một mô hình vì chúng ta không có các nhãn đủ tin cậy như trong trường hợp của học có giám sát.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Hệ số Silhouette - Bằng việc kí hiệu a và b là khoảng cách trung bình giữa một điểm mẫu với các điểm khác trong cùng một lớp, và giữa một điểm mẫu với các điểm khác thuộc cụm kế cận gần nhất, hệ số silhouette s đối với một điểm mẫu đơn được định nghĩa như sau:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Chỉ số Calinski-Harabaz - Bằng việc kí hiệu k là số cụm, các chỉ số Bk và Wk về độ phân tán giữa và trong một cụm lần lượt được định nghĩa như là

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; Chỉ số Calinski-Harabaz s(k) cho biết khả năng phân cụm tốt đến đâu của một mô hình phân cụm, như là với score cao hơn, sẽ kém hơn và việc phân cụm tốt hơn. Nó được định nghĩa như sau:

<br>

**29. Dimension reduction**

&#10230; Giảm số chiều dữ liệu

<br>

**30. Principal component analysis**

&#10230; Principal component analysis

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; Là một kĩ thuật giảm số chiều dữ liệu, kĩ thuật này sẽ tìm các hướng tối đa hoá phương sai để chiếu dữ liệu trên đó. 

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Giá trị riêng, vector riêng - Cho ma trận A∈Rn×n, λ là giá trị riêng của A nếu tồn tại một vector z∈Rn∖{0}, gọi là vector riêng, mà ta có như sau:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Định lý Spectral - Với A∈Rn×n. Nếu A đối xứng thì A có thể chéo hoá bởi một ma trận trực giao U∈Rn×n. Bằng việc kí hiệu Λ=diag(λ1,...,λn), ta có:

<br>

**34. diagonal**

&#10230; đường chéo

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Chú thích: vector riêng tương ứng với giá trị riêng lớn nhất được gọi là vector riêng chính của ma trận A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k
dimensions by maximizing the variance of the data as follows:**

&#10230; Thuật toán - Principal Component Analysis (PCA) là một kĩ thuật giảm số chiều dữ liệu, nó sẽ chiếu dữ liệu lên k chiều bằng cách tối đa hoá phương sai của dữ liệu như sau:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Bước 1: Chuẩn hoá dữ liệu để có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Bước 2: Tính Σ=1mm∑i=1x(i)x(i)T∈Rn×n, là đối xứng với các giá trị riêng thực.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Bước 3: Tính u1,...,uk∈Rn là k vector riêng trực giao của Σ, tức các vector trực giao riêng của k giá trị riêng lớn nhất.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Bước 4: Chiếu dữ liệu lên spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Thủ tục này tối đa hoá phương sai giữa các không gian k-chiều.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Dữ liệu trong không gian đặc trưng, Tìm các thành phần chính, Dữ liệu trong không gian các thành phần chính]

<br>

**43. Independent component analysis**

&#10230; Independent component analysis

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; Là một kĩ thuật tìm các nguồn tạo cơ bản.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Giả định - Chúng ta giả sử rằng dữ liệu x của chúng ta được tạo ra bởi vector nguồn n-chiều s=(s1,...,sn), với si là các biến ngẫu nhiên độc lập, thông qua một ma trận mixing và non-singular A như sau:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; Mục tiêu là tìm ma trận unmixing W=A−1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Giải thuật Bell và Sejnowski ICA - Giải thuật này tìm ma trận unmixing W bằng các bước dưới đây:

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Ghi xác suất của x=As=W−1s như là:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Ghi log likelihood cho dữ liệu huấn luyện {x(i),i∈[[1,m]]} của chúng ta và bằng cách kí hiệu g là hàm sigmoid như là:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Vì thế, quy tắc học của stochastic gradient ascent là cho mỗi ví dụ huấn luyện x(i), chúng ta cập nhật W như sau:

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; Machine Learning cheatsheets hiện đã có bản [tiếng Việt].

<br>

**52. Original authors**

&#10230; Các tác giả

<br>

**53. Translated by X, Y and Z**

&#10230; Được dịch bởi X, Y và Z

<br>

**54. Reviewed by X, Y and Z**

&#10230; Được review bởi X, Y và Z

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Giới thiệu, Động lực, Bất đẳng thức Jensen]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Phân cụm, Tối đa hoá kì vọng, k-means, Hierarchical clustering, Các chỉ số]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Giảm số chiều dữ liệu, PCA, ICA]
