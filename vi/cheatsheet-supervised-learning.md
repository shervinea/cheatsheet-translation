**1. Supervised Learning cheatsheet**

&#10230; Cheatsheet học có giám sát

<br>

**2. Introduction to Supervised Learning**

&#10230; Giới thiệu về học có giám sát

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; Cho một tập hợp các điểm dữ liệu {x(1),...,x(m)} tương ứng với đó là tập các đầu ra {y(1),...,y(m)}, chúng ta muốn xây dựng một bộ phân loại học được cách dự đoán y từ x.

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Loại dự đoán - Các loại mô hình dự đoán được tổng kết trong bảng bên dưới: 

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Hồi quy, Phân loại, Đầu ra, Các ví dụ]

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Liên tục, Lớp, Hồi quy tuyến tính, Hồi quy Logistic, SVM, Naive Bayes]

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Loại mô hình - Các mô hình khác nhau được tổng kết trong bảng bên dưới:

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Mô hình phân biệt, Mô hình sinh, Mục tiêu, Những gì học được, Hình minh hoạ, Các ví dụ]

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [Ước lượng trực tiếp P(y|x), Ước lượng P(x|y) để tiếp tục suy luận P(y|x), Biên quyết định, Phân bố xác suất của dữ liệu, Hồi quy, SVMs, GDA, Naive Bayes]

<br>

**10. Notations and general concepts**

&#10230; Các kí hiệu và khái niệm tổng quát

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; Hypothesis - Hypothesis được kí hiệu là hθ, là một mô hình mà chúng ta chọn. Với dữ liệu đầu vào cho trước x(i), mô hình dự đoán đầu ra là hθ(x(i)).

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; Hàm mất mát - Hàm mất mát là một hàm số dạng: L:(z,y)∈R×Y⟼L(z,y)∈R lấy đầu vào là giá trị dự đoán được z tương ứng với đầu ra thực tế là y, hàm có đầu ra là sự khác biệt giữa hai giá trị này. Các hàm mất mát phổ biến được tổng kết ở bảng dưới đây:

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [Least squared error, Mất mát Logistic, Mất mát Hinge, Cross-entropy]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Hồi quy tuyến tính, Hồi quy Logistic, SVM, Mạng neural]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Hàm giá trị (Cost function) - Cost function J thường được sử dụng để đánh giá hiệu năng của mô hình và được định nghĩa với hàm mất mát L như sau:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Gradient descent - Bằng việc kí hiệu α∈R là tốc độ học, việc cập nhật quy tắc/ luật cho gradient descent được mô tả với tốc độ học và cost function J như sau:

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Chú ý: Stochastic gradient descent (SGD) là việc cập nhật tham số dựa theo mỗi ví dụ huấn luyện, và batch gradient descent là dựa trên một lô (batch) các ví dụ huấn luyện.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Likelihood - Likelihood của một mô hình L(θ) với tham số θ được sử dụng để tìm tham số tối ưu θ thông qua việc cực đại hoá likelihood. Trong thực tế, chúng ta sử dụng log-likelihood ℓ(θ)=log(L(θ)) đễ dễ dàng hơn trong việc tôi ưu hoá. Ta có:

<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Giải thuật Newton - Giải thuật Newton là một phương thức số tìm θ thoả mãn điều kiện ℓ′(θ)=0. Quy tắc cập nhật của nó là như sau:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Chú ý: Tổng quát hoá đa chiều, còn được biết đến như là phương thức Newton-Raphson, có quy tắc cập nhật như sau:

<br>

**21. Linear models**

&#10230; Các mô hình tuyến tính

<br>

**22. Linear regression**

&#10230; Hồi quy tuyến tính

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; Chúng ta giả sử ở đây rằng y|x;θ∼N(μ,σ2)

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Phương trình chuẩn - Bằng việc kí hiệu X là ma trận thiết kế, giá trị của θ làm cực tiểu hoá cost function là một phương pháp dạng đóng như sau:

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; Giải thuật LMS - Bằng việc kí hiệu α là tốc độ học, quy tắc cập nhật của giải thuật Least Mean Squares (LMS) cho tập huấn luyện của m điểm dữ liệu, còn được biết như là quy tắc học Widrow-Hoff, là như sau:

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Chú ý: Luật cập nhật là một trường hợp đặc biệt của gradient ascent.

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; LWR - Hồi quy trọng số cục bộ, còn được biết với cái tên LWR, là biến thể của hồi quy tuyến tính, nó sẽ đánh trọng số cho mỗi ví dụ huấn luyện trong cost function của nó bởi w(i)(x), được định nghĩa với tham số τ∈R như sau:

<br>

**28. Classification and logistic regression**

&#10230; Phân loại và logistic hồi quy

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Hàm Sigmoid - Hàm sigmoid g, còn được biết đến như là hàm logistic, được định nghĩa như sau:

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Hồi quy logistic - Chúng ta giả sử ở đây rằng y|x;θ∼Bernoulli(ϕ). Ta có công thức như sau:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Chú ý: không có giải pháp dạng đóng cho trường hợp của hồi quy logistic.

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Hồi quy Softmax - Hồi quy softmax, còn được gọi là hồi quy logistic đa lớp, được sử dụng để tổng quát hoá hồi quy logistic khi có nhiều hơn 2 lớp đầu ra. Theo quy ước, chúng ta thiết lập θK=0, làm cho tham số Bernoulii ϕi của mỗi lớp i bằng với:

<br>

**33. Generalized Linear Models**

&#10230; Mô hình tuyến tính tổng quát

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Họ số mũ - Một lớp của phân phối được cho rằng thuộc về họ số mũ nếu nó có thể được viết dưới dạng một thuật ngữ của tham số tự nhiên, cũng được gọi là tham số kinh điển (canonical parameter) hoặc hàm kết nối, η, một số liệu thống kê đầy đủ T(y) và hàm phân vùng log (log-partition function) a(η) sẽ có dạng như sau:

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Chú ý: chúng ta thường có T(y)=y. Đồng thời, exp(−a(η)) có thể được xem như là tham số chuẩn hoá sẽ đảm bảo rằng tổng các xác suất là một.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Ở đây là các phân phối mũ phổ biến nhất được tổng kết ở bảng bên dưới:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Phân phối, Bernoulli, Gaussian, Poisson, Geometric]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Giả thuyết GLMs - Mô hình tuyến tính tổng quát (GLM) với mục đích là dự đoán một biến ngẫu nhiên y như là hàm cho biến x∈Rn+1 và dựa trên 3 giả thuyết sau:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Chú ý: Bình phương nhỏ nhất thông thường và logistic regression đều là các trường hợp đặc biệt của các mô hình tuyến tính tổng quát.

<br>

**40. Support Vector Machines**

&#10230; Máy vector hỗ trợ

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; Mục tiêu của máy vector hỗ trợ là tìm ra dòng tối đa hoá khoảng cách nhỏ nhất tới dòng.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Optimal margin classifier - Optimal margin classifier h là như sau:

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; với (w,b)∈Rn×R là giải pháp cho vấn đề tối ưu hoá sau đây:

<br>

**44. such that**

&#10230; như là:

<br>

**45. support vectors**

&#10230; vector hỗ trợ

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Chú ý: đường thẳng có phương trình là wTx−b=0.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Mất mát Hinge - Mất mát Hinge được sử dụng trong thiết lập của SVMs và nó được định nghĩa như sau:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Kernel (nhân) - Cho trước feature mapping ϕ, chúng ta định nghĩa kernel K như sau:

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; Trong thực tế, kernel K được định nghĩa bởi K(x,z)=exp(−||x−z||22σ2) được gọi là Gaussian kernal và thường được sử dụng.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Phân tách phi tuyến, Việc sử dụng một kernel mapping, Biến quyết định trong không gian gốc]

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Chú ý: chúng ta nói rằng chúng ta sử dụng "kernel trick" để tính toán cost function sử dụng kernel bởi vì chúng ta thực sự không cần biết đến ánh xạ tường minh ϕ, nó thường khá phức tạp. Thay vào đó, chỉ cần biết giá trị K(x,z).

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagrangian - Chúng ta định nghĩa Lagrangian L(w,b) như sau:

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Chú ý: hệ số βi được gọi là bội số Lagrange.

<br>

**54. Generative Learning**

&#10230; Generative Learning

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Một mô hình sinh đầu tiên cố gắng học cách dữ liệu được sinh ra thông qua việc ước lượng P(x|y), sau đó chúng ta có thể sử dụng P(x|y) để ước lượng P(y|x) bằng cách sử dụng luật Bayes.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Gaussian Discriminant Analysis

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Thiết lập - Gaussian Discriminant Analysis giả sử rằng y và x|y=0 và x|y=1 là như sau:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Sự ước lượng - Bảng sau đây tổng kết các ước lượng mà chúng ta tìm thấy khi tối đa hoá likelihood:

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Giả thiết - Mô hình Naive Bayes giả sử rằng các features của các điểm dữ liệu đều độc lập với nhau:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Giải pháp - Tối đa hoá log-likelihood đưa ra những lời giải sau đây, với k∈{0,1},l∈[[1,L]]

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Chú ý: Naive Bayes được sử dụng rộng rãi cho bài toán phân loại văn bản và phát hiện spam.

<br>

**63. Tree-based and ensemble methods**

&#10230; Các phương thức Tree-based và ensemble

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Các phương thức này có thể được sử dụng cho cả bài toán hồi quy lẫn bài toán phân loại.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - Cây phân loại và hồi quy (CART), thường được biết đến là cây quyết định, có thể được biểu diễn dưới dạng cây nhị phân. Chúng có các ưu điểm có thể được diễn giải một cách dễ dàng.

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Rừng ngẫu nhiên - Là một kĩ thuật dựa trên cây (tree-based), sử dụng số lượng lớn các cây quyết định để lựa chọn ngẫu nhiên các tập thuộc tính. Ngược lại với một cây quyết định đơn, kĩ thuật này khá khó diễn giải nhưng do có hiệu năng tốt nên đã trở thành một giải thuật khá phổ biến hiện nay.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Chú ý: rừng ngẫu nhiên là một loại giải thuật ensemble.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Boosting - Ý tưởng của các phương thức boosting là kết hợp các phương pháp học yếu hơn để tạo nên phương pháp học mạnh hơn. Những phương thức chính được tổng kết ở bảng dưới đây:

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Adaptive boosting, Gradient boosting]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; Các trọng số có giá trị lớn được đặt vào các phần lỗi để cải thiện ở bước boosting tiếp theo

<br>

**71. Weak learners trained on remaining errors**

&#10230; Các phương pháp học yếu huấn luyện trên các phần lỗi còn lại 

<br>

**72. Other non-parametric approaches**

&#10230; Các cách tiếp cận phi-tham số khác

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-nearest neighbors - Giải thuật k-nearest neighbors, thường được biết đến là k-NN, là cách tiếp cận phi-tham số, ở phương pháp này phân lớp của một điểm dữ liệu được định nghĩa bởi k điểm dữ liệu gần nó nhất trong tập huấn luyện. Phương pháp này có thể được sử dụng trong quá trình thiết lập cho bài toán phân loại cũng như bài toán hồi quy.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Chú ý: Tham số k cao hơn, bias cao hơn, tham số k thấp hơn, phương sai cao hơn

<br>

**75. Learning Theory**

&#10230; Lý thuyết học

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Union bound - Cho k sự kiện là A1,...,Ak. Ta có:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Bất đẳng thức Hoeffding - Cho Z1,..,Zm là m biến iid được đưa ra từ phân phối Bernoulli của tham số ϕ. Cho ˆϕ là trung bình mẫu của chúng và γ>0 cố định. Ta có:

<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Chú ý: bất đẳng thức này còn được biết đến như là ràng buộc Chernoff.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Lỗi huấn luyện (Training error) - Cho trước classifier h, ta định nghĩa training error ˆϵ(h), còn được biết đến là empirical risk hoặc empirical error, như sau:

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:**

&#10230; Probably Approximately Correct (PAC) - PAC là một framework với nhiều kết quả về lí thuyết học đã được chứng minh, và có tập hợp các giả thiết như sau:

<br>

**81: the training and testing sets follow the same distribution**

&#10230; tập huấn luyện và test có cùng phân phối

<br>

**82. the training examples are drawn independently**

&#10230; các ví dụ huấn luyện được tạo ra độc lập

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Shattering (Chia nhỏ) - Cho một tập hợp S={x(1),...,x(d)}, và một tập hợp các classifiers H, ta nói rằng H chia nhỏ S nếu với bất kì tập các nhãn {y(1),...,y(d)} nào, ta có:

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Định lí giới hạn trên - Cho H là một finite hypothesis class mà |H|=k với δ, kích cỡ m là cố định. Khi đó, với xác suất nhỏ nhất là 1−δ, ta có:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC dimension - Vapnik-Chervonenkis (VC) dimension của class infinite hypothesis H cho trước, kí hiệu là VC(H) là kích thước của tập lớn nhất được chia nhỏ bởi H.

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Chú ý: VC dimension của H={tập hợp các linear classifiers trong 2 chiều} là 3.

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Định lí (Vapnik) - Cho H với VC(H)=d và m là số lượng các ví dụ huấn luyện. Với xác suất nhỏ nhất là 1−δ, ta có:

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [Giới thiệu, Loại dự đoán, Loại mô hình]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [Các kí hiệu và các khái niệm tổng quát, hàm mất mát, gradient descent, likelihood]

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [Các mô hình tuyến tính, hồi quy tuyến tính, hồi quy logistic, các mô hình tuyến tính tổng quát]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [Máy vector hỗ trợ, Optimal margin classifier, Mất mát Hinge, Kernel]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [Cây và các phương pháp ensemble, CART, Rừng ngẫu nhiên, Boosting]

<br>

**94. [Other methods, k-NN]**

&#10230; [Các phương thức khác, k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [Lí thuyết học, Bất đẳng thức Hoeffding, PAC, VC dimension]
