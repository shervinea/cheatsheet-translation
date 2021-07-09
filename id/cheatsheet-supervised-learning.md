**1. Supervised Learning cheatsheet**

&#10230; **Supervised Learning cheatsheet**

<br>

**2. Introduction to Supervised Learning**

&#10230; **2. Pengenalan Supervised Learning**

<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; **3. Diberikan sebuah kumpulan data poin {x(1),....,x(m)} yang berasosiasi dengan hasil {y(1),....,y(m)}, kita ingin membuat klasifikasi yang mempelajari bagaimana memprediksi nilai y dari x.**

<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; **4. Jenis prediksi ― Perbedaan jenis model prediksi diringkas dalam tabel berikut:**

<br>

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; **5. [Regresi, klasifikasi, hasil, contoh]**

<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; **6. [Continues, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

<br>

**7. Type of model ― The different models are summed up in the table below:**

&#10230; **7. Jenis model ― Perbedaan antar model diringkas dalam tabel berikut:**

<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; **8. [Discriminative model, Generative model, Tujuan, Apa yang telah dipelajari, Ilustrasi, Contoh]**

<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; **9. [Estimasi langsung P(y|x), Estimasi P(x|y) untuk mendeduksi P(y|x), Decision boundary, Probabilitas distribusi data, Regresi, SVM, GDA, Naive Bayes]**

<br>

**10. Notations and general concepts**

&#10230; **10. Notasi dan konsep umum**

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230; **11. Hipotesis ― Hipotesis dinotasikan dengan hθ dan model yang kita pilih. Untuk input data x(i), hasil prediksi model adalah hθ(x(i)).**

<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230; **12. Loss function ― Fungsi loss adalah sebuah fungsi L:(z,y)∈R×Y⟼L(z,y)∈R yang mengambil input sebagai prediksi nilai z yang berkorespondensi dengan nilai real y dan memberikan output perbedaan antara keduanya. Fungsi loss yang umum adalah sebagai berikut:**

<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; **13. [Leas squared error, Logistic loss, Hinge loss, Cross-entropy]**

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; **14. [Linear regression, Logistic regression, SVM, Neural network]**

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; **15. Const function ― Funsgi cost j adalah uum digunakan untuk mengukur performa sebuah model, dan mendefinisikannya dengan fungsi loss L sebagai berikut:**

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; **16. Gradient descent ― α∈R adalah tingkat pembelajaran (learning rate), aturan untuk memperbarui gradient descent diekspresikan dengan hubungan antara learning rate dan fungsi cost J sebagai berikut:**

<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; **17. Catatan: Stochastic gradient descent (SGD) memperbarui parameter berdasarkan setiap contoh data latih, dan batch gradient descent adalah batch pada setiap contoh training**.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; **q8. Likelihood ― Likelihood dari model L(θ) diberikan parameter θ digunakan untuk mencari parameter optimal θ dengan memaksimalkan nilai likelihood. Dalam praktiknya, kita menggunakan log-likehood
ℓ(θ)=log(L(θ)) yang memudahkan untuk optimalisasi.**
<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; **19. Algoritma Newton ― Algoritma newton adalah metode numerik yang mencari 0 sehingga ℓ′(θ)=0. Algoritma ini memperbarui dengan cara berikut:**

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; **20. Catatan: generalisasi multidimensional, juga disebut sebagai Metode Newton-Raphson, cara kerjanya sebagai berikut:**

<br>

**21. Linear models**

&#10230; **21. Model linear**

<br>

**22. Linear regression**

&#10230; **22. Regresi linear**

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230; **23. Asumsinya sebagai berikut: y|x;θ∼N(μ,σ2)** 

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; **24. Persamaan normal ― Dengan X sebagai desain matriks, nilai 0 digunakan untuk meminimalisir nilai fungsi cost sehingga mendekati bentuk solusi:** 

<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; **25. Algoritma LMS ― α adalah learning rate, perbaruan algoritma LMS untuk data training m, yang disebut juga Widrow-Hoff learning:**

<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; **26. Catatan: perbaruan rule adalah contoh dari gradient ascent.**

<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; **27. LWR ― Locally Weighted Regression, disebut juga LWR, adalah varian dari regersi linear yang bobot pada setiap data training dalam cost functionnya dinotasikan w(i)(x), yang didefinisikan dengan parameter τ∈R sebagai:** 

<br>

**28. Classification and logistic regression**

&#10230; **28. Klasifikasi dan logistic regression**

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; **29. Fungsi sigmoid ― fungsi sigmoid g, disebut juga fungsi logistic, didefinisikan sebagai berikut:**

<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; **30. Logistic regression ― kita asumsikan bahwa y|x;θ∼Bernoulli(ϕ). Dengan bentuk sebagai berikut:**

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; **31. Catatan: tidak ada bentuk solusi tertutup untuk kasus logistic regression.**

<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; **32. Softmax regression ― Softmax regression disebut juga sebagai multiclass logistic regression, digunakan untuk membuat logistic regression ketika terdapat lebih dari dua kelas output. Secara umum, kita men-set θK=0, yang membuat Bernoulli parameter ϕi pada setiap kelas i sama dengan:**

<br>

**33. Generalized Linear Models**

&#10230; **33. Generalized Linear Models**

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; **34. Keluarga eksponensial ― Sebuah kelas distribusi disebut keluarga eksponensial jika ditulis dalam sebuah parameter natural, disebut juga sebagai parameter canonical atau link function, η, statistik yang memadai T(y) dan fungsi log-partition a(η) adalah sebagai berikut:**

<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; **35. Catatan: kita akan sering memiliki T(y)=y. Juga, exp(−a(η)) dapat dilihat sebagai parameter normalisasi yang memastikan bahwa jumlah dari nilai probabilitasnya adalah satu.**

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; **36. Distribusi eksponensial yang paling umum digunakan terdapat dalam tabel berikut:** 

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; **37. [Distribusi, Bernoulli, Gaussian, Poisson, Geometric]**

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; **38. Asumsi GLM ― Generalized Linear Models (GLM) bertujuan untuk memprediksi sebuah random variabel y sebagai fungsi fo x∈Rn+1 dan bergantung pada 3 asumsi berikut ini:**

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; **39. Catatan: ordinary least squares dan logistrik regression adalah contoh spesial dari generalized linear mmodels.**

<br>

**40. Support Vector Machines**

&#10230; **40. Support Vector Machines**

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; **41. Tujuan dari SVM adalah untuk menentukan garis yang memiliki jarak seminimum mungkin ke garis**

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; **42. Optimal margin classifier ― Optimal margin classifier h adalah sebagai berikut:**

<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; **43: dimana (w, b)∈Rn×R adalah solusi dari masalah optimalisasi sebagai berikut:**

<br>

**44. such that**

&#10230; **44. Misalnya**

<br>

**45. support vectors**

&#10230; **45. Support vectors**

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; **46. Catatan: garis didefinisikan sebagai wTx−b=0.**

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; **47. Hinge loss ― Hinge loss digunaka untuk pengaturan dari SVM dan definisikan sebagai berikut:**

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; **48. Kernel ― Diberikan sebuah fitur mapping ϕ, kita mendefinisikan kernel K sebagai berikut:**

<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; **48. Pada praktiknya, kernel K didefinisikan dengan K(x,z)=exp(−||x−z||22σ2) yang disebut sebagai Gaussian kernel dan yang paling umum digunakan.**

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; **50. [Non-linear separability, Penggunan kernel mapping, Decision boundary di original space]**

<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; **51. Catatan: kita katakan bahwa kita menggunakan "trik kernel" yaitu menghitung fungsi coss menggunakan kernel karena kita tidak perlu mengetahui mapping eksplisit ϕ, dimana itu sangat kompleks. Sehingga hanya dibutuhkan nilai K(x,z).**

<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; **52. Lagrangian ― Kita mendefiniskan Lagrangian L(w,b) sebagai berikut:**

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; **Catatan: Koefisien βi disebut sebagai Lagrange multipliers.**

<br>

**54. Generative Learning**

&#10230; **54. Generative Learning**

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; **55. Sebuah generative model pertama kali digunakan untuk mempelajari bagaimana data dihasilkan dengan mengestimasi P(x|y), yang kemudian digunakan untuk mengestimasi P(y|x) dengan aturan Bayes.**

<br>

**56. Gaussian Discriminant Analysis**

&#10230; **55. Gaussian Discriminant Analysis**

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; **57. Pengaturan ― Asumsi dari gaussian discriminant Analysis adalah y dan x|y=0 dan x|y=1 adalah sebagai berikut:** 

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; **58. Estimasi ― Tabel berikut meringkas estimasi ketika melakukan maksimalisasi kemungkinan:**

<br>

**59. Naive Bayes**

&#10230; **59. Naive Bayes**

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; **60. Asumsi ― Model Naive Bayes menduga bahwa fitur dari setiap data point adalah independent:**

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; **61. Solusi ― Memaksimalkan log-likelihood memberikan solusi berikut, dengan k∈{0,1},l∈[[1,L]]**

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; **62. Catatan: Naive Bayes umum digunakan untuk klasifikasi teks dan deteksi spam.**

<br>

**63. Tree-based and ensemble methods**

&#10230; **63. Tree-based dan ensemble methods**

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; **64. Metode ini digunakan untuk permasalahan regresi dan klasifikasi.**

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; **65. CART ― Klasifikasi dan pohon regresi (CART), umumnya disebut sebagai decision trees, direpresentasikan sebagai pohon binary. Memiliki keuntungan yang sangat dapat diinterprestasi.**

<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; **66. Random forest ― Merupakan teknik tree-based yang menggunakan angka tertinggi dari decision-tress yang secara random dipilih dari sekumpulan fitur. Berbeda dengan simple decision tree, ini sangat tidak mudah diinterpretasi namum secara umum memiliki performa yang sangat bagus, dan ini adalah salah satu algoritma yang populer.**

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; **67. Catatan: random forest adalah salahsatu jenis metode ensemble.**

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; **68. Boosting ― ide dari metode boosting adalah untuk mengkombinasi beberapa kelemahan learner untuk membentuk leaner yang kuat. Beberapa tujuan utamanya adalah diringkas dalam tabel berikut:**

<br>

**69. [Adaptive boosting, Gradient boosting]**

&#10230; **69. [Adaptive boosting, Gradient boosting]**

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; **70. Bobot yang tinggi diletakkan di tempat yang memiliki error untuk meningkatkan tahapan boosting berikutnya.**

<br>

**71. Weak learners trained on remaining errors**

&#10230; **71. Learner yang lemah berada di tempat yang error**

<br>

**72. Other non-parametric approaches**

&#10230; **72. Pendekatan non-parametrik lain**

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; **73. K-nearest neighbors ― Algoritma k-nearest neighbors, biasa disebut k-NN, adalah pendekatan non-parametrik dimana respon terhadap data point ditentukan oleh apa yang terjadi di sekitar k dalam data latih. Algoritma ini digunakan pada klasifikasi dan regresi.**

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; **74. Catatan: Semakin tinggi parameter k, semakin tinggi bias, dan semakin rendah parameter k, semakin tinggi variansinya.**

<br>

**75. Learning Theory**

&#10230; **75. Teori pembelajaran**

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; **76. Union bound ― Dimana A1,...,Ak dan event k. Kita memiliki:**

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; **77. Ketidaksamaan Hoeffding ― Dimana Z1,...,Zm dan variabel iid didapat dari distribusi Bernoulli dengan parameter ϕ. ^ϕ adalah rerata sampel dan γ>0 adalah tetap. Sehingga:**
<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; **78. Catatan: ketidaksamaan ini juga dikenal sebagai Chernoff bound.**

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; **79. Error latih ― diberikan klasifier h, kita mendefinisikan training error sebagai ˆϵ(h), disebut juga empirical risk atau empirical error, dikenal sebagai berikut:**

<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions:**

&#10230; **80. Probably Approximately Correct (PAC) ― PAC adalah sebuah framework yang dihasilkan dari learning theory, dan memiliki beberapa asumsi:** 

<br>

**81. the training and testing sets follow the same distribution.**

&#10230; **81. Data training dan testing mengikuti distribusi yang sama.**

<br>

**82. the training examples are drawn independently**

&#10230; **82. Contoh data training dihasilkan secara independen**

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; **83. Shattering ― Diberikan sebuah set S={x(1),...,x(d)}, dan sebuah set klasifier H, kita  dapat katakan bahwa H shatter S apabila setiap set dari label {y(1),...,y(d)}, sehingga:**

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; **84. Teorem Upper bound ― Diberikan H merupakan kelas hipotesis dimana |H|=k, δ, dan sampel m adalah tetap. Sehingga probabilitas 1-δ adalah:**

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; **85. Dimensi VC ― Dimensi Vapnik-Chervonenkis (VC) dari suah hipotesis kelas H tak terhingga. VC (H) adalah ukuran set terbesar yang di-shatter oleh H.**

<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; **86. Catatan: dimensi VC dari H={set dari klasifier linear dalam 2 dimensi} adalah 3.**

<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; **87. Teorem (Vapnik) ― Diberikan H, dimana VC(H)=d dan m adalah jumlah data training. Dengan probabilitas paling minimal 1−δ, sehingga:**

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; **88. [Introduction, Jenis prediksi, jenis model]**

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; **89. [Notasi dan konsep umum, fungsi loss, gradient descent, likelihood]**

<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; **90. [Model linear, regeresi linear, regresi logistik, generalized linear models]**

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; **91. [Support vector machines. Optimal margin classifier, Hinge loss, Kernel]**

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; **92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; **93. [Tress and ensemble methods, CART, Random forest, Boosting]**

<br>

**94. [Other methods, k-NN]**

&#10230; **94. [Other methods, k-NN]**

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; **95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**
