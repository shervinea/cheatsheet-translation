**1. Supervised Learning cheatsheet**

&#10230; Gözetimli Öğrenme El kitabı

<br> 

**2. Introduction to Supervised Learning**

&#10230; Gözetimli Öğrenmeye Giriş

<br> 

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

&#10230; {y(1),...,y(m)} çıktı kümesi ile ilişkili olan {x(1),...,x(m)} veri noktalarının kümesi göz önüne alındığında, y'den x'i nasıl tahmin edebileceğimizi öğrenen bir sınıflandırıcı tasarlamak istiyoruz. 

<br> 

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

&#10230; Tahmin türü ― Farklı tahmin modelleri aşağıdaki tabloda özetlenmiştir: 

<br> 

**5. [Regression, Classifier, Outcome, Examples]**

&#10230; [Regresyon, Sınıflandırıcı, Çıktı , Örnekler]

<br> 

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

&#10230; [Sürekli, Sınıf, Lineer regresyon (bağlanım), Lojistik regresyon (bağlanım), Destek Vektör Makineleri (DVM), Naive Bayes]

<br> 

**7. Type of model ― The different models are summed up in the table below:**

&#10230; Model türleri ― Farklı modeller aşağıdaki tabloda özetlenmiştir:

<br> 

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

&#10230; [Ayırt edici model, Üretici model, Amaç, Öğrenilenler, Örnekleme, Örnekler]

<br> 

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, SVMs, GDA, Naive Bayes]**

&#10230; [ Doğrudan tahmin P (y|x), P (y|x)'i tahmin etmek için P(x|y)'i tahmin etme, Karar Sınırı, Verilerin olasılık dağılımı, Regresyon, Destek Vektör Makineleri, Gauss Diskriminant Analizi, Naive Bayes] 

<br> 

**10. Notations and general concepts**

&#10230; Gösterimler ve genel konsept

<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

&#10230;  Hipotez ― Hipotez hθ olarak belirtilmiştir ve bu bizim seçtiğimiz modeldir. Verilen x(i) verisi için modelin tahminlediği çıktı hθ(x(i))'dir.

<br> 

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

&#10230;  Kayıp fonksiyonu ― L:(z,y)∈R×Y⟼L(z,y)∈R şeklinde tanımlanan bir kayıp fonksiyonu y gerçek değerine karşılık geleceği öngörülen z değerini girdi olarak alan ve ne kadar farklı olduklarını gösteren bir fonksiyondur. Yaygın kayıp fonksiyonları aşağıdaki tabloda özetlenmiştir:

<br> 

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

&#10230; [En küçük kareler hatası, Lojistik yitimi (kaybı), Menteşe yitimi (kaybı), Çapraz entropi]

<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

&#10230; [Lineer regresyon (bağlanım), Lojistik regresyon (bağlanım), Destek Vektör Makineleri, Sinir Ağı]

<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

&#10230; Maliyet fonksiyonu ― J maliyet fonksiyonu genellikle bir modelin performansını değerlendirmek için kullanılır ve L kayıp fonksiyonu aşağıdaki gibi tanımlanır:

<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

&#10230; Bayır inişi ― α∈R öğrenme oranı olmak üzere, bayır inişi için güncelleme kuralı olarak ifade edilen öğrenme oranı ve J maliyet fonksiyonu aşağıdaki gibi ifade edilir:
<br> 


**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

&#10230; Not: Stokastik bayır inişi her eğitim örneğine bağlı olarak parametreyi günceller, ve yığın bayır inişi bir dizi eğitim örneği üzerindedir.

<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

&#10230; Olabilirlik - θ parametreleri verilen bir L (θ) modelinin olabilirliğini,olabilirliği maksimize ederek en uygun θ  parametrelerini bulmak için kullanılır. bulmak için kullanılır. Uygulamada, optimize edilmesi daha kolay olan log-olabilirlik ℓ (θ) = log (L (θ))'i kullanıyoruz. Sahip olduklarımız:

<br>      

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

&#10230; Newton'un algoritması - ℓ′(θ)=0 olacak şekilde bir θ bulan nümerik bir yöntemdir. Güncelleme kuralı aşağıdaki gibidir:

<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

&#10230; Not: Newton-Raphson yöntemi olarak da bilinen çok boyutlu genelleme aşağıdaki güncelleme kuralına sahiptir:

<br>

**21. Linear models**

&#10230; Lineer modeller

<br>

**22. Linear regression**

&#10230; Lineer regresyon

<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

&#10230;y|x;θ∼N(μ,σ2) olduğunu varsayıyoruz

<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

&#10230; Normal denklemler - X matris tasarımı olmak üzere, maliyet fonksiyonunu en aza indiren θ değeri X'in matris tasarımını not ederek, maliyet fonksiyonunu en aza indiren θ değeri kapalı formlu bir çözümdür:

<br>  

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

&#10230; En Küçük Ortalama Kareler algoritması (Least Mean Squares-LMS) - α öğrenme oranı olmak üzere, m veri noktasını içeren eğitim kümesi için Widrow-Hoff öğrenme oranı olarak bilinen En Küçük Ortalama Kareler Algoritmasının güncelleme kuralı aşağıdaki gibidir:

<br> 

**26. Remark: the update rule is a particular case of the gradient ascent.**

&#10230; Not: güncelleme kuralı, bayır yükselişinin özel bir halidir.

<br> 

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

&#10230; Yerel Ağırlıklı Regresyon (Locally Weighted Regression-LWR) - LWR olarak da bilinen Yerel Ağırlıklı Regresyon ağırlıkları her eğitim örneğini maliyet fonksiyonunda w (i) (x) ile ölçen doğrusal regresyonun bir çeşididir.

<br> 

**28. Classification and logistic regression**

&#10230; Sınıflandırma ve lojistik regresyon

<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

&#10230; Sigmoid fonksiyonu - Lojistik fonksiyonu olarak da bilinen sigmoid fonksiyonu g, aşağıdaki gibi tanımlanır:

<br> 

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

&#10230; Lojistik regresyon - y|x;θ∼Bernoulli(ϕ) olduğunu varsayıyoruz. Aşağıdaki forma sahibiz:

<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

&#10230; Not: Lojistik regresyon durumunda kapalı form çözümü yoktur.

<br> 

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

&#10230; Softmax regresyonu - Çok sınıflı lojistik regresyon olarak da adlandırılan Softmax regresyonu 2'den fazla sınıf olduğunda lojistik regresyonu genelleştirmek için kullanılır. Genel kabul olarak, her i sınıfı için Bernoulli parametresi ϕi'nin eşit olmasını sağlaması için θK=0 olarak ayarlanır.

<br>

**33. Generalized Linear Models**

&#10230; Genelleştirilmiş Lineer Modeller

<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

&#10230; Üstel aile - Eğer kanonik parametre veya bağlantı fonksiyonu olarak adlandırılan doğal bir parametre η, yeterli bir istatistik T (y) ve aşağıdaki gibi bir log-partition fonksiyonu a (η) şeklinde yazılabilirse, dağılım sınıfının üstel ailede olduğu söylenir:

<br> 

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

&#10230; Not: Sık sık T (y) = y olur. Ayrıca, exp (−a (η)), olasılıkların birleştiğinden emin olan normalleştirme parametresi olarak görülebilir.

<br>

**36. Here are the most common exponential distributions summed up in the following table:**

&#10230; Aşağıdaki tabloda özetlenen en yaygın üstel dağılımlar:

<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

&#10230; [Dağılım, Bernoulli, Gauss, Poisson, Geometrik]

<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

&#10230; Genelleştirilmiş Lineer Modellerin  (Generalized Linear Models-GLM) Yaklaşımları - Genelleştirilmiş Lineer Modeller x∈Rn+1 için rastgele bir y değişkenini tahminlemeyi hedeflen ve aşağıdaki 3 varsayıma dayanan bir fonksiyondur:

<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

&#10230; Not: sıradan en küçük kareler ve lojistik regresyon, genelleştirilmiş doğrusal modellerin özel durumlarıdır.

<br>

**40. Support Vector Machines**

&#10230; Destek Vektör Makineleri

<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

&#10230; Destek Vektör Makinelerinin amacı minimum mesafeyi maksimuma çıkaran doğruyu bulmaktır.

<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

&#10230; Optimal marj sınıflandırıcısı - h optimal marj sınıflandırıcısı şöyledir:

<br> 

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

&#10230; burada (w,b)∈Rn×R, aşağıdaki optimizasyon probleminin çözümüdür:

<br>

**44. such that**

&#10230; öyle ki

<br>

**45. support vectors**

&#10230; destek vektörleri

<br>

**46. Remark: the line is defined as wTx−b=0.**

&#10230; Not: doğru wTx−b=0 şeklinde tanımlanır.

<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

&#10230; Menteşe yitimi (kaybı) - Menteşe yitimi Destek Vektör Makinelerinin ayarlarında kullanılır ve aşağıdaki gibi tanımlanır:

<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

&#10230; Çekirdek - ϕ gibi bir özellik haritası verildiğinde, K olarak tanımlanacak çekirdeği tanımlarız:

<br>  
**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

&#10230; Uygulamada, K (x, z) = exp (- || x − z || 22σ2) tarafından tanımlanan çekirdek K, Gauss çekirdeği olarak adlandırılır ve yaygın olarak kullanılır.

<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

&#10230; [Lineer olmayan ayrılabilirlik, Çekirdek Haritalamının Kullanımı, Orjinal uzayda karar sınırı]

<br> 

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

&#10230; Not: Çekirdeği kullanarak maliyet fonksiyonunu hesaplamak için "çekirdek numarası" nı kullandığımızı söylüyoruz çünkü genellikle çok karmaşık olan ϕ açık haritalamasını bilmeye gerek yok. Bunun yerine, yalnızca K(x,z) değerlerine ihtiyacımız vardır.

<br> 

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

&#10230; Lagranj - Lagranj L(w,b) şeklinde şöyle tanımlanır: 

<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

&#10230; Not: βi katsayılarına Lagranj çarpanları denir.

<br>

**54. Generative Learning**

&#10230; Üretici Öğrenme

<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

&#10230; Üretken bir model, önce Bayes kuralını kullanarak P (y | x) değerini tahmin etmek için kullanabileceğimiz P (x | y) değerini tahmin ederek verilerin nasıl üretildiğini öğrenmeye çalışır.

<br>

**56. Gaussian Discriminant Analysis**

&#10230; Gauss Diskriminant (Ayırtaç) Analizi

<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

&#10230; Yöntem - Gauss Diskriminant Analizi y ve x|y=0 ve x|y=1 'in şu şekilde olduğunu varsayar:

<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

&#10230; Tahmin - Aşağıdaki tablo, olasılığı en üst düzeye çıkarırken bulduğumuz tahminleri özetlemektedir:

<br>

**59. Naive Bayes**

&#10230; Naive Bayes

<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

&#10230; Varsayım - Naive Bayes modeli, her veri noktasının özelliklerinin tamamen bağımsız olduğunu varsayar:

<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

&#10230; Çözümler - Log-olabilirliğinin k∈{0,1},l∈[[1,L]] ile birlikte aşağıdaki çözümlerle maksimize edilmesi:

<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

&#10230; Not: Naive Bayes, metin sınıflandırması ve spam tespitinde yaygın olarak kullanılır.

<br>

**63. Tree-based and ensemble methods**

&#10230; Ağaç temelli ve topluluk yöntemleri

<br>

**64. These methods can be used for both regression and classification problems.**

&#10230; Bu yöntemler hem regresyon hem de sınıflandırma problemleri için kullanılabilir.

<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

&#10230; CART - Sınıflandırma ve Regresyon Ağaçları (Classification and Regression Trees (CART)), genellikle karar ağaçları olarak bilinir, ikili ağaçlar olarak temsil edilirler.

<br> 

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

&#10230; Rastgele orman - Rastgele seçilen özelliklerden oluşan çok sayıda karar ağacı kullanan ağaç tabanlı bir tekniktir.
Basit karar ağacının tersine, oldukça yorumlanamaz bir yapıdadır ancak genel olarak iyi performansı onu popüler bir algoritma yapar.

<br>

**67. Remark: random forests are a type of ensemble methods.**

&#10230; Not: Rastgele ormanlar topluluk yöntemlerindendir.

<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

&#10230; Artırım - Artırım yöntemlerinin temel fikri bazı zayıf öğrenicileri biraraya getirerek güçlü bir öğrenici oluşturmaktır. Temel yöntemler aşağıdaki tabloda özetlenmiştir:

<br> 

**69. [Adaptive boosting, Gradient boosting]**

&#10230; [Adaptif artırma, Gradyan artırma]

<br>

**70. High weights are put on errors to improve at the next boosting step**

&#10230; Yüksek ağırlıklar bir sonraki artırma adımında iyileşmesi için hatalara maruz kalır.

<br>

**71. Weak learners trained on remaining errors**

&#10230; Zayıf öğreniciler kalan hatalar üzerinde eğitildi

<br>

**72. Other non-parametric approaches**

&#10230; Diğer parametrik olmayan yaklaşımlar

<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-en yakın komşular - genellikle k-NN olarak adlandırılan k- en yakın komşular algoritması, bir veri noktasının tepkisi eğitim kümesindeki kendi k komşularının doğası ile belirlenen parametrik olmayan bir yaklaşımdır. Hem sınıflandırma hem de regresyon yöntemleri için kullanılabilir.

<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Not: k parametresi ne kadar yüksekse, yanlılık okadar  yüksek ve k parametresi ne kadar düşükse, varyans o kadar yüksek olur.

<br>  

**75. Learning Theory**

&#10230; Öğrenme Teorisi

<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

&#10230; Birleşim sınırı - A1,...,Ak k olayları olsun. Sahip olduklarımız:

<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

&#10230; Hoeffding eşitsizliği - Z1, .., Zm, ϕ parametresinin Bernoulli dağılımından çizilen değişkenler olsun. Örnek ortalamaları mean ve γ>0 sabit olsun. Sahip olduklarımız:

<br> 

**78. Remark: this inequality is also known as the Chernoff bound.**

&#10230; Not: Bu eşitsizlik, Chernoff sınırı olarak da bilinir.

<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

&#10230; Eğitim hatası - Belirli bir h sınıflandırıcısı için, ampirik risk veya ampirik hata olarak da bilinen eğitim hatasını ˆϵ (h) şöyle tanımlarız:

<br> 

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

&#10230; Olası Yaklaşık Doğru (Probably Approximately Correct (PAC)) ― PAC, öğrenme teorisi üzerine sayısız sonuçların kanıtlandığı ve aşağıdaki varsayımlara sahip olan bir çerçevedir:
<br> 


**81: the training and testing sets follow the same distribution **

&#10230; eğitim ve test kümeleri aynı dağılımı takip ediyor

<br>

**82. the training examples are drawn independently**

&#10230; eğitim örnekleri bağımsız olarak çizilir

<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

&#10230; Parçalanma ― S={x(1),...,x(d)} kümesi ve H sınıflandırıcıların kümesi verildiğinde, H herhangi bir etiketler kümesi S'e parçalar.

<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

&#10230; Üst sınır teoremi ― |H|=k , δ ve örneklem sayısı m'nin sabit olduğu sonlu bir hipotez sınıfı H olsun. Ardından, en az 1−δ olasılığı ile elimizde:

<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

&#10230; VC boyutu ― VC(H) olarak ifade edilen belirli bir sonsuz H hipotez sınıfının Vapnik-Chervonenkis (VC) boyutu,  H tarafından parçalanan en büyük kümenin boyutudur.

<br> 

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

&#10230; Not: H = {2 boyutta doğrusal sınıflandırıcılar kümesi}'nin VC boyutu 3'tür.

<br> 

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

&#10230; Teorem (Vapnik) - H, VC(H)=d ve eğitim örneği sayısı m verilmiş olsun. En az 1−δ olasılığı ile, sahip olduklarımız:

<br>

**88. [Introduction, Type of prediction, Type of model]**

&#10230; [Giriş, Tahmin türü, Model türü]

<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

&#10230; [Notasyonlar ve genel kavramlar,kayıp fonksiyonu, bayır inişi, olabilirlik]

<br> 

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

&#10230; [Lineer modeller, Lineer regresyon, lojistik regresyon, genelleştirilmiş lineer modeller]

<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

&#10230; [Destek vektör makineleri, optimal marj sınıflandırıcı, Menteşe yitimi, Çekirdek]

<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

&#10230; [Üretici öğrenme, Gauss Diskriminant Analizi, Naive Bayes]

<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

&#10230; [Ağaçlar ve topluluk yöntemleri, CART, Rastegele orman, Artırma]

<br>

**94. [Other methods, k-NN]**

&#10230; [Diğer yöntemler, k-NN]

<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

&#10230; [Öğrenme teorisi, Hoeffding eşitsizliği, PAC, VC boyutu]
