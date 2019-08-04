**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**

&#10230; Makine Öğrenmesi ile Refleks tabanlı modeller

<br>


**2. Linear predictors**

&#10230; Doğrusal öngörücüler

<br>


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

&#10230; Bu bölümde, girdi-çıktı çiftleri olan örneklerden geçerek, deneyim ile gelişebilecek refleks bazlı modelleri göreceğiz.

<br>


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

&#10230; Öznitelik vektörü - x girişinin öznitelik vektörü ϕ (x) olarak not edilir ve şöyledir:

<br>


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

&#10230; Puan - Bir örneğin s(x, w)si ni (ϕ(x),y))∈Rd×R, w∈Rd doğrusal ağırlık modeline bağlı olarak:

<br>


**6. Classification**

&#10230; Sınıflandırma

<br>


**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

&#10230; Doğrusal sınıflandırıcı - Bir ağırlık vektörü w∈Rd ve bir öznitelik vektörü ϕ(x)∈Rd verildiğinde, ikili doğrusal sınıflandırıcı fw şöyle verilir:

<br>


**8. if**

&#10230;

<br> Eğer


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

&#10230; 

<br>


**10. Regression**

&#10230; Bağlanım (Regression)

<br>


**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**

&#10230; Doğrusal bağlanım (Linear regression) - w∈Rd bir ağırlık vektörü ve bir öznitelik vektörü ϕ(x)∈Rd verildiğinde, fw olarak belirtilen ağırlıkların doğrusal bir bağlanım" çıktısı şöyle verilir:

<br>


**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**

&#10230; Artık (Residual) - Artık res(x,y,w)∈R, fw(x) tahmininin y hedefini aştığı miktar olarak tanımlanır:

<br>


**13. Loss minimization**

&#10230; Kayıp minimizasyonu

<br>


**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**

&#10230; Kayıp fonksiyonu - Kayıp fonksiyonu Loss(x,y,w), x girişinden y çıktısının öngörme görevindeki model ağırlıkları ile ne kadar mutsuz olduğumuzu belirler. Bu değer eğitim sürecinde en aza indirmek istediğimiz bir miktar.

<br>


**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**

&#10230; Sınıflandırma durumu - Doğru etiket y∈{−1,+1} değerinin x örneğinin doğrusal ağırlık w modeliyle sınıflandırılması fw(x)≜sign(s(x,w)) belirleyicisi ile yapılabilir. Bu durumda, sınıflandırma kalitesini ölçen bir fayda ölçütü m(x,y,w) marjı ile verilir ve aşağıdaki kayıp fonksiyonlarıyla birlikte kullanılabilir:

Doğru etiket y örneğinin x değerinin doğrusal ağırlık w modeli ile sınıflandırılması f öngörüsü ile yapılabilir.

<br>


**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**
 
&#10230; [Ad, Örnekleme, Sıfır-bir kayıp, Menteşe kaybı, Lojistik kaybı]

<br>


**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**

&#10230; Regresyon durumu - Doğru etiket y∈R değerinin x örneğinin bir doğrusal ağırlık modeli w ile öngörülmesi fw(x)≜s(x,w) öngörüsü ile yapılabilir. Bu durumda, regresyonun kalitesini ölçen bir fayda ölçütü res(x,y,w) marjı ile verilir ve aşağıdaki kayıp fonksiyonlarıyla birlikte kullanılabilir:

<br>


**18. [Name, Squared loss, Absolute deviation loss, Illustration]**

&#10230; [Ad, Kareler kaybı, Mutlak sapma kaybı, Örnekleme]

<br>


**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**

&#10230; Kayıp minimize etme çerçevesi (framework) - Bir modeli eğitmek için, eğitim kaybını en aza indirmek istiyoruz;

<br>


**20. Non-linear predictors**

&#10230; Doğrusal olmayan öngörücüler

<br>


**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-en yakın komşu - Yaygın olarak k-NN olarak bilinen k-en yakın komşu algoritması, bir veri noktasının tepkisinin eğitim kümesinden k komşularının yapısı tarafından belirlendiği parametrik olmayan bir yaklaşımdır. Hem sınıflandırma hem de regresyon ayarlarında kullanılabilir.

<br>


**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Not: k parametresi ne kadar yüksekse, önyargı (bias) o kadar yüksek ve k parametresi ne kadar düşükse, varyans o kadar yüksek olur.

<br>


**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Yapay sinir ağları - Yapay sinir ağları katmanlarla oluşturulmuş bir model sınıfıdır. Yaygın olarak kullanılan sinir ağları, evrişimli ve tekrarlayan sinir ağlarını içerir. Yapay sinir ağları mimarisi etrafındaki kelime bilgisi aşağıdaki şekilde tanımlanmıştır:

<br>


**24. [Input layer, Hidden layer, Output layer]**

&#10230; [Giriş katmanı, Gizli katman, Çıkış katmanı]

<br>


**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; i, ağın i. katmanı ve j, katmanın j. gizli birimi olacak şekilde aşağıdaki gibi ifade edilir:

<br>


**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**

&#10230; w, b, x, z değerlerinin sırasıyla nöronun ağırlık, önyargı (bias), girdi ve aktive edilmemiş çıkışını olarak ifade eder.

<br>


**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**

&#10230; Yukarıdaki kavramlara daha ayrıntılı bir bakış için, Gözetimli Öğrenme el kitabına göz atın!

<br>


**28. Stochastic gradient descent**

&#10230; Stokastik gradyan inişi (Bayır inişi)

<br>


**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**

&#10230; Gradyan inişi (Bayır inişi) - η∈R öğrenme oranını (aynı zamanda adım boyutu olarak da bilinir) dikkate alınarak, gradyan inişine ilişkin güncelleme kuralı, öğrenme oranı ve Loss(x,y,w) kayıp fonksiyonu ile aşağıdaki şekilde ifade edilir:

<br>


**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**

&#10230; Stokastik güncellemeler - Stokastik gradyan inişi (SGİ / SGD), bir seferde bir eğitim örneğinin (ϕ(x),y)∈Değitim parametrelerini günceller. Bu yöntem bazen gürültülü, ancak hızlı güncellemeler yol açar.

<br>


**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**

&#10230; Yığın güncellemeler - Yığın gradyan inişi (YGİ / BGD), bir seferde bir grup örnek (örneğin, tüm eğitim kümesi) parametrelerini günceller. Bu yöntem daha yüksek bir hesaplama maliyetiyle kararlı güncelleme talimatlarını hesaplar.

<br>


**32. Fine-tuning models**

&#10230; İnce ayar modelleri

<br>


**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**

&#10230; Hipotez sınıfı - Bir hipotez sınıfı F, sabit bir ϕ (x) ve değişken w ile olası öngörücü kümesidir:

<br>


**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**

&#10230; Lojistik fonksiyon - Ayrıca sigmoid fonksiyon olarak da adlandırılan lojistik fonksiyon σ, şöyle tanımlanır:

<br>


**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**

&#10230; Not: σ′(z)=σ(z)(1−σ(z)) şeklinde ifade edilir.

<br>


**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**

&#10230; Geri yayılım - İleriye geçiş, i'de yer alan alt ifadenin değeri olan fi ile yapılırken, geriye doğru geçiş gi=∂out∂fi aracılığıyla yapılır ve fi'nin çıkışı nasıl etkilediğini gösterir.

<br>


**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**

&#10230; Yaklaşım ve kestirim hatası - Yaklaşım hatası ϵapprox, F tüm hipotez sınıfının hedef öngörücü g∗ ne kadar uzak olduğunu gösterirken, kestirim hatası ϵest öngörücüsü ^f, F hipotez sınıfının en iyi yordayıcısı f∗'ya göre ne kadar iyi olduğunu gösterir.
<br>


**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Düzenlileştirme (Regularization) - Düzenlileştirme prosedürü, modelin verilerin aşırı öğrenmesinden kaçınmayı amaçlar ve böylece yüksek değişkenlik sorunlarıyla ilgilenir. Aşağıdaki tablo, yaygın olarak kullanılan düzenlileştirme tekniklerinin farklı türlerini özetlemektedir:

<br>


**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Katsayıları 0'a düşürür, Değişken seçimi için iyi, Katsayıları daha küçük yapar, Değişken seçimi ile küçük katsayılar arasında ödünleşim]

<br>


**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**

&#10230; Hiperparametreler - Hiperparametreler öğrenme algoritmasının özellikleridir ve öznitelikler dahildir, λ normalizasyon parametresi, yineleme sayısı T, adım büyüklüğü η, vb.

<br>


**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Kümeler - Bir model seçerken, veriyi aşağıdaki gibi 3 farklı parçaya ayırırız:

<br>


**42. [Training set, Validation set, Testing set]**

&#10230; [Eğitim kümesi, Doğrulama kümesi, Test kümesi]

<br>


**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**

&#10230; [Model eğitilir, Veri kümesinin genellikle %80'i, Model değerlendirilir, Veri kümesinin genellikle %20'si, Ayrıca tutma veya geliştirme kümesi olarak da adlandırılır, Model tahminlerini verir, Görünmeyen veriler]

<br>


**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Model seçildikten sonra, tüm veri kümesi üzerinde eğitilir ve görünmeyen test kümesinde test edilir. Bunlar aşağıdaki şekilde gösterilmektedir:

<br>


**45. [Dataset, Unseen data, train, validation, test]**

&#10230; [Veri kümesi, Görünmeyen veriler, eğitim, doğrulama, test]

<br>


**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**

&#10230; Yukarıdaki kavramlara daha ayrıntılı bir bakış için, Makine Öğrenmesi ipuçları ve püf noktaları el kitabını göz atın!

<br>


**47. Unsupervised Learning**

&#10230; Gözetimsiz Öğrenme

<br>


**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**

&#10230; Gözetimsiz öğrenme yöntemlerinin sınıfı, zengin gizli yapılara sahip olabilecek verilerin yapısını keşfetmeyi amaçlamaktadır.

<br>


**49. k-means**

&#10230; k-ortalama

<br>


**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**

&#10230; Kümeleme - Dtrain giriş noktalarından oluşan bir eğitim kümesi göz önüne alındığında, kümeleme algoritmasının amacı, her bir ϕ(xi) noktasını zi∈{1,...,k} kümesine atamaktır.

<br>


**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**

&#10230; Amaç fonksiyonu - Ana kümeleme algoritmalarından biri olan k-ortalama için kayıp fonksiyonu şöyle ifade edilir:
 
<br>


**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritma - Küme merkezlerini μ1,μ2,...,μk∈Rn kümesini rasgele başlattıktan sonra, k-ortalama algoritması yakınsayana kadar aşağıdaki adımı tekrarlar:

<br>


**53. and**

&#10230; ve 

<br>


**54. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Başlatma anlamına gelir, Kümeleme görevi, Güncelleme, Yakınsama anlamına gelir]

<br>


**55. Principal Component Analysis**

&#10230; Temel Bileşenler Analizi

<br>


**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Özdeğer, özvektör - Bir A∈Rn×n matrisi verildiğinde, z∈Rn∖{0} olacak şekilde bir vektör varsa λ, A'nın bir öz değeri olduğu söylenir, aşağıdaki gibi ifade edilir:

<br>


**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spektral teoremi - A∈Rn×n olsun. A simetrik ise, o zaman A gerçek ortogonal matris U∈Rn×n olacak şekilde köşegenleştirilebilir. Λ=diag(λ1,...,λn) formülü dikkate alınarak aşağıdaki gibi ifade edilir:

<br>


**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Not: En büyük özdeğerle ilişkilendirilen özvektör, A matrisinin temel özvektörüdür.

<br>


**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritma - Temel Bileşenler Analizi (PCA) prosedürü, verilerin varyansını en üst düzeye çıkararak k boyutlarına indirgeyen bir boyut küçültme tekniğidir:

<br>


**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Adım 1: Verileri ortalama 0 ve 1 standart sapma olacak şekilde normalize edin.

<br>


**61. [where, and]**

&#10230; [koşul, ve]

<br>


**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**

&#10230; [Adım 2: Hesaplama Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, ki bu, gerçek özdeğerlerle simetriktir., Adım 3: Hesaplama u1,...,uk∈Rn k'nin ortogonal ana özvektörleri, yani k en büyük özdeğerlerin ortogonal özvektörleri., Adım 4: spanR(u1,...,uk)'daki verilerin izdüşümünü al.

<br>


**63. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Bu prosedür, tüm k boyutlu uzaylar arasındaki farkı en üst düzeye çıkarır.

<br>


**64. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Öznitelik uzayındaki veriler, Asıl bileşenleri bulma, Asıl bileşenler uzayındaki veriler]

<br>


**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**

&#10230; Yukarıdaki kavramlara daha ayrıntılı bir genel bakış için, Gözetimsiz Öğrenme el kitaplarına göz atın!

<br>


**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin]**

&#10230; [Doğrusal öngörücüler, Öznitelik vektörü, Doğrusal sınıflandırıcı/regresyon, Margin]

<br>


**67. [Loss minimization, Loss function, Framework]**

&#10230; [Kayıp minimizasyonu, Kayıp fonksiyonu, Çerçeve (Framework)]

<br>


**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**

&#10230; [Doğrusal olmayan öngörücüler, k-en yakın komşular, Yapay sinir ağları]

<br>


**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**

&#10230; [Stokastik Dereceli Azalma/Bayır İnişi, Gradyan, Stokastik güncellemeler, Yığın (Batch) güncellemeler]

<br>


**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**

&#10230; [Hassas ayar modeller, Hipotez sınıfı, Geri yayılım, Düzenlileştirme (Regularization), Kelime dizisi]

<br>


**71. [Unsupervised Learning, k-means, Principal components analysis]**

&#10230; [Gözetimsiz Öğrenme, k-ortalama, Temel bileşenler analizi]

<br>


**72. View PDF version on GitHub**

&#10230; GitHub'da PDF sürümünü görüntüleyin

<br>


**73. Original authors**

&#10230; Orijinal yazarlar

<br>


**74. Translated by X, Y and Z**

&#10230; X, Y ve Z tarafından çevrilmiştir

<br>


**75. Reviewed by X, Y and Z**

&#10230; X, Y ve Z tarafından gözden geçirilmiştir

<br>


**76. By X and Y**

&#10230; X ve Y ile

<br>


**77. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Yapay Zeka el kitabı şimdi [hedef dilde] mevcuttur.

