**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; Derin öğrenme püf noktaları ve ipuçları el kitabı

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Derin Öğrenme

<br>


**3. Tips and tricks**

&#10230; Püf noktaları ve ipuçları

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [Veri işleme, Veri artırma, Küme normalizasyonu]

<br>

**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [Bir sinir ağının eğitilmesi, Dönem (Epok), Mini-küme, Çapraz-entropy yitimi (kaybı), Geriye yayılım, Gradyan (Bayır) iniş, Ağırlıkların güncellenmesi, Gradyan (Bayır) kontrolü]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [Parametrelerin ayarlanması, Xavier başlatma, Transfer öğrenme, Öğrenme oranı, Uyarlamalı öğrenme oranları]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [Düzenlileştirme, Seyreltme, Ağırlıkların düzeltilmesi, Erken durdurma]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [İyi örnekler, Küçük kümelerin aşırı öğrenmesi, Gradyan kontrolü]

<br>


**9. View PDF version on GitHub**

&#10230; GitHub'da PDF sürümünü görüntüleyin

<br>


**10. Data processing**

&#10230; Veri işleme

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; Veri artırma ― Derin öğrenme modelleri genellikle uygun şekilde eğitilmek için çok fazla veriye ihtiyaç duyar. Veri artırma tekniklerini kullanarak mevcut verilerden daha fazla veri üretmek genellikle yararlıdır. Temel işlemler aşağıdaki tabloda özetlenmiştir. Daha doğrusu, aşağıdaki girdi görüntüsüne bakıldığında, uygulayabileceğimiz teknikler şunlardır:

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [Orijinal, Çevirme, Rotasyon (Yönlendirme), Rastgele kırpma/kesme]
 
<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [Herhangi bir değişiklik yapılmamış görüntü, Görüntünün anlamının korunduğu bir eksene göre çevrilmiş görüntü, Hafif açılı döndürme, Yanlış yatay kalibrasyonu simule eder, Görüntünün bir bölümüne rastgele odaklanma, Arka arkaya birkaç rasgele kesme yapılabilir]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [Renk değişimi, Gürültü ekleme, Bilgi kaybı, Kontrast değişimi]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; [RGB'nin nüansları biraz değiştirilmesi, Işığa maruz kalırken oluşabilecek gürültü, Gürültü ekleme, Girdilerin kalite değişkenliğine daha fazla toleranslı olması, Yok sayılan görüntüler, Görüntünün parçalardaki olası kayıplarını kopyalanması, Gün içindeki ışık ve renk değişimim kontrolü]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; Not: Veriler genellikle eğitim sırasında artırılır.

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Küme normalleştirme - Bu, {xi} kümesini normalleştiren, β hiperparametresinin bir adımıdır. μB ve σ2B'ye dikkat ederek, kümeyi düzeltmek istediklerimizin ortalaması ve varyansı şu şekilde yapılır:

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Genellikle tam-tüm bağlı/evrişimli bir katmandan sonra ve doğrusal olmayan bir katmandan önce yapılır. Daha yüksek öğrenme oranlarına izin vermeyi ve başlangıç durumuna güçlü bir şekilde bağımlılığı azaltmayı amaçlar.

<br>


**19. Training a neural network**

&#10230; Bir sinir ağının eğitilmesi

<br>


**20. Definitions**

&#10230; Tanımlamalar

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; Dönem (Epok/Epoch) ― Bir modelin eğitimi kapsamında, modelin ağırlıklarını güncellemek için tüm eğitim setini kullandığı bir yinelemeye ifade etmek için kullanılan bir terimdir.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; Mini-küme gradyan (bayır) iniş ― Eğitim aşamasında, ağırlıkların güncellenmesi genellikle hesaplama karmaşıklıkları nedeniyle bir kerede ayarlanan tüm eğitime veya gürültü sorunları nedeniyle bir veri noktasına dayanmaz. Bunun yerine, güncelleme adımı bir toplu işdeki veri noktalarının sayısının ayarlayabileceğimiz bir hiperparametre olduğu mini kümelerle yapılır. Veriler mini-kümeler halinde alınır.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; Yitim fonksiyonu  ― Belirli bir modelin nasıl bir performans gösterdiğini ölçmek için, L yitim (kayıp) fonksiyonu genellikle y gerçek çıktıların, z model çıktıları tarafından ne kadar doğru tahmin edildiğini değerlendirmek için kullanılır.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Çapraz-entropi kaybı ― Yapay sinir ağlarında ikili sınıflandırma bağlamında, çapraz entropi kaybı L (z, y) yaygın olarak kullanılır ve şöyle tanımlanır:

<br>


**25. Finding optimal weights**

&#10230; Optimum ağırlıkların bulunması

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; Geriye yayılım ― Geri yayılım, asıl çıktıyı ve istenen çıktıyı dikkate alarak sinir ağındaki ağırlıkları güncellemek için kullanılan bir yöntemdir. Her bir ağırlığa göre türev, zincir kuralı kullanılarak hesaplanır.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; Bu yöntemi kullanarak, her ağırlık kurala göre güncellenir:

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Ağırlıkların güncellenmesi ― Bir sinir ağında, ağırlıklar aşağıdaki gibi güncellenir:

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [Adım 1: Bir küme eğitim verisi alın ve kaybı hesaplamak için ileriye doğru ilerleyin, Step 2: Her ağırlığa göre kaybın derecesini elde etmek için kaybı tekrar geriye doğru yayın, Adım 3: Ağın ağırlıklarını güncellemek için gradyanları kullanın.]


<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [İleri yayılım, Geriye yayılım, Ağırlıkların güncellenmesi]

<br>


**31. Parameter tuning**

&#10230; Parametre ayarlama

<br>


**32. Weights initialization**

&#10230; Ağırlıkların başlangıçlandırılması

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Xavier başlangıcı (ilklendirme) ― Ağırlıkları tamamen rastgele bir şekilde başlatmak yerine, Xavier başlangıcı, mimariye özgü özellikleri dikkate alan ilk ağırlıkların alınmasını sağlar.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; Transfer öğrenme ― Bir derin öğrenme modelini eğitmek çok fazla veri ve daha da önemlisi çok zaman gerektirir. Kullanım durumumuza yönelik eğitim yapmak ve güçlendirmek için günler/haftalar süren dev veri setleri üzerinde önceden eğitilmiş ağırlıklardan yararlanmak genellikle yararlıdır. Elimizdeki ne kadar veri olduğuna bağlı olarak, aşağıdakilerden yararlanmanın farklı yolları:

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [Eğitim boyutu, Görselleştirme, Açıklama]

<br>


**36. [Small, Medium, Large]**

&#10230; [Küçük, Orta, Büyük]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [Tüm katmanlar dondurulur, Softmax'taki ağırlıkları eğitilir, Çoğu katmanlar dondurulur, son katmanlar ve softmax katmanı ağırlıklar ile eğitilir, Önceden eğitilerek elde edilen ağırlıkları kullanarak katmanlar ve softmax için kullanır]

<br>


**38. Optimizing convergence**

&#10230; Yakınsamayı optimize etmek

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Öğrenme oranı (adımı) ― Genellikle α veya bazen η olarak belirtilen öğrenme oranı, ağırlıkların hangi hızda güncellendiğini belirler. Sabitlenebilir veya uyarlanabilir şekilde değiştirilebilir. Mevcut en popüler yöntemin adı Adam'dır ve öğrenme hızını ayarlayan bir yöntemdir.

<br>

**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; Uyarlanabilir öğrenme oranları ― Bir modelin eğitilmesi sırasında öğrenme oranının değişmesine izin vermek eğitim süresini kısaltabilir ve sayısal optimum çözümü iyileştirebilir. Adam optimizasyonu yöntemi en çok kullanılan teknik olmasına rağmen, diğer yöntemler de faydalı olabilir. Bunlar aşağıdaki tabloda özetlenmiştir:

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [Yöntem, Açıklama, w'ların güncellenmesi, b'nin güncellenmesi]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [Momentum, Osilasyonların azaltılması/yumuşatılması, SGD (Stokastik Gradyan/Bayır İniş) iyileştirmesi, Ayarlanacak 2 parametre]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, Ortalama Karekök yayılımı, Osilasyonları kontrol ederek öğrenme algoritmasını hızlandırır]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [Adam, Uyarlamalı Moment tahmini/kestirimi, En popüler yöntem, Ayarlanacak 4 parametre]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; Not: diğer yöntemler içinde Adadelta, Adagrad ve SGD.

<br>


**46. Regularization**

&#10230; Düzenlileştirme

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; Seyreltme ― Seyreltme, sinir ağlarında, p>0 olasılıklı nöronları silerek eğitim verilerinin fazla kullanılmaması için kullanılan bir tekniktir. Modeli, belirli özellik kümelerine çok fazla güvenmekten kaçınmaya zorlar.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; Not: Çoğunlukla derin öğrenme kütüphanleri, 'keep' ('tutma') parametresi 1−p aracılığıyla seyreltmeyi parametrize eder.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; Ağırlık düzenlileştirme ― Ağırlıkların çok büyük olmadığından ve modelin eğitim setine uygun olmadığından emin olmak için, genellikle model ağırlıklarında düzenlileştirme teknikleri uygulanır. Temel olanlar aşağıdaki tabloda özetlenmiştir:

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230; [LASSO, Ridge, Elastic Net]

<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Katsayıları 0'a düşürür, Değişken seçimi için iyi, Katsayıları daha küçük yapar, Değişken seçimi ile küçük katsayılar arasında ödünleşim sağlar]

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230; Erken durdurma ― Bu düzenleme tekniği, onaylama kaybı bir stabilliğe ulaştığında veya artmaya başladığında eğitim sürecini durdurur.

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230; [Hata, Geçerleme/Doğrulama, Eğitim, erken durdurma, Epochs]

<br>


**53. Good practices**

&#10230; İyi uygulamalar

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230; Küçük kümelerin ezberlenmesi ― Bir modelde hata ayıklama yaparken, modelin mimarisinde büyük bir sorun olup olmadığını görmek için hızlı testler yapmak genellikle yararlıdır. Özellikle, modelin uygun şekilde eğitilebildiğinden emin olmak için, ezberleyecek mi diye görmek için ağ içinde bir mini küme ile eğitilir. Olmazsa, modelin normal boyutta bir eğitim setini bırakmadan, küçük bir kümeyi bile ezberleyecek kadar çok karmaşık ya da yeterince karmaşık olmadığı anlamına gelir. 

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230; Gradyanların kontrolü ― Gradyan kontrolü, bir sinir ağının geriye doğru geçişinin uygulanması sırasında kullanılan bir yöntemdir. Analitik gradyanların değerini verilen noktalardaki sayısal gradyanlarla karşılaştırır ve doğruluk için bir kontrol rolü oynar.

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230; [Tip, Sayısal gradyan, Analitik gradyan]

<br>


**57. [Formula, Comments]**

&#10230; [Formül, Açıklamalar]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230; [Maliyetli; Kayıp, boyut başına iki kere hesaplanmalı, Analitik uygulamanın doğruluğunu anlamak için kullanılır, Ne çok küçük (sayısal dengesizlik) ne de çok büyük (zayıf gradyan yaklaşımı) seçimi yapılmalı, bunun için ödünleşim gerekir]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230; ['Kesin' sonuç, Doğrudan hesaplama, Son uygulamada kullanılır]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].

&#10230; Derin Öğrenme el kitabı şimdi [hedef dilde] mevcuttur.

**61. Original authors**

&#10230; Orijinal yazarlar

<br>

**62.Translated by X, Y and Z**

&#10230; X, Y ve Z tarafından çevirildi

<br>

**63.Reviewed by X, Y and Z**

&#10230; X, Y ve Z tarafından gözden geçirildi

<br>

**64.View PDF version on GitHub**

&#10230; GitHub'da PDF sürümünü görüntüleyin

<br>

**65.By X and Y**

&#10230; X ve Y tarafından

<br>
