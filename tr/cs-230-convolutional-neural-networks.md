**1. Convolutional Neural Networks cheatsheet**

&#10230; Evrişimli Sinir Ağları el kitabı

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Derin Öğrenme

<br>


**3. [Overview, Architecture structure]**

&#10230; [Genel bakış, Mimari yapı]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [Katman tipleri, Evrişim, Ortaklama, Tam bağlantı]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [Filtre hiperparametreleri, Boyut, Adım aralığı/Adım kaydırma, Ekleme/Doldurma]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [Hiperparametrelerin ayarlanması, Parametre uyumluluğu, Model karmaşıklığı, Receptive field]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [Aktivasyon fonksiyonları, Düzeltilmiş Doğrusal Birim, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [Nesne algılama, Model tipleri, Algılama, Kesiştirilmiş Bölgeler, Maksimum olmayan bastırma, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [Yüz doğrulama/tanıma, Tek atış öğrenme, Siamese ağ, Üçlü yitim/kayıp]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [Sinirsel stil aktarımı, Aktivasyon, Stil matrisi, Stil/içerik maliyet fonksiyonu]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [İşlemsel püf nokta mimarileri, Çekişmeli Üretici Ağ, ResNet, Inception Ağı]

<br>


**12. Overview**

&#10230; Genel bakış

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Geleneksel bir CNN (Evrişimli Sinir Ağı) mimarisi - CNN'ler olarak da bilinen evrişimli sinir ağları, genellikle aşağıdaki katmanlardan oluşan belirli bir tür sinir ağıdır:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; Evrişim katmanı ve ortaklama katmanı, sonraki bölümlerde açıklanan hiperparametreler ile ince ayar (fine-tuned) yapılabilir.

<br>


**15. Types of layer**

&#10230; Katman tipleri

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Evrişim katmanı (CONV) ― Evrişim katmanı (CONV) evrişim işlemlerini gerçekleştiren filtreleri, I girişini boyutlarına göre tararken kullanır. Hiperparametreleri F filtre boyutunu ve S adımını içerir. Elde edilen çıktı O, öznitelik haritası veya aktivasyon haritası olarak adlandırılır.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Not: evrişim adımı, 1B ve 3B durumlarda da genelleştirilebilir (B: boyut).

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Ortaklama (POOL) - Ortaklama katmanı (POOL), tipik olarak bir miktar uzamsal değişkenlik gösteren bir evrişim katmanından sonra uygulanan bir örnekleme işlemidir. Özellikle, maksimum ve ortalama ortaklama, sırasıyla maksimum ve ortalama değerin alındığı özel ortaklama türleridir.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Tip, Amaç, Görsel Açıklama, Açıklama]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Maksimum ortaklama, Ortalama ortaklama, Her ortaklama işlemi, geçerli matrisin maksimum değerini seçer, Her ortaklama işlemi, geçerli matrisin değerlerinin ortalaması alır.]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [Algılanan özellikleri korur, En çok kullanılan, Boyut azaltarak örneklenmiştelik öznitelik haritası, LeNet'te kullanılmış]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230; Tam Bağlantı (FC) ― Tam bağlı katman (FC), her girişin tüm nöronlara bağlı olduğu bir giriş üzerinde çalışır. Eğer varsa, FC katmanları genellikle CNN mimarisinin sonuna doğru bulunur ve sınıf skorları gibi hedefleri optimize etmek için kullanılabilir.

<br>


**23. Filter hyperparameters**

&#10230; Hiperparametrelerin filtrelenmesi

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; Evrişim katmanı, hiperparametrelerinin ardındaki anlamı bilmenin önemli olduğu filtreler içerir.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Bir filtrenin boyutları - C kanalları içeren bir girişe uygulanan F×F boyutunda bir filtre, I×I×C boyutundaki bir girişte evrişim gerçekleştiren ve aynı zamanda bir çıkış özniteliği haritası üreten F aktivitesi (aktivasyon olarak da adlandırılır) O) O×O×1 boyutunda harita.

<br>


**26. Filter**

&#10230; Filtre

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Not: F×F boyutunda K filtrelerinin uygulanması, O×O×K boyutunda bir çıktı öznitelik haritasının oluşmasını sağlar.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Adım aralığı ― Evrişimli veya bir ortaklama işlemi için, S adımı (adım aralığı), her işlemden sonra pencerenin hareket ettiği piksel sayısını belirtir.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230; Sıfır ekleme/doldurma ― Sıfır ekleme/doldurma, girişin sınırlarının her bir tarafına P sıfır ekleme işlemini belirtir. Bu değer manuel olarak belirlenebilir veya aşağıda detaylandırılan üç moddan biri ile otomatik olarak ayarlanabilir:

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Mod, Değer, Görsel Açıklama, Amaç, Geçerli, Aynı, Tüm]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Ekleme/doldurma yok, Boyutlar uyuşmuyorsa son evrişimi düşürür, Öznitelik harita büyüklüğüne sahip ekleme/doldurma ⌈IS⌉, Çıktı boyutu matematiksel olarak uygundur, 'Yarım' ekleme olarak da bilinir, Son konvolüsyonların giriş sınırlarına uygulandığı maksimum ekleme, Filtre girişi uçtan uca "görür"]

<br>


**32. Tuning hyperparameters**

&#10230; Hiperparametreleri ayarlama

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Evrişim katmanında parametre uyumu - Girdinin hacim büyüklüğü I uzunluğu, F filtresinin uzunluğu, P sıfır ekleme miktarı, S adım aralığı, daha sonra bu boyut boyunca öznitelik haritasının O çıkış büyüklüğü belirtilir:

<br>


**34. [Input, Filter, Output]**

&#10230; [Giriş, Filtre, Çıktı]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Not: çoğunlukla, Pstart=Pend≜P, bu durumda Pstart+Pend'i yukarıdaki formülde 2P ile değiştirebiliriz.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Modelin karmaşıklığını anlama - Bir modelin karmaşıklığını değerlendirmek için mimarisinin sahip olacağı parametrelerin sayısını belirlemek genellikle yararlıdır. Bir evrişimsli sinir ağının belirli bir katmanında, aşağıdaki şekilde yapılır:

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Görsel Açıklama, Giriş boyutu, Çıkış boyutu, Parametre sayısı, Not]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [Filtre başına bir bias(önyargı) parametresi, Çoğu durumda, S<F, K için ortak bir seçenek 2C'dir.]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [Ortaklama işlemi kanal bazında yapılır, Çoğu durumda S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [Giriş bağlantılanmış, Nöron başına bir bias parametresi, tam bağlantı (FC) nöronlarının sayısı yapısal kısıtlamalardan arındırılmış]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Evrişim sonucu oluşan haritanın boyutu ― K katmanında filtre çıkışı, k-inci aktivasyon haritasının her bir pikselinin 'görebileceği' girişin Rk×Rk olarak belirtilen alanını ifade eder. Fj, j ve Si katmanlarının filtre boyutu, i katmanının adım aralığı ve S0=1 (ilk adım aralığının 1 seçilmesi durumu) kuralıyla, k katmanındaki işlem sonucunda elde edilen aktivasyon haritasının boyutları bu formülle hesaplanabilir:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; Aşağıdaki örnekte, F1=F2=3 ve S1=S2=1 için R2=1+2⋅1+2⋅1=5 sonucu elde edilir.

<br>


**43. Commonly used activation functions**

&#10230; Yaygın olarak kullanılan aktivasyon fonksiyonları

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Düzeltilmiş Doğrusal Birim ― Düzeltilmiş doğrusal birim katmanı (ReLU), (g)'nin tüm elemanlarında kullanılan bir aktivasyon fonksiyonudur. Doğrusal olmamaları ile ağın öğrenmesi amaçlanmaktadır. Çeşitleri aşağıdaki tabloda özetlenmiştir:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230;[ReLU, Sızıntı ReLU, ELU, ile]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Doğrusal olmama karmaşıklığı biyolojik olarak yorumlanabilir, Negatif değerler için ölen ReLU sorununu giderir, Her yerde türevlenebilir]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax ― Softmax adımı, x∈Rn skorlarının bir vektörünü girdi olarak alan ve mimarinin sonunda softmax fonksiyonundan p∈Rn çıkış olasılık vektörünü oluşturan genelleştirilmiş bir lojistik fonksiyon olarak görülebilir. Aşağıdaki gibi tanımlanır:

<br>


**48. where**

&#10230; buna karşılık

<br>


**49. Object detection**

&#10230; Nesne algılama

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Model tipleri ― Burada, nesne tanıma algoritmasının doğası gereği 3 farklı kestirim türü vardır. Aşağıdaki tabloda açıklanmıştır:

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Görüntü sınıflandırma, Sınıflandırma ve lokalizasyon (konumlama), Algılama]

<br>


**52. [Teddy bear, Book]**

&#10230; [Oyuncak ayı, Kitap]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Bir görüntüyü sınıflandırır, Nesnenin olasılığını tahmin eder, Görüntüdeki bir nesneyi algılar/tanır, Nesnenin olasılığını ve bulunduğu yeri tahmin eder, Bir görüntüdeki birden fazla nesneyi algılar, Nesnelerin olasılıklarını ve nerede olduklarını tahmin eder]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [Geleneksel CNN, Basitleştirilmiş YOLO (You-Only-Look-Once), R-CNN (R: Region - Bölge), YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Algılama ― Nesne algılama bağlamında, nesneyi konumlandırmak veya görüntüdeki daha karmaşık bir şekli tespit etmek isteyip istemediğimize bağlı olarak farklı yöntemler kullanılır. İki ana tablo aşağıdaki tabloda özetlenmiştir:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Sınırlayıcı kutu ile tespit, Karakteristik nokta algılama]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Görüntüde nesnenin bulunduğu yeri algılar, Bir nesnenin şeklini veya özelliklerini algılar (örneğin gözler), Daha ayrıntılı]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Kutu merkezi (bx,by), yükseklik bh ve genişlik bw, Referans noktalar (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Kesiştirilmiş Bölgeler - Kesiştirilmiş Bölgeler, IoU (Intersection over Union) olarak da bilinir, Birleştirilmiş sınırlama kutusu, tahmin edilen sınırlama kutusu (Bp) ile gerçek sınırlama kutusu Ba üzerinde ne kadar doğru konumlandırıldığını ölçen bir fonksiyondur. Olarak tanımlanır:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Not: Her zaman IoU∈ [0,1] ile başlarız. Kural olarak, Öngörülen bir sınırlama kutusu Bp, IoU (Bp, Ba)⩾0.5 olması durumunda makul derecede iyi olarak kabul edilir.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Öneri (Anchor) kutular, örtüşen sınırlayıcı kutuları öngörmek için kullanılan bir tekniktir. Uygulamada, ağın aynı anda birden fazla kutuyu tahmin etmesine izin verilir, burada her kutu tahmini belirli bir geometrik öznitelik setine sahip olmakla sınırlıdır. Örneğin, ilk tahmin potansiyel olarak verilen bir formun dikdörtgen bir kutusudur, ikincisi ise farklı bir geometrik formun başka bir dikdörtgen kutusudur.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Maksimum olmayan bastırma - Maksimum olmayan bastırma tekniği, nesne için yinelenen ve örtüşen öneri kutuları içinde en uygun temsilleri seçerek örtüşmesi düşük olan kutuları kaldırmayı amaçlar. Olasılık tahmini 0.6'dan daha düşük olan tüm kutuları çıkardıktan sonra, kalan kutular ile aşağıdaki adımlar tekrarlanır:

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Verilen bir sınıf için, Adım 1: En büyük tahmin olasılığı olan kutuyu seçin., Adım 2: Önceki kutuyla IoU⩾0.5 olan herhangi bir kutuyu çıkarın.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Kutu tahmini/kestirimi, Maksimum olasılığa göre kutu seçimi, Aynı sınıf için örtüşme kaldırma, Son sınırlama kutuları]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO ― You Only Look Once (YOLO), aşağıdaki adımları uygulayan bir nesne algılama algoritmasıdır:

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Adım 1: Giriş görüntüsünü G×G kare parçalara (hücrelere) bölün., Adım 2: Her bir hücre için, aşağıdaki formdan y'yi öngören bir CNN çalıştırın: k kez tekrarlayın]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; pc'nin bir nesneyi algılama olasılığı olduğu durumlarda, bx, by, bh, bw tespit edilen olası sınırlayıcı kutusunun özellikleridir, cl, ..., cp, p sınıflarının tespit edilen one-hot temsildir ve k öneri (anchor) kutularının sayısıdır.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Adım3: Potansiyel yineli çakışan sınırlayıcı kutuları kaldırmak için maksimum olmayan bastırma algoritmasını çalıştır.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Orijinal görüntü, GxG kare parçalara (hücrelere) bölünmesi, Sınırlayıcı kutu kestirimi, Maksimum olmayan bastırma]
 
<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Not: pc=0 olduğunda, ağ herhangi bir nesne algılamamaktadır. Bu durumda, ilgili bx, ..., cp tahminleri dikkate alınmamalıdır.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN - Evrişimli Sinir Ağları ile Bölge Bulma (R-CNN), potansiyel olarak sınırlayıcı kutuları bulmak için görüntüyü bölütleyen (segmente eden) ve daha sonra sınırlayıcı kutularda en olası nesneleri bulmak için algılama algoritmasını çalıştıran bir nesne algılama algoritmasıdır.

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Orijinal görüntü, Bölütleme (Segmentasyon), Sınırlayıcu kutu kestirimi, Maksimum olmayan bastırma]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; Not: Orijinal algoritma hesaplamalı olarak maliyetli ve yavaş olmasına rağmen, yeni mimariler algoritmanın Hızlı R-CNN ve Daha Hızlı R-CNN gibi daha hızlı çalışmasını sağlamıştır.

<br>


**74. Face verification and recognition**

&#10230; Yüz doğrulama ve tanıma

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Model tipleri ― İki temel model aşağıdaki tabloda özetlenmiştir:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Yüz doğrulama, Yüz tanıma, Sorgu, Kaynak, Veri tabanı]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [Bu doğru kişi mi?, Bire bir arama, Veritabanındaki K kişilerden biri mi?, Bire-çok arama]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; Tek Atış (Onr-Shot) Öğrenme - Tek Atış Öğrenme, verilen iki görüntünün ne kadar farklı olduğunu belirleyen benzerlik fonksiyonunu öğrenmek için sınırlı bir eğitim seti kullanan bir yüz doğrulama algoritmasıdır. İki resme uygulanan benzerlik fonksiyonu sıklıkla kaydedilir (resim 1, resim 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Siyam (Siamese) Ağı - Siyam Ağı, iki görüntünün ne kadar farklı olduğunu ölçmek için görüntülerin nasıl kodlanacağını öğrenmeyi amaçlar. Belirli bir giriş görüntüsü x(i) için kodlanmış çıkış genellikle f(x(i)) olarak alınır.

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Üçlü kayıp - Üçlü kayıp ℓ, A (öneri), P (pozitif) ve N (negatif) görüntülerinin üçlüsünün gömülü gösterimde hesaplanan bir kayıp fonksiyonudur. Öneri ve pozitif örnek aynı sınıfa aitken, negatif örnek bir diğerine aittir. α∈R+ marjın parametresini çağırarak, bu kayıp aşağıdaki gibi tanımlanır:

<br>


**81. Neural style transfer**

&#10230; Sinirsel stil transferi (aktarımı)

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Motivasyon ― Sinirsel stil transferinin amacı, verilen bir C içeriğine ve verilen bir S stiline dayanan bir G görüntüsü oluşturmaktır.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [İçerik C, Stil S, Oluşturulan görüntü G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Aktivasyon ― Belirli bir l katmanında, aktivasyon [l] olarak gösterilir ve nH×nw×nc boyutlarındadır

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; İçerik maliyeti fonksiyonu ― İçerik maliyeti fonksiyonu Jcontent(C,G), G oluşturulan görüntüsünün, C orijinal içerik görüntüsünden ne kadar farklı olduğunu belirlemek için kullanılır.Aşağıdaki gibi tanımlanır:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; Stil matrisi - Stil matrisi G[l], belirli bir l katmanının her birinin G[l]kk′ elemanlarının k ve k′ kanallarının ne kadar ilişkili olduğunu belirlediği bir Gram matristir. A[l] aktivasyonlarına göre aşağıdaki gibi tanımlanır:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Not: Stil görüntüsü ve oluşturulan görüntü için stil matrisi, sırasıyla G[l] (S) ve G[l] (G) olarak belirtilmiştir.

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Stil maliyeti fonksiyonu - Stil maliyeti fonksiyonu Jstyle(S,G), oluşturulan G görüntüsünün S stilinden ne kadar farklı olduğunu belirlemek için kullanılır. Aşağıdaki gibi tanımlanır:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Genel maliyet fonksiyonu - Genel maliyet fonksiyonu, α, β parametreleriyle ağırlıklandırılan içerik ve stil maliyet fonksiyonlarının bir kombinasyonu olarak tanımlanır:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Not: yüksek bir α değeri modelin içeriğe daha fazla önem vermesini sağlarken, yüksek bir β değeri de stile önem verir.

<br>


**91. Architectures using computational tricks**

&#10230; Hesaplama ipuçları kullanan mimariler

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230; Çekişmeli Üretici Ağlar - GAN olarak da bilinen çekişmeli üretici ağlar, modelin üretici denen ve gerçek imajı ayırt etmeyi amaçlayan ayırıcıya beslenecek en doğru çıktının oluşturulmasını amaçladığı üretici ve ayırt edici bir modelden oluşur.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [Eğitim, Gürültü, Gerçek dünya görüntüsü, Üretici, Ayırıcı, Gerçek Sahte]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; Not: GAN'ın kullanım alanları, yazıdan görüntüye, müzik üretimi ve sentezi.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet ― Artık Ağ mimarisi (ResNet olarak da bilinir), eğitim hatasını azaltmak için çok sayıda katman içeren artık bloklar kullanır. Artık blok aşağıdaki karakterizasyon denklemine sahiptir:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception Ağ ― Bu mimari inception modüllerini kullanır ve özelliklerini çeşitlendirme yoluyla performansını artırmak için farklı evrişim kombinasyonları denemeyi amaçlamaktadır. Özellikle, hesaplama yükünü sınırlamak için 1x1 evrişm hilesini kullanır.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Derinöğrenme el kitabı artık kullanıma hazır [hedef dilde].

<br>


**98. Original authors**

&#10230; Orijinal yazarlar

<br>


**99. Translated by X, Y and Z**

&#10230; X, Y ve Z tarafından çevirildi

<br>


**100. Reviewed by X, Y and Z**

&#10230; X, Y ve Z tarafından kontrol edildi

<br>


**101. View PDF version on GitHub**

&#10230; GitHub'da PDF sürümünü görüntüleyin

<br>


**102. By X and Y**

&#10230; X ve Y ile

<br>
