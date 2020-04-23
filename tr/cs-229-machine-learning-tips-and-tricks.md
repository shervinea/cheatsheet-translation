**1. Machine Learning tips and tricks cheatsheet**

&#10230;  Makine Öğrenmesi ipuçları ve püf noktaları el kitabı

<br>

**2. Classification metrics**

&#10230; Sınıflandırma metrikleri

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; İkili bir sınıflandırma durumunda, modelin performansını değerlendirmek için gerekli olan ana metrikler aşağıda verilmiştir.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Karışıklık matrisi - Karışıklık matrisi, bir modelin performansını değerlendirirken daha eksiksiz bir sonuca sahip olmak için kullanılır. Aşağıdaki şekilde tanımlanmıştır:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Tahmini sınıf, Gerçek sınıf]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Ana metrikler - Sınıflandırma modellerinin performansını değerlendirmek için aşağıda verilen metrikler yaygın olarak kullanılmaktadır:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Metrik, Formül, Açıklama]

<br>

**8. Overall performance of model**

&#10230; Modelin genel performansı

<br>

**9. How accurate the positive predictions are**

&#10230; Doğru tahminlerin ne kadar kesin olduğu

<br>

**10. Coverage of actual positive sample**

&#10230; Gerçek pozitif örneklerin oranı

<br>

**11. Coverage of actual negative sample**

&#10230; Gerçek negatif örneklerin oranı

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Dengesiz sınıflar için yararlı hibrit metrik

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; İşlem Karakteristik Eğrisi (ROC) ― İşlem Karakteristik Eğrisi (receiver operating curve), eşik değeri değiştirilerek Doğru Pozitif Oranı-Yanlış Pozitif Oranı grafiğidir. Bu metrikler aşağıdaki tabloda özetlenmiştir:

<br>

**14. [Metric, Formula, Equivalent]**
 
&#10230; [Metrik, Formül, Eşdeğer]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; Eğri Altında Kalan Alan (AUC) ― Aynı zamanda AUC veya AUROC olarak belirtilen işlem karakteristik eğrisi altındaki alan, aşağıdaki şekilde gösterildiği gibi İşlem Karakteristik Eğrisi (ROC)'nin altındaki alandır:

<br>

**16. [Actual, Predicted]**

&#10230; [Gerçek, Tahmin Edilen]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Temel metrikler - Bir f regresyon modeli verildiğinde aşağıdaki metrikler genellikle modelin performansını değerlendirmek için kullanılır:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Toplam karelerinin toplamı, Karelerinin toplamının açıklaması, Karelerinin toplamından artanlar]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Belirleme katsayısı - Genellikle R2 veya r2 olarak belirtilen belirleme katsayısı, gözlemlenen sonuçların model tarafından ne kadar iyi kopyalandığının bir ölçütüdür ve aşağıdaki gibi tanımlanır:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Ana metrikler - Aşağıdaki metrikler, göz önüne aldıkları değişken sayısını dikkate alarak regresyon modellerinin performansını değerlendirmek için yaygın olarak kullanılır:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; burada L olabilirlik ve ˆσ2, her bir yanıtla ilişkili varyansın bir tahminidir.

<br>

**22. Model selection**

&#10230; Model seçimi

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Kelime Bilgisi - Bir model seçerken, aşağıdaki gibi sahip olduğumuz verileri 3 farklı parçaya ayırırız:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Eğitim seti, Doğrulama seti, Test seti]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Model eğitildi, Model değerlendirildi, Model tahminleri gerçekleştiriyor]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Genelde veri kümesinin %80'i, Genelde veri kümesinin %20'si]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Ayrıca doğrulama için bir kısmını bekletme veya geliştirme seti olarak da bilinir, Görülmemiş veri]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Model bir kere seçildikten sonra, tüm veri seti üzerinde eğitilir ve görünmeyen test setinde test edilir. Bunlar aşağıdaki şekilde gösterilmiştir:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Çapraz doğrulama ― Çapraz doğrulama, başlangıçtaki eğitim setine çok fazla güvenmeyen bir modeli seçmek için kullanılan bir yöntemdir. Farklı tipleri aşağıdaki tabloda özetlenmiştir:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [k − 1 katı üzerinde eğitim ve geriye kalanlar üzerinde değerlendirme, n − p gözlemleri üzerine eğitim ve kalan p üzerinde değerlendirme]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Genel olarak k=5 veya 10, Durum p=1'e bir tanesini dışarıda bırak denir]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; En yaygın olarak kullanılan yöntem k-kat çapraz doğrulama olarak adlandırılır ve k-1 diğer katlarda olmak üzere, bu k sürelerinin hepsinde model eğitimi yapılırken, modeli bir kat üzerinde doğrulamak için eğitim verilerini k katlarına ayırır. Hata için daha sonra k-katlar üzerinden ortalama alınır ve çapraz doğrulama hatası olarak adlandırılır.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Düzenlileştirme (Regularization) - Düzenlileştirme prosedürü, modelin verileri aşırı öğrenmesinden kaçınılmasını ve dolayısıyla yüksek varyans sorunları ile ilgilenmeyi amaçlamaktadır. Aşağıdaki tablo, yaygın olarak kullanılan düzenlileştirme tekniklerinin farklı türlerini özetlemektedir:


<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Değişkenleri 0'a kadra küçült, Değişken seçimi için iyi, Katsayıları daha küçük yap, Değişken seçimi ile küçük katsayılar arasındaki çelişki]


<br>

**35. Diagnostics**

&#10230; Tanı

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Önyargı - Bir modelin önyargısı, beklenen tahmin ve verilen veri noktaları için tahmin etmeye çalıştığımız doğru model arasındaki farktır.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Varyans - Bir modelin varyansı, belirli veri noktaları için model tahmininin değişkenliğidir.
 
<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Önyargı/varyans çelişkisi - Daha basit model, daha yüksek önyargı, ve daha karmaşık model, daha yüksek varyans.


<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Belirtiler, Regresyon illüstrasyonu, sınıflandırma illüstrasyonu, derin öğrenme illüstrasyonu, olası çareler]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Yüksek eğitim hatası, Test hatasına yakın eğitim hatası, Yüksek önyargı, Eğitim hatasından biraz daha düşük eğitim hatası, Çok düşük eğitim hatası, Eğitim hatası test hatasının çok altında, Yüksek varyans]


<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Model karmaşıklaştığında, Daha fazla özellik ekle, Daha uzun eğitim süresi ile eğit, Düzenlileştirme gerçekleştir, Daha fazla bilgi edin]


<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Hata analizi - Hata analizinde mevcut ve mükemmel modeller arasındaki performans farkının temel nedeni analiz edilir.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Ablatif analiz - Ablatif analizde mevcut ve başlangıç modelleri arasındaki performans farkının temel nedeni analiz edilir.

<br>

**44. Regression metrics**

&#10230; Regresyon metrikleri

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Sınıflandırma metrikleri, karışıklık matrisi, doğruluk, kesinlik, geri çağırma, F1 skoru, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Regresyon metrikleri, R karesi, Mallow'un CP'si, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Model seçimi, çapraz doğrulama, düzenlileştirme]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Tanı, Önyargı/varyans çelişkisi, hata/ablatif analiz]
