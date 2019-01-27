**1. Recurrent Neural Networks cheatsheet**

&#10230; Tekrarlayan Yapay Sinir Ağları (Recurrent Neural Networks-RNN) El Kitabı

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Derin Öğrenme

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230; [Genel bakış, Mimari yapı, RNN'lerin uygulamaları, Kayıp fonksiyonu, Geriye Yayılım]

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230; [Uzun vadeli bağımlılıkların ele alınması, Ortak aktivasyon fonksiyonları, Gradyanın kaybolması / patlaması, Gradyan kırpma, GRU / LSTM, Kapı tipleri, Çift Yönlü RNN, Derin RNN]

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230; [Kelime gösterimini öğrenme, Notasyonlar, Gömme matrisi, Word2vec, Skip-gram, Negatif örnekleme, GloVe]

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230; [Kelimeleri karşılaştırmak, Cosine benzerliği, t-SNE]

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230; [Dil modeli, n-gram, Karışıklık]

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230; [Makine çevirisi, Işın araması, Uzunluk normalizasyonu, Hata analizi, Bleu skoru]

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230; [Dikkat, Dikkat modeli, Dikkat ağırlıkları]

<br>


**10. Overview**

&#10230; Genel Bakış

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230; Geleneksel bir RNN mimarisi - RNN'ler olarak da bilinen tekrarlayan sinir ağları, gizli durumlara sahipken önceki çıktıların girdi olarak kullanılmasına izin veren bir sinir ağları sınıfıdır. Tipik olarak aşağıdaki gibidirler:

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230; Her bir t zamanında, a<t> aktivasyonu ve y<t> çıktısı aşağıdaki gibi ifade edilir:

<br>


**13. and**

&#10230; ve

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230; burada Wax,Waa,Wya,ba,by geçici olarak paylaşılan katsayılardır ve g1,g2 aktivasyon fonksiyonlarıdır.

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230; Tipik bir RNN mimarisinin artıları ve eksileri aşağıdaki tabloda özetlenmiştir:

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230; [Avantajlar, Herhangi bir uzunluktaki girdilerin işlenmesi imkanı, Girdi büyüklüğüyle artmayan model boyutu, Geçmiş bilgileri dikkate alarak hesaplama, Zaman içinde paylaşılan ağırlıklar]

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230; [Dezavantajları, Yavaş hesaplama, Uzun zaman önceki bilgiye erişme zorluğu, Mevcut durum için gelecekteki herhangi bir girdinin düşünülememesi]

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230; RNN'lerin Uygulamaları ― RNN modelleri çoğunlukla doğal dil işleme ve konuşma tanıma alanlarında kullanılır. Farklı uygulamalar aşağıdaki tabloda özetlenmiştir:

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230; [RNN Türü, Örnekleme, Örnek]

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230; [Bire bir, Bire çok, Çoka bir, Çoka çok]

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230; [Geleneksel sinir ağı, Müzik üretimi, Duygu sınıflandırma, İsim varlık tanıma, Makine çevirisi]

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230; Kayıp fonksiyonu ― Tekrarlayan bir sinir ağı olması durumunda, tüm zaman dilimlerindeki L kayıp fonksiyonu, her zaman dilimindeki kayıbı temel alınarak aşağıdaki gibi tanımlanır:

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230; Zamanla geri yayılım ― Geriye yayılım zamanın her noktasında yapılır. T zaman diliminde, ağırlık matrisi W'ye göre L kaybının türevi aşağıdaki gibi ifade edilir:

<br>


**24. Handling long term dependencies**

&#10230; Uzun vadeli bağımlılıkların ele alınması

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230; Yaygın olarak kullanılan aktivasyon fonksiyonları ― RNN modüllerinde kullanılan en yaygın aktivasyon fonksiyonları aşağıda açıklanmıştır:

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230; [Sigmoid, Tanh, RELU]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230; Kaybolan / patlayan gradyan ― Kaybolan ve patlayan gradyan fenomenlerine RNN'ler bağlamında sıklıkla rastlanır. Bunların olmasının nedeni, katman sayısına göre katlanarak azalan / artan olabilen çarpımsal gradyan nedeniyle uzun vadeli bağımlılıkları yakalamanın zor olmasıdır.

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230; Gradyan kırpma ― Geri yayılım işlemi sırasında bazen karşılaşılan patlayan gradyan sorunuyla başa çıkmak için kullanılan bir tekniktir. Gradyan için maksimum değeri sınırlayarak, bu durum pratikte kontrol edilir.

<br>


**29. clipped**

&#10230; kırpılmış

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230; Giriş Kapıları Çeşitleri ― Kaybolan gradyan problemini çözmek için bazı RNN türlerinde belirli kapılar kullanılır ve genellikle iyi tanımlanmış bir amaca sahiptir. Genellikle Γ olarak ifade edilir ve şuna eşittir:

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230; burada W, U, b kapıya özgü katsayılardır ve σ ise sigmoid fonksiyondur. Temel olanlar aşağıdaki tabloda özetlenmiştir:

<br>


**32. [Type of gate, Role, Used in]**

&#10230; [Kapının tipi, Rol, Kullanılan]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230; [Güncelleme kapısı, Uygunluk kapısı, Unutma kapısı, Çıkış kapısı]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230; [Şimdi ne kadar geçmiş olması gerekir?, Önceki bilgiyi bırak?, Bir hücreyi sil ya da silme?, Bir hücreyi ortaya çıkarmak için ne kadar?]

<br>


**35. [LSTM, GRU]**

&#10230; [LSTM, GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230; GRU/LSTM ― Geçitli Tekrarlayan Birim (Gated Recurrent Unit-GRU) ve Uzun Kısa Süreli Bellek Birimleri (Long Short-Term Memory-LSTM), geleneksel RNN'lerin karşılaştığı kaybolan gradyan problemini ele alır, LSTM ise GRU'nun genelleştirilmiş halidir. Her bir mimarinin karakterizasyon denklemlerini özetleyen tablo aşağıdadır:

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230; [Karakterizasyon, Geçitli Tekrarlayan Birim (GRU), Uzun Kısa Süreli Bellek (LSTM), Bağımlılıklar]

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230; Not: ⋆ işareti iki vektör arasındaki birimsel çarpımı belirtir.

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230; RNN varyantları ― Aşağıdaki tablo, diğer yaygın kullanılan RNN mimarilerini özetlemektedir:

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230; [Çift Yönlü (Bidirectional-BRNN), Derin (Deep-DRNN)]

<br>


**41. Learning word representation**

&#10230; Kelime temsilini öğrenme

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230; Bu bölümde V kelimeleri, |V| ise kelimelerin boyutlarını ifade eder.

<br>


**43. Motivation and notations**

&#10230; Motivasyon ve notasyon

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230; Temsil etme teknikleri ― Kelimeleri temsil etmenin iki temel yolu aşağıdaki tabloda özetlenmiştir:

<br>


**45. [1-hot representation, Word embedding]**

&#10230; [1-hot gösterim, Kelime gömme]

<br>


**46. [teddy bear, book, soft]**

&#10230; [oyuncak ayı, kitap, yumuşak]

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230;

<br> [ow not edildi, Naive yaklaşım, benzerlik bilgisi yok, ew not edildi, kelime benzerliği dikkate alınır]


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230; Gömme matrisi ― Belirli bir w kelimesi için E gömme matrisi, 1-hot temsilini ew gömmesi sayesinde aşağıdaki gibi eşleştiren bir matristir:

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230; Not: Gömme matrisinin öğrenilmesi hedef / içerik olabilirlik modelleri kullanılarak yapılabilir.

<br>


**50. Word embeddings**

&#10230; Kelime gömmeleri

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230; Word2vec ― Word2vec, belirli bir kelimenin diğer kelimelerle çevrili olma olasılığını tahmin ederek kelime gömmelerini öğrenmeyi amaçlayan bir çerçevedir. Popüler modeller arasında skip-gram, negatif örnekleme ve CBOW bulunur.

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230; [Sevimli ayıcık okuyor, ayıcık, yumuşak, Farsça şiir, sanat]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230; [Proxy görevinde ağı eğitme, üst düzey gösterimi çıkartme, Kelime gömme hesaplama]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230; Skip-gram ― Skip-gram word2vec modeli verilen herhangi bir t hedef kelimesinin c gibi bir bağlam kelimesi ile gerçekleşme olasılığını değerlendirerek kelime gömmelerini öğrenen denetimli bir öğrenme görevidir.

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230; Not: Softmax bölümünün paydasındaki tüm kelime dağarcığını toplamak, bu modeli hesaplama açısından maliyetli kılar. CBOW, verilen bir kelimeyi tahmin etmek için çevreleyen kelimeleri kullanan başka bir word2vec modelidir.

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230; Negatif örnekleme - Belirli bir bağlamın ve belirli bir hedef kelimenin eşzamanlı olarak ortaya çıkmasının muhtemel olup olmadığının değerlendirilmesini, modellerin k negatif örnek kümeleri ve 1 pozitif örnek kümesinde eğitilmesini hedefleyen, lojistik regresyon kullanan bir ikili sınıflandırma kümesidir. Bağlam sözcüğü c ve hedef sözcüğü t göz önüne alındığında, tahmin şöyle ifade edilir:

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230; Not: Bu yöntem, skip-gram modelinden daha az hesaplamalıdır.

<br>


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230; GloVe ― Kelime gösterimi için Global vektörler tanımının kısaltılmış hali olan GloVe, eşzamanlı bir X matrisi kullanan ki burada her bir Xi,j , bir hedefin bir j bağlamında gerçekleştiği sayısını belirten bir kelime gömme tekniğidir. Maliyet fonksiyonu J aşağıdaki gibidir:

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and ? play in this model, the final word embedding e(final)w is given by:**

&#10230; f, Xi,j=0⟹f(Xi,j)=0 olacak şekilde bir ağırlıklandırma fonksiyonudur.
Bu modelde e ve θ'nin oynadığı simetri göz önüne alındığında, e (final) w'nin kelime gömmesi şöyle ifade edilir:

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230; Not: Öğrenilen kelime gömme bileşenlerinin ayrı ayrı bileşenleri tam olarak yorumlanamaz.

<br>


**60. Comparing words**

&#10230; Kelimelerin karşılaştırılması

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230; Kosinüs benzerliği ― w1 ve w2 kelimeleri arasındaki kosinüs benzerliği şu şekilde ifade edilir:

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230; Not: θ, w1 ve w2 kelimeleri arasındaki açıdır.

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230; t-SNE ― t-SNE (t-dağıtımlı Stokastik Komşu Gömme), yüksek boyutlu gömmeleri daha düşük boyutlu bir alana indirmeyi amaçlayan bir tekniktir. Uygulamada, kelime uzaylarını 2B alanda görselleştirmek için yaygın olarak kullanılır.

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230; [edebiyat, sanat, kitap, kültür, şiir, okuma, bilgi, eğlendirici, sevimli, çocukluk, kibar, ayıcık, yumuşak, sarılmak, sevimli, sevimli]

<br>


**65. Language model**

&#10230; Dil modeli

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230; Genel bakış - Bir dil modeli P(y) cümlesinin olasılığını tahmin etmeyi amaçlar.

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230; n-gram modeli ― Bu model, eğitim verilerindeki görünüm sayısını sayarak bir ifadenin bir korpusta ortaya çıkma olasılığını ölçmeyi amaçlayan naif bir yaklaşımdır.

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230; Karışıklık - Dil modelleri yaygın olarak, PP olarak da bilinen karışıklık metriği kullanılarak değerlendirilir ve bunlar T kelimelerinin sayısıyla normalize edilmiş veri setinin ters olasılığı olarak yorumlanabilir. Karışıklık, daha düşük, daha iyi ve şöyle tanımlanır:

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230; Not: PP, t-SNE'de yaygın olarak kullanılır.

<br>


**70. Machine translation**

&#10230; Makine çevirisi

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230; Genel bakış ― Bir makine çeviri modeli, daha önce yerleştirilmiş bir kodlayıcı ağına sahip olması dışında, bir dil modeline benzer. Bu nedenle, bazen koşullu dil modeli olarak da adlandırılır. Amaç şu şekilde bir cümle bulmaktır:

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230; Işın arama ― Makine çevirisinde ve konuşma tanımada kullanılan ve x girişi verilen en olası cümleyi bulmak için kullanılan sezgisel bir arama algoritmasıdır.

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k-1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**
  
&#10230; [Adım 1: En olası B kelimeleri bulun y<1>, 2. Adım: Koşullu olasılıkları hesaplayın y|x,y<1>, ..., y, 3. Adım: En olası B kombinasyonlarını koruyun x,y<1>, ..., y, İşlemi durdurarak sonlandırın]

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230; Not: Eğer ışın genişliği 1 olarak ayarlanmışsa, bu naif (naive) bir açgözlü (greedy) aramaya eşdeğerdir.

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230; Işın genişliği ― Işın genişliği B, ışın araması için bir parametredir. Daha yüksek B değerleri daha iyi sonuç elde edilmesini sağlar fakat daha düşük performans ve daha yüksek hafıza ile. Küçük B değerleri daha kötü sonuçlara neden olur, ancak hesaplama açısından daha az yoğundur. B için standart bir değer 10 civarındadır.

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230; Uzunluk normalizasyonu ― Sayısal stabiliteyi arttırmak için, ışın arama genellikle, aşağıdaki gibi tanımlanan normalize edilmiş log-olabilirlik amacı olarak adlandırılan normalize edilmiş hedefe uygulanır:

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230; Not: α parametresi yumuşatıcı olarak görülebilir ve değeri genellikle 0,5 ile 1 arasındadır.

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230; Hata analizi ― Kötü bir çeviri elde edildiğinde, aşağıdaki hata analizini yaparak neden iyi bir çeviri almadığımızı araştırabiliriz:

<br>


**79. [Case, Root cause, Remedies]**

&#10230; [Durum, Ana neden, Çözümler]

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230; [Işın arama hatası, RNN hatası, Işın genişliğini artırma, farklı mimariyi deneme, Düzenlileştirme, Daha fazla bilgi edinme]

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230; Bleu puanı ― İki dilli değerlendirme alt ölçeği (bleu) puanı, makine çevirisinin ne kadar iyi olduğunu, n-gram hassasiyetine dayalı bir benzerlik puanı hesaplayarak belirler. Aşağıdaki gibi tanımlanır:

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230; pn, n-gramdaki bleu skorunun sadece aşağıdaki şekilde tanımlandığı durumlarda:

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230; Not: Yapay olarak şişirilmiş bir bleu skorunu önlemek için kısa öngörülen çevirilere küçük bir ceza verilebilir.

<br>


**84. Attention**

&#10230; Dikkat

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y should pay to the activation a<t′> and c the context at time t, we have:**

&#10230; Dikkat modeli ― Bu model, bir RNN'de girişin önemli olduğu düşünülen belirli kısımlarına dikkat etmesine olanak sağlar,sonuçta ortaya çıkan modelin pratikteki performansını arttırır. α<t,t′> ile ifade edilen dikkat miktarı, a<t′> aktivasyonu ve t zamanındaki c bağlamını y çıktısı olarak verir.

<br>


**86. with**

&#10230; ile

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230; Not: Dikkat skorları, görüntü altyazılama ve makine çevirisinde yaygın olarak kullanılır.

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230; Sevimli bir oyuncak ayı Fars edebiyatı okuyor.

<br>


**89. Attention weight ― The amount of attention that the output y should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230; Dikkat ağırlığı ― Y çıktısının a<t′> aktivasyonuna vermesi gereken dikkat miktarı, aşağıdaki gibi hesaplanan α<t,t′> ile verilir:

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230; Not: hesaplama karmaşıklığı Tx'e göre ikinci derecedendir.

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Derin Öğrenme el kitapları şimdi [hedef dilde] mevcuttur.

<br>

**92. Original authors**

&#10230;Orijinal yazarlar

<br>

**93. Translated by X, Y and Z**

&#10230; X, Y ve Z tarafından çevrilmiştir.

<br>

**94. Reviewed by X, Y and Z**

&#10230; X, Y ve Z tarafından gözden geçirilmiştir.

<br>

**95. View PDF version on GitHub**

&#10230; GitHub'da PDF versiyonunu görüntüleyin.

<br>

**96. By X and Y**

&#10230; X ve Y tarafından

<br>
