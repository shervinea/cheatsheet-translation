**Recurrent Neural Networks translation**

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230;Recurrent Neural Network cheatsheet

<br>


**2. CS 230 - Deep Learning**

&#10230;CS 230 - Deep Learning

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230;[Overview, Struktur Arkitektur, Penerapan RNN, Fungsi loss, Backpropagation]

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230;[Menangani kertergantungan jangka panjang, Fungsi aktifasi yang umum, Gradient yang vanish atau explode, Mengklip gradient, GRU/LSTM, Jenis-jenis gerbang, Bidirectional RNN, Deep RNN]

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230;[Belajar representasi kata-kata, Notasi-notasi, Matriks embedding, Word2vec, Skip-gram, Negative sampling, GloVe]

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230;[Comparing words, Cosine similaritas, t-SNE]

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230;[Model bahasa, n-gram, Perplexity]

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230;[Mesin perjemah, Beam search, Normalisasi panjang, Analisis Eror, Bleu score]

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230;[Attention, Attention model, Attention weight]

<br>


**10. Overview**

&#10230;[Overview]

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230;[Arsitektur dari sebuah tradisional RNN - Recurrent neural network, juga dikenal sebagai RNN, adalah sebuah kelas dari neural network yang membolehkan penggunaan keluaran sebelumnya sebagai input meski memiliki hidden states. Recurrent neural network dapat dituliskan sebagai berikut:]

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230;[Untuk setiap timestep t, aktivasi a<t> dan keluaran y<t> diekspresikan sebagai berikut:]

<br>


**13. and**

&#10230; dan

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230;dimana Wax,Waa,Wya,ba,by adalah koefisien yang dipakai bersama untuk setiap waktu dan g1,gw adalah fungsi-fungsi aktifasi.

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230;Keuntungan dan kerugian dari ariketur RNN yang tipikal dirangkum pada tabel dibawah ini:

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230;[Keuntungan, Kemampuan untuk memproses masukan dengan panjang seberapapun, Ukuran model tidak meningkat dengan peningkatan ukuran dari masukan, Komputasi mempertimbangkan informasi yang lalu, Weight yang sama digunakan sepanjang waktu]

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230;[Kerugian, Komputasi yang lambat, Kesulitan dalam mengakses informasi yang dahulu sekali, Tidak mempertimbangkan masukan selajutnya untuk state yang sekarang]

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230;Penggunaan dari RNN - Model RNN biasa digunakan dalam ilmu natural language processing dan rekognisi suara. Berbegai aplikasi RNN dirangkum pada tabel dibawah ini:

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230;[Jenis RNN, Ilustrasi, Contoh]

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230;[Satu-ke-satu, Satu-ke-banyak, Banyak-ke-satu, Banyak-ke-banyak]

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230;[Neural network tradisional, Membuat musik, Klasifikasi sentiment, Rekognisi entitas nama, Mesin penerjemah]

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230;[Fungsi loss, pada kasus recurrent neural network, fungsi loss L dari keseluruhan waktu didefinisikan berdasarkan pada loss pada setiap waktu sebagai berikut.]

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230;Backpropagation sepanjang waktu - Backpropagation dilakukan pada setiap poin disetiap waktu. Pada timestep T, derivative dari loss L terhadap matriks W dapat diekspresikan sebagai berikut:

<br>


**24. Handling long term dependencies**

&#10230;Menangani ketergantungan jangka panjang

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230;Fungsi-fungsi aktifasi yang biasa dipakai - Fungsi-fungsi aktifasi yang paling sering digunakan di module RNN dijelaskan dibawah ini:

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230;[Sigmpid, Tanh RELU]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230;Gradient yang vanish/explode - Fenomena gradient yang vanish(menghilang) dan explode(meledak) sering terjadi pada RNNs. Alasan mengapa fenomena tersebut terjadi adalah dikarenakan RNN sulit untuk mengerti ketergantungan jangka panjang pada data karena gradient multiplikatif yang dapat secara eksponensial menurun atau meningkat terhadap jumah layer.

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230;Mengklipping gradient - Adalah sebuah teknik yang digunakan untuk mengatasi permsaalahan gradient yang explode yang terkadang terjadi saat melakukan backpropagation. Dengan membatasi nilai maksimum dari gradient, fenomena ini terkontrol pada praktiknya.

<br>


**29. clipped**

&#10230; diklip

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230;Jenis-jenis gerbang - Untuk mengatasi permasalahan gradient yang vanish, gerbang-gerbang spesifik digunakan pada beberapa tipe RNN dan dengan tujuan tertentu. Gerbang tersebut biasanya dinotasikan sebagai Γ dan sama dengan:

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230;dimana W,U,b adalah koefisien yang spesifik terhadap gerbang dan σ yang merupakan fungsi sigmoid. Gerbang-gerbang yang utama dirangkum pada tabel dibawah ini:

<br>


**32. [Type of gate, Role, Used in]**

&#10230;[Jenis gerbang, Peranan, Digunakan di]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230;[Gerbang pembaharuan, Gerbang relevance, Gerbang forget, Gerbang keluaran]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230;[Seberapa banyak informasi sebelumnya berarti sekarang?, Hapus informasi sebelumnya?, Hapus sebuah sel atau tidak?, Seberapa banyak sebuah sel harus diperlihatkan?]

<br>


**35. [LSTM, GRU]**

&#10230;[LSTM, GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230;GRU/LSTM - Gated Recurrent Unit (GRU) and Long Short-Term Memory unit menangangi permasalahan gradient yang vanish yang biasa dialami RNN tradisional, dengan LSTM sebagai versi generalisasi dari GRU. Dibawah ini adalah tabel yang merangkum persamaan yang mengkarakterisasi setiap arsitektur.

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230;[Karakterisasi, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Ketergantungan]

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230;Perlu diperhatikan: tang * melambangkan perkalian element-wise antara dua vektor.

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230;Jenis-jenis dari RNN - Tabel dibawah ini merangkum arsitektur-arsitektur RNN lainnya yang biasa dipakai:

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230;[Bidirectional (BRNN), Deep (DRNN)]

<br>


**41. Learning word representation**

&#10230;Belajar representasi kata

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230;Pada sesi ini, kita menotasikan V sebagai kosa kata dan |V| sebagai ukuran kosa kata tersebut.

<br>


**43. Motivation and notations**

&#10230;Motivasi dan notasi-notasi

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230;Teknik representasi - Dua cara utama dalam merepresentasikan kata-kata dirangkum pada tabel dibawah ini:

<br>


**45. [1-hot representation, Word embedding]**

&#10230;[Representasi 1-hot, Word embedding]

<br>


**46. [teddy bear, book, soft]**

&#10230;[boneka beruang, buku, lembut]

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230;[Perhatikan ow, Pendakatan yang naif, tidak ada persamaan informasi, Perhatikan ew, Mempertimbangkan kemiripan kata]

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230;Matriks embedding - Untuk sebuah kata, matriks embedding E adalah sebuah matriks yang memetakan representasi 1-hot dari kata tersebut ow ke embedingnya sebagai berikut:

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230;Perlu diperhatikan: menentukan nilai matriks embedding dapat dilakukan menggunakan target/konteks model likelihood.

<br>


**50. Word embeddings**

&#10230;Word embeddings

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230;Word2vec - Word2vec adalah sebuah framework yang ditujukan untuk mengetahui word embeddings dengan mengestimasi likelihood dari sebuah kata yang dikelilingi oleh kata-kata lain. Model-model yang terkenal meliputi skip-gram, negative sampling dan CBOW.

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230;[Sebuah boneka beruang sedang membaca, boneka beruang, lembut, sajak Persia, seni]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230;[Train network pada tugas proxy, Ekstrak representasi level tinggi, Hitung word embeddings]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230;Skip-gram - The skip-gram model word2vec adalah sebuah tugas supervised learning algoritma yang mempelajari word embeddings dengan menaksir likelihood dari sebuah target word t yang menyangkut sebuah konteks c. Dengan menuliskan θt sebagai sebuah parameter terasosiasi dengan t, probabilitas P(t|c) dituliskan sebagai berikut:

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230;Perlu diperhatikan: jumlah dari keseluruhan kosa kata pada pembagi pada bagian softmax membuat proses komputasi model ini membutuhkan resource (waktu dan memori) yang banyak. CBOW adalah model word2vec lainnya yang menggunakan kata-kata sekitar untuk memprediksi sebuah kata.

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230;Negative sampling - Adalah sebuah set dari biner classifier yang menggunakan logistic regressions yang bertujuan untuk menilai seberapa besar sebuah konteks dan sebuah target kata untuk muncul secara bersamaan, dengan model-model yang ditrained pada set k sampel negative dan 1 sample positif. Diberikan sebuah kata konteks c dan sebuah target kata k, prediksi diekspresikan sebagai berikut:

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230;Perlu diperhatikan: komputasi model ini membutuhkan resource yang lebih sedikit dibandingkan model skip-gram.

<br>


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230;bis. Glove - Model Glove, kependekan dari global vectors for word representation, adalah teknik word embedding yang menggunakan sebuah matriks co-occurence X dimana setiap Xi,j melambangkan seberapa banyak sebuah target i terjadi dengan sebuah konteks j. Fungsi lossnya J dituliskan sebagai berikut:

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230;dimana f adalah sebuah fungsi pembobotan sepertiXi, j=0⟹f(Xi,j)=0.
Untuk simetri e dan θ yang berperan pada model ini, word embedding akhir e(akhir) w diformulasikan sebagai berikut:
<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230;Perlu diperhatikan: komponen-komponen individu dari word embeddings yang dipelajari tidak haruslah bisa diinterprestasi.

<br>


**60. Comparing words**

&#10230;Comparing words

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230;Cosine similarity - Cosine similarity antara kata w1 dan w2 dapat diekspresikan sebagai berikut:

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230;Perlu diperhatikan: θ adalah sudut antara kata w1 dan w2.

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230;t-SNE - t-SNE (t-distributed Stochastic Neighbor Embedding) adalah sebuah teknik yang bertujuan untuk mentransformasi high-dimensi embeddings ke sebuah ruang dimensi yang lebih rendah. Pada praktiknya, t-SNE biasa digunakan untuk mengvisualisasi word vektor pada ruang 2D.

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230;[literatur, seni, buku, budaya, sajak, baca, pengetahuan, menghibur, memikat, masa kecil, baik, boneka beruang, lembut, peluk, lucu, menggemaskan]

<br>


**65. Language model**

&#10230;[Model bahasa]

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230;Overview - Sebuah model bahasa bertujuan untuk mengestimasi probabilitas dari sebuah kalimat P(y).

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230;model n-gram - Model ini adalah sebuah metode naiv yang bertujuan untuk menkuantifikasi probabilitas bahwa sebuah ekspreksi muncul di sebuah korpus dengan menghitung jumlah kemunculan ekspresi tersebut pada data training.

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230;Perplexity - Model bahasa yang biasa diperhitungkan menggunakan metrik perplexity, juna dikenal sebagai PP, yang dapat diinterpretasi sebagai probability inverse dari dataset dinormalisasi oleh jumlah kata T. Semakin rendah perplexity semakin baik dan perplexity didefinisikan sebagain berikut:

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230;Perlu diperhatikan: PP biasa digunakan di t-SNE.

<br>


**70. Machine translation**

&#10230;Mesin penerjemah

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230;Overview -  Sebuah model mesin penerjemah serupa dengan model bahasa terkecuali mesin penerjemah memiliki sebuah network encoder yang sebelumnya telah ditrained. Untuk alasan tersebut, mesin penerjemah seringkali disebut sebagai sebuah model bahasa kondisional. Tujuan dari mesin penerjemah adalah untuk menemukan sebuah kalimat y seperti:

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230;Beam search - Adalah sebuah algoritma pencara heuristik yang digunakan pada mesin penerjemah dan rekognisi suara untuk menemukan kemungkinan kalimat y terhadap sebuah masukan yang diberikan.

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230;[Langkah 1: Temukan B kata yang paling memungkinkan y<1>, Langkah 2: Menghitung probabilitas kondisional y<k>|x,y<1>,...,y<k−1>, Langkah 3: Pertahankan B kombinasi teratas x,y<1>,...,y<k> Selesaikan proses pada sebuah stop word (kata pemberhenti)]

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230;Perlu diperhatikan: jika sebuah beam width diset dengan nilai 1, maka algoritma ini sama dengan algoritma naive greedy search.

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230;Beam width - Beam width B adalah sebuah parameter untuk algoritma beam search. Nilai besar B menghasilkan hasil yang lebih baik tetapi menyembabkan performa yang lebih lambat dan peningkatan penggunaan memory. Nilai kecil B menghasilakn hasil yang lebih buruk dengan komputasi yang membutukan lebih sedikit resource. Nilai standar untuk B adalah sekitar 10.

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230;Normalisasi length - Untuk meningkatkan stabiliti numerik, beam search biasanya diaplikasikan pada objektif yang dinormalisasi sebagai berikut, biasa disebut normalized log-likelihood objective, didefinisikan sebagai:

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230;Perlu diperhatikan: parameter α dapat diartikan sebagai sebuah softener, dan nilainya berkisar antara 0.5 dan 1.

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230;Analisis eror - Ketika mendapatkan prediksi terjemahan ˆy yang tidak sesuai, kita dapat mengetahui mengapa kita tidak mendapatkan terjemahan yang sesuai y∗ dengan melakukan analisis error berikut:

<br>


**79. [Case, Root cause, Remedies]**

&#10230;[Masalah, Akar permasalahan, Solusi]

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230;[Kesalahan beam search, kesalahan RNN, Tingkatkan nilai beam width, Coba arsitektur yang berbeda, Regularisasi, Dapatkan data lebih banyak]

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230;Skor Blue - Skor bilingual evaluation understudy (bleu) menghitung seberapa bagus sebuah mesin penerjemah dengan menghitung kemiripan skor berdasarkan n-gram presisi. Skor bleu didefinisikan sebagai berikut:

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230;dimana pn is the skor bleu pada n-gram hanya didefinisikan sebagai berikut:

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230;Perlu diperhatikan: pinalti singkat dapat diterapkan pada prediksi translasi yang pendek untuk mencegah skor bleu terinflansi secara artifisial.

<br>


**84. Attention**

&#10230;Attention

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230;Model attention -  Model ini menjadikan RNN dapat memperhatikan bagian tertentu dari masukan yang dipertimbangkan sebagai bagian yang penting, yang meningkatkan performa dari model pada penerapannya. Dengan mendefinisikan α<t,t′> sebagai jumlah attention yang keluaran y<t> harus berikan terhadap aktifasi a<t′> dan koteks c<t> pada time t, kita memiliki:

<br>


**86. with**

&#10230;dengan

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230;Perlu diperhatikan: skor attention biasanya digunakan pada proses pemberian keterangan pada gambar dan mesin penerjemah.

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230;Sebuah boneka beruang yang lucu sedang membaca literatur Persia.

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230;Weight attention - Jumlah dari attention yang keluaran y<t> harus berikan terhadap aktifasi a<t′> yang tentukan oleh α<t,t′> dan ditulis sebagai berikut:

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230;Perlu diperhatikan: kompleksitas komputasi adalah kuadratik terhadap Tx.

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230;Deep Learning cheatsheet sekarang tersedia di [id].

<br>

**92. Original authors**

&#10230;Penulis orisinil

<br>

**93. Translated by X, Y and Z**

&#10230;Diterjemahkan oleh X, Y, dan Z

<br>

**94. Reviewed by X, Y and Z**

&#10230;Diulas oleh X, Y, dan Z

<br>

**95. View PDF version on GitHub**

&#10230;Lihat versi PDF pada GitHub

<br>

**96. By X and Y**

&#10230;Oleh X dan Y

<br>
