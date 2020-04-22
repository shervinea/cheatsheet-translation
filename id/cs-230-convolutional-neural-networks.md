**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230;Cheatsheet Convolutional Neural Network

<br>


**2. CS 230 - Deep Learning**

&#10230;Deep Learning

<br>


**3. [Intisari, Struktur arsitektur]**

&#10230;[Overview, Struktur Arsitektur]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230;[Jenis-jenis layer, Konvolusi, Pooling, Fully connected]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230;[Hiperparameter filter, Dimensi, Stride, Padding]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230;[Penyetelan hiperparameter, Kesesuaian parameter, Kompleksitas model, Receptive field]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230;[Fungsi-fungsi aktifasi, Rectified Linear Unit, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230;[Deteksi objek, Tipe-tipe model, Deteksi, Intersection over Union, Non-max suppression, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230;[Verifikasi/pengenal wajah, One shot learning, Siamese network, Loss triplet]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230;[Transfer neural style, Aktifasi, Matriks style, Fungsi cost style/konten]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230;[Arkitektur trik komputasional, Generative Adversarial Net, ResNet, Inception Network]

<br>


**12. Overview**

&#10230;Ringkasan

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230;Arkitektur dari sebuah tradisional CNN - Convolutional neural network, juga dikenal sebagai CNN, adalah sebuah tipe khusus dari neural network yang secara umum terdiri dari layer-layer berikut:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230;Layer konvolusi and layer pooling dapat disesuaikan terhadap hiperparameter yang dijelaskan pada bagian selanjutnya.

<br>


**15. Types of layer**

&#10230;Jenis-jenis layer

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230;Layer convolution - Layer convolution (CONV) menggunakan banyak filter yang dapat melakukan operasi konvolusi karena CONV memindai input I dengan memperhatikan dimensinya. Hiperparameter dari CONV meliputi ukuran filter F dan stride S. Keluaran hasil O disebut feature map atau activation map.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230;Catatan: tahap konvolusi dapat digeneralisasi juga dalam kasus 1D dan 3D.

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230;Pooling (POOL) - Layer pooling adalah sebuah operasi downsampling, biasanya diaplikasikan setelah lapisan konvolusi, yang menyebabkan invarian spasial. Pada khususnya, pooling max dan average merupakan jenis-jenis pooling spesial di mana masing-masing nilai maksimal dan rata-rata diambil.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230;[Jenis, Tujuan, Ilustrasi, Komentar]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230;[Max pooling, Average pooling, Setiap operasi pooling mewakili nilai maksimal dari tampilan terbaru, setiap operasi pooling meratakan nilai-nilai dari tampilan terbaru]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230;[Mempertahankan fitur yang terdeteksi, yang paling sering digunakan, Downsamples feature map, dipakai di LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230;Fully Connected (FC) - Fully connected layer (FC) menangani sebuah masukan dijadikan 1D ddi mana setiap masukan terhubung ke seluruh neuron. Bila ada, lapisan-lapisan FC biasanya ditemukan pada akhir arsitektur CNN dan dapat digunakan untuk mengoptimalkan hasil seperti skor-skor kelas (pada kasus klasifikasi).

<br>


**23. Filter hyperparameters**

&#10230;Hiperparameter filter

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230;Layer konvolusi mengandung penyaring yang penting untuk dimengerti tentang maksud dari penyaring hiperparameter tersebut.
<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230;Dimensi dari sebuah filter - Sebuah filter dengan ukuran FxF diaplikasikan pada sebuah input yang memuat C channel memiliki volume FxFxC yang melakukan konvolusi pada sebuah input masukan dengan ukuran IxIxC dan menghasilkan sebuah keluaran feature map (juga dikenal activation map) dengan ukuran O×O×1

<br>


**26. Filter**

&#10230;Filter

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230;Catatan: pengaplikasian dari penyaring F dengan ukuran FxF menghasilkan sebuah keluaran fitur peta dengan ukuran O×O×K.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230;Stride - Untuk sebuah konvolusi atau sebauh operasi pooling, stide S melambangkan jumlah pixel yang dilewati window setelah setiap operasi.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230;Zero-padding - Zero-padding melambangkan proses penambahan P nilai 0 pada setiap sisi akhir dari masukan. Nilai dari zero-padding dapat dispesifikasikan secara manual atau secara otomatis melalui salah satu dari tiga mode yang dijelaskan dibawah ini:

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230;[Mode, Nilai, Ilustrasi, Tujuan, Valid, Same, Full]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230;[No padding, Hapus konvolusi terakhir jika dimensi tidak sesuai, Padding yang menghasilkan feature map dengan ukuran ⌈IS⌉, Ukuran keluaran cocok secara matematis, Juga disebut 'half' padding, Maximum padding menjadikan akhir konvolusi dipasangkan pada batasan dari input, Filter 'melihat' masukan end-to-end]

<br>


**32. Tuning hyperparameters**

&#10230;Menyetel hiperparameter

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230;Kompabilitas parameter pada lapisan konvolusi - Dengan menuliskan I sebagai panjang dari ukuran volume masukan, F sebagai panjang dari filter, P sebagai jumlah dari zero padding, S sebagai stride, maka ukuran keluaran 0 dari feature map pada dimensi tersebut ditandai dengan:

<br>


**34. [Input, Filter, Output]**

&#10230;[Masukan, Filter, Keluaran]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230;Catatan: sering, Pstart=Pend≜P, pada kasus tersebut kita dapat mengganti Pstart+Pend dengan 2P pada formula di atas.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230;Memahami kompleksitas dari model - Untuk menilai kompleksitas dari sebuah model, sangatlah penting untuk menentukan jumlah parameter yang arsitektur dari model akan miliki. Pada sebuah convolutional neural network, hal tersebut dilakukan sebagai berikut:

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230;[Ilustrasi, Ukuran masukan, Ukuran keluaran, Jumlah parameter, Catatan]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230;[Satu parameter bias per filter, Pada banyak kasus, S>F, sebuah pilihan umum untuk K adalah 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230;[Operasi pooling yang dilakukan dengan channel-wise, Pada banyak kasus, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230;[Masukan diratakan, satu parameter bias untuk setiap neuron, Jumlah dari neuron FC adalah terbebas dari batasan struktural]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230;Receptive field - Receptive field pada layer k adalah area yang dinotasikan RkxRk dari masukan yang setiap pixel dari k-th activation map dapat "melihat". Dengan menyebut Fj (sebagai) ukuran penyaring dari lapisan j dan Si (sebagai) nilai stride dari lapisan i dan dengan konvensi 50=1, receptive field pada lapisan k dapat dihitung dengan formula:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230;Pada contoh dibawah ini, kita memiliki F1=F2=3 dan S1=S2=1, yang menghasilkan R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230;Fungsi-fungsi aktifasi yang biasa dipakai

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230;Rectified Linear Unit - Layer rectified linear unit (ReLU) adalah sebuat fungsi aktivasi g yang digunakan pada seluruh elemen volume. Unit ini bertujuan untuk menempatkan non-linearitas pada jaringan. Variasi-variasi ReLU ini dirangkum pada tabel di bawah ini:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230;[ReLU, Leaky ReLU, ELU, dengan]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230;[Kompleksitas non-linearitas yang dapat ditafsirkan secara biologi, Menangani permasalahan dying ReLU yang bernilai negatif, Yang dapat dibedakan di mana pun]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230;Softmax - Langkah softmax dapat dilihat sebagai sebuah fungsi logistik umum yang berperan sebagai masukan dari nilai skor vektor x∈Rn dan mengualarkan probabilitas produk vektor p∈Rn melalui sebuah fungsi softmax pada akhir dari jaringan arsitektur. Softmax didefinisikan sebagai berikut:

<br>


**48. where**

&#10230;Di mana

<br>


**49. Object detection**

&#10230;Deteksi objek

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230;Tipe-tipe model - Ada tiga tipe utama dari algoritma rekognisi objek, yang mana hakikat yang diprediksi tersebut berbeda. Tipe-tipe tersebut dijelaskan pada tabel di bawah ini:

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230;[Klasifikasi gambar, Klasifikasi w. lokalisasi, Deteksi]

<br>


**52. [Teddy bear, Book]**

&#10230;[Boneka beruang, Buku]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230;[Mengklasifikasikan sebuah gambar, Memprediksi probabilitas dari objek, Mendeteksi objek pada sebuah gambar, Memprediksi probabilitas dari objek dan lokasinya pada gambar, Mendeteksi hingga beberapa objek pada sebuah gambar, Memprediksi probabilitas dari objek-objek dan dimana lokasi mereka]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230;[CNN tradisional, Simplified YOLO, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230;Deteksi - Pada objek deteksi, metode yang berbeda digunakan tergantung apakah kita hanya ingin untuk mengetahui lokasi objek atau mendeteksi sebuah bentuk yang lebih rumit pada gambar. Dua metode yang utama dirangkum pada tabel dibawah ini:

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230;[Deteksi bounding box, Deteksi landmark]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230;[Mendeteksi bagian dari gambar dinama objek berlokasi, Mendetek bentuk atau karakteristik dari sebuah objek (contoh: mata), Lebih granular]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230;[Pusat dari box (bx,by), tinggi bh dan lebah bw, Poin referensi (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230;[Intersection over Union - Intersection over Union, juga dikenal sebagai IoU, adalah sebuah fungsi yang mengkuantifikasi seberapa benar posisi dari sebuah prediksi bounding box Bp terhadap bounding box yang sebenarnya Ba. IoU didefinisikan sebagai berikut:]

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230;Perlu diperhatikan: kita selalu memiliki nilai IoU∈[0,1]. Umumnya, sebuah prediksi bounding box dianggap cukup bagus jika IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230;Anchor boxes ― Anchor boxing adalah sebuah teknik yang digunakan untuk memprediksi bounding box yang overlap. Pada pengaplikasiannya, network diperbolehkan untuk memprediksi lebih dari satu box secara bersamaan, dimana setiap prediksi box dibatasi untuk memiliki kumpulan properti geometri. Contohnya, prediksi pertama dapat berupa sebuah box persegi panjang untuk sebuah bentuk, sedangkan prediksi kedua adalah persegi panjang lainnya dengan bentuk geometri yang berbeda.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230;Non-max suppression ― Teknik non-max suppression bertujuan untuk menghapus duplikasi bounding box yang overlap satu sama lain dari sebuah objek yang sama dengan memilih box yang paling representatif. Setelah menghapus seluruh box dengan prediksi probability lebih kecil dari 0.6, langkah berikut diulang selama terdapat box tersisa.

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230;[Untuk sebuah kelas, Langkah 1: Pilih box dengan probabilitas prediksi tertinggi., Langkah 2: Singkirkan box manapun yang yang memiliki IoU⩾0.5 dengan box yang dipilih pada tahap 1.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230;[Prediksi-prediksi box, Seleksi box dari probabilitas tertinggi, Penghapusan overlap pada kelas yang sama, Bounding box akhir]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230;YOLO - You Only Look Once (YOLO) adalah sebuah algoritma deteksi objek yang melakukan langkah-langkah berikut:

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230;Langkah 1: Bagi gambar masukan kedalam sebuah grid dengan ukuran GxG, Langkah 2: Untuk setiap sel grid, gunakan sebuah CNN yang memprediksi y dengan bentuk sebagai berikut; lakukan sebanyak k kali]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230;dimana pc adalah deteksi probabilitas dari sebuah objek, bx,by,bh,bw adalah properti dari box bounding yang terdeteksi, c1,...,cp adalah representasi one-hot yang mana p classes terdeteksi, dan k adalah jumlah box anchor.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230;Langkah 3: Jalankan algoritma non-max suppression yang menghapus duplikasi potensial yang mengoverlap box bounding yang sebenarnya.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230;[Gambar asli, Pembagian kedalam grid berukuran GxG, Prediksi box bounding, Non-max suppression]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230;Perlu diperhatikan: ketika pc=0, maka netwok tidak mendeteksi objek apapun. Pada kasus seperti itu, prediksi yang bersangkutan bx,...,cp harus diabaikan.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230;R-CNN ― Region with Convolutional Neural Networks (R-CNN) adalah sebuah algoritma objek deteksi yang pertama-tama mensegmentasi gambar untuk menemukan potensial box-box bounding yang relevan dan selanjutnya menjalankan algoritma deteksi untuk menemukan objek yang paling memungkinkan pada box-box bounding tersebut..

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230;[Gambar asli, Segmentasi, Prediksi box bounding, Non-max suppressio]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230;Perlu diperhatikan: meskipun algoritma asli dari R-CNN membutuhkan komputasi resource yang besar dan lambar, arsitektur terbaru memungkinan algoritma untuk memiliki performa yang lebih cepat, yang dikenal sebagai Fast R-CNN dan Faster R-CNN.

<br>


**74. Face verification and recognition**

&#10230;Verifikasi wajah dan rekognisi

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230;Jenis-jenis model - Dua jenis tipe utama dirangkum pada tabel dibawah ini:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230;[Ferivikasi wajah, Rekognisi wajah, Query, Referensi, Database]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230;[Apakah ini adalah orang yang sesuai?, One-to-one lookup, Apakah ini salah satu dari K orang pada database?, One-to-many lookup]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230;One Shot Learning ― One Shot Learning adalah sebuah algoritma verifikasi wajah yang menggunakan sebuah training set yang terbatas untuk belajar fungsi kemiripan yang mengkuantifikasi seberapa berbeda dua gambar yang diberikan. Fungsi kemiripan yang diaplikasikan pada dua gambar sering dinotasikan sebagai d(image 1,image 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230;Siamese Network ― Siamese Networks didesain untuk mengkodekan gambar dan mengkuantifikasi seberapa berbeda dua buah gambar. Untuk sebuah gambar masukan x(i), keluaran yang dikodekan sering dinotasikan sebagai f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230;Loss triplet - Loss triplet adalah sebuah fungsi loss yang dihitung pada representasi embedding dari sebuah tiga pasang gambar A (anchor), P (positif) dan N (negatif). Sampel anchor dan positif berdasal dari sebuah kelas yang sama, sedangkan sampel negatif berasal dari kelas yang lain. Dengan menuliskan α∈R+ sebagai parameter margin, fungsi loss ini dapat didefinisikan sebagai berikut:

<br>


**81. Neural style transfer**

&#10230;Transfer neural style

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230;Motivasi: Tujuan dari mentransfer Neural style adalah untuk menghasilakn sebuah gambar G berdasarkan sebuah konten dan sebuah style S.

<br>


**83. [Content C, Style S, Generated image G]**

&#10230;[Konten C, Style S, gambar yang dihasilkan G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230;Aktifasi - Pada sebuah layer l, aktifasi dinotasikan sebagai a[l] dan berdimensi nH×nw×nc

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230;Fungsi cost content - Fungsi cost content Jcontent(C,G) digunakan untuk menghitung perbedaan antara gambar yang dihasilkan dan gambar konten yang sebenarnya C. Fungsi cost content didefinsikan sebagai berikut:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230;Matriks style - Matriks style G[l] dari sebuah layer l adalah sebuah matrix Gram dimana setiap elemennya G[l]kk′ mengkuantifikasi seberapa besar korelasi antara channel k dan k'. Matriks style didefinisikan terhadap aktifasi a[l] sebagai berikut:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230;Perlu diperhatikan: matriks style untuk gambar style dan gambar yang dihasilkan masing-masing dituliskan sebagai G[l] (S) dan G[l] (G).

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230;Fungsi cost style - Fungsi cost style Jstyle(S,G) digunakan untuk menentukan perbedaan antara gambar yang dihasilkan G dengan style yang diberikan S. Fungsi tersebut definisikan sebagai berikut:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230;Fungsi cost overall - Fungsi cost overall didefinisikan sebagai sebuah kombinasi dari fungsi cost konten dan syle, dibobotkan oleh parameter α,β, sebagai berikut:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230;Perlu diperhatikan: semakin tinggi nilai α akan membuat model lebih memperhatikan konten sedangkan semakin tinggi nilai β akan membuat model lebih memprehatikan style.

<br>


**91. Architectures using computational tricks**

&#10230;Arsitektur menggunakan trik komputasi.

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230;Generative Adversarial Network - Generative adversarial networks, juga dikenala sebagai GANs, terdiri dari sebuah generatif dan diskriminatif  model , dimana generatif model didesain untuk menghasilkan keluaran palsu yang mendekati keluaran sebenarnya yang akan diberikan kepada diskriminatif model yang didesain untuk membedakan gambar palsu dan gambar sebenarnya.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230;[Training, Noise, Gambar real-world, Generator, Discriminator, Real Fake]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230;Perlu diperhatikan: penggunaan dari variasi GANs meliputi sistem yang dapat mengubah teks ke gambar, dan menghasilkan dan mensintese musik.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet ― Arsitektur Residual Network (juga disebut ResNet) menggunakan blok-blok residual dengan jumlah layer yang banyak untuk mengurangi training error. Blok residual memiliki karakteristik formula sebagai berikut:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230;Inception Network ― Arsitektur ini menggunakan modul inception dan didesain dengan tujuan untuk meningkatkan performa network melalu diversifikasi fitur dengan menggunakan CNN yang berbeda-beda. Khususnya, inception model menggunakan trik 1×1 CNN untuk membatasi beban komputasi.

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230;Deep Learning cheatsheet sekarang tersedia di [Bahasa Indonesia]

<br>


**98. Original authors**

&#10230;Penulis asli

<br>


**99. Translated by X, Y and Z**

&#10230;Diterjemahkan oleh X, Y dan Z

<br>


**100. Reviewed by X, Y and Z**

&#10230;Diulas oleh X, Y dan Z

<br>


**101. View PDF version on GitHub**

&#10230;Lihat versi PDF pada GitHub

<br>


**102. By X and Y**

&#10230;Oleh X dan Y

<br>
