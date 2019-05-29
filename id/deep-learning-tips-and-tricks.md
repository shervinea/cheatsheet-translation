**Deep Learning Tips and Tricks translation**

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230;Deep Learning tip dan trik cheatsheet

<br>


**2. CS 230 - Deep Learning**

&#10230;Deep learning

<br>


**3. Tips and tricks**

&#10230;Tip and trick

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230;[Preprosessing data, Augmentasi Data, Normalisasi batch]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230;[Training neural network, Epoch, Mini-batch, Loss cross-entropy, Backpropagation, Gradient descent, Updating weights, Cek gradient]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230;[Penyetelan parameter, Inisialisasi Xavier, Transfer learning, Learning rate, Adaptive learning rates]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230;[Regularisasi, Dropout, Regularisasi weight, Early stopping]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230;[Good practices, Overfitting small batch, Cek gradient]

<br>


**9. View PDF version on GitHub**

&#10230;Lihat versi PDF pada GitHub

<br>


**10. Data processing**

&#10230;Preprosessing Data

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230;Augmentasi data - Deep learning model biasanya membutuhkan banyak data untuk mencapai hasil yang memuaskan dalam training. Adalah sangat bermanfaat untuk mendapatkan data lebih dari data yang tersedia menggunakan teknik augmentasi data. Teknik-teknik yang utama dirangkum pada tabel dibawah ini. Untuk input berupa gambar, dibawah ini adalah beberapa teknik yang dapat kita gunakan:

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230;[Original, Pembalikan, Rotasi, Random crop]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230;[Gambar tanpa modifikasi apapun, Dibalik terhadap sebuah aksis yang mana wujud gambar tersebut dipertahankan. Rotasi dengan sudut yang kecil, Mengsimulasi kalibrasi horizon yang tidak tepat, Fokus acak terhadap satu bagian dari gambar, Beberapa crops acak yang dilakukan pada sebuah baris]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230;[Perubahan warna, Penambahan noise, Menghapus informasi, Merubah kontras]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230;[Nilai RGB diubah sedikit, Menampak noise yang dapat terjadi dengan paparan cahaya, Penambahan noise, Lebih tolerant terhadap variasi kualitas dari input, Beberapa bagian dari gambar diabaikan, Meniru kemungkinan hilangnya bagian dari gambar, Perubahan terang cahaya, Mengkontrol perbedaan pada bentangan perubahan waktu dalam sehari]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230;Perlu diperhatikan: data biasanya hanya diaugmentasi pada proses training.

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;Normalisasi batch - Normalisasi batch adalah sebuah step dari penggunaan parameter γ,β yang mengnormalisasi batch {xi} (sampel dari keseluruhan data). Dengan menuliskan μB,σ2B sebagain nilai rata-rata dan variansi dari batch kita ingin untuk membenarkan nilai batch tersebut, hal tersebut dilakukan sebagai berikut:

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;Biasanya ditaruh setelah layer fully connected atau convolutional dan sebelum layer non-linear dan bertujuan untuk memungkinkannya penggunakan nilai learning rate yang lebih besar dan mengurangi ketergantungan pada nilai inisialisasi parameter.

<br>


**19. Training a neural network**

&#10230;Training sebuah neural network

<br>


**20. Definitions**

&#10230;Definisi-definisi

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230;Epoch  - Pada konteks training sebuah model, epoch adalah istilah yang digunakan untuk menunjukan satu iterasi dimana model melihat keseluruhan data training untuk memperbaharui weight-nya.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230;Mini-batch gradient descent - Selama proses training, pembaharuan nilai weight biasanya tidak berdasarkan keseluruhan data training pada satu waktu dikarenakan kompleksitas perhitungan atau berdasarkan satu data point dikarenakan permasalahan noise. Sebagai gantinya, pembaharuan dilakukan menggunakan mini-batches, dimana jumlah data points pada sebuah batch adalah hyperparameter yang dapat kita sesuaikan.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230;Fungsi loss - Untuk mengkuantifikasi seberapa bagus performa dari sebuah model, fungsi loss L biasanya digunakan untuk mengevaluasi sejauh mana output sebenarnya y diperdiksi secara tepat oleh keluaran model z.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;Loss cross-entropy - Pada kasus klasifikasi biner di neural network, loss cross-entropy L(z,y) biasa didefinisikan sebagai berikut:

<br>


**25. Finding optimal weights**

&#10230;Menemukan weight yang optimal

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230;Backpropagation - Backpropagation adalah sebuah metode untuk memperbaharui nilai weight pada neural network dengan mempertimbangkan nilai keluaran sebenarnya dan keluaran yang diharapkan. Turunan terhadap setiap nilai weight dihitung menggunakan chain rule.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230;Menggunakan metode ini, setiap weight diperbaharui dengan aturan:

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;Memperbaharui weight - Pada neural network, nilai weight diperbaharui seperti berikut:

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230;[Langkah 1: Ambil sebuah batch dari training data dan lakukan forward propagation untuk menghitung loss, Langkah 2: Backpropagate nilai loss untuk mendapatkan gradient dari loss terhadap setiap weight, Langkah 3: Gunakan gradient untuk memperbaharui nilai weight dari network.]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230;[Forward propagation, Backpropagation, Perubahan weight]

<br>


**31. Parameter tuning**

&#10230;Penyesuaian nilai parameter

<br>


**32. Weights initialization**

&#10230;Inisialisasi weight

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230;Penginisialisasian Xavier - Sebagai ganti dari teknik inisialisi yang semata-mata secara acak, penginisialisian Xavier memungkinkan proses inisialisi nilai weight yang mempertimbangkan keunikan karaktersitik pada arsitektur model.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230;

<br>


**35. [Training size, Illustration, Explanation]**

&#10230;[Ukuran training data, Illustrasi, Penjelasan]

<br>


**36. [Small, Medium, Large]**

&#10230;[Kecil, Menengah, Besar]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230;[Bekukan seluruh layer, train weight pada layer softmax, Bekukan keseluruhan layer, train weight pada layer-layer terakhir dan softmax, Train weight pada seluruh layer dan layer softmax dengan menginisialisasi weight dengan model yang telah ditrain sebelumnya.]

<br>


**38. Optimizing convergence**

&#10230;Konvergensi yang optimal

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.
**

&#10230;Learning rate - Learning rate, sering dinotasikan sebagai α atau η, mengindikasikan sebepara besar nilai weight diubah. Nilai dari learning rate dapat diset fix atau diubah secara adaptiv. Metode sekarang yang paling terkenal adalah Adam, yang merupakan sebuah metode yang menyelaraskan nilai learning rate.

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230;Learning rate yang adaptiv - Menjadikan nilai learning-rate berubah-ubah saat men-training sebuah model dapat mengurangi waktu training dan menambah keoptimal solusi secara numerik. Meskipun Adam optimizer adalah teknik yang paling banyak digunakan, metode-metode lainnya dapat juga berguna. Metode-metode tersebut dirangkum pada tabel dibawah ini:

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230;[Metode, Penjelasan, Perubahan w, Perubahan b]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230;[Momentum, Dampens oscillations, Pemutahiran SDG, 2 parameter harus disesuaikan]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230;[RMSprop, Root Mean Square propagation, Mempercepat algoritma learning dengan mengkontrol osilasi]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230;[Adam, Etimasi Adaptive Moment, Metode paling terkenal, 4 parameter harus disesuaikan]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230;Perlu diperhatikan: metode-metode lain termasuk Adadelta, Adagrad dan SGD.

<br>


**46. Regularization**

&#10230;Regularisasi

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230;Dropout - Dropout adalah sebuah teknik yang digunakan pada neural network untuk mencegah model overfit pada training data dengan mendrop out neuron-neuron dengan probabilitas p>0. Dropout mencegah model tergantung terlalu besar terhadap bagian khusus dari fitur pada data.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230;Perlu diperhatikan: kebanyak deep learning framework menset parameter dropout melalui 'keep' parameter 1-p.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230;Regularisasi weight - Untuk memastikan bahwa nilai weight tidak terlalu besar sehingga model tidak overfit terhadap training data, teknik-teknik regularisasi biasanya dilakukan pada weight model. Teknik-teknik yang utama dirangkum pada tabel dibawah ini:

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230;[LASSO, Ridge, Elastic Net]

<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;bis. Menyusutkan nilai koefisien ke 0, Bagus untuk pemilihan variabel, Membuat nilai koefisien lebih kecil, Tradeoff antara pemilihan variabel dan koefisien yang kecil]

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230;Early stopping - Teknik regularisasi ini menghentikan proses training segera setelah nilai dari loss validasi mencapai plateau (tidak mengecil) atau mulai membesar.

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230;[Error, Validasi, Trainig, early stopping, Epochs]

<br>


**53. Good practices**

&#10230;Good practices

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230;Overfitting small batch - Ketika melakukan debug pada sebuah model, sangatlah penting untuk melakukan segera test untuk mengetahui apakah ada permasalahan besar pada arsitektur dari model. Pada khususnya, untuk memastikan bahwa model dapat ditrained secara seharusnya, sebuah mini-batch diberikan ke network untuk melihat jika network dapat overfit pada batch tersebut. Jika network tidak overfit, hal tersebut berarti model terlalu kompleks atau tidak cukup kompleks bahkan untuk overfit terhadap batch yang kecil, apalagi ukuran normal sebuah training data.

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230;Chek gradient - Check gradient adalah metode yang digunakan saat mengimplementasi backward pass pada sebuah neural network. Chek gradient membandingkan nilai gradient analitik terhadap nilai gradient numerik pada titik tertentu dan memiliki perananan sebagai perkiraan kebenaran.

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230;[Tipe, Gradient numerik, Gradient Analitik]

<br>


**57. [Formula, Comments]**

&#10230;[Formula, Komentar]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230;[Mebutuhkan komputasi yang besar; loss harus dihitung dua kali untuk setiap dimensi, Digunakan untuk memverifikasi kebeneran dari implementasi analitik, Trade-off pada memilih nilai h tidak terlalu kecil (tidak stabil secara numerik) atau tidak telalu besar (aproksimasi nilai gradient yang tidak bagus)]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230;['Tepat' hasil, Komputasi langsung, Digunakan pada implementasi akhir]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].

&#10230;Deep Learning cheatsheet sekarang tersedia di [id]


**61. Original authors**

&#10230;Penulis orisinil

<br>

**62.Translated by X, Y and Z**

&#10230;Diterjemahkan oleh X, Y dan Z

<br>

**63.Reviewed by X, Y and Z**

&#10230;Ditinjau kembali oleh X, Y and Z

<br>

**64.View PDF version on GitHub**

&#10230;Lihat versi PDF pada GitHub

<br>

**65.By X and Y**

&#10230;Oleh X dan Y

<br>
