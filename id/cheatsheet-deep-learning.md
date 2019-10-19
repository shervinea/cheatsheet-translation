**1. Deep Learning cheatsheet**

&#10230; **1. Catatan ringkas Deep Learning**

<br>

**2. Neural Networks**

&#10230; **2. Neural Networks**

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; **3. Neural networks merupakan sebuah kelas model yang disusun atas beberapa layer. Jenis umum dari neural networks yang umum digunakan adalah convolutional (CNN) dan recurrent neural networks (RNN).**

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; **4. Arsitektur - Beberapa istilah yang umum digunakan dalam arsitektur neural network dijelaskan pada gambar di bawah ini**

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; **5. [Input layer, hidden layer, output layer]**

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; **6. Dengan i adalah layer ke-i dari network dan j adalah unit hidden layer ke-j, maka:**

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; **7. Catatan: w, b, z adalah weight, bias, dan output.**

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; **8. Fungsi aktivasi - Fungsi aktivasi di unit hidden terakhir berfungsi untuk menunjukkan kompleksitas non-linear terhadap model. Beberapa yang umum digunakan:**

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; **9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;**10. Cross-entroy loss - Dalam konteks neural networks, cross-entroy loss L(z,y) sangat umum digunakan untuk mendefinisikan:**

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;**11. Learning rate - Learning rate, sering dinotasikan sebagai α atau η, mendefinisikan seberapa cepat nilai weight diperbaharui. Learning rate bisa diset dengan nilai fix atau dirubah secara adaptif. Metode yang paling terkenal saat ini adalah Adam. Sebuah method yang merubah learning rate secara adaptif.**

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;**12. Backpropagation - Backpropagation adalah metode untuk memperbarui bobot dalam neural networks dengan memperhitungkan output aktual dan output yang diinginkan. Bobot w dihitung dengan menggunakan aturan rantai turunan dalam bentuk berikut:**

<br>

**13. As a result, the weight is updated as follows:**

&#10230; **13. Sebagai hasilnya, nilai bobot diperbaharui sebagai berikut:**

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;**14. Memperbaharui nilai weights - Dalam neural network, nilai weights diperbarui nilainya dengan cara berikut:**

<br>

**15. Step 1: Take a batch of training data.**

&#10230;**15. Langkah 1: Mengambil batch (sampel data) dari keseluruhan training data.**

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;**16. Langkah 2: Melakukan forward propagation untuk mendapatkan nilai loss berdasarkan nilai masukan (input).**

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; **17. Langkah 3: Melakukan backpropagate terhadap loss untuk mendapatkan gradient.**

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230;**18. Langkah 4: Menggunakan gradient untuk untuk memperbarui nilai weights dari network.**

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;**19. Dropout - Dropout adalah teknik untuk mencegah overfit terhadap data training dengan menghilangkan satu atau lebih unit layer dalam neural network. Pada praktiknya, neurons di-drop dengan probabilitas p atau dipertahankan dengan probabilitas 1-p** 

<br>

**20. Convolutional Neural Networks**

&#10230; **20. Convolutional Neural Networks**

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; **21. Kebutuhan layer convolutional - W adalah ukuran volume input, F adalah ukuran dari layer neuron convolutional, P adalah jumlah zero padding, maka jumlah neurons N yang sesuai dengan ukuran dimensi masukan adalah:**

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; **22. Batch normalization - Adalah salah satu step hyperparameter γ,β yang menormalisasikan batch {xi}. Dengan mendefiniskan μB,σ2B sebagai nilai rata-rata dan variansi dari batch yang ingin kita normalisasi, hal tersebut dapat dilakukan dengan cara:** 

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; **23. Biasanya diaplikasikan setelah layer fully connected dan sebelum layer non-linear, yang bertujuan agar memungkinkannya penggunaan nilai learning rates yang lebih tinggi dan mengurangi ketergantungan pada nilai inisialisasi parameter neural network.**
 

<br>

**24. Recurrent Neural Networks**

&#10230; **24. Recurrent Neural Networks (RNN)**

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; **25. Jenis-jenis gates - Terdapat beberapa jenis gates dalam Recurrent Neural Network:**

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; **26. [Input gate (gerbang masuk), forget gate (gerbang lupa), gate, output gate (gerbang keluar)]**

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; **27, [Dituliskan ke dalm sel atau tidak?, Hapus sel atau tidak?, Berapa banyak yang harus ditulis ke dalam sel?, Berapa banyak yang dibutuhkan untuk mengungkap sel?]**

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; **28. LSTM (Long short-term memory) - LSTM layer adalah salahsatu model RNN yang dibuat untuk menyelesaikan masalah hilangnya gradien dengan menambahkan gerbang 'lupa'.**

<br>

**29. Reinforcement Learning and Control**

&#10230; **29, Reinforcement Learning dan Kontrol**

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; **30. Tujuan dari reinforcement learning adalah agar agen bisa membaur dan beradaptasi dengan lingkungannya.**

<br>

**31. Definitions**

&#10230;

<br> **31. Definisi**

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; **32. Markov decision processes (MDP) - Proses pengambilan keputusan Markov (MDP) adalah sebuah 5-tuple (S,A,{Psa},γ,R) dimana:** 

<br>

**33. S is the set of states** 

&#10230; **33. S adalah himpunan dari keadaan (states)**

<br>

**34. A is the set of actions**

&#10230; **34. A adalah himpunan dari aksi/tindakan**

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; **35. {Psa} merupakan probabilitas perubahan kejadian untuk s∈S dan a∈A** 

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; **36. γ∈[0,1[ merupakan faktor potongan]]**

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; **37. R:S×A⟶R atau R:S⟶R adalah fungsi penghargaan (reward) yang akan ditingkatkan nilainya oleh algoritma**

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; **38. Policy - Policy π adalah sebuah fungsi π:S⟶A yang memetakan keadaan (S) ke tindakan (A).**

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; **39. Catatan: Kita menjalankan sebuah policy π jika diberikan keadaan S, maka tindakan a = π (s)**

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; **40. Fungsi nilai - Diberikan sebuah policy π dan sebuah keadaan S, maka kita mendefinisikan nilai fungsi Vπ dengan sebagai berikut:**

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; **41. Persamaan Bellman - Persamaan Optimal Bellman mengkarakterisasi fungsi nilai Vπ∗ dari sebuah optimal policy π∗:** 

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; **42. Catatan: Nilai optimal policy dari π∗ untuk sebuah keadaan S adalah sebagai berikut:**

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; **43. Nilai perulangan algoritma - Nilai perulangan algoritma dibagai atas dua tahap:**

<br>

**44. 1) We initialize the value:**

&#10230; **44. 1) Menginisialisasi nilai:**

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; **45. 2) Melakukan iterasi berdasarkan nilai sebelumnya:**

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; **46. Estimasi kemungkinan maksimum ― Estimasi kemungkinan maksimum untuk probabilitas transisi keadaan S adalah sebagai berikut:**

<br>

**47. times took action a in state s and got to s′**

&#10230; **47. Waktu yang dibutuhkan aksi A dalam keadaan S untuk menuju keadaan S'**

<br>

**48. times took action a in state s**

&#10230; **48. Waktu yang dibutuhkan aksi A di keadaa S**

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; **49. Q-learning ― Q-learning adalah sebuah estimasi model bebas dari Q, yang didapat dari beikut:**

<br>

**50. View PDF version on GitHub**

&#10230; **50. Lihat versi PDF di GitHub**

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; **51 [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; **52, [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; **53. [Recurrent Neural Networks, Gates, LSTM]**

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; **54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**
