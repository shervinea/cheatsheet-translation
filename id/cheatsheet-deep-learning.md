**1. Deep Learning cheatsheet**

&#10230;Deep Learning cheatsheet

<br>

**2. Neural Networks**

&#10230;Jaringan Saraf

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230;Jaringan Saraf adalah sebuah kelas model yang dibentuk dari beberapa lapisan (layer). Tipe-tipe jaringan ini meliputi neural network convolution dan recurrent.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230;Arsitektur - Kosa kata pada aritekture neural network dijelaskan pada gambar di bawah ini:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230;[Layer masukan, layer hidden (layer antara layer masukan dan keluaran), layer keluaran]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;Dengan menuliskan i sebagai layer ke-i dari network dan j sebagai hidden unit ke-j dari layer, kita mendapatkan

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230;dimana w, b, z adalah weight, bias, dan keluaran neural network.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230;Fungsi aktifasi- Fungsi aktifasi digunakan setelah sebuah hidden unit agar model dapat memodelkan permasalahan non-linear. Berikut adalah fungsi aktifasi yang sering digunakan:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;[Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;Fungsi loss cross-entropy - Pada neural network, loss cross-entropy L(z,y) adalah fungsi loss yang sering digunakan dan didefinisikan sebagai berikut:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;Learning rate- Learning rate, sering dituliskan sebagai α atau η, mendefinisikan seberapa cepat nilai weight diperbaharui. Learning rate bisa diset dengan nilai fix atau dirubah secara adaptif. Metode yang paling terkenal saat ini adalah Adam, sebuah method yang merubah learning rate secara adaptif.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;Backpropagation - Backpropagation adalah method untuk mengubah nilai weight pada neural network dengan mempertimbangkan perbedaan dari keluaran prediksi dan keluaran yang diinginkan. Turunan terhadap weight w dihitung menggunakan chain rule dan dapat diformulakan dengan:

<br>

**13. As a result, the weight is updated as follows:**

&#10230;Oleh karena itu, nilai weight diubah dengan formula sebagai berikut

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;Mengubah nilai weight - Pada neural network, nilai weight diubah dengan tahapan sebagai berikut:

<br>

**15. Step 1: Take a batch of training data.**

&#10230;Ambil sebuah batch (sample, contoh dari 100 training data ambil 50 data) dari training data.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;Lakukan perhitungan forward propagation dan hitung loss berdasarkan keluaran prediksi dan yang diinginkan.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230;Lakukan perhitungan backpropagate dengan nilai loss untuk mendapatkan nilai gradien.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230;Ubah nilai weight berdasarkan nilai gradien.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;Dropout - Dropout adalah sebuah teknik yang digunakan untuk mencegah overfitting pada saraf tiruan dengan men-drop out unit pada sebuah neural network. Pada pengaplikasiannya, neuron di drop dengan probabilitas p atau dipertahankan dengan probabilitas 1-p

<br>

**20. Convolutional Neural Networks**

&#10230;Convolutional Neural Network

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230;Ketentuan layer convolutional - Dengan mendefinisikan W sebagai ukuran volume (dimensi) dari masukan, F sebagai jumlah neuron (yang diinginkan) pada layer convolutional, P sebagai jumlah zero padding (penambahan nilai zero pada masukan), maka jumlah neuron N yang sesuai dengan ukuran dimensi masukan adalah:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalisasi batch - Normalisasi batch adalah sebuah langkah untuk menormalisasi batch {xi}. Dengan mendefinisikan μB,σ2B sebagai nilai rata-rata dan variansi dari batch yang ingin kita normalisasi, hal tersebut dapat dilakukan dengan cara:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;Batch normalisasi biasa ditempatkan setelah sebuah layer fully-connected atau convolutional dan sebelum sebuah non-linear layer yang bertujun untuk memungkinkannya penggunaan nilai learning rate yang lebih tinggi dan mengurangi ketergantungan model pada nilai inisialisasi parameter.

<br>

**24. Recurrent Neural Networks**

&#10230;Recurrent Neural Networks

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;Tipe-tipe gerbang - Dibawah ini merupakan beberapa jenis gerbang yang biasa ditemui pada recurrent neural network:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230;[Gerbang masukan, gerbang lupa (untuk melupakan informasi), gerbang, gerbang keluaran]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;Menulis ke sel atau tidak?, Hapus sebuah sel atau tidak?, Berapa banyak penulisan ke sel?, Berapa banyak yang disampaikan ke sel?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;LSTM - Long short-term memory (LSTM) network adalah sebuah tipe model dari RNN yang mencegah permasalahan vanishing gradien (nilai gradien menjadi 0)  dengan menambahkan gerbang 'lupa':

<br>

**29. Reinforcement Learning and Control**

&#10230;Reinforcement Learning dan Kontrol

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;Tujuan dari reinforcement learning adalah menjadikan sebuah agen (contoh: robot) dapat belajar untuk menyesuaikan diri terhadap lingkungan sekelilingnya.

<br>

**31. Definitions**

&#10230;Definisi-definisi

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;Markov decision processes- Markov decision process (MDP) adalah sebuah 5-tuple(S,A,{Psa},γ,R) dimana:

<br>

**33. S is the set of states**

&#10230;S adalah set dari state-state (tahap-tahap)

<br>

**34. A is the set of actions**

&#10230;A adalah set dari aksi-aksi

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;{Psa} adalah transisi probabilitas dari satu state ke state lainnya untuk s∈S and a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230;γ∈[0,1[ adalah faktor diskon

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;R:S×A⟶R or R:S⟶R adalah fungsi reward (hadiah) yang algoritma ingin untuk maksimalkan nilai keluarannya

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;Kebijakan - sebuah kebijakan adalah sebuah fungsi π:S⟶A yang memetakan state-state ke aksi-aksi.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230;Perlu diperhatikan: kita mengatakan bahwa kita mengeksekusi sebuah kebijakan π jika sebuah state s maka kita melakukan aksi a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;Fungsi value - Untuk sebuah kebijakan π dan sebuah state s, kita mendefinisikan fungsi value Vπ sebagai berikut:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;Bellman equation - Persamaan optimal Bellman menandakan fungsi value Vπ∗ dari kebijakan yang optimal π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;Perlu diperhatikan: kita mendefinisikan bahwa kebijakan optimal π∗ untuk state yang diberikan sebagai:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;Algoritma value iteration - Algoritma value iteration memiliki dua tahap:

<br>

**44. 1) We initialize the value:**

&#10230;Kita menginialisasi value

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;Kita melakukan iterasi value berdasarkan value sebelumnya

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;Estimasi maksimum likelihood - Estimasi maksimum likelihood untuk probabilitas transisi antara state-state didefinisikan sebagai berikut:

<br>

**47. times took action a in state s and got to s′**

&#10230;Jumlah melakukan aksi a pada state s dan menuju state s'

<br>

**48. times took action a in state s**

&#10230;Jumlah melakukan aksi a pada state s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;Q-learning - Q-learning adalah teknik estimasi tanpa menggunakan model dari Q, yang diformulasikan sebagai berikut:

<br>

**50. View PDF version on GitHub**

&#10230;Lihat versi PDF pada GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;[Neural Network, Arsitektur, Fungsi-fungsi Aktifasi, Bakcpropagation, Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;[Convolutional Neural Network, Layer Convolutional, Normalisasi Batch]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230;[Recurrent Neural Network, Gerbang-gerbang, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;[Reinforcement learning, Markov decision processes, Iterasi Value/policy, Approximate dynamic programming, Policy search]
