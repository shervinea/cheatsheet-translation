**1. Deep Learning cheatsheet**

&#10230; Derin Öğrenme El Kitabı
<br>

**2. Neural Networks**

&#10230; Sinir Ağları

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Sinir ağları, katmanlarla inşa edilen bir modeller sınıfıdır. Sinir ağlarının yaygın kullanılan çeşitleri evrişimsel sinir ağları ve yinelenen sinir ağlarını içerir.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Mimari ― Sinir ağları mimarisi aşağıdaki figürde açıklanmaktadır:  

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; Giriş katmanı, gizli katman, ürün katmanı

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Ağın i. sırasındaki katmana i ve katmandaki j. sırasındaki gizli birime j dersek, elimizde: 

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; burada w, b, z değerleri sırasıyla ağırlık, yönelim ve ürünü temsil eder.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Etkinleştirme fonksiyonu ― Etkinleştirme fonksiyonları gizli birimlerin sonunda, modele lineer olmayan karmaşıklıklar katmak için kullanılır. Aşağıda en yaygın kullanılanlarını görebilirsiniz: 

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Çapraz-entropi kaybı ― Sinir ağları içeriğinde, çapraz-entropi kaybı L(z,y) sık olarak kullanılır ve aşağıdaki gibi tanımlanır:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Öğrenme derecesi ― Öğrenme derecesi, sıklıkla α veya bazen η olarak belirtilir, ağırlıkların hangi tempoda güncellendiğini gösterir. Bu derece sabit olabilir veya uyarlamalı olarak değişebilir. Mevcut en gözde yöntem Adam olarak adlandırılan ve öğrenme oranını uyarlayan bir yöntemdir.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Geri yayılım ― Geri yayılım sinir ağındaki ağırlıkları güncellemek için kullanılan ve bunu yaparken de asıl sonuç ile istenilen sonucu hesaba katan bir yöntemdir. Ağırlık w değerine göre türev, zincir kuralı kullanılarak hesaplanılır ve aşağıdaki şekildedir: 

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Sonuç olarak, ağırlık güncellenmesi aşağıdaki gibidir: 

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Ağırlıkları güncelleme ― Sinir ağında ağırlıklar, aşağıdaki gibi güncellenir: 

<br>

**15. Step 1: Take a batch of training data.**

&#10230; 1. Adım: Bir grup eğitim verisi alınır

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 2.Adım: Denk gelen kaybı elde etmek için, ileri yayılım gerçekleştirilir. 

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 3.Adım: Gradyanları elde etmek için kayba geri yayılım uygulanır.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; 4.Adım: Ağın ağırlıklarını güncellemek için gradyanlar kullanılır.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Düşürme ― Düşürme, eğitim verisinin aşırı uymasını engellemek için sinir ağındaki birimleri düşürmek yoluyla uygulanan bir tekniktir. Pratikte, nöronlar ya p olasılığıyla düşürülür ya da 1-p olasılığıyla tutulur.

<br>

**20. Convolutional Neural Networks**

&#10230; Evrişimsel Sinir Ağları

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Evrişimsel katman gereksinimleri ― Girdi boyutuna W, evrişimsel katman nöronlarının boyutlarına F, sıfır dolgulama miktarına P dersek, 

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;

<br>

**24. Recurrent Neural Networks**

&#10230;

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230;

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;

<br>

**29. Reinforcement Learning and Control**

&#10230;

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;

<br>

**31. Definitions**

&#10230;

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;

<br>

**33. S is the set of states**

&#10230;

<br>

**34. A is the set of actions**

&#10230;

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230;

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;

<br>

**39. Remark: we say that we execute a given policy π if given a state a we take the action a=π(s).**

&#10230;

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;

<br>

**44. 1) We initialize the value:**

&#10230;

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;

<br>

**47. times took action a in state s and got to s′**

&#10230;

<br>

**48. times took action a in state s**

&#10230;

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
