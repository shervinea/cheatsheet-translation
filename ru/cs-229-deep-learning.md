**Deep learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning)

<br>

**1. Deep Learning cheatsheet**

&#10230; Шпаргалка по глубокому обучению

<br>

**2. Neural Networks**

&#10230; Нейронные сети

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Нейронные сети - это класс моделей, построенных с использованием слоёв. Обычно используемые типы нейронных сетей включают сверточные и рекуррентные нейронные сети.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Архитектура ― Словарь архитектур нейронных сетей описан на рисунке ниже:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Входной слой, Скрытый слой, Выходной слой]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Отметив i-й слой сети и j-ю скрытую единицу слоя, мы имеем:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; где мы отмечаем w,b,z вес, смещение и выход соответственно.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; As a result, the weight is updated as follows:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Updating weights ― In a neural network, weights are updated as follows:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Step 1: Take a batch of training data.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Step 2: Perform forward propagation to obtain the corresponding loss.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Step 3: Backpropagate the loss to get the gradients.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Step 4: Use the gradients to update the weights of the network.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p

<br>

**20. Convolutional Neural Networks**

&#10230; Convolutional Neural Networks

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.

<br>

**24. Recurrent Neural Networks**

&#10230; Recurrent Neural Networks

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Input gate, forget gate, gate, output gate]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.

<br>

**29. Reinforcement Learning and Control**

&#10230; Reinforcement Learning and Control

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; The goal of reinforcement learning is for an agent to learn how to evolve in an environment.

<br>

**31. Definitions**

&#10230; Definitions

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:

<br>

**33. S is the set of states**

&#10230; S is the set of states

<br>

**34. A is the set of actions**

&#10230; A is the set of actions

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} are the state transition probabilities for s∈S and a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ is the discount factor

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Policy ― A policy π is a function π:S⟶A that maps states to actions.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Примечание: we say that we execute a given policy π if given a state s we take the action a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Примечание: we note that the optimal policy π∗ for a given state s is such that:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Value iteration algorithm ― The value iteration algorithm is in two steps:

<br>

**44. 1) We initialize the value:**

&#10230; 1) We initialize the value:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) We iterate the value based on the values before:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:

<br>

**47. times took action a in state s and got to s′**

&#10230; times took action a in state s and got to s′

<br>

**48. times took action a in state s**

&#10230; times took action a in state s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:

<br>

**50. View PDF version on GitHub**

&#10230; View PDF version on GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Convolutional Neural Networks, Convolutional layer, Batch normalization]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Recurrent Neural Networks, Gates, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]
