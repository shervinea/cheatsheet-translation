**1. Deep Learning cheatsheet**

&#10230; Pense-bête de Deep Learning

<br>

**2. Neural Networks**

&#10230; Réseau de neurones

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Les réseaux de neurones sont une classe de modèles qui sont construits à l'aide de couches de neurones. Les réseaux de neurones convolutionnels ainsi que les réseaux de neurones récurrents font parti des principaux types de réseaux de neurones.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Architecture ― Le vocabulaire autour des architectures des réseaux de neurones est décrit dans la figure ci-dessous:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Couche d'entrée, couche cachée, couche de sortie]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; En notant i la ième couche du réseau et j la jième unité de couche cachée, on a: 

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; où l'on note w, b, z le coefficient, biais et la sortie respectivement.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Les fonctions d'activation sont utilisées à la fin d'une unité de couche cachée pour introduire des complexités non-linéaires au modèle. Voici les plus fréquentes :

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoïde, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Dans le contexte des réseaux de neurones, la fonction objectif de cross-entropie L(z,y) est communément utilisée et est définie de la manière suivante : 

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Taux d'apprentissage ― Le taux d'apprentissage, souvent noté α ou parfois η, indique à quelle vitesse les coefficients se font actualiser, qui peut être une quantité fixe ou variable. L'une des méthodes les plus populaires actuelles est Adam, qui a un taux d'apprentissage adaptatif.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Rétropropagation du gradient ― La rétropropagation du gradient est une méthode destinée à actualiser les coefficients d'un réseau de neurones en prenant en compte la sortie obtenue et la sortie désirée. La dérivée par rapport au coefficient w est calculée à l'aide de la règle de la dérivation de la chaîne et est de la forme suivante :

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Ainsi, le coefficient est actualisé de la manière suivante :

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Actualiser les coefficients ― Dans un réseau de neurones, les coefficients sont actualisés comme suit :

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Étape 1 : Prendre un groupe d'observations appartenant au données du training set.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Étape 2 : Réaliser la propagation avant pour obtenir le loss correspondant.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Étape 3 : Effectuer une rétropropagation du loss pour obtenir les gradients.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Étape 4 : Utiliser les gradients pour actualiser les coefficients du réseau.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout ― Le dropout est une technique qui est destinée à empêcher à overfitter le training data en abandonnant des unités dans un réseau de neurones. En pratique, les neurones sont soit abandonnés avec une probabilité p ou gardés avec une probabilité 1-p 

<br>

**20. Convolutional Neural Networks**

&#10230; Convolutional Neural Networks

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Pré-requis de la couche convolutionelle ― Si on note W la taille du volume d'entrée, F la taille de la couche de neurones convolutionelle, P la quantité de zero padding, alors le nombre de neurones N qui tient dans un volume donné est tel que :

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalisation de batch ― C'est une étape d'hyperparamètre γ,β qui normalise le batch {xi}. En notant μB,σ2B la moyenne et la variance de ce que l'on veut corriger au batch, ceci est fait de la manière suivante :

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; C'est normalement effectué après un fully connected/couche convolutionelle et avant une couche de non-linéarité et a pour but de permettre un taux d'apprentissage plus grand et réduire une dépendance trop forte de l'initialisation.

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
