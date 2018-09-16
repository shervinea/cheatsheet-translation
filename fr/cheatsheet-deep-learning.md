**1. Deep Learning cheatsheet**

&#10230; Pense-bête de Deep Learning

<br>

**2. Neural Networks**

&#10230; Réseau de neurones

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Les réseaux de neurones (en anglais *neural networks*) sont une classe de modèles qui sont construits à l'aide de couches de neurones. Les réseaux de neurones convolutionnels (en anglais *convolutional neural networks*) ainsi que les réseaux de neurones récurrents (en anglais *recurrent neural networks*) font parti des principaux types de réseaux de neurones.

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

&#10230; où l'on note w, b, z le coefficient, le biais ainsi que la variable sortie respectivement.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Fonction d'activation ― Les fonctions d'activation sont utilisées à la fin d'une unité de couche cachée pour introduire des complexités non linéaires au modèle. En voici les plus fréquentes :

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoïde, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Dans le contexte des réseaux de neurones, la fonction objectif de cross-entropie L(z,y) est communément utilisée et est définie de la manière suivante : 

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Taux d'apprentissage ― Le taux d'apprentissage (appelé en anglais *learning rate*), souvent noté α ou parfois η, indique la vitesse à laquelle les coefficients évoluent. Cette quantité peut être fixe ou variable. L'une des méthodes les plus populaires à l'heure actuelle s'appelle Adam, qui a un taux d'apprentissage qui s'adapte au file du temps.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Rétropropagation du gradient ― La rétropropagation du gradient (en anglais *backpropagation*) est une méthode destinée à mettre à jour les coefficients d'un réseau de neurones en comparant la sortie obtenue et la sortie désirée. La dérivée par rapport au coefficient w est calculée à l'aide du théorème de dérivation des fonctions composées, et s'écrit de la manière suivante :

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

&#10230; Dropout ― Le dropout est une technique qui est destinée à empêcher le sur-ajustement sur les données de training en abandonnant des unités dans un réseau de neurones. En pratique, les neurones sont soit abandonnés avec une probabilité p ou gardés avec une probabilité 1-p 

<br>

**20. Convolutional Neural Networks**

&#10230; Réseaux de neurones convolutionels (en anglais *Convolutional Neural Networks*, *CNN*)

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Pré-requis de la couche convolutionelle ― Si l'on note W la taille du volume d'entrée, F la taille de la couche de neurones convolutionelle, P la quantité de zero padding, alors le nombre de neurones N qui tient dans un volume donné est tel que :

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalisation de batch ― C'est une étape possédant les paramètres γ,β qui normalise le batch {xi}. En notant μB,σ2B la moyenne et la variance de ce que l'on veut corriger au batch, ceci est fait de la manière suivante :

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Cela est normalement effectué après une couche fully-connected/couche convolutionelle et avant une couche de non-linéarité et a pour but de permettre un taux d'apprentissage plus grand et de réduire une dépendance trop forte à l'initialisation.

<br>

**24. Recurrent Neural Networks**

&#10230; Réseaux de neurones récurrents (en anglais *Recurrent Neural Networks*, *RNN*)

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Types de porte ― Voici les différents types de porte que l'on rencontre dans un réseau de neurones récurrent typique :

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Porte d'entrée, porte d'oubli, porte, porte de sortie]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Écrire dans la cellule ou non ?, Supprimer la cellule ou non ? Combien écrire à la cellule ? A quel point révéler la cellule ?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― Un réseau de long court terme (en anglais *long sort-term memory*, *LSTM*) est un type de modèle RNN qui empêche le phénomène de *vanishing gradient* en ajoutant des portes d'oubli.

<br>

**29. Reinforcement Learning and Control**

&#10230; Reinforcement Learning et Control

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; Le but du reinforcement learning est pour un agent d'apprendre comment évoluer dans un environnement.

<br>

**31. Definitions**

&#10230; Définitions

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Processus de décision markovien ― Un processus de décision markovien (MDP) est décrite par 5 quantités (S,A,{Psa},γ,R), où :

<br>

**33. S is the set of states**

&#10230; S est l'ensemble des états

<br>

**34. A is the set of actions**

&#10230; A est l'ensemble des actions

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} sont les probabilités d'états de transition pour s∈S et a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ est le taux d'actualisation (en anglais *discount factor*)

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R ou R:S⟶R est la fonction de récompense que l'algorithme veut maximiser

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Politique ― Une politique π est une fonction π:S⟶A qui lie les états aux actions.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Remarque : on dit que l'on effectue une politique donnée π si étant donné un état s, on prend l'action a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Fonction de valeurs ― Pour une politique donnée π et un état donné s, on définit la fonction de valeurs Vπ comme suit :

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Équation de Bellman ― Les équations de Bellman optimales caractérisent la fonction de valeurs Vπ∗ de la politique optimale π∗ :

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Remarque : on note que la politique optimale π∗ pour un état donné s est tel que :

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algorithme d'itération sur la valeur ― L'algorithme d'itération sur la valeur est faite de deux étapes :

<br>

**44. 1) We initialize the value:**

&#10230; 1) On initialise la valeur :

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) On itère la valeur en se basant sur les valeurs précédentes :

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Maximum de vraisemblance ― Les estimations du maximum de vraisemblance pour les transitions de probabilité d'état sont comme suit :

<br>

**47. times took action a in state s and got to s′**

&#10230; nombre de fois où l'action a dans l'état s est prise pour arriver à l'état s'

<br>

**48. times took action a in state s**

&#10230; nombre de fois où l'action a dans l'état s est prise

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning ― Le Q-learning est une estimation non-paramétrique de Q, qui est faite de la manière suivante :
