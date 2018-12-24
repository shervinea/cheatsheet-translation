**1. Deep Learning cheatsheet**

&#10230; Deep Learning Spickzettel

<br>

**2. Neural Networks**

&#10230; Neuronale Netze

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Neuronale Netze sind eine Klasse von Modellen die in Schichten aufgebaut sind. Gängige Typen neuronaler Netze sind unter anderem faltende und rekurrente neuronale Netze.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Architektur - Das Vokabular rund um Architekturen neuronaler Netze ist in folgender Abbildung beschrieben: 

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Eingangsschicht, versteckte Schicht, Ausgangsschicht]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Sei i die i-te Schicht des Netzes und j die j-te versteckte Einheit der Schicht, so ist:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; wobei w, b und z jeweils Gewichtung, Vorspannung und Ergebnis bezeichnen.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Aktivierungsfunktion - Aktivierungsfunktionen werden am Ende einer versteckten Schicht benutzt um nicht-lineare Verflechtungen in das Model einzubringen. Hier sind die die gebräuchlisten:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, undichte ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Kreuzentropieverlust - Im Kontext neuronaler Netze wird der Kreuzentropieverlust L(z,y) gebräuchlicherweise benutzt und definiert wie folgt:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Lernrate - Die Lernrate, oft mit α oder manchmal mit η bezeichnet, gibt an mit welcher Schnelligkeit die Gewichtungen aktualisiert werden. Die Lernrate kann konstant oder anpassend variierend sein. Die aktuell populärste Methode, Adam, ist eine Methode die die Lernrate anpasst.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Fehlerrückführung - Fehlerrückführung aktualisiert die Gewichtungen in neuronalen Netzen durch Einberechnung der tatsächlichen und der gewünschten Ausgabe. Die Ableitung nach der Gewichtung w wird mit Hilfe der Kettenregel berechnet und hat die folgende Form:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Im Ergebnis wird das Gewicht wie folgt aktualisiert:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Das Aktualisieren der Gewichtungen - In neuronalen Netzen werden die Gewichtungen wie folgt aktualisiert:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Schritt 1: Nimm einen Stapel von Lerndaten.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Schritt 2: Führe Vorwärtsausbreitung durch um den entsprechenden Verlust zu erhalten.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Schritt 3: Fehlerrückführe den Verlust um die Gradienten zu erhalten.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Schritt 4: Verwende die Gradienten um die Gewichtungen im Netz zu aktualisieren.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Aussetzen - Aussetzen ist eine Technik um eine Überanpassung der Lerndaten zu verhindern bei der Einheiten in einem neuronalen Netz ausfallen. In der Praxis werden Neuronen entweder mit Wahrscheinlichkeit p ausgesetzt oder mit Wahrscheinlichkeit 1-p behalten.

<br>

**20. Convolutional Neural Networks**

&#10230; Faltende neuronale Netzwerke

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Vorraussetzung für eine faltende Schicht - Sei W das Eingangsvolumen, f die Größe der Neuronen der faltenden Schicht, P die Anzahl der aufgefüllten Nullen, dann ist die Anzahl der Neuronen N die in ein gegebenes Volumen passen:  

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Bündelnormalisierung - Ein Schritt des Hyperparameters γ,β welcher das Bündel {xi} normalisiert. Es seien μB der Mittelwert und σ2B die Varianz dessen um was das Bündel korrigiert werden soll, dann gilt folgendes:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Geschieht üblicherweise nach einer vollständig verbundenen/faltenden Schicht und vor einer nicht-linearen Schicht und bezweckt eine höhere Lernrate und eine Reduzierung der starken Abhängigkeit von der Initialisierung.

<br>

**24. Recurrent Neural Networks**

&#10230; Rekurrente neuronale Netze

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Typen von Gattern - Dies sind die verschiedenen Typen der Gatter die wir in typischen rekurrenten neuronalen Netzen vorfinden:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Eingangsgatter, Vergißgatter, Gatter, Ausgangsgatter]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Zelle beschreiben oder nicht?, Zelle löschen oder nicht?, Wieviel in die Zelle schreiben?, Wieviel um die Zelle aufzudecken?]

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

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

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

<br>

**50. View PDF version on GitHub**

&#10230;

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230;

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
