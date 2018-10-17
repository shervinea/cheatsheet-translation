**1. Deep Learning cheatsheet**

&#10230; Formulario di apprendimento profondo

<br>

**2. Neural Networks**

&#10230; Reti Neurali

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Le reti neurali sono una classe di modelli composte da vari livelli. Le reti neurali convoluzionali e le reti neurali ricorrenti sono tipi di reti neurali usati comunemente.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Architettura ― Il vocabolario riguardante le architetture delle reti neurali è illustrato nella figura sottostante:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Livello di ingresso, livello nascosto, livello di uscita]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Dato i i-esimo livello della rete e j j-esima unità nascosta del livello, abbiamo:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; dove indichiamo con w, b, z rispettivamente il peso, il bias e l'uscita.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Funzione d'attivazione ― Le funzioni di attivazione sono usate alla fine di una unità nascosta per introdurre complessità non lineare al modello. Di seguito le più comuni:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoide, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;  Funzione di perdita entropia incrociata ― In ambito reti neurali, l'entropia incrociata L(z,y) è comunemente usata ed è definita come segue:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Tasso di apprendimento ― Il tasso di apprendimento, spesso chiamato α o a volte η, indica con quale passo sono aggiornati i pesi. Questo può essere fisso o cambiato in maniera adattiva. Il metodo attualmente più popolare si chiama Adam, un metodo che adatta il tasso di apprendimento.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Retropropagazione dell'errore ― La retropropagazione dell'errore è un metodo per aggiornare i pesi della rete neurale tenendo in considerazione l'uscita effettiva e l'uscita desiderata. La derivata rispetto al peso w è calcolata utilizzando la regola della catena ed è del seguente tipo:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Di conseguenza, il peso è aggiornato come segue:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Aggiornare i pesi ― In una rete neurale, i pesi sono aggiornati nel seguente modo:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Passo 1: Prendere un gruppo di dati per l'apprendimento.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Passo 2: Effettuare la propagazione in avanti per ottenere il corrispondente valore della funzione di perdita.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Passo 3: Effettuare la retropropagazione per ottenere i gradienti.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Passo 4: Utilizzare i gradienti per aggiornare i pesi della rete.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout ― Il dropout è una tecnica per prevenire l'eccessivo adattamento ai dati di allenamento attraverso la disattivazione di alcune unità in una rete neurale. In pratica, i neuroni sono o disattivati con probabilità p o mantenuti attivi con probabilità 1-p

<br>

**20. Convolutional Neural Networks**

&#10230; Reti neurali convoluzionali

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Requisito per un livello convoluzionale ― Chiamando W la dimensione in ingresso, F la dimensione dei neuroni del livello convoluzionale, P la quantità di margine nullo, allora il numero di neuroni N che entra in un certo volume è tale che:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalizzazione di batch ― È un passo di iperparametri γ,β che normalizza il batch {xi}. Denotando μB,σ2B la media e la varianza di ciò che vogliamo correggere nel batch, è eseguita nel seguente modo:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; È solitamente utilizzata dopo un livello completamente connesso o convoluzionale e prima di una non linearità e mira a consentire maggiori tassi di apprendimento e a ridurre la forte dipendenza dall'inizializzazione.

<br>

**24. Recurrent Neural Networks**

&#10230; Reti Neurali Ricorrenti

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Tipi di porte ― Seguono i diversi tipi di porte che vediamo in una tipica rete neurale ricorrent:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Porta d'ingresso, porta di cancellatura, porta, porta di uscita]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Scrivere la cellula o no?, Cancellare la cellula o no? Quanto scrivere sulla cellula? Quanto rivelare la cellula?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― Una memoria a breve-lungo termine è un tipo di modello ricorrente che impedisce il fenomeno della scomparsa del gradiente aggiungendo porte di cancellatura.

<br>

**29. Reinforcement Learning and Control**

&#10230; Apprendimento per rinfornzo e Controllo

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; L'obiettivo dell'apprendimento per rinfornzo è che un agente apprenda come evolvere in un ambiente.

<br>

**31. Definitions**

&#10230; Definizioni

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Processo decisionale di Markov ― Un processo decisionale di Markov (MDP) è descritto da una n-upla di 5 elementi (S,A,{Psa},γ,R), dove:

<br>

**33. S is the set of states**

&#10230; S è l'insieme degli stati

<br>

**34. A is the set of actions**

&#10230; A è l'insieme delle azioni

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} sono le probabilità di transizione per s∈S e a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ è il fattore di sconto

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R o R:S⟶R è la funzione di ricompensa che l'algoritmo vuole massimizzare

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Politica ― Una politica π è una funzione π:S⟶A che mappa gli stati alle azioni.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Osservazione: diciamo di seguire una politica π se, dato uno stato s, eseguiamo l'azione a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Funzione di valore ― Per una data politica π e un dato stato s, definiamo la funzione di valore Vπ nel seguente modo:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Equazione di Bellman ― Le equazioni di ottimalità di Bellman caratterizzano la funzione di valore Vπ∗ della politica ottima π∗ :

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Osservazione: notiamo che la politica ottima π∗ per un dato stato s è tale che:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algoritmo di iterazione dei valori ― L'algoritmo di iterazione dei valori avviene in due passi:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Inizializziamo i valori:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) Iteriamo i valori a partire dai valori precedenti:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Stima di massima verosimiglianza ― Le stime di massima verosimiglianza per le transizioni di stato sono le seguenti:

<br>

**47. times took action a in state s and got to s′**

&#10230; volte in cui è stata eseguita l'azione a nello stato s e si è arrivati allo stato s'

<br>

**48. times took action a in state s**

&#10230; volte in cui è stata eseguita l'azione a nello stato s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning ― Il Q-learning è una stima senza modello dell'ambiente di Q, effettuata nel seguente modo:
