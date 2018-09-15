**1. Deep Learning cheatsheet**

&#10230; Deep Learning - ściąga

<br>

**2. Neural Networks**

&#10230; Sieci neuronowe

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Sieci neuronowe to klasa modeli zbudowanych z warstw. Często wykorzystywane rodzaje sieci neuronowych to splotowe i rekurencyjne sieci neuronowe.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Architektura - słownictwo związane z sieciami neuronowymi jest opisane poniżej:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Warstwa wejściowa, warstwa ukryta, warstwa wyjściowa]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Przez i rozumiemy i-tą warstwę sieci a przez j, j-ty neuron warstwy, mamy więc:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; gdzie w to wagi (współczynniki), b to wyraz wolny funkcji i z to wynik.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Funkcja aktywacji - Funkcje aktywacji stosowane są po wyliczeniu warstwy ukrytej w celu wprowadzenia nieliniowości do modelu. Oto najczęściej stosowane:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Koszt logarytmiczny (Cross-entropy loss) ― W kontekście sieci neuronowych koszt logarytmiczny L(z,y) jest często stosowany i wygląda następująco:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Współczynnik uczenia ― Współczynnik uczenia, często zapisywany jako α lub rzadziej η, określa z jaką szybkością będą aktualizowane wagi. Może on mieć wartość stałą lub zmienną. Obecnie najpopularniejszą metodą optymalizacji funkcji kosztu jest metoda Adam, która dostosowuje wartość współczynnika uczenia.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Propagacja wsteczna ― Propagacja wsteczna jest metodą aktualizacji wag w sieci neuronowej, która bierze pod uwagę różnice pomiędzy wynikiem uzyskanym, a oczekiwanym (koszt). Pochodna cząstkowa względem wagi w jest liczona z wykorzystaniem zasady złożenia pochodnych funkcji i wygląda następująco:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; W wyniku czego, wagi są aktualizowane w następujący sposób:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Aktualizacja wag ― W sieci neuronowej, wagi są aktualizowane w następujący sposób: 

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Krok 1: Pobierz pakiet danych treningowych.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Krok 2: dokonaj propagacji do przodu aby uzyskać wartość kosztu.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Step 3: Z wykorzystaniem propagacji wstecznej użyj koszt aby uzyskać gradient.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Krok 4: Wykorzystaj gradient aby zaktualizować wagi w sieci neuronowej.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout ― Dropout jest techniką zapobiegania nadmiernemu dopasowaniu (overfitting) do danych treningowych poprzez pomijanie niektórych neuronów w sieci. W praktyce, neurony są pomijane z prawdopodobieństwem p lub nie są pomijane z prawdopodobieństwem 1-p

<br>

**20. Convolutional Neural Networks**

&#10230; Konwolucyjne Sieci Neuronowe

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Wymagania warstwy konwolucyjnej ― Zauwżając, że W to rozmiar danych wejściowych, F to rozmiar neuronów warstwy konwolucyjnej, P rozmiar uzupełnienia zerami, to wymaganą ilość neuronów określamy następująco:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalizacja pakietu (Batch normalization) ― Jest to krok w którym hiperparametry γ,β są wykorzystywane do normalizacji pakietu {xi}. Zauważając, że μB to średnia, a σ2B to wariancja, to normalizacja pakiet wygląda następująca:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Jest ona zazwyczaj stosowana po warstwie pełnej lub konwolucyjnej, a przed zastosowaniem nieliniowej funkcji aktywacyjnej i ma na celu umożliwienie stosowania dużego współczynnika uczenia i zmniejszenia zależności od inicjalizacji.

<br>

**24. Recurrent Neural Networks**

&#10230; Rekurencyjne Sieci Neuronowe

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Rodzaje bramek ― Przedstawiamy różne rodzaje bramek, które możemy spotkać w typowych sieciach rekurencyjnych (RNN):

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Bramka wejściowa, bramka zapominajca, bramka, bramka wyjściowa]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Pisać do komórki, czy nie?, Wyczyścić komówke, czy nie?, Jak dużo zapisać do komórki?, Jak dużo ujawnić komórce?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― Długa krótkoterminowa sieć neuronowa (LSTM) to rodzaj sieci rekurencyjnej (RNN), która radzi sobie z problemem zanikającego gradientu poprzez wykorzystanie bramek zapominających.

<br>

**29. Reinforcement Learning and Control**

&#10230; Uczenie Wspomagane i Kontrola

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; Celem uczenia wspomaganego jest nauczenie agenta tego, w jaki sposób ewoluować w danym środowisku.

<br>

**31. Definitions**

&#10230; Definicje:

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Proces decyzyjny Markowa ― Proces decyzyjny markowa (MDP) jest 5-krotką (S,A,{Psa},γ,R), gdzie: 

<br>

**33. S is the set of states**

&#10230;

<br> S jest zbiorem stanów

**34. A is the set of actions**

&#10230; A jest zbiorem działań

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} to zbiór prawdopodobieństw przejść pomiędzy stanami gdzie s∈S i a∈A 

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ jest współczynnikiem dyskontującym. 

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R lub R:S⟶R to funkcja nagrody, którą algorytm ma za zadanie zmaksymalizować.

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Strategia - Strategia π jest funkcją π:S⟶A, która mapuje stany na działania.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Przypomnienie: mówimy, że wykonujemy daną strategię π w danym stanie s, gdy wykonujemy działanie a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Funkcja wartości ― Dla danej strategii π w danym stanie s, definiujemy wartość funkcji Vπ w następujący sposób:    

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Równanie Bellmana - Optymalne równania Bellmana charakteryzują wartość funkcji Vπ∗ optymalnej strategii π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Przypomnienie: zauważmy, że optymalna strategia π∗ dla danego stanu s jest taka, że:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algorytm iteracyjnego ustalania wartości zmiennej - algorytm ten składa się z dwóch kroków:

<br>

**44. 1) We initialize the value:**

&#10230; Inicjalizujemy zmienną wartością:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; W iteracyjny sposób ustalamy wartość zmiennej w oparciu o wartość poprzedniej zmiennej:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Szacowanie maksymalnego prawdopodobieństwa - Szacowanie maksymalnego prawdopodobieństwo dla poszczególnych przejść pomiędzy stanami wygląda następująco:

<br>

**47. times took action a in state s and got to s′**

&#10230; ile razy podjęto działanie a w stanie s i otrzymano stan s'

<br>

**48. times took action a in state s**

&#10230; ile razu podjęto działanie a w stanie s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning ― Q-learning jest bezmodelowym sposobem estymowania Q, który wygląda następująco: 
