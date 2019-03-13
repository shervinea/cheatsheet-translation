**1. Deep Learning cheatsheet**

&#10230; Mély tanulás (Deep Learning) segédanyag

<br> 

**2. Neural Networks**

&#10230; Neurális hálózatok

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; A neurális hálózatok különböző rétegekből felépülő modellcsaládot jelentenek. A legelterjedtebb típusok: konvolúciós és rekurrens neurális hálózatok.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Architektúra ― A neurális hálózatok felépítésével kapcsolatos legfőbb elnevezések az alábbi ábrán láthatóak:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Bemeneti réteg, rejtett réteg, kimeneti réteg]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Ha i-vel jelöljük az i-edik réteget és j-vel a réteg j-edik rejtett egységét, akkor:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; ahol w jelöli a súlyvektort, b az eltolásvektort (bias) és z a kimeneti vektort.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Aktivációs függvény ― Az aktivációs függvényeket a rejtett egységek végén használjuk, így elérve, hogy a modell nemlineáris függvényeket is tudjon approximálni. Az alábbi táblázat tartalmazza a leggyakoribbakat:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Szigmoid (logisztikus függvény), Tangens hiperbolikus, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Kereszt-entrópia veszteségfüggvény ― A neurális hálózatok kontextusában gyakran használatos a kereszt-entrópia veszteségfüggvény (jel.: L(z,y)), melyet így definiálunk:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Tanulási faktor ― A tanulási faktor (jel.: α vagy néha η) azt jelöli, hogy milyen ütemű a súlyfrissítés. Ez történhet rögzített vagy adaptív módon. A jelenleg legnépszerűbb optimalizációs módszer az Adam, mely adaptívan frissíti a súlyokat.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Hibavisszaterjesztés ― A hibavisszaterjesztési (backpropagation) algoritmussal a neurális hálózatbeli súlyokat frissítjük; figyelembe véve a kiszámolt és a tényleges output közti eltérést. A veszteségfüggvény súlyokra vonatkozó deriváltját a láncszabály segítségével számolhatjuk ki, az alábbi formula alapján:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Eszerint a súlyok frissítési szabálya:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; A tanítás folyamata ― Neurális hálózatokban a súlyokat az alábbi lépések alapján frissítjük:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; 1. lépés: Vegyünk egy kötegnyi tanítóadatot.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 2. lépés: Hierarchikusan terjesszük végig a bemeneteket a hálózaton, így kapjuk a megfelelő veszteséget.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 3. lépés: Terjesszük vissza a hibát ― eközben számítjuk a gradienseket.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; 4. lépés: A gradienseket felhasználva frissítsük a hálózatbeli súlyokat.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Kiejtés (dropout) ― A kiejtéses regularizáció technikáját a túlillesztés elkerülése érdekében alkalmazhatjuk. Ennek folyamán a neurális hálózat véletlenszerűen kiválasztott egységeit kiejtjük a tanítás folyamatából. Ez a gyakorlatban azt jelenti, hogy a neuronokat p valószínűséggel kiejtjük (azaz 1-p valószínűséggel megtartjuk).

<br>

**20. Convolutional Neural Networks**

&#10230; Konvolúciós neurális hálózatok

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Feltétel a konvolúciós rétegre ― Jelölje W az inputméretet, F a konvolúciós réteg neuronjainak méretét, P a nullákkal való feltöltés (zero padding) mértékét. Ekkor az illesztő neuronok számára (jel.: N) az alábbi összefüggés adódik:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Kötegnormalizálás ― Segítségével az {xi} köteget tudjuk normalizálni. Az alábbi képletben μB, illetve σ2B jelöli a köteg várható értékét, illetve a szórásnégyzetét:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Leginkább a teljesen kapcsolt/konvolúciós réteg után és a nemlineáris réteg előtt alkalmazzuk. Célja, hogy magasabb tanulási faktort tudjunk alkalmazni, és kevésbé függjön a tanítás az inicializációtól.

<br>

**24. Recurrent Neural Networks**

&#10230; Rekurrens neurális hálózatok (RNN)

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Kapuk típusai ― Az alábbiakban olyan különféle kaputípusokat mutatunk be, melyekkel egy tipikus rekurrens neurális hálózatban találkozhatunk:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Bemeneti kapu, felejtő kapu, kapu, kimeneti kapu]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Írjunk a cellába?, Töröljük a cellát?, Mennyit írjunk a cellába?, Mennyire fedjük fel a cellát?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― Rövid-hosszútávú memória (LSTM) hálózatok az RNN-típusú modellek közé tartoznak. A "felejtő" kapu hozzáadásával próbálják kiküszöbölni az eltűnő gradiens problémáját.

<br>

**29. Reinforcement Learning and Control**

&#10230; Megerősítéses tanulás és kontroll

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; A megerősítéses tanulás célja, hogy egy ágens megtanulja, hogyan fejlődjön egy környezetben.

<br>

**31. Definitions**

&#10230; Definíciók

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Markov döntési folyamatok ― A Markov döntési folyamat (MDF) olyan (S,A,{Psa},γ,R) ötös, melyre:

<br>

**33. S is the set of states**

&#10230; S az állapothalmaz

<br>

**34. A is the set of actions**

&#10230; A a lépések halmaza

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} jelöli az állapotátmenetek valószínűségeit, ahol s∈S és a∈A.

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ a diszkont faktor

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R vagy R:S⟶R a jutalmazó függvény, amelyet az algoritmus maximalizálni kíván

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Eljárás (policy) ― Eljárásnak nevezzük a π:S⟶A függvényeket.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Megjegyzés: azt mondjuk, hogy végrehajtunk egy adott eljárást, ha egy adott s állapot esetén az a=π(s) lépést választjuk.

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Értékfüggvény ― Adott π eljárás és s állapot. Ekkor definiáljuk a Vπ értékfüggvényt az alábbi módon:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Bellman-egyenlet ― A Bellman-egyenleteket határozzák meg az optimális π∗ eljárás Vπ∗ értékfüggvényét:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Megjegyzés: egy adott s állapot esetén π∗-gal jelöljük az optimális eljárásmódot, és az alábbi képlettel számolhatjuk ki: 

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Értékiteráció algortimusa ― Az értékiterációs algoritmus két lépésből áll:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Inicializáljuk az értéket:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) A korábbi értékek alapján iteráljuk az értéket:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Maximum likelihood becslés ― Az állapotátmenetek valószínűsűgének maximum likelihood becslése az alábbi alapján számítható:

<br>

**47. times took action a in state s and got to s′**

&#10230; ennyiszer választottuk az a lépést s állapotban és kerültünk s′ állapotba.

<br>

**48. times took action a in state s**

&#10230; ennyiszer választottuk az a lépést az s állapotban.

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-tanulás ― A Q-tanulás a Q minőségfüggvény egy modellmentes becslése:

<br>

**50. View PDF version on GitHub**

&#10230; Tekintsd meg a PDF-verziót GitHubon!

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Neurális hálózatok, Architektúra, Aktivációs függvény, Hibavisszaterjesztés, Kiejtés]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Konvolúciós neurális hálózatok, Konvolúciós réteg, Kötegnormalizálás]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Rekurrens neurális hálózatok, Kapuk, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Megerősítéses tanulás, Markov döntési folyamatok, Értékiteráció, Approximate dynamic programming, Eljáráskeresés]
