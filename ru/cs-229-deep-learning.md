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

&#10230; Функция активации - используются в конце скрытого блока, чтобы внести в модель нелинейность. Вот самые распространенные:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Сигмоида, Tanh, ReLU, ReLU с утечкой]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Функция потерь на основе перекрестной энтропии ― в контексте нейронных сетей обычно используются потери кросс-энтропии L(z,y), которые определяются следующим образом:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Скорость обучения - часто обозначаемая как α или иногда η, указывает, с какой скоростью обновляются веса. Её можно исправить или адаптивно изменить. Самый популярный в настоящее время метод называется Adam (адаптивные моменты), он адаптирует скорость обучения.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Обратное распространение ― это метод обновления весов в нейронной сети с учетом фактического и желаемого результата. Производная по весу w вычисляется с использованием цепного правила и имеет следующий вид:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; В результате вес обновляется следующим образом:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Обновление весов ― В нейронной сети веса обновляются следующим образом:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Шаг 1. Возьмите пакет обучающих данных.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Шаг 2: Выполните прямое распространение, чтобы получить соответствующие значения функции стоимости.


<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Шаг 3: Выполните обратное распространение ошибки, чтобы получить градиенты.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Шаг 4. Используйте градиенты, чтобы обновить веса сети.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Прореживание (Dropout) ― это метод, предназначенный для предотвращения переобучения на обучающих данных путем исключения блоков в нейронной сети. На практике нейроны либо отбрасываются с вероятностью p, либо сохраняются с вероятностью 1−p

<br>

**20. Convolutional Neural Networks**

&#10230; Сверточные нейронные сети

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Требования к сверточному слою ― обозначим W - размер входного объема, F - размер нейронов сверточного слоя, P - величину дополнения нулями, тогда количество нейронов N, которые помещаются в данный объем, будет таким:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Пакетная нормировка ― метод адаптивной перепараметризации γ,β, который нормирует пакет {xi}. Обозначим μB,σ2B как среднее значение и дисперсию, которые мы хотим скорректировать для пакета, это делается следующим образом:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Обычно это делается после полносвязного / сверточного слоя и до функции нелинейности и направлено на повышение скорости обучения и уменьшение сильной зависимости от инициализации.

<br>

**24. Recurrent Neural Networks**

&#10230; Рекуррентные нейронные сети

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Типы вентилей ― Вот различные типы вентилей, с которыми мы сталкиваемся в типичной рекуррентной нейронной сети:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Входной вентиль, Вентиль забывания, Вентиль обновления, Вентиль выхода]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Писать в ячейку или нет?, Стереть ячейку или нет?, Сколько писать в ячейку?, Насколько раскрыть ячейку?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM ― Сеть с долгой кратковременной памятью (LSTM) - это тип модели RNN, которая позволяет избежать проблемы исчезающего градиента, добавляя вентиль «забывания».

<br>

**29. Reinforcement Learning and Control**

&#10230; Обучение с подкреплением и контроль

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; Цель обучения с подкреплением - научить агента развиваться в окружающей среде.

<br>

**31. Definitions**

&#10230; Определения

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Марковские процессы принятия решений - Марковский процесс принятия решений (MDP) представляет собой кортеж из 5 составляющих (S,A,{Psa},γ,R), где :

<br>

**33. S is the set of states**

&#10230; S - множество всех состояний

<br>

**34. A is the set of actions**

&#10230; A - множество всех действий

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} - вероятности перехода состояний для s∈S и a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ - коэффициент дисконтирования

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R или R:S⟶R - функция вознаграждения, которую алгоритм хочет максимизировать

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Политика ― Политика π - это функция π:S⟶A, которая отображает состояния в действия.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Примечание: мы говорим, что выполняем данную политику π, если для данного состояния s мы предпринимаем действие a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Функция ценности ― для данной политики π и данного состояния s мы определяем функцию ценности Vπ следующим образом:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Уравнение Беллмана - Оптимальные уравнения Беллмана характеризуют функцию цены Vπ∗ оптимальной политики π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Примечание: отметим, что оптимальная политика π∗ для данного состояния s такова, что:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Алгоритм итерации ценностей ― алгоритм итерации ценностей состоит из двух этапов:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Инициализируем ценность:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) Итерация ценности на основе ценностей до:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Оценка максимального правдоподобия - оценки максимального правдоподобия для вероятностей перехода между состояниями следующие:

<br>

**47. times took action a in state s and got to s′**

&#10230; раз предприняли действие a в состоянии s и добрались до s′

<br>

**48. times took action a in state s**

&#10230; раз предприняли действия a в состоянии s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-обучение ― это безмодельная оценка Q, которая выполняется следующим образом:

<br>

**50. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Нейронные Сети, Архитектура, Функция активации, Обратное распространение, Прореживание]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Сверточные Нейронные Сети, Сверточный слой, Пакетная нормировка]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Рекуррентные Нейронные Сети, Вентили, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Обучение с подкреплением, Марковский процесс принятия решений, Итерация ценности/политики, Приближенное динамическое программирование, Политика поиска]