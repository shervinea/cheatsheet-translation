**1. Deep Learning cheatsheet**

&#10230; Hoja de referencia de Aprendizaje Profundo (Deep learning)

<br>

**2. Neural Networks**

&#10230; Redes Neuronales

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Las redes neuronales son una clase de modelos construidos a base de capas. Los tipos más utilizados de redes neuronales incluyen las redes neuronales convolucionales y las redes neuronales recurrentes.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Arquitectura - El vocabulario en torno a arquitecturas de redes neuronales se describe en la siguiente figura:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Capa de entrada, capa oculta, capa de salida]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Denotando i en la i-ésima capa de la red y j en la j-ésima unidad oculta de la capa, tenemos:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; donde w, b y z son el peso, el sesgo y la salida, respectivamente.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Función de activación - Las funciones de activación son utilizadas al final de una unidad oculta para introducir complejidades no lineales al modelo. A continuación las más comunes:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoide, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Pérdida de entropía cruzada - En el contexto de las redes neuronales, la pérdida de entropía cruzada L(z,y) es utilizada comúnmente y definida de la siguiente manera:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Velocidad de aprendizaje - La velocidad de aprendizaje, denotada como α o algunas veces η, indica a que ritmo los pesos son actualizados. Este valor puede ser fijo o cambiar de forma adaptativa. El método más popular en este momento es llamado Adam, que es un método que adapta la velocidad de aprendizaje.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Retropropagación - La retropropagación, o propagación inversa, es un método de actualización de los pesos en una red neuronal, teniendo en cuenta la salida actual y la salida esperada. La derivada respecto al peso w es calculada utilizando la regla de la cadena y se expresa de la siguiente forma:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Como resultado, el peso es actualizado de la siguiente forma:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Actualizando pesos - En una red neuronal, los pesos son actualizados de la siguiente forma:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Paso 1: Tomar un lote de los datos de entrenamiento.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Paso 2: Realizar propagación hacia adelante para obtener la pérdida correspondiente.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Paso 3: Propagar inversamente la pérdida para obtener los gradientes.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Paso 4: Utiliza los gradientes para actualizar los pesos de la red.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;Abandono - El retiro es una técnica para prevenir el sobreajuste de los datos de aprendizaje descartando unidades en una red neuronal. En la práctica, las neuronas son retiradas con una probabilidad de p o se mantienen con una probabilidad de 1-p.

<br>

**20. Convolutional Neural Networks**

&#10230; Redes neuronales convolucionales.

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Requisito de la capa convolucional. Notando que W es el volumen de la entrada, F el tamaño de las neuronas de la capa convolucional, P la cantidad de relleno con ceros, entonces el número de neuronas N que entran en el volumen dado es tal que:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalización por lotes - Es un paso de híperparámetro y,β que normaliza el lote {xi}. Denotando μB,σ2B la media y la varianza de lo que queremos corregir en el lote, se realiza de la siguiente manera:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Se realiza usualmente después de una capa completamente conectada/convolucional y antes de una capa no-lineal y su objetivo es permitir velocidades de aprendizaje más altas y reducir su fuerte dependencia sobre la inicialización.

<br>

**24. Recurrent Neural Networks**

&#10230;Redes Neuronales Recurrentes

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Tipos de puertas - A continuación, tenemos los diferentes tipos de puertas que encontramos en una red neuronal recurrente típica:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Puerta de entrada, puerta de olvido, puerta, puerta de salida]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [¿Escribir o no en la celda?, ¿Borrar o no la celda?, ¿Cuánto escribir en la celda?, ¿Cuánto revelar la celda?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - Una red de memoria de corto y largo plazo (LSTM por sus siglas en inglés) es un tipo de modelo de red neuronal recurrente que evita el problema del gradiente desvaneciente añadiendo puertas de 'olvido'.

<br>

**29. Reinforcement Learning and Control**

&#10230; Aprendizaje por refuerzo y control.

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; El objetivo del aprendizaje por refuerzo es hacer que un agente aprenda como evolucionar en un ambiente.

<br>

**31. Definitions**

&#10230; Definiciones

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Procesos de decisión de Markov - Un proceso de decisión de Markov (MDP por sus siglas en inglés) es una 5-tupla (S,A,{Psa},γ,R) donde:

<br>

**33. S is the set of states**

&#10230; S es el conjunto de estados

<br>

**34. A is the set of actions**

&#10230; A es el conjunto de acciones

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa] son las probabilidades de transición de estado para s∈S y a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ es el factor de descuento

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R o R:S⟶R es la función recompensa que el algoritmo pretende maximizar**

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Política - Una política π es una función π:S⟶A que asigna estados a acciones.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Observación: decimos que ejecutamos una política π dada si dado un estado a tomamos la acción a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Función valor - Para una política dada π y un estado dado s, definimos el valor de la función Vπ de la siguiente manera:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; La ecuación de Bellman - Las ecuaciones óptimas de Bellman, caracterizan la función valor Vπ* de la política óptima π*:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Observación: denotamos que la política óptima π* para un estado dado s es tal que:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algoritmo de iteración valor - El algoritmo de iteración valor es en dos pasos:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Inicializamos el valor:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; Iteramos el valor con base en los valores de antes:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Estimación por máxima verosimilitud - Las estimaciones por máxima verosimilitud para las probabilidades de transición de estado son como se muestra a continuación:

<br>

**47. times took action a in state s and got to s′**

&#10230; veces que se tomó la acción a en el estado s y llevó a s'

<br>

**48. times took action a in state s**

&#10230; veces que se tomó la acción a en el estado s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**


&#10230; Q-learning - Q-learning es una estimación libre de modelo de Q, que se realiza de la siguiente forma:

<br>

**50. View PDF version on GitHub**

&#10230; Ver la versión PDF en GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Redes Neuronales, Arquitectura, Función de activación, Retropropagación, Retiro]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; Redes Neuronales Convolucionales, Capa convolucional, Normalización de lotes]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Redes Neuronales Recurrentes, Puertas, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Aprendizaje por refuerzo, Procesos de decisión de Markov, Iteración de valor/política, Programación dinámica de aproximación, búsqueda de políticas]
