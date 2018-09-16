**1. Deep Learning cheatsheet**

&#10230; Hoja de referencia de Aprendizaje Profundo

<br>

**2. Neural Networks**

&#10230; Redes Neuronales

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Las redes neuronales (*neural networks* en inglés) son una clase de modelos organizados en capas. Las redes neuronales convolucionales (*convolutional neural networks* en inglés), así como las redes neuronales recurrentes (*recurrent neural networks* en inglés), son las más comúnmente usadas.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Arquitectura -- El vocabulario que rodea a las redes neuronales se describe en la figura de abajo:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Capa de entrada, capa oculta, capa de salida]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Si denotamos i a la i-ésima capa de la red y j a la j-ésima unidad oculta de la capa, tenemos:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; donde denotamos w, b, z a los pesos, sesgo (*bias* en inglés) y salida respectivamente.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Función de activación -- Las funciones de activación se usan al final de una capa oculta para introducir complejidades no lineales en el modelo. Las más comunes son:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoide, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Error de *cross-entropy* -- En el contexto de las redes neuronales, se usa comúnmente el error de *cross-entropy* L(z,y) y se define de la manera siguiente:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Velocidad de aprendizaje (*learning rate* en inglés) - La velocidad de aprendizaje, a menudo denotada como α o a veces η, indica a qué velocidad los pesos se actualizan. Esta puede ser fija o cambiar de manera adaptativa. El método más popular se denomina *Adam*, que es un método que adapta la velocidad de aprendizaje.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Propagación inversa (*backpropagation* en inglés) - La propagación inversa es un método para actualizar los pesos en una red neuronal teniendo en cuenta la salida actual y la salida deseada. La derivada respecto del peso w se calcula usando la regla de la cadena y es de la forma siguiente:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Como resultado, los pesos se actualizan como sigue:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Actualizando los pesos -- En una red neuronal, los pesos se actualizan como sigue:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Paso 1: Quédate con un lote (*batch* en inglés) de los datos de entrenamiento.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Paso 2: Realiza la propagación hacia adelante para obtener el error correspondiente.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Paso 3: Propagación inversa del error para obtener los gradientes.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Usa los gradientes para actualizar los pesos de la red.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout -- Dropout es una técnica que pretende evitar el sobreajuste (*overfitting* en inglés) sobre los datos de entrenamiento eliminando unidades en la red neuronal. En la práctica, las neuronas, o bien se eliminan con probabilidad p, o se mantienen con probabilidad 1-p 

<br>

**20. Convolutional Neural Networks**

&#10230; Redes Neuronales Convolucionales

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Requisito de una capa convolucional -- Si denotamos W al tamaño del volumen de entrada, F al tamaño de las neuronas de la capa convolucional y P a la cantidad de relleno (*padding* en inglés) de ceros, entonces el número de neuronas N que caben en el volumen dado es: 

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalización por lotes (*batch normalization* en inglés) -- Es un conjunto de parámetros de entrenamiento γ,β que normaliza el lote {xi}. Denotando μB,σ2B a la media y a la varianza de lo que queremos corregir en el lote, se realiza como sigue:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Se suele realizar después de una capa densa/convolucional y antes de una no linealidad y tiene por objetivo permitir una velocidad de aprendizaje más alta y reducir la fuerte dependencia en la inicialización. 

<br>

**24. Recurrent Neural Networks**

&#10230; Redes Neuronales Recurrentes

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Tipos de puertas -- Aquí vemos los diferentes tipos de puertas que encontramos en una red neuronal recurrente típica:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Puerta de entrada, puerta del olvido (*forget gate* en inglés), puerta, puerta de salida]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [¿Escribir a una celda o no?, ¿Borrar una celda o no?, ¿Cuánto esribir a una celda?, ¿Cuánto desvelar a una celda?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; Una red de memoria a largo y corto plazo (*long short-term memory*, *LSTM* por sus siglas en inglés) es un tipo de modelo de red neuronal recurrente que evita el problema de desvanecimiento de los gradientes (*vanishing gradient problem* en inglés) añadiendo puertas 'forget'. 

<br>

**29. Reinforcement Learning and Control**

&#10230; Aprendizaje por Refuerzo y Control

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; El objetivo del aprendizaje por refuerzo es que un agente aprenda como evolucionar en un entorno.

<br>

**31. Definitions**

&#10230; Definiciones

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Procesos de decisión de Markov -- Un problema de decisión de Markov (*MDP* por sus siglas en inglés) es una tupla de 5 elementos (S,A,{Psa},γ,R) donde:

<br>

**33. S is the set of states**

&#10230; S es el conjunto de estados

<br>

**34. A is the set of actions**

&#10230; A es el conjunto de acciones

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} son las probabilidades de transición de estados para s∈S y a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1 es el factor de descuento (*discount factor* en inglés)

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize** 

&#10230; R:S×A⟶R or R:S⟶R es a la función de recompensa que el algoritmo quiere maximizar

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Política -- Una política π es una función π:S⟶A asigna estados a acciones.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Observación: decimos que ejecutamos una política π dada n si dado un estado s escogemos la acción a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Función de valor -- Dada una política π y un estado dado s, definimos la función valor Vπ como sigue: 

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Ecuación de Bellman -- Las ecuaciones de Bellman óptimas caracterizan la función valor Vπ de la política óptima π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Observación: indicamos que la política óptima π∗ para un estado dado s es tal que:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algoritmo de iteración de valor - El algoritmo de iteración de valor se realiza en dos pasos:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Inicializamos el valor:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; Iteramos el valor basándonos en los valores anteriores:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Estimación por máxima verosimilitud (*maximum likelihood estimate* en inglés) -- Las estimaciones por máxima verosimilitud para las probabilidades de transición de estados son como sigue:

<br>

**47. times took action a in state s and got to s′**

&#10230; veces que se tomó la acción a en el estado s y se llegó a s'

<br>

**48. times took action a in state s**

&#10230; veces que se tomó la acción a en el estado s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning -- Q-learning una estimación de Q independiente del modelo, que se realiza como sigue:

<br>

**50. View PDF version on GitHub**

&#10230; Ver la versión PDF en GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Redes Neuronales, Arquitectura, Función de activación, Propagación inversa (*backpropagation*), Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Redes Neuronales Convolucionales, Capa convolucional, Normalización por lotes]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Red Neuronal Recurrente, Puertas, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Aprendizaje por refuerzo, Procesos de decisión de Markov, iteración valor/política, Programación dinámica aproximada, Búsqueda de política] 

&#10230;
