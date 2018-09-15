**1. Deep Learning cheatsheet**

&#10230; Resumo de Aprendizagem Profunda

<br>

**2. Neural Networks**

&#10230; Redes Neurais

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Redes neurais são uma classe de modelos que são construídos com camadas. Os tipos de redes neurais comumente utilizadas incluem redes neurais convolucionais e recorrentes.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Arquitetura - O vocabulário em torno das arquiteruras de redes neurais é descrito na figura abaixo:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Camada de entrada, camada escondida, camada de saída]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Note que i é a i-ésima camada da rede e j a j-ésima unidade escondida da camada, nós temos:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; onde notamos w, b, z, o peso, o viés e a saída respectivamente. 

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Função de ativação - Funções de ativação são usadas no fim de uma unidade escondida para introduzir complexidades não lineares ao modelo. Aqui estão as mais comuns:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoide, Tanh (tangente hiperbólica), ReLu (unidade linear retificada), Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Perda de entropia cruzada - No contexto de redes neurais, a perda de entropia cruzada L(z,y) é comumente utilizada e é definida como se segue: 

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Taxa de apredizado - A taxa de aprendizado, frequentemente notada por α ou às vezes η, indica a que ritmo os pesos são atualizados. Isso pode ser fixado ou alterado de modo adaptativo. O método atual mais popular é chamado Adam, o qual é um método que adapta a taxa de apredizado.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Retropropagação - Retropropagação é um método para atualizar os pesos em uma rede neural levando em conta a saída atual e a saída desejada. A derivada relativa ao peso w é computada utilizando a regra da cadeia e é da seguinte forma:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Como resultado, o peso é atualizado como se segue:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Atualizando os pesos - Em uma rede neural, os pesos são atualizados como a seguir:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Passo 1: Pegue um lote de dados de treinamento.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Passo 2: Realize propagação para frente a fim de obter a perda correspondente. 

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Passo 3: Retropropague a perda para obter os gradientes.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Passo 4: Use os gradientes para atualizar os pesos da rede.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Abandono - Abandono é uma técnica que pretende prevenir o sobreajuste dos dados de treinamente abandonando unidades na rede neural. Na prática, neurônios são ou abandonados com a propabilidade p ou mantidos com a propabilidade 1-p

<br>

**20. Convolutional Neural Networks**

&#10230; Redes Neurais Convolucionais

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Requisito de camada convolucional - Note que W é o tamanho do volume de entrada, F o tamanho dos neurônios da camada convolucional, P a quantidade de preenchimento de zeros, então o número de neurônios N que cabem em um dado volume é tal que:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Normalização em lote - É uma etapa de hiperparâmetro γ,β que normaliza o lote {xi}. Note que μB,σ2B são a média e a variância daquilo que queremos conectar ao lote, isso é feito como se segue:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Isso é usualmente feito após de uma totalmente conectada/camada concolucional e antes de uma camada não linear e objetiva permitir maiores taxas de apredizado e reduzir a forte dependência na inicialização.

<br>

**24. Recurrent Neural Networks**

&#10230; Redes Neurais Recorrentes

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Tipos de portas - Aqui estão os diferentes tipos de portas que encontramos em uma rede neural recorrente típica:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Porta de entrada, porta esquecida, porta, porta de saída]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Escrever na célula ou não?, Apagar a célula ou não?, Quanto escrever na célula?, Quanto revelar da célula?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - Uma rede de memória de longo prazo (LSTM) é um tipo de modelo de rede neural recorretne (RNN) que evita o problema do desaparecimento da gradiente adicionando portas de 'esquecimento'.

<br>

**29. Reinforcement Learning and Control**

&#10230; Aprendizado e Controle Reforçado

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; O objetivo do aprendizado reforçado é fazer um agente aprender como evoluir em um ambiente.

<br>

**31. Definitions**

&#10230; Definições

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Processos de decisão de Markov - Um processo de decição de Markov (MDP) é uma tupla de 5 elementos (S,A,{Psa},γ,R) onde:

<br>

**33. S is the set of states**

&#10230; S é o conjunto de estados

<br>

**34. A is the set of actions**

&#10230; A é conjunto de ações

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; Psa são as probabilidade de transição de estado para s∈S e a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ é o fator de desconto

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R ou R:S⟶R é a função de recompensa que o algoritmo quer maximizar

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Diretriz - Uma diretriz π é a função π:S⟶A que mapeia os estados a ações.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Observação: dizemos que executamos uma dada diretriz π se dado um estado s nós tomamos a ação a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Função de valor - Para uma dada diretriz π e um dado estado s, nós definimos a função de valor Vπ como a seguir:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Equação de Bellman - As equações de Bellman ótimas caracterizam a função de valor Vπ∗ para a ótima diretriz π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Observação: notamos que a ótima diretriz π∗ para um dado estado s é tal que: 

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Algoritmo de iteração de valor - O algoritmo de iteração de valor é realizado em duas etapas:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Inicializamos o valor:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) Iteramos o valor baseado nos valores anteriores

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Máxima probabilidade estimada - A máxima probabildiade estima para o estado de transição de probabilidades como se segue:

<br>

**47. times took action a in state s and got to s′**

&#10230; vezes que a ação a entrou no estado s e obteve s′

<br>

**48. times took action a in state s**

&#10230; vezes que a ação a entrou no estado s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Aprendizado Q - Aprendizado Q é um modelo livre de estimativa de Q, o qual é feito como se segue:
