1. **Deep Learning cheatsheet**

&#10230; 深度学习简明指南

<br>

2. **Neural Networks**

&#10230; 神经网络

<br>

3. **Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; 神经网络是一类由层构建成的模型。通常使用的神经网络包括卷积和循环神经网络。

<br>

4. **Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; 架构 - 神经网络架构相关的词汇在下列图中描述：

<br>

5. **[Input layer, hidden layer, output layer]**

&#10230; [输入层，隐藏层，输出层]

<br>

6. **By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; 记 i 为网络的第 i 层，j 为该层的第 j 个隐藏元，我们有：

<br>

7. **where we note w, b, z the weight, bias and output respectively.**

&#10230; 其中我们记 w, b, z 分别为权重，偏差和输出。

<br>

8. **Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 激活函数 - 激活函数在隐藏元的尾部使用，来给模型引入非线性复杂度。下面是最常见的：

<br>

9. **[Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

10. **Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 交叉熵损失函数 - 在神经网络语境下，交叉熵损失函数 L(z,y) 常被使用，其定义为：

<br>

11. **Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 学习率 - 学习率，常记作 α 或者 η，表示权重更新的幅度。可以固定或者适应性变化。当前常见的流行方法称为 Adam，这是一种适应性变化学习率的方法。

<br>

12. **Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; 反向传播 - 反向传播是在神经网络中通过考虑实际输出和预想输出信息的更新权重的方法。关于权重 w 的导数使用链式法则计算，形如：

<br>

13. **As a result, the weight is updated as follows:**

&#10230; 因此，权重按照下面方式更新：

<br>

14. **Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 更新权重 - 在一个神经网络中，权重按照如下方式更新：

<br>

15. **Step 1: Take a batch of training data.**

&#10230; 步骤 1：获取一个批量训练数据。

<br>

16. **Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 步骤 2：执行前向传播获得相应的损失函数。

<br>

17. **Step 3: Backpropagate the loss to get the gradients.**

&#10230; 步骤 3：反向传播损失函数获得梯度。

<br>

18. **Step 4: Use the gradients to update the weights of the network.**

&#10230; 步骤 4：使用梯度更新网络权重。

<br>

19. **Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; 随机丢弃 - 随机丢弃是一个通过丢弃神经网络中的隐藏元来防止过匹配的技术。在实践中，神经网络元或者以概率 p 丢弃，或者以 1-p 概率保留。

<br>

20. **Convolutional Neural Networks**

&#10230; 卷积神经网络

<br>

21. **Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 卷积层的要求 - 记 W 为输入的大小，F 为卷积层神经元的大小，P 为零垫充的数量，那么匹配给定数据的量的神经元数量 N 有： 

<br>

22. **Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 批规范化 - 规范化批量数据 {xi} 包含超参数 γ,β 的一步。记 μB,σ2B 为我们想要更正的均值和方差，按照如下方式进行：

<br>

23. **It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 通常在一个全连接/卷积后，在一个非线性层前完成，为了允许更高的学习率和降低对初始化的依赖。

<br>

24. **Recurrent Neural Networks**

&#10230; 循环神经网络

<br>

25. **Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; 门的类型 - 在一个典型的循环神经网络中有不同的门：

<br>

26. **[Input gate, forget gate, gate, output gate]**

&#10230; [输入门，忘记门，门，输出门]

<br>

27. **[Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [是否写入内部状态？，是否清除内部状态？，写入多少信息入内部状态？，读出内部状态多少信息？]

<br>

28. **LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - 一个长-短期记忆（LSTM）网络是循环神经网络的一种类型，通过添加‘忘记’门来防止梯度消失问题。

<br>

29. **Reinforcement Learning and Control**

&#10230; 强化学习和控制

<br>

30. **The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 强化学习的目标是让一个智能体学习如何在一个环境中进化。

<br>

31. **Definitions**

&#10230; 定义

<br>

32. **Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; 马尔科夫决策过程 - 一个马尔科夫决策过程（MDP）是一个 5-元组 (S,A,{Psa},γ,R) 其中：

<br>

33. **S is the set of states**

&#10230; S 是状态集

<br>

34. **A is the set of actions**

&#10230; A 是行动集

<br>

35. **{Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} 是对 s∈S 和 a∈A 状态转移概率

<br>

36. **γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ 为折扣因子

<br>

37. **R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R 或者 R:S⟶R 是算法希望最大化的奖励函数

<br>

38. **Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; 策略 - 策略 π 是一个函数 π:S⟶A 其映射状态到行动上。

<br>

39. **Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; 注：我们称执行一个给定策略 π 当给定一个状态 s 我们执行行动 a=π(s)。

<br>

40. **Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; 值函数 - 对一个给定策略 π 和一个给定状态 s，我们定义值函数 Vπ 如下：

<br>

41. **Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; 贝尔曼方程 - 最优贝尔曼方程刻画了最优策略 π∗ 的值函数 Vπ∗：

<br>

42. **Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; 注：记对一个给定状态 s 的最优策略 π∗ 有：

<br>

43. **Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; 值迭代算法 - 值迭代算法为两步：

<br>

44. **1) We initialize the value:**

&#10230; 我们初始值：

<br>

45. **2) We iterate the value based on the values before:**

&#10230; 我们基于之前的值进行值的迭代：

<br>

46. **Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; 极大似然估计 - 对状态转移概率的极大似然估计如下：

<br>

47. **times took action a in state s and got to s′**

&#10230; 在状态 s 下执行行动 a 达到状态 s' 的次数

<br>

48. **times took action a in state s**

&#10230; 在状态 s 下执行行动 a 的次数

<br>

49. **Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-学习 - Q-学习是一种免模型估计 Q 的方法，按照如下方式进行：

<br>

50. **View PDF version on GitHub**

&#10230; 在 Github 上看 PDF 版本

<br>

51. **[Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [神经网络，架构，激活函数，反向传播，随机丢弃]

<br>

52. **[Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [卷积神经网络，卷积层，批规范化]

<br>

53. **[Recurrent Neural Networks, Gates, LSTM]**

&#10230; [循环神经网络，门，LSTM]

<br>

54. **[Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [强化学习，马尔科夫决策过程，值/策略迭代，近似动态规划，策略搜索]
