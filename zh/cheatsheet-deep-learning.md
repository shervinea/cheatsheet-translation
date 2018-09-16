1. **Deep Learning cheatsheet**

&#10230; 深度学习速查表

<br>

2. **Neural Networks**

&#10230; 神经网络

<br>

3. **Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; 神经网络是一类按层结构搭建的模型。常用的神经网络包括卷积神经网络和递归神经网络。

<br>

4. **Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; 架构 —— 下表列举了用来描述神经网络架构的词汇：

<br>

5. **[Input layer, hidden layer, output layer]**

&#10230; [输入层，隐层，输出层]

<br>

6. **By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; 已知***i***是网络的第i层，***j***是网络的第j层，我们有：

<br>

7. **where we note w, b, z the weight, bias and output respectively.**

&#10230; 我们用w, b, z分别表示权重，偏差和输出。

<br>

8. **Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 激活函数 —— 激活函数被用在隐含单元之后来向模型引入非线性复杂度。比较常见的如下所示：

<br>

9. **[Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [逻辑函数(Sigmoid), 双曲正切函数(Tanh), 线性整流函数(ReLU), 带泄露线性整流函数(Leaky ReLU)]

<br>

10. **Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 交叉墒损失 —— 在神经网络中，交叉墒损失L(z, y)通常如下定义：

<br>

11. **Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 学习率 —— 学习率，通常记为α或η，表示权重更新的速度。它可以被修复或自适应改变。现阶段最常用的方法是一种调整学习率的算法，叫做Adam。

<br>

12. **Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; 反向传播 —— 反向传播是一种通过考虑实际输出和期望输出来更新神经网络中权重的方法。权重*w*的导数由链式法则计算，模式如下：

<br>

13. **As a result, the weight is updated as follows:**

&#10230; 作为结果，权重更新如下：

<br>

14. **Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 更新权重 —— 在一个神经网络中，权重如下所示更新：

<br>

15. **Step 1: Take a batch of training data.**

&#10230; 第一步：分出第一批次的训练数据。

<br>

16. **Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 第二步：通过前向传播来得到相关损失。

<br>

17. **Step 3: Backpropagate the loss to get the gradients.**

&#10230; 第三步：通过反向传播损失来得到梯度。

<br>

18. **Step 4: Use the gradients to update the weights of the network.**

&#10230; 第四步：利用梯度更新网络的权重。

<br>

19. **Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; 降层(Dropout) —— 降层是一种通过丢弃神经网络单元来防止训练数据过拟合的技术。实际上，神经元以概率p被丢弃或以概率1-p被保留。

<br>

20. **Convolutional Neural Networks**

&#10230; 卷积神经网络

<br>

21. **Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 卷积神经网络要求 —— 记W为输入图像尺寸，F为卷积层神经元尺寸，P为零填充的大小，那么给定的输入图像能够容纳的神经元数目N为：

<br>

22. **Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 批标准化 —— 它是超参数γ, β标准化样本批{xi}的一个步骤。将我们希望修正这一批样本的均值和方差记作μB, σ2B，会得到：

<br>

23. **It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 它通常在非线性层之前和全连接／卷积层后应用，目的是允许更高的学习率和减少初始化的强相关。

<br>

24. **Recurrent Neural Networks**

&#10230; 递归神经网络

<br>

25. **Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; 门控的种类 —— 在一个典型的递归神经网络中有多种门控结构：

<br>

26. **[Input gate, forget gate, gate, output gate]**

&#10230; [输入门，忘记门，门控，输出门]

<br>

27. **[Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [要不要写入单元？要不要清空单元？在单元写入多少？从单元泄露多少？]

<br>

28. **LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM —— 长短期记忆网络是递归神经网络的一种，它可以通过增加“遗忘”门控来避免梯度消失问题。

<br>

29. **Reinforcement Learning and Control**

&#10230; 强化学习和控制

<br>

30. **The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 强化学习的目标是让代理学习如何在环境中进化。

<br>

31. **Definitions**

&#10230; 定义

<br>

32. **Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; 马尔可夫决策过程 —— 一个马尔可夫决策过程(MDP)是一个5维元组 (S, A, {Psa}, γ, R)，即：

<br>

33. **S is the set of states**

&#10230; S是状态的集合

<br>

34. **A is the set of actions**

&#10230; A是动作的集合

<br>

35. **{Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa}是对于s属于S并且a属于A的状态转换概率

<br>

36. **γ∈[0,1] is the discount factor**

&#10230; γ∈[0,1]是折扣系数

<br>

37. **R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R或R:S⟶R是算法希望最大化的回报函数

<br>

38. **Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; 策略 —— 策略π是映射状态到动作的π:S⟶A函数。

<br>

39. **Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; 注意：如果对于一个指定的状态s我们完成了行动a=π(s)，我们认为执行了一个指定的策略π。

<br>

40. **Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; 价值函数 —— 对于一个指定的策略π和指定的状态s，我们定义如下价值函数Vπ：

<br>

41. **Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; 贝尔曼方程 —— 最优贝尔曼方程描述了最优策略π∗的价值方程Vπ∗：

<br>

42. **Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; 注意：我们注意到对于一个特定的状态s的最优策略π∗是：

<br>

43. **Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; 值迭代算法 —— 值迭代算法分为两步：

<br>

44. **1) We initialize the value:**

&#10230; 1）首先我们初始化值：

<br>

45. **2) We iterate the value based on the values before:**

&#10230; 2）我们通过之前的值进行迭代：

<br>

46. **Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; 极大似然估计 —— 状态转移概率的极大似然估计如下：

<br>

47. **times took action a in state s and got to s′**

&#10230; 状态s下进行动作a并且进入状态s‘的次数：

<br>

48. **times took action a in state s**

&#10230; 状态s下进行动作a的次数

<br>

49. **Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q学习 —— Q学习是一种Q的无模型(model-free)预测，如下所示：
