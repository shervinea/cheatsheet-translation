1. **Deep Learning cheatsheet**

&#10230;
深度學習參考手冊
<br>

2. **Neural Networks**

&#10230;
神經網路
<br>

3. **Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230;
神經網路是一種透過 layer 來建構的模型。經常被使用的神經網路模型包括了卷積神經網路 (CNN) 和遞迴式神經網路 (RNN)。
<br>

4. **Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230;
架構 - 神經網路架構所需要用到的詞彙描述如下：
<br>

5. **[Input layer, hidden layer, output layer]**

&#10230;
[輸入層、隱藏層、輸出層]
<br>

6. **By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;
我們使用 i 來代表網路的第 i 層、j 來代表某一層中第 j 個隱藏神經元的話，我們可以得到下面得等式：
<br>

7. **where we note w, b, z the weight, bias and output respectively.**

&#10230;
其中，我們分別使用 w 來代表權重、b 代表偏差、z 代表輸出。
<br>

8. **Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230;
激勵函數 - 激勵函數是為了在神經元中帶入非線性的轉換而設計。底下是一些常見函數：
<br>

9. **[Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;
[Sigmoid, Tanh, ReLU, Leaky ReLU]
<br>

10. **Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;
交叉熵損失函式
<br>

11. **Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;
學習速率 - 學習速率通常用 α 或 η 來表示，目的是用來控制權重更新的速度。學習速度可以是一個固定值，或是隨著訓練的過程改變。現在最熱門的方法叫做 Adam，是一種隨著訓練過程改變的學習速度。
<br>

12. **Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;
反向傳播演算法 - 反向傳播演算法是一種在神經網路中用來更新權重的方法，更新的基準是根據神經網路的實際輸出值和期望輸出值之間的關係。權重的導數是根據連鎖律 (chain rule) 來計算，通常會表示成下面的形式：
<br>

13. **As a result, the weight is updated as follows:**

&#10230;
因此，權重會透過以下的方式來更新：
<br>

14. **Updating weights ― In a neural network, weights are updated as follows:**

&#10230;
更新權重 - 在神經網路中，權重的更新會透過以下步驟進行：
<br>

15. **Step 1: Take a batch of training data.**

&#10230;
步驟一：取出一個批次 (batch) 的資料
<br>

16. **Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;
步驟二：執行前向傳播演算法 (forward propagation) 來得到對應的損失值
<br>

17. **Step 3: Backpropagate the loss to get the gradients.**

&#10230;
步驟三：將損失值透過反向傳播演算法來得到梯度
<br>

18. **Step 4: Use the gradients to update the weights of the network.**

&#10230;
步驟四：使用梯度來更新網路的權重
<br>

19. **Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;
Dropout - Dropout 是一種透過丟棄一些神經元，來避免過擬和的技巧。在實務上，神經元會透過機率值的設定來決定要丟棄或保留
<br>

20. **Convolutional Neural Networks**

&#10230;
卷積神經網絡
<br>

21. **Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230;
卷積層的需求 - 我們使用 W 來表示輸入的尺寸、F 代表卷積層的 filter 尺寸、P 代表使用的 pad 數量，S 代表 stride 的數量，則輸出的尺寸可以透過以下的公式表示：
<br>

22. **Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;
批次正規化 (Batch normalization) - 它是一個藉由 γ,β 兩個超參數來正規化每個批次 {xi} 的過程。每一次正規化的過程，我們使用 μB,σ2B 分別代表平均數和變異數。請參考以下公式：
<br>

23. **It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;
批次正規化的動作通常在全連接層/卷積層之後、在非線性層之前進行。目的在於容許更高的學習速率，並且減少對於初始化資料的依賴
<br>

24. **Recurrent Neural Networks**

&#10230;
遞歸神經網路 (RNN)
<br>

25. **Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;
閘的種類 - 在傳統的遞歸神經網路中，你會遇到幾種閘：
<br>

26. **[Input gate, forget gate, gate, output gate]**

&#10230;
輸入閘、遺忘閥、閘、輸出閘
<br>

27. **[Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;
要不要將資料寫入到記憶區塊中？要不要將存在在記憶區塊中的資料清除？要寫多少資料到記憶區塊？要不要將資料從記憶區塊中取出？
<br>

28. **LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;
長短期記憶模型 - 長短期記憶模型是一種遞歸神經網路，藉由導入遺忘閘的設計來避免梯度消失的問題
<br>

29. **Reinforcement Learning and Control**

&#10230;
強化學習及控制
<br>

30. **The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;
強化學習的目標就是為了讓代理 (agent) 能夠學習在環境中進化
<br>

31. **Definitions**

&#10230;
定義
<br>

32. **Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;
馬可夫決策過程 - 一個馬可夫決策過程 (MDP) 包含了五個元素：
<br>

33. **S is the set of states**

&#10230;
S 是一組狀態的集合
<br>

34. **A is the set of actions**

&#10230;
A 是一組行為的集合
<br>

35. **{Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;
{Psa} 指的是，當 s∈S、a∈A 時，狀態轉移的機率
<br>

36. **γ∈[0,1[ is the discount factor**

&#10230;
γ∈[0,1[ 是衰減係數
<br>

37. **R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;
R:S×A⟶R 或 R:S⟶R 指的是獎勵函數，也就是演算法想要去最大化的目標函數
<br>

38. **Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;
策略 - 一個策略 π 指的是一個函數 π:S⟶A，這個函數會將狀態映射到行為
<br>

39. **Remark: we say that we execute a given policy π if given a state a we take the action a=π(s).**

&#10230;
注意：我們會說，我們給定一個策略 π，當我們給定一個狀態 s 我們會採取一個行動 a=π(s)
<br>

40. **Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;
價值函數 - 給定一個策略 π 和狀態 s，我們定義價值函數 Vπ 為：
<br>

41. **Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;
貝爾曼方程 - 最佳的貝爾曼方程是將價值函數 Vπ∗ 和策略 π∗ 表示為：
<br>

42. **Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;
注意：對於給定一個狀態 s，最佳的策略 π∗ 是：
<br>

43. **Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;
價值迭代演算法 - 價值迭代演算法包含兩個步驟：
<br>

44. **1) We initialize the value:**

&#10230;
1) 針對價值初始化：
<br>

45. **2) We iterate the value based on the values before:**

&#10230;
根據之前的值，迭代此價值的值：
<br>

46. **Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;
最大概似估計 - 針對狀態轉移機率的最大概似估計為：
<br>

47. **times took action a in state s and got to s′**

&#10230;
從狀態 s 到 s′ 所採取行為的次數
<br>

48. **times took action a in state s**

&#10230;
從狀態 s 所採取行為的次數
<br>

49. **Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
Q-learning 演算法 - Q-learning 演算法是針對 Q 的一個 model-free 的估計，如下：

50. **View PDF version on GitHub**

&#10230;
前往 GitHub 閱讀 PDF 版本
<br>

51. **[Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;
[神經網路, 架構, 激勵函數, 反向傳播演算法, Dropout]
<br>

52. **[Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;
[卷積神經網絡, 卷積層, 批次正規化]
<br>

53. **[Recurrent Neural Networks, Gates, LSTM]**

&#10230;
[遞歸神經網路 (RNN), 閘, 長短期記憶模型]
<br>

54. **[Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
[強化學習, 馬可夫決策過程, 價值/策略迭代, 近似動態規劃, 策略搜尋]