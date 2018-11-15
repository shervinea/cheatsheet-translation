**1. Deep Learning cheatsheet**

&#10230; ディープラーニングチートシート

<br>

**2. Neural Networks**

&#10230; ニューラルネットワーク

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; ニューラルネットワークとは複数の層を用いて組まれる数学モデルです。代表的なネットワークとして畳み込みと再帰型ニューラルネットワークが挙げられます。

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; 構造　－ ニューラルネットワークを組む上で重要な用語は以下の図により説明されます：

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [入力層, 隠れ層, 出力層]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; iをネットワーク上のi層目の層とし、隠れ層のj個目のユニットをjとすると：

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; この場合重み付けをw、バイアス項をb、出力をzとします。

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 活性化関数　ー ユニットの出力に非線形性を与える関数を活性化関数といいます。一般的には以下の関数がよく使われます：

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid(シグモイド関数), Tanh(双曲線関数), ReLU(ランプ関数), Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 交差エントロピーロス　ー ニューラルネットにおいて交差エントロピーロスL(z,y)は頻繁に使われ、以下のように定義されています：

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 学習率　ー αやηで表される学習率は勾配法による重み付けのアップデートをする速度を表します。学習率は固定または適応的に変更することができます。現在一般的に使われている学習法はAdam（アダム）であり、学習率を適用させる方法です。

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; 誤差逆伝播法（backpropagation）ー 誤差逆伝播法はニューラルネットの期待される出力値と実際の出力の差異を考慮し重み付けのアップデートをする方法の一つです。重みwに関する導関数は連鎖規則を使用して計算され、次の形式で表される：

<br>

**13. As a result, the weight is updated as follows:**

&#10230; 結果、重みは以下のようにアップデートされます：

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 重みアップデート　ー ニューラルネットでは以下のように重みがアップデートされます：

<br>

**15. Step 1: Take a batch of training data.**

&#10230; ステップ１： 訓練データを１バッチ用意する。

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; ステップ２：　フォワードプロパゲーションを行い誤差を求める。

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 求められた誤差を用い、傾斜を計算する。

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; 傾斜を使い誤差が小さくなるように重みを調整する。

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;ドロップアウト　ー ドロップアウトはニューラルネット内の一部のユニットを非活性化させることにより過学習を防ぐテクニックである。実際には、ニューロンはある確率pで非活性、1-pの確率で活性化されるようになってる。

<br>

**20. Convolutional Neural Networks**

&#10230; 畳み込みニューラルネットワーク

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 畳み込みレイヤーの条件　ー Wを入力サイズ、Fを畳み込みレイヤーニューロンのサイズ、Pをゼロパディングの量とすると、与えられた体積に収まるニューロン数Nは次のようになります。

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; バッチ正規化　ー バッチ{xi}を正規化するハイパーパラメータγ、βのステップです。バッチに修正したい平均値と分散値をμB,σ2Bとすると、正規化は以下のように行われます:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; これは通常、学習率を高め、初期値への依存性を減らすことを目的でFully Connected層と畳み込み層の後、非線形化を行う前に行われます。

<br>

**24. Recurrent Neural Networks**

&#10230; 再帰型ニューラルネットワーク (RNN)

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; ゲートの種類　ー　典型的なRNNに使われるゲートです:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [入力ゲート, 忘却ゲート, ゲート, 出力ゲート]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [セルに追加するべき？, セルを削除するべき？, 情報をどれだけセルに追加するべき？, セルをどの程度通すべき？]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - A long short-term memory (LSTM) ネットワークは勾配消失問題を解決するために忘却ゲートが追加されているRNNの一種です。

<br>

**29. Reinforcement Learning and Control**

&#10230; 強化学習とコントロール

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 強化学習はある環境内においてエージェントが学習し、進化することを目標とします。

<br>

**31. Definitions**

&#10230; 定義

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; マルコフ決定過程 ー  マルコフ決定過程(Markov decision process; MDP)を5タプル(S,A,{Psa},γ,R)としたとき：

<br>

**33. S is the set of states**

&#10230; Sは状態の有限集合

<br>

**34. A is the set of actions**

&#10230; Aは行動の有限集合

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa}は状態s∈Sと行動a∈Aの状態遷移確率

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[は割引因子と呼ばれる値

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R or R:S⟶Rはアルゴリズムが最大化したい報酬関数

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; 政策 - 政策πは状態と行動を写像する関数π:S⟶A

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; 備考: 状態sを与えられた際に行動a=π(s)を行うことを政策πを実行すると言う。

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; 価値関数 - ある政策πとある状態sにおいて価値関数Vπを以下のように定義する：

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; ベルマン方程式 - 政策πをとった価値関数Vπ∗に対する最適なベルマン方程式：

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; 備考: 与えられた状態sに対する最適方針π*はこのようになります：

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; 価値反復法アルゴリズム - 価値反復法アルゴリズムは２段階で行われます：

<br>

**44. 1) We initialize the value:**

&#10230; 1) 値を初期化する。

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) 前の値を元に値を繰り返す：

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; 最尤推定 ー 状態遷移確率の最尤推定(maximum likelihood estimate; MLE)：

<br>

**47. times took action a in state s and got to s′**

&#10230; 状態sで行動aを行い状態s′に遷移した回数

<br>

**48. times took action a in state s**

&#10230; 状態sで行動aを行った回数

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;  Q学習　ー Q学習は数学モデルを使わないQ値の評価手法であり、以下のように行われる：

<br>

**50. View PDF version on GitHub**

&#10230; GitHubでPDF版を見る

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [ニューラルネットワーク, アーキテクチャ, 活性化関数, 誤差逆伝播法, ドロップアウト]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [畳み込みニューラルネットワーク, 畳み込み層, バッチノーマライゼーション]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [再帰型ニューラルネットワーク, ゲート, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [強化学習, マルコフ決定過程, バリュー/ポリシー反復, 近似動的計画法, ポリシーサーチ]
