**1. Deep Learning cheatsheet**

&#10230; ディープラーニング虎の巻

<br>

**2. Neural Networks**

&#10230; ニューラルネットワーク

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; ニューラルネットワークは階層で構成されるモデルの1つです. 一般に使用されるニューラルネットワークには畳み込みや再帰型ニューラルネットワークがあります. 

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; アーキテクチャ - 下図で示されるニューラルネットワークのアーキテクチャ周辺のボキャブラリ:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [入力層, 隠れ層, 出力層]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; i はネットワークの階層の i 番目,  j はその階層の隠れ層の j 番目の隠れユニットとしたとき, 次式: 

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; ここで w, b, z をそれぞれ重み, バイアス, 出力とする.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 活性化関数 - 活性化関数はモデルに非線形な複雑さ導入するために隠れユニットの最後で使われます. 最も一般的な活性化関数: 

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 交差エントロピー損失 - ニューラルネットワークの文脈で, 一般的に使われる交差エントロピー損失 L(z,y)の定義は以下: 

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 学習率 - α あるいは時々 η と記述される学習率は, 重さが更新されるペースを示しています. これは修正される可能性があり, あるいは適応的に変化します. 現在の最もよく利用される手法は Adam で, これは学習率を順応させる手法です.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; バックプロパゲーション - バックプロパゲーション（誤差逆伝播法）は実際の出力と求める出力を考慮することによって, ニューラルネットワークの重みを更新する手法です. 重み w について, チェーンルール(連鎖率）を用いて計算されるの微分係数は次式: 

<br>

**13. As a result, the weight is updated as follows:**

&#10230; 結果として, 更新される重さは以下: 

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 更新された重み - あるニューラルネットワークで, 更新される重みは以下: 

<br>

**15. Step 1: Take a batch of training data.**

&#10230; ステップ 1: まとまった訓練データを入手します. 

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; ステップ 2: 該当する損失を獲得するためフォワードプロパゲーション(順伝播)を実行します. 

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 勾配を得るために損失をバックプロパゲーション(逆伝播)します。

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; ネットワークの重みを更新するために勾配を利用します。

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; ドロップアウト - ドロップアウトはニューラルネットワークのユニットをドロップアウトすることによって, 訓練データへのオーバーフィット(過学習)を抑制することを意味する技術です. 

<br>

**20. Convolutional Neural Networks**

&#10230; 畳み込みニューラルネットワーク

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 畳み込み層の要件 - W は入力データの数, F は畳み込み層のニューロンの数, P はゼロパディングの量, そしてニューロンの数 N に見合う量は次式: 

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; バッチ正規化 - それはハイパーパラメータ γ,β を用いて正規化するバッチ処理　{xi} の方法です. μB,σ2B で表す正確な平均と分散を用いたバッチ処理は, 次式: 

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; それは通常, 全結合/畳み込み層の後, 非線形の層の前で処理され, 高い学習率と強い初期値依存性を軽減します. 

<br>

**24. Recurrent Neural Networks**

&#10230; 再帰型ニューラルネットワーク

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; ゲートの種類 - ここでは典型的な再帰型ニューラルネットワークを例に挙げ, 異なる種類のゲートを紹介します. 

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [入力ゲート, 忘却ゲート, ゲート, 出力ゲート]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [セルに書き込みするか否か?, セルから削除するか否か?, セルにどのくらい書き込むか?, どのくらいセルから出力するか?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - Long Short-Term Memory (LSTM) ネットワークは, '忘却'ゲートの追加により勾配消失問題を防ぐタイプの RNN モデルです. 

<br>

**29. Reinforcement Learning and Control**

&#10230; 強化学習とその制御

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 強化学習のゴールは, エージェントにある環境で進化する方法を学ばせることです.  

<br>

**31. Definitions**

&#10230; 定義

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; マルコフ決定過程 - マルコフ決定過程 (Markov Deision Process; MDP) は, 次の5つ組のタプル (S,A,{Psa},γ,R): 

<br>

**33. S is the set of states**

&#10230; S は状態の集合

<br>

**34. A is the set of actions**

&#10230; A は行動の集合

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} は s∈S and a∈A の状態遷移確率

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1] は割引因子

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R or R:S⟶R はアルゴリズムが最大化すべき報酬関数

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; ポリシー(政策) - ポリシー π は状態に合わせるための行動を取る関数 π:S⟶A です.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; 注釈: もし状態 s が与えられたら, 与えられたポリシー π に従って行動 a=π(s) を実行します. 

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; 価値関数 - 与えられたポリシー π と与えられた状態 s に対して, 定義された価値関数 Vπ は以下: 

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; ベルマン方程式 - 最適化ポリシー π∗ を用いた価値関数 Vπ∗ の特徴を持つ最適化ベルマン方程式:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;

<br>

**44. 1) We initialize the value:**

&#10230;

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;

<br>

**47. times took action a in state s and got to s′**

&#10230;

<br>

**48. times took action a in state s**

&#10230;

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;

<br>

**50. View PDF version on GitHub**

&#10230;

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230;

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
