**Recurrent Neural Networks translation**

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230;リカレントニューラルネットワーク　チートシート

<br>


**2. CS 230 - Deep Learning**

&#10230;ディープラーニング

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230;概要、アーキテクチャの構造、RNNの応用アプリケーション、損失関数、逆伝播

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230;長期依存性関係の処理、活性化関数、勾配喪失と発散、勾配クリッピング、GRU/LTSM、ゲートの種類、双方向性RNN、ディープ(深層学習)RNN

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230;単語出現の学習、ノーテーション、埋め込み行列、Word2vec、スキップグラム、ネガティブサンプリング、グローブ

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230;単語の比較、コサイン類似度、t-SNE

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230;言語モデル、n-gramモデル、パープレキシティ

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230;機械翻訳、ビームサーチ、言語長正規化、エラー分析、ブルースコア(機械翻訳比較スコア)

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230;アテンション、アテンションモデル、アテンションウェイト

<br>


**10. Overview**

&#10230;概要

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230;一般的なRNNのアーキテクチャ - RNNとして知られるリカレントニューラルネットワークは、隠れ層の状態を利用して、前の出力を次の入力として取り扱うことを可能にするニューラルネットワークの一種です。一般的なモデルは下記のようになります。

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230;それぞれの時点 t において活性化関数の状態 a<t> と出力 y<t> は下記のように表現されます。　

<br>


**13. and**

&#10230;そして

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230;Wax,Waa,Wya,baは全ての時点で共有される係数であり、g1,g2は活性化関数です。

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230;一般的なRNNのアーキテクチャ利用の長所・短所については下記の表にまとめられています。

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230;長所、任意の長さの入力を処理できる、入力サイズに応じてモデルサイズが大きくならない、計算は時系列情報を考慮している、重みは全ての時点で共有される

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230;短所、遅い計算、長い時間軸での情報の利用が困難、現在の状態から将来の入力を予測不可能

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230;RNNの応用 - RNNモデルは主に自然言語処理と音声認識の分野で使用されます。以下の表に、さまざまな応用例がまとめられています。

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230;RNNの種類、図、例

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230;一対一、一対多、多対一、多対多

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230;伝統的なニューラルネットワーク、音楽生成、感情分類、固有表現認識、機械翻訳

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230;損失関数 - リカレントニューラルネットワークの場合、時間軸全体での損失関数Lは、各時点での損失に基づき、次のように定義されます。

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230;時間軸での誤差逆伝播法 - 誤差逆伝播(バックプロパゲーション)が各時点で行われます。時刻 T における、重み行列 W に関する損失 L の導関数は以下のように表されます。

<br>


**24. Handling long term dependencies**

&#10230;長期依存関係の処理

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230;一般的に使用される活性化関数 - RNNモジュールで使用される最も一般的な活性化関数を以下に説明します。

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230;[シグモイド、Tanh、RELU]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230;勾配消失と勾配爆発について - 勾配消失と勾配爆発の現象は、RNNでよく見られます。これらの現象が起こる理由は、掛け算の勾配が層の数に対して指数関数的に減少/増加する可能性があるため、長期の依存関係を捉えるのが難しいからです。

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230;勾配クリッピング - 誤差逆伝播法を実行するときに時折発生する勾配爆発問題に対処するために使用される手法です。勾配の上限値を定義することで、実際にこの現象が抑制されます。

<br>


**29. clipped**

&#10230;クリップド

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230;ゲートの種類 - 勾配消失問題を解決するために、特定のゲートがいくつかのRNNで使用され、通常明確に定義された目的を持っています。それらは通常Γと記され、以下のように定義されます。

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230;ここで、W、U、bはゲート固有の係数、σはシグモイド関数です。主なものは以下の表にまとめられています。

<br>


**32. [Type of gate, Role, Used in]**

&#10230;[ゲートの種類、役割、下記で使用される]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230;[更新ゲート、関連ゲート、忘却ゲート、出力ゲート]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230;[過去情報はどのくらい重要ですか？、前の情報を削除しますか？、セルを消去しますか？しませんか？、セルをどのくらい見せますか？]

<br>


**35. [LSTM, GRU]**

&#10230;[LSTM GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230;GRU/LSTM - ゲート付きリカレントユニット（GRU）およびロングショートタームメモリユニット（LSTM）は、従来のRNNが直面した勾配消失問題を解決しようとします。LSTMはGRUを一般化したものです。以下は、各アーキテクチャを特徴づける式をまとめた表です。

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230;特徴づけ、ゲート付きリカレントユニット（GRU）、ロングショートタームメモリ（LSTM）、依存関係

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230;備考：記号*は2つのベクトル間の要素ごとの乗算を表します。

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230;RNNの変種 - 以下の表は、一般的に使用されている他のRNNアーキテクチャをまとめたものです。

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230;[双方向(BRNN)、ディープ(DRNN)] 

<br>


**41. Learning word representation**

&#10230;単語表現の学習

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230;この節では、Vを語彙、そして|V|を語彙のサイズとして定義します。

<br>


**43. Motivation and notations**

&#10230;動機と表記

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230;表現のテクニック - 単語を表現する2つの主な方法は、以下の表にまとめられています。

<br>


**45. [1-hot representation, Word embedding]**

&#10230;[1-hot表現、単語埋め込み]

<br>


**46. [teddy bear, book, soft]**

&#10230;テディベア、本、柔らかい

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230;[owと表記される、素朴なアプローチ、類似性情報なし、ewと表記される、単語の類似性を考慮に入れる]

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230;埋め込み行列 - 与えられた単語wに対して、埋め込み行列Eは、以下のように1-hot表現owを埋め込み行列ewに写像します。

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230;注：埋め込み行列は、ターゲット/コンテキスト尤度モデルを使用して学習できます。

<br>


**50. Word embeddings**

&#10230;単語の埋め込み

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230;Word2vec - Word2vecは、ある単語が他の単語の周辺にある可能性を推定することで、単語の埋め込みの重みを学習することを目的としたフレームワークです。人気のあるモデルは、スキップグラム、ネガティブサンプリング、およびCBOWです。

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230;[かわいいテディベアが読んでいる、テディベア、柔らかい、ペルシャ詩、芸術]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230;[代理タスクでのネットワークの訓練、高水準表現の抽出、単語埋め込み重みの計算]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230;スキップグラム - スキップグラムword2vecモデルは、あるコンテキスト単語を与え、ターゲット単語t の出現確率を計算することで単語の埋め込みを学習する教師付き学習タスクです。時点tと関係するパラメーターθtと表記すると、確率P(t|c) は下記のように表現されます。

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230;注意：softmax部分の分母全体の語彙全体を合計すると、モデルの計算コストは高くなります。 CBOWは、ある単語を予測するため周辺単語を使用する別のタイプのword2vecモデルです。

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230;ネガティブサンプリング -  k個のネガティブな例と1つのポジティブな例で訓練されたモデルで、ある与えられた文脈とターゲット単語の出現確率を評価するロジスティック回帰を使用するバイナリ分類器です。単語cとターゲット語tが与えられると、予測は次のように表現されます。

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230;注意：この計算コストは、スキップグラムモデルよりも少ないです。

<br>


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230;GloVe  -  GloVeモデルは、単語表現のためのグローバルベクトルの略で、共起行列Xを使用する単語の埋め込み手法です。ここで、各Xi、jは、ターゲットiがコンテキストjで発生した回数を表します。そのコスト関数Jは以下の通りです。

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230;ここで、fはXi、j =0⟹f（Xi、j）= 0となるような重み関数です。このモデルでeとθが果たす対称性を考えると、e（final）wが最後の単語の埋め込みになります。

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230;注意：学習された単語の埋め込みの個々の要素は、必ずしも関係性がある必要はないです。

<br>


**60. Comparing words**

&#10230;単語の比較

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230;コサイン類似度 - 単語w1とw2のコサイン類似度は次のように表されます。

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230;注意：θはワードw1とw2の間の角度です。

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230; t-SNE − t-SNE（ｔ−分布確率的近傍埋め込み）は、高次元埋め込みから低次元埋め込み空間への次元削減を目的とした技法です。実際には、2次元空間で単語ベクトルを視覚化するために使用されます。

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230;[文学、芸術、本、文化、詩、読書、知識、娯楽、愛らしい、幼年期、親切、テディベア、ソフト、抱擁、かわいい、愛らしい] 

<br>


**65. Language model**

&#10230;言語モデル

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230;概要 - 言語モデルは文の確率P(y)を推定することを目的としています。

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230;n-gramモデル - このモデルは、トレーニングデータでの出現数を数えることによって、コーパス表現の出現確率を定量化することを目的とした単純なアプローチです。

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230;パープレキシティ - 言語モデルは、一般的にPPとも呼ばれるパープレキシティメトリックを使用して評価されます。これは、ワード数Tにより正規化されたデータセットの確率の逆数と解釈できます。パープレキシティの数値はより低いものがより選択しやすい単語として評価されます(訳注:10であれば10個の中から1つ選択される、10000であれば10000個の中から1つ)、評価式は下記のようになります。

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230;備考：PPはt-SNEで一般的に使用されています。

<br>


**70. Machine translation**

&#10230;機械翻訳

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230;概要 - 機械翻訳モデルは、エンコーダーネットワークのロジックが最初に付加されている以外は、言語モデルと似ています。このため、条件付き言語モデルと呼ばれることもあります。目的は次のような文yを見つけることです。

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230;ビーム検索 - 入力xが与えられたとき最も可能性の高い文yを見つける、機械翻訳と音声認識で使用されるヒューリスティック探索アルゴリズムです。

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230;［ステップ１：単語y<1>と高い確率を持つ上位Ｂ個の組み合わせを見つける。ステップ２：条件付き確率y<k>|x,y<1>,...,y<k−1>を計算する。ステップ３：上位Ｂ個の組み合わせx,y<1>,...,y<k>を保持しながら、あるストップワードでプロセスを終了する]

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230;注意：ビーム幅が1に設定されている場合、これは単純な貪欲法と同等の結果を導きます。

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230;ビーム幅 - ビーム幅Bはビームサーチのパラメータです。 Bの値を大きくするとより良い結果が得られますが、探索パフォーマンスは低下し、メモリ使用量が増加します。 Bの値が小さいと結果が悪くなりますが、計算量は少なくなります。 Bの標準推奨値は10前後です。

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230;文章の長さの正規化 - 数値の安定性を向上させるために、ビームサーチは通常次のような正規化、特に対数尤度正規化された探索対象物に対して適用されます。

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230;注意：パラメーターαは緩衝パラメーターと見なされ、その値は通常0.5から1の間です。

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230;エラー分析 - 予測ˆyの翻訳が誤りである場合、その文の後に続く誤り分析を実行することで訳文y*がなぜ不正解であるかを理解することが可能です。

<br>


**79. [Case, Root cause, Remedies]**

&#10230;[症例、根本原因、改善策]

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230;[ビーム検索の誤り、RNNの誤り、ビーム幅の拡大、さまざまなアーキテクチャを試す、正規化、データをさらに取得] 

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230;Bleuスコア - バイリンガル正確性の代替評価（bleu）スコアは、n-gramの精度に基づき類似性スコアを計算することで、機械翻訳がどれほど優れているかを定量化します。以下のように定義されています。

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230;ここで、pnは、唯一定義されたn-gramでのbleuスコアです。定義は下記のようになります。

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230;注：人為的に水増しされたブルースコアを防ぐために、短い翻訳評価には簡潔なペナルティが適用される場合があります。

<br>


**84. Attention**

&#10230;アテンション

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230;アテンションモデル - このモデルはRNNが重要であると考えられる特定の入力部分に注目することで、モデルの実際の性能結果を向上させます。時点tにおける出力y<t>が、活性化関数a<t'>およびコンテキストc <t>に注目するとき、α<t、t'>はアテンション量と定義されます。式は次のようになります。

<br>


**86. with**

&#10230;ウェイト

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230;注：アテンションスコアは、一般的に画像のキャプション作成および機械翻訳で使用されています。*

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230;かわいいテディベアがペルシャ文学を読んでいます。

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230;アテンションの重み - 出力y<t>が活性化関数a<t'>で表現されるアテンションのウェイト量α<t,t>は、次のように計算されます。

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230;注意：この計算の複雑さはTxの２次関数です。

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230;ディープラーニングのチートシートが[対象言語]で利用可能になりました。

<br>

**92. Original authors**

&#10230;原作者

<br>

**93. Translated by X, Y and Z**

&#10230;X,YそしてZにより翻訳されました。

<br>

**94. Reviewed by X, Y and Z**

&#10230;X,YそしてZにより校正されました。

<br>

**95. View PDF version on GitHub**

&#10230;GitHubでPDF版を見る

<br>

**96. By X and Y**

&#10230;XそしてYによる。

<br>
