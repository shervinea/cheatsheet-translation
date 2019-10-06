**Recurrent Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230;
循环神经网络简明指南
<br>


**2. CS 230 - Deep Learning**

&#10230;
CS 230 - 深度学习
<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230;
[概述, 网络结构, RNN的应用, 损失函数, 反向传播]
<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230;
[处理长时间依赖性, 常见激活函数, 梯度消失/梯度爆炸, 梯度裁剪, GRU/LSTM, 门类型, 双向RNN, 深度RNN]
<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230;
[词表示学习, 注解, 嵌入矩阵, Word2vec, Skip-gram, 负采样, GloVe]
<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230;
[词比较, 余弦相似度, t-SNE]
<br>


**7. [Language model, n-gram, Perplexity]**

&#10230;
[语言模型, n-gram, 困惑度]
<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230;
[机器翻译, 集束搜索/束搜索, 长度归一化, 误差分析, Bleu分数]
<br>


**9. [Attention, Attention model, Attention weights]**

&#10230;
[注意力机制, 注意力模型, 注意力权重]
<br>


**10. Overview**

&#10230;
概述
<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230;
传统RNN的结构 - 循环神经网络（Recurrent Neural Networks,RNNs）, 是一类可以将之前的输出作为后续隐藏状态的输入的神经网络。通常可表示为以下形式：
<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230;
对于每一个时间步t,激活值a<t>和输出y<t>可表示如下：
<br>


**13. and**

&#10230;
并且
<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230;
其中Wax,Waa,Wya,ba是相关的系数矩阵, 在时间尺度上被整个网络共享；g1,g2是相关的激活函数。
<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230;
一个典型的RNN体系结构的优点和缺点可概括如下表：
<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230;
[优点, 可处理任何长度的输入, 模型大小不会随输入大小增加, 计算考虑历史信息, 权重在时间尺度上被整个网络共享]
<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230;
[缺点, 计算缓慢, 难以访问长时间的历史信息, 难以考虑未来时间步的输入对当前状态的影响]
<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230;
RNNs的应用 - RNN模型常用于自然语言处理和语音识别, 下表总结了RNN模型的不同应用场景：
<br>


**19. [Type of RNN, Illustration, Example]**

&#10230;
[RNN的类型, 图形表示, 示例]
<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230;
[一对一, 一对多, 多对一, 多对多]
<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230;
[传统神经网络, 音乐生成, 情感分类, 命名实体识别, 机器翻译]
<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230;
损失函数 - 在循环神经网络的情况下, 所有时间步长的损失函数L是基于每个时间步长的损失来定义的, 其表示如下：
<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230;
随时间反向传播算法(BPTT) - 反向传播在每个时间点完成。在时间步T, 损失函数L相对于权重矩阵W的导数表示如下：
<br>


**24. Handling long term dependencies**

&#10230;
解决长时间依赖问题
<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230;
常用的激活函数 - 在RNN模型中常用的激活函数如下所示：
<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230;
[Sigmoid, Tanh, RELU]
<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230;
梯度消失/梯度爆炸 - 梯度消失和梯度爆炸现象常出现在RNN模型中。其原因是该模型结构难以捕获长期依赖性, 因为乘法梯度会随着层数增加而呈指数递减/递增。
<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230;
梯度裁剪 - 该方法是用于解决进行反向传播时时而出现梯度爆炸问题的技术。通过限制梯度的最大值, 这种现象在实际中得到了相应的控制。
<br>

**29. clipped**

&#10230;
裁剪
<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230;
门类型 - 为了解决消失梯度问题, 在某些类型的RNN中使用了特定的门, 并且通常有明确的目的。它们通常被写为Γ：
<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230;
其中W,U,b是针对特定门的系数, σ是sigmoid激活函数。其主要的门类型可概括如下：
<br>


**32. [Type of gate, Role, Used in]**

&#10230;
[门类型, 角色, 被用于]
<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230;
[更新门, 关联门, 遗忘门, 输出门]
<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230;
[过去多久的信息对现在来说是重要的？, 是否丢失以前的信息？,是否擦除该单元？, 展示单元的多少信息？]
<br>


**35. [LSTM, GRU]**

&#10230;
[LSTM, GRU]
<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230;

<br>GRU/LSTM ― 门控循环单元(GRU)和长短时记忆单元(LSTM)可解决传统RNNs中遇到的梯度消失问题, 其中GRU是LSTM的一种推广。下表总结了每种结构的特性方程：


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230;

<br>特性, 门控循环单元(GRU), 长短时记忆网络(LSTM), 依赖项


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230;

<br>注：符号⋆表示两个向量之间的元素相乘。


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230;

<br>RNN模型的变种 - 下表列出了其他常用的RNN结构: 


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230;

<br>[双向RNN(Bidirectional RNN, BRNN), 深度RNN(Deep RNN, DRNN)]


**41. Learning word representation**

&#10230;

<br>词表示学习


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230;

<br>在本节中，我们用V来表示词汇，用|V|来表示词汇大小。


**43. Motivation and notations**

&#10230;

<br>动机和注解


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230;

<br>表示技术 - 两种主要的词表示方法的总结如下表所示：


**45. [1-hot representation, Word embedding]**

&#10230;

<br>[独热表示(one-hot), 词嵌入(word embedding)]


**46. [teddy bear, book, soft]**

&#10230;

<br>[泰迪熊, 书, 柔软的]


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230;

<br>[以ow表示, 朴素方法, 没有相似信息, 以ew表示, 考虑词汇之间的相似性]


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230;

<br>嵌入矩阵 - 对于给定的词汇w, 将该词汇的one-hot表示ow映射至词嵌入表示ew的嵌入矩阵E满足下式：


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230;

<br>注：使用目标/上下文似然模型可以学习嵌入矩阵。


**50. Word embeddings**

&#10230;

<br>词嵌入


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230;

<br>Word2vec ― Word2vec是一个旨在于通过估计给定词汇被其他词汇包围的可能性来学习词嵌入的框架。流行的模型包括skip-gram, 负采样和CBOW(Continuous Bag-of-Words Model)。


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230;

<br>[一只可爱的泰迪熊正在阅读, 泰迪熊, 柔软的, 波斯诗歌, 艺术]


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230;

<br>[通过代理任务训练网络, 提取高级表示, 计算词嵌入]


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230;

<br>Skip-gram ― skip-gram word2vec模型是一个通过评估任意给定目标词汇t与上下文词汇c一起出现的可能性来学习词嵌入的监督式学习框架。记与时间t相关联的参数为θt, 概率P(t|c)可写作：


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230;

<br>注：在softmax部分的分母中总计所有词汇使得模型的计算代价十分高昂。CBOW是另一个word2vec模型，其使用周围的单词来预测给定的单词。


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230;

<br>负采样 - 它是基于逻辑回归的二分类器集合，旨在于评估给定上下文和给定目标词是如何同时出现的，其中模型被训练在k个反例和1个正例的集合上。对于一个给定的上下文单词c和一个目标单词t，其预测可由以下表达式进行表示：


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230;

<br>注：该模型相比skip-gram模型而言，其计算代价更小。


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230;

<br>GloVe ― GloVe模型，是词表示的全局向量(global vectors for word representation)的简称, 是一种使用共现矩阵X的词嵌入技术，其中Xi,j表示的是目标词汇i与上下文j共同出现的次数。其代价函数J可写为：


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230;

<br>其中f是加权函数使得Xi,j=0⟹f(Xi,j)=0。考虑到e和θ在该模型中的对称性，最终嵌入的单词e(final)w由下式给出：


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230;

<br>注：所学单词的嵌入表示的各个部分不一定是可解释的。


**60. Comparing words**

&#10230;

<br>词比较


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230;

<br>余弦相似度 - 单词w1和w2之间的余弦相似度可表示如下：


**62. Remark: θ is the angle between words w1 and w2.**

&#10230;

<br>注：θ是词w1和w2之间的夹角。


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230;

<br>t-SNE ― 全称为t-distributed Stochastic Neighbor Embedding。t-SNE是一种将高维嵌入表示降维至低维空间的技术。实际上，其常用于将词向量在2D空间中的可视化。


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230;

<br>[文学，艺术，书籍，文化，诗歌，阅读，知识，娱乐，惹人爱的、童年、善良、泰迪熊、柔软、拥抱、可爱、讨人喜欢的。]


**65. Language model**

&#10230;

<br>语言模型


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230;

<br>概述 - 语言模型的目标在于估计句子的概率P(y)


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230;

<br>n-gram模型 - 该模型的思想很朴素，旨在通过计算一个词汇表达式(词汇组合)在训练数据中出现的次数来量化该表达式出现在语料库中的概率。


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230;

<br>困惑度-语言模型通常使用困惑度来进行度量，其也被称为PP，它可以被解释为利用词的数量进行归一化的数据集的逆概率。困惑度越低越好，其定义如下：


**69. Remark: PP is commonly used in t-SNE.**

&#10230;

<br>注：PP常用于t-SNE模型中。


**70. Machine translation**

&#10230;

<br>机器翻译


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230;

<br>概述 - 机器翻译模型与语言模型类似，只是其前面有一个编码器网络。因此，机器翻译模型有时被称为条件语言模型。该模型目标是找到一个句子y，以便：


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230;

<br>波束搜索 - 它是一种启发式搜索算法，用于机器翻译和语音识别，以找到给定输入x的最有可能的句子y。


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230;

<br>[第1步：寻找最相似的B个单词y<1>, 第2步：计算条件概率y<k>|x,y<1>,...,y<k−1>, 第3步：保持最相似的B个组合x,y<1>,...,y<k>,在停止词汇处结束进程]


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230;

<br>注：如果束宽设置为1,则其与朴素贪婪搜索等价。


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230;

<br>束宽 - 束宽B是束搜索的参数。B的值越大，搜索结果越好，但是其性能会变慢并且内存占用增加，B的值越小，搜索结果越差，但是计算代价小。B的标准值大约为10。


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230;

<br>长度归一化 - 为提高数值稳定性，束搜索常被应用于以下归一化目标，常称为归一化对数似然目标，定义如下：


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230;

<br>注：参数α可看做软化器，其值在0.5 ~ 1之间。


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230;

<br>误差分析 - 当所预测得到的翻译ˆy很差时，有人会想，为什么我们没有通过执行以下错误分析得到一个好的翻译y：


**79. [Case, Root cause, Remedies]**

&#10230;

<br>[具体情况、根本原因、补救措施]


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230;

<br>[波束搜索故障，RNN故障，增加波束宽度，尝试不同架构，正则化，获取更多数据] 


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230;

<br>bleu分数 ― 双语评估替补(bilingual evaluation understudy, bleu)分数通过基于n-gram精度计算相似度分数来量化机器翻译的好坏。其定义如下： 


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230;

<br>其中pn是n-gram上的bleu分数，定义如下：


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230;

<br>注：简洁的惩罚项可以应用于短预测翻译，以防止人为夸大bleu分数。


**84. Attention**

&#10230;

<br>注意力机制


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230;

<br>注意力模型 - 该模型允许RNN关注被认为是重要的输入的特定部分，从而提高了所得到的模型在实际中的性能。通过注意α<t,t′>输出上下文的时间t，我们得到： 


**86. with**

&#10230;

<br>和


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230;

<br>注：注意力分数常用于图像字幕和机器翻译。


**88. A cute teddy bear is reading Persian literature.**

&#10230;

<br>一只可爱的泰迪熊正在阅读波斯文学书。


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230;

<br>注意力权重 - 输出y<t>对激活量a<t′>的注意力程度(即注意力权重)由α<t,t′>给出，其计算如下：


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230;

<br>注：计算复杂度是Tx的平方。


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230;

<br>现已提供[目标语言]版本的深度学习简明指南。

**92. Original authors**

&#10230;

<br>原作者

**93. Translated by X, Y and Z**

&#10230;

<br>翻译自 X,Y和Z

**94. Reviewed by X, Y and Z**

&#10230;

<br>审阅自X,Y和Z

**95. View PDF version on GitHub**

&#10230;

<br>在Github上查看PDF版本

**96. By X and Y**

&#10230;

<br>由X和Y
