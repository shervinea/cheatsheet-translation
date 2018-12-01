**Recurrent Neural Networks translation**


**1.** Recurrent Neural Networks cheatsheet

 &#10230;


**2.** CS 230 - Deep Learning

 &#10230;


**3.** [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]

 &#10230;


**4.** [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]

 &#10230;


**5.** [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]

 &#10230;


**6.** [Comparing words, Cosine similarity, t-SNE]

 &#10230;


**7.** [Language model, n-gram, Perplexity]

 &#10230;


**8.** [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]

 &#10230;


**9.** [Attention, Attention model, Attention weights]

 &#10230;


**10.** Overview

 &#10230;


**11.** Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:

 &#10230;


**12.** For each timestep t, the activation a<t> and the output y<t> are expressed as follows:

 &#10230;


**13.** and

 &#10230;


**14.** where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.

 &#10230;


**15.** The pros and cons of a typical RNN architecture are summed up in the table below:

 &#10230;


**16.** [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]

 &#10230;


**17.** [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]

 &#10230;


**18.** Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:

 &#10230;


**19.** [Type of RNN, Illustration, Example]

 &#10230;


**20.** [One-to-one, One-to-many, Many-to-one, Many-to-many]

 &#10230;


**21.** [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]

 &#10230;


**22.** Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:

 &#10230;


**23.** Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:

 &#10230;


**24.** Handling long term dependencies

 &#10230;


**25.** Commonly used activation functions ― The most common activation functions used in RNN modules are described below:

 &#10230;


**26.** [Sigmoid, Tanh, RELU]

 &#10230;


**27.** Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.

 &#10230;


**28.** Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.

 &#10230;


**29.** clipped

 &#10230;


**30.** Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:

 &#10230;


**31.** where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:

 &#10230;


**32.** [Type of gate, Role, Used in]

 &#10230;


**33.** [Update gate, Relevance gate, Forget gate, Output gate]

 &#10230;


**34.** [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]

 &#10230;


**35.** [LSTM, GRU]

 &#10230;


**36.** GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:

 &#10230;


**37.** [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]

 &#10230;


**38.** Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.

 &#10230;


**39.** Variants of RNNs ― The table below sums up the other commonly used RNN architectures:

 &#10230;


**40.** [Bidirectional (BRNN), Deep (DRNN)]

 &#10230;


**41.** Learning word representation

 &#10230;


**42.** In this section, we note V the vocabulary and |V| its size.

 &#10230;


**43.** Motivation and notations

 &#10230;


**44.** Representation techniques ― The two main ways of representing words are summed up in the table below:

 &#10230;


**45.** [1-hot representation, Word embedding]

 &#10230;


**46.** [teddy bear, book, soft]

 &#10230;


**47.** [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]

 &#10230;


**48.** Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:

 &#10230;


**49.** Remark: learning the embedding matrix can be done using target/context likelihood models.

 &#10230;


**50.** Word embeddings

 &#10230;


**51.** Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.

 &#10230;


**52.** [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]

 &#10230;


**53.** [Train network on proxy task, Extract high-level representation, Compute word embeddings]

 &#10230;


**54.** Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:

 &#10230;


**55.** Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.

 &#10230;


**56.** Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:

 &#10230;


**57.** Remark: this method is less computationally expensive than the skip-gram model.

 &#10230;


**58.** where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:

 &#10230;


**59.** Remark: the individual components of the learned word embeddings are not necessarily interpretable.

 &#10230;


**60.** Comparing words

 &#10230;


**61.** Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:

 &#10230;


**62.** Remark: θ is the angle between words w1 and w2.

 &#10230;


**63.** t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.

 &#10230;


**64.** [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]

 &#10230;


**65.** Language model

 &#10230;


**66.** Overview ― A language model aims at estimating the probability of a sentence P(y).

 &#10230;


**67.** n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.

 &#10230;


**68.** Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:

 &#10230;


**69.** Remark: PP is commonly used in t-SNE.

 &#10230;


**70.** Machine translation

 &#10230;


**71.** Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:

 &#10230;


**72.** Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.

 &#10230;


**73.** [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]

 &#10230;


**74.** Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.

 &#10230;


**75.** Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.

 &#10230;


**76.** Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:

 &#10230;


**77.** Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.

 &#10230;


**78.** Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:

 &#10230;


**79.** [Case, Root cause, Remedies]

 &#10230;


**80.** [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]

 &#10230;


**81.** Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:

 &#10230;


**82.** where pn is the bleu score on n-gram only defined as follows:

 &#10230;


**83.** Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.

 &#10230;


**84.** Attention

 &#10230;


**85.** Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:

 &#10230;


**86.** with

 &#10230;


**87.** Remark: the attention scores are commonly used in image captioning and machine translation.

 &#10230;


**88.** A cute teddy bear is reading Persian literature.

 &#10230;


**89.** Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:

 &#10230;


**90.** Remark: computation complexity is quadratic with respect to Tx.

 &#10230;


**91.** The Deep Learning cheatsheets are now available in [target language].

&#10230;


**92.** Original authors

&#10230;


**93.** Translated by X, Y and Z

&#10230;


**94.** Reviewed by X, Y and Z

&#10230;


**95.** View PDF version on GitHub

&#10230;


**96.** By X and Y

&#10230;
