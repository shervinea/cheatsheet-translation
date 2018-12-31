**Recurrent Neural Networks translation**

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230; 1. 순환 신경망 치트시트

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - 딥 러닝

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230; [개요, 아키텍처 구조, RNN의 응용, 손실 함수, 역전파]

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230; [장기 의존성 처리, 일반적인 활성 함수, 그래디언트 소실/폭발, 그래디언트 클리핑, GRU/LSTM, 게이트 형식, 양방향 RNN, 심층 RNN]

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230; [학습 단어 표현, 표기법, 임베딩 매트릭스, Word2vec, 스킵-그램, 네거티브 샘플링, 글로브]

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230; [단어 비교, 코사인 유사도, t-SNE]

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230; [언어 모델, n-그램, 퍼플렉시티]

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230; [기계 번역, 빔 서치, 길이 정규화, 오류 분석, 블루 점수]

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230; [주의, 주의 모델, 주의 가중치]

<br>


**10. Overview**

&#10230; 개요

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230; 전통적인 RNN의 아키텍처 - RNNs이라고 알려진 순환 신경망은 이전의 출력을 은닉층에서 입력으로 사용하게 하는 신경망의 한 종류입니다. 이는 일반적으로 다음과 같습니다:

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230; 각 시점 t에 대해 활성화 a<t>와 출력 y<t>는 다음과 같이 표현됩니다.

<br>


**13. and**

&#10230; 와

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230; 여기서 Wax, Waa, Wya, ba, by는 시간적으로 공유되는 계수이고, g1, g2는 활성화 함수입니다.

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230; 일반적인 RNN 아키텍처의 장단점은 아래 표에 요약되어 있습니다:

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230; [장점, 모든 길이의 입력을 처리할 수 있는 가능성, 입력 크기에 따라 모델 크기가 증가하지 않음, 계산 시 이력 정보가 고려됨, 시간에 따라 가중치가 공유됨]

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230; [단점, 계산 속도가 느림, 오래 전 정보에 대한 접근이 어려움, 현재 상태에 대한 향후 입력을 고려할 수 없음]

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230; RNN의 응용 - RNN 모델은 자연어 처리 및 음성 인식 분야에서 주로 사용됩니다. 다양한 응용 프로그램이 아래표에 요약되어 있습니다:

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230; [RNN의 유형, 일러스트레이션, 예제]

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230; [일-대-일, 일-대-다, 다-대-일, 다-대-다]

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230; [전통적인 신경망, 음악 생성, 감성 분류, 이름 엔터티 인식, 기계 번역]

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230; 손실 함수 - 순환 신경망의 경우, 모든 시점의 손실 함수 L은 다음과 같이 매 시점의 손실을 기준으로 정의됩니다:

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230; 시간에 따른 역전파 - 역전파는 각 시점에 수행됩니다. T시점에서, 가중치 행렬 W에 대한 손실 L의 도함수는 다음과 같이 표현됩니다:

<br>


**24. Handling long term dependencies**

&#10230; 장기 의존성 처리

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230; 일반적으로 사용되는 활성화 함수 - RNN 모듈에서 사용되는 가장 일반적인 활성화 함수는 다음과 같습니다:

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230; [시그모이드, 하이퍼볼릭탄젠트, 렐루]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230; 그래디언트 소실/폭발 - 그래디언트가 소실되거나 폭발하는 현상은 RNNs에서 종종 발생합니다. 이와 같은 현상들이 발생하는 이유는 층의 수에 따라 기하 급수적으로 감소하거나 증가할 수 있는 곱셈 그래디언트로 인해 장기 종속성을 포착하기가 어렵기 때문입니다.

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230; 그래디언트 클리핑 - 역전파를 수행할 때 종종 마주치는 그래디언트 폭발 문제를 처리하기 위한 테크닉입니다. 그래디언트의 최대값을 캡핑하면 이 현상이 실제로 제어가 됩니다.

<br>


**29. clipped**

&#10230; 클립

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230; 게이트 유형 - 그래디언트 소실 문제를 해결하기 위해 특정 게이트가 일부 유형의 RNN에서 사용되고 일반적으로 잘 정의된 목적을 가지고 있습니다. 보통 Γ로 표시되며 다음과 같습니다:

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230; 여기서 W, U, b는 게이트에 고유한 계수이고 σ는 시그모이드 함수입니다. 주요 내용은 아래 표에 요약되어 있습니다:

<br>


**32. [Type of gate, Role, Used in]**

&#10230; [게이트 유형, 역할, 사용]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230; [업데이트 게이트, 관련도 게이트, 게이트 망각, 게이트 출력]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230; [과거가 현재 얼마나 중요한가?, 이전 정보를 버릴 것인가?, 셀을 지울 것인가 말 것인가?, 셀을 얼만큼 공개할 것인가?]

<br>


**35. [LSTM, GRU]**

&#10230; [LSTM, GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230; GRU/LSTM - Gated Recurrent Unit (GRU) 및 Long Short-Term Memory units (LSTM)은 전통적인 RNN에서 발생하는 그래디언트 소실 문제를 처리합니다. LSTM은 GRU의 일반화된 형태입니다. 다음은 각 아키텍처의 특성화 방정식을 요약 한 표입니다.

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230; [특성화, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), 종속성]

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230; 비고: * 기호는 두 벡터 사이의 원소 단위의 곱셈을 나타냅니다.

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230; RNN의 변형 - 아래 표는 일반적으로 사용되는 다른 RNN 아키텍처를 요약한 것입니다:

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230; [양방향 RNN (BRNN), 심층 RNN (DRNN)]

<br>


**41. Learning word representation**

&#10230; 단어 표현 학습

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230; 이 절에서, 우리는 어휘는 V, 차수는 |V|로 표기합니다.

<br>


**43. Motivation and notations**

&#10230; 동기 부여 및 표기법

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230; 표현 기법 - 단어를 표현하는 두 가지 주요 방법이 아래 표에 요약되어 있습니다:

<br>


**45. [1-hot representation, Word embedding]**

&#10230; [원-핫 표현, 단어 임베딩]

<br>


**46. [teddy bear, book, soft]**

&#10230; [테디 베어(teddy bear), 책(book), soft(부드러운)]

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230; [ow 표기, 나이브 접근법, 유사도 정보 없음, ew 표기, 단어 유사도 고려]

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230; 임베딩 매트릭스 - 주어진 단어 w에 대해 임베딩 매트릭스 E는 다음과 같이 임베디드 ew에 원-핫 표시를 매핑하는 매트릭스입니다:

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230; 비고: 임베딩 매트릭스 학습은 목표/상황 가능도 모델을 사용하여 수행 할 수 있습니다.

<br>


**50. Word embeddings**

&#10230; 단어 임베딩

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230; Word2vec - Word2vec는 주어진 단어가 다른 단어로 둘러싸여 있을 가능성을 추정하여 단어 임베딩 학습을 목표로 하는 프레임워크입니다. 인기있는 모델에는 스킵-그램(skip-gram), 네거티브 샘플링(negative sampling) 그리고 CBOW가 있습니다.

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230; [귀여운 테디 베어는 독서 중, 테디 베어, 부드러운, 페르시안 시, 예술]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230; [프록시 작업에 대한 네트워크 학습, 고급 표현 추출, 단어 임베딩 계산]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230; 스킵-그램(Skip-gram) - 

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230;

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230;

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230;

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230;

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230;

<br>


**60. Comparing words**

&#10230;

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230;

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230;

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230;

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230;

<br>


**65. Language model**

&#10230;

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230;

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230;

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230;

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230;

<br>


**70. Machine translation**

&#10230;

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230;

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230;

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230;

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230;

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230;

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230;

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230;

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230;

<br>


**79. [Case, Root cause, Remedies]**

&#10230;

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230;

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230;

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230;

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230;

<br>


**84. Attention**

&#10230;

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230;

<br>


**86. with**

&#10230;

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230;

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230;

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230;

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230;

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230;

<br>

**92. Original authors**

&#10230;

<br>

**93. Translated by X, Y and Z**

&#10230;

<br>

**94. Reviewed by X, Y and Z**

&#10230;

<br>

**95. View PDF version on GitHub**

&#10230;

<br>

**96. By X and Y**

&#10230;

<br>
