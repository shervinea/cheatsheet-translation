**Recurrent Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230; Шпаргалка по Рекуррентным Нейронным Сетям

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Глубокое Обучение

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230; [Обзор, Структура архитектуры, Приложения RNN, Функция потерь, Обратное распространение ошибки]

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230; [Обработка долгосрочных зависимостей, Общие функции активации, Исчезающий/увеличивающийся градиент, Отсечение градиента, GRU/LSTM, Типы вентилей, Двунаправленный RNN, Глубокая RNN.]

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230; [Обучение представления слов, Обозначения, Embedding matrix, Word2vec, Skip-gram, Отрицательная выборка, GloVe]

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230; [Сравнение слов, Косинусное сходство, t-SNE]

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230; [Языковая модель, n-граммы, Метрика Perplexity]

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230; [Машинный перевод, Лучевой поиск, Нормализация длины, Анализ ошибок, Метрика BLEU]

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230; [Внимание, Модель внимания, Веса внимания]

<br>


**10. Overview**

&#10230; Обзор

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230; Архитектура классической RNN ― Рекуррентные нейронные сети, также известные как RNN, представляют собой класс нейронных сетей, которые позволяют использовать предыдущие выходы в качестве входов, имея скрытые состояния. Обычно они следующие:

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230; Для каждого временного шага t активация a<t> и выход y<t> выражаются следующим образом:

<br>


**13. and**

&#10230; и

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230; где Wax,Waa,Wya,ba,by являются коэффициентами, которые являются одинаковыми для всех ячеек RNN во времени, и функциями активации g1,g2.

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230; Плюсы и минусы типичной архитектуры RNN перечислены в таблице ниже:

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230; [Преимущества, Возможность обработки входа любой длины, Размер модели не увеличивается с размером входных данных, При расчетах учитывается историческая информация, Веса одинаковы во времени]

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230; [Недостатки, Вычисления идут медленно, Сложность доступа к очень давней информации, Невозможно рассмотреть какие-либо будущие входные данные для текущего состояния]

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230; Применение RNN ― Модели RNN в основном используются в области обработки естественного языка и распознавания речи. Различные приложения приведены в таблице ниже:

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230; [Тип RNN, Иллюстрация, Пример]

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230; [One-to-one, One-to-many, Many-to-one, Many-to-many]

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230; [Классическая нейронная сеть, Генерация музыки, Определение эмоциональной окраски, Распознавание именованных сущностей, Машинный перевод]

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230; Функция потерь ― В случае рекуррентной нейронной сети функция потерь L всех временных шагов определяется на основе значений функции потерь на каждом временном шаге следующим образом:

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230; Обратное распространение ошибки во времени ― Обратное распространение ошибки выполняется в каждый момент времени. На временном шаге T производная потерь L по матрице весов W выражается следующим образом:

<br>


**24. Handling long term dependencies**

&#10230; Работа с долгосрочными зависимостями

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230; Часто используемые функции активации ― Наиболее распространенные функции активации, используемые в модулях RNN, описаны ниже:

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230; [Sigmoid, Tanh, RELU]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230; Исчезающий/взрывающийся градиент ― Явления исчезающих и взрывающихся градиентов часто встречаются в контексте RNN. Причина, по которой они происходят, заключается в том, что трудно зафиксировать долгосрочные зависимости из-за мультипликативного градиента, который может экспоненциально уменьшаться/увеличиваться по отношению к количеству слоев.

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230; Отсечение градиента ― это метод, используемый для решения проблемы взрывающихся градиентов, иногда возникающей при выполнении обратного распространения ошибки. Ограничивая градиент по выбранному максимальному значению, это явление можно контролировать на практике.

<br>


**29. clipped**

&#10230; отсечено

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230; Типы вентилей ― чтобы решить проблему исчезающего градиента, в некоторых типах RNN используются специфичные вентили, которые имеют четко заданную цель. Обычно они обозначаются Γ и равны:

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230; где W,U,b ― коэффициенты, относящиеся к вентилю, а σ - функция сигмоиды. Основные из них приведены в таблице ниже:

<br>


**32. [Type of gate, Role, Used in]**

&#10230; [Тип вентиля, Роль, Используется в]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230; [Вентиль обновления (Update), Вентиль сброса (Relevance), Вентиль забывания (Forget), Вентиль выхода (Output)]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230; [Насколько прошлое должно иметь значение сейчас?, Отбросить предыдущую информацию?, Стереть ячейку или нет?, Насколько раскрыть ячейку?]

<br>


**35. [LSTM, GRU]**

&#10230; [LSTM, GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230; GRU/LSTM ― Вентильный Рекуррентный Блок (Gated Recurrent Unit, GRU) и Блок с Долгой Краткосрочной Памятью (Long Short-Term Memory units, LSTM) имеет дело с проблемой исчезающего градиента, с которой сталкиваются традиционные RNN, причем LSTM является обобщением GRU. Ниже представлена таблица, в которой перечислены характеризующие уравнения каждой архитектуры:

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230; [Характеристика, Вентильный Рекуррентный Блок (GRU), Блок с Долгой Краткосрочной Памятью (LSTM), Зависимости]

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230; Примечание: знак ⋆ означает поэлементное умножение двух векторов.

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230; Варианты RNN ― В таблице ниже перечислены другие часто используемые архитектуры RNN:

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230; [Двунаправленная (Bidirectional/BRNN), Глубокая (Deep/DRNN)]

<br>


**41. Learning word representation**

&#10230; Получение представления слов

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230; В этом разделе мы обозначаем словарь V и его размер |V|.

<br>


**43. Motivation and notations**

&#10230; Мотивация и обозначения

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230; Методы представления ― два основных способа представления слов подытожены в таблице ниже:

<br>


**45. [1-hot representation, Word embedding]**

&#10230; [One-hot представление, Представление слова]

<br>


**46. [teddy bear, book, soft]**

&#10230; [плюшевый мишка, книжка, мягкий]

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230; [Обозначено ow, Наивный подход, нет информации о сходстве, Обозначено ew, Учитывает сходство слов]

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230; Матрица представлений слов (embedding matrix) ― для данного слова w матрица представлений E является матрицей, которая отображает его one-hot представление ow на его представление ew следующим образом:

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230; Примечание: получить матрицу представлений можно путем обучения моделей целевого/контекстного правдоподобия.

<br>


**50. Word embeddings**

&#10230; Векторные представления слов

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230; Word2vec ― это модель, предназначенная для получения представлений слов путем оценки вероятности того, что конкретное слово окружено другими словами. Популярные модели включают Skip-gram, Negative sampling и CBOW.

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230; [Милый плюшевый мишка читает, плюшевый мишка, мягкий, персидская поэзия, искусство]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230; [Обучить сеть на вспомогательной задаче, Извлечь высокоуровневое представление, Вычислить представления слов]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230; Skip-gram ― Модель skip-gram word2vec - это алгоритм обучения с учителем, который выучивает представления слов, оценивая правдоподобие того, что любое заданное целевое слово t встречается с контекстным словом c. Обозначим параметр, связанный с t, как θt, тогда вероятность P(t|c) определяется выражением:

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230; Примечание: суммирование по всему словарю в знаменателе части softmax делает эту модель дорогостоящей в вычислительном отношении. CBOW - это еще одна модель word2vec, использующая окружающие слова для предсказания данного слова.

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230; Negative sampling ― это набор бинарных классификаторов, использующих логистические регрессии, целью которых является оценка того, как данный контекст и заданные целевые слова могут появляться одновременно, при этом модели обучаются на наборах из k отрицательных примеров и 1 положительного примера. Учитывая контекстное слово c и целевое слово t, прогноз выражается следующим образом:

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230; Примечание: этот метод менее затратен с точки зрения вычислений, чем модель Skip-gram.

<br>


**57 GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230; GloVe ― Модель GloVe, сокращение от глобальных векторов для представления слов, является методом получения представлений слов, который использует матрицу совпадения X, где каждый Xi,j обозначает количество раз, когда цель i встречалась с контекстом j. Его функция стоимости J выглядит следующим образом:

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0. Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230; где f - такая взвешивающая функция, что Xi,j=0⟹f(Xi,j)=0. Учитывая симметрию, которую играют e и θ в этой модели, последнее представление слов e(final)w задается выражением:

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230; Примечание: отдельные компоненты векторов представлений слов не обязательно поддаются интерпретации.

<br>


**60. Comparing words**

&#10230; Сравнение слов

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230; Косинусное сходство ― косинусное сходство между словами w1 и w2 выражается следующим образом:

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230; Примечание: θ - угол между словами w1 и w2.

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230; t-SNE ― t-распределенное стохастическое соседнее представление (t-distributed Stochastic Neighbor Embedding, t-SNE) - это метод, направленный на сокращение высоко-размерного пространства представлений до пространства меньшей размерности. На практике он обычно используется для визуализации векторов слов в 2D-пространстве.

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230; [литература, искусство, книга, культура, стихотворение, чтение, знания, развлекательный, милый, детство, добрый, плюшевый мишка, мягкий, обнять, милый, очаровательный]

<br>


**65. Language model**

&#10230; Языковая модель

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230; Обзор ― языковая модель предназначена для оценки вероятности предложения P(y).

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230; Модель n-gram ― эта модель представляет собой наивный подход, направленный на количественную оценку вероятности того, что выражение появляется в корпусе, путем подсчета его количества появлений в обучающих данных.

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230; Метрика Perplexity ― языковые модели обычно оцениваются с помощью метрики perplexity, также известной как PP, которую можно интерпретировать как обратную вероятность набора данных, нормализованную на количество слов T. Perplexity таково, что чем оно ниже, тем лучше, и определяется следующим образом:

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230; Примечание: PP обычно используется в t-SNE.

<br>


**70. Machine translation**

&#10230; Машинный перевод

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230; Обзор ― модель машинного перевода похожа на языковую модель, за исключением того, что перед ней размещена сеть кодировщика. По этой причине её иногда называют условной языковой моделью. Цель состоит в том, чтобы найти такое предложение y, что:

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230; Лучевой поиск ― это алгоритм эвристического поиска, используемый в машинном переводе и распознавании речи для поиска наиболее вероятного предложения y при вводе x.

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230; [Шаг 1: Найти top B наиболее вероятных слов y<1>, Шаг 2: Вычислить условные вероятности y<k>|x,y<1>,...,y<k−1>, Шаг 3: Сохранить top B комбинации x,y<1>,...,y<k>, Завершить процесс на стоп-слове]

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230; Примечание: если ширина луча установлена на 1, то это равносильно наивному жадному поиску.

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230; Ширина луча ― Ширина луча B является параметром лучевого поиска. Большие значения B дают лучший результат, но с меньшим качеством предсказаний и увеличенным объёмом памяти. Маленькие значения B приводят к худшим результатам, но требуют меньших вычислительных затрат. Стандартное значение B составляет около 10.

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230; Нормализация длины ― Чтобы улучшить численную стабильность, лучевой поиск обычно применяется к следующей нормализованной цели, часто называемой нормализованной целью логарифмического правдоподобия, определяемой как:

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230; Примечание: параметр α можно рассматривать как смягчитель, и его значение обычно составляет от 0.5 до 1.

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230; Анализ ошибок ― При получении предсказанного перевода ˆy, который является плохим, можно задаться вопросом, почему мы не получили хороший перевод y∗ , выполнив следующий анализ ошибок:

<br>


**79. [Case, Root cause, Remedies]**

&#10230; [Случай, Первопричина, Исправления]

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230; [Ошибка лучевого поиска, Неисправность RNN, Увеличить ширину луча, Попробовать другую архитектуру, Регуляризировать, Взять больше данных]

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230; Метрика BLEU ― bilingual evaluation understudy количественно определяет, насколько хорош машинный перевод, путем вычисления оценки сходства на основе точности n-грамм. Это определяется следующим образом:

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230; где pn - это оценка по n-граммам, определяемая следующим образом:

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230; Примечание: к коротким предсказанным переводам может применяться штраф за краткость, чтобы предотвратить искусственно завышенную оценку BLEU.

<br>


**84. Attention**

&#10230; Внимание

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230; Модель внимания ― эта модель позволяет RNN обращать внимание на определенные части входных данных, которые считаются важными, что на практике улучшает качество предсказаний полученной модели. Обозначим α<t,t′> количество внимания, которое выход y<t> должен уделять активации a<t′> и c<t> контексту в момент времени t, у нас есть:

<br>


**86. with**

&#10230; с

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230; Примечание: оценки внимания обычно используются в задачах добавления субтитров к изображениям и машинном переводе.

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230; Милый плюшевый мишка читает персидскую литературу.

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230; Вес внимания ― количество внимания, которое выход y<t> должен уделять активации a<t′>, задается выражением α<t,t′>, вычисляемым следующим образом:

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230; Примечание: сложность вычислений квадратична относительно Tx.

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Шпаргалки по глубокому обучению теперь доступны на русском языке.

<br>

**92. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>

**93. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>

**94. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>

**95. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>

**96. By X and Y**

&#10230; По X и Y

<br>
