**Recurrent Neural Networks translation**

<br>

**1. Recurrent Neural Networks cheatsheet**

&#10230; Pense-bête de réseaux de neurones récurrents

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Apprentissage profond

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

&#10230; [Vue d'ensemble, Structure d'architecture, Applications des RNNs, Fonction de loss, Backpropagation]

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

&#10230; [Dépendances à long terme, Fonctions d'activation communes, Gradient qui disparait/explose, Coupure de gradient, GRU/LSTM, Types de porte, RNN bi-directionnel, RNN profond]

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

&#10230; [Apprentissage de la représentation de mots, Notations, Matrice de représentation, Word2vec, Skip-gram, Échantillonnage négatif, GloVe]

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

&#10230; [Comparaison des mots, Similarité cosinus, t-SNE]

<br>


**7. [Language model, n-gram, Perplexity]**

&#10230; [Modèle de langage, n-gram, Perplexité]

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

&#10230; [Traduction machine, Recherche en faisceau, Normalisation de longueur, Analyse d'erreur, Score bleu]

<br>


**9. [Attention, Attention model, Attention weights]**

&#10230; [Attention, Modèle d'attention, Coefficients d'attention]

<br>


**10. Overview**

&#10230; Vue d'ensemble

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

&#10230; Architecture d'un RNN traditionnel ― Les réseaux de neurones récurrents (en anglais <i>recurrent neural networks</i>), aussi appelés RNNs, sont une classe de réseaux de neurones qui permettent aux prédictions antérieures d'être utilisées comme entrées, par le bias d'états cachés (en anglais <i>hidden states</i>). Ils sont de la forme suivante :

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

&#10230; À l'instant t, l'activation a<t> et la sortie y<t> sont de la forme suivante :

<br>


**13. and**

&#10230; et

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

&#10230; où Wax,Waa,Wya,ba,by sont des coefficients indépendents du temps et où g1,g2 sont des fonctions d'activation.

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

&#10230; Les avantages et inconvénients des architectures de RNN traditionnelles sont résumés dans le tableau ci-dessous :

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

&#10230; [Avantages, Possibilité de prendre en compte des entrées de toute taille, La taille du modèle n'augmente pas avec la taille de l'entrée, Les calculs prennent en compte les informations antérieures, Les coefficients sont indépendents du temps]

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

&#10230; [Inconvénients, Le temps de calcul est long, Difficulté d'accéder à des informations d'un passé lointain, Impossibilité de prendre en compte des informations futures un état donné]

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

&#10230; Applications des RNNs ― Les modèles RNN sont surtout utilisés dans les domaines du traitement automatique du langage naturel et de la reconnaissance vocale. Le tableau suivant détaille les applications principales à retenir :

<br>


**19. [Type of RNN, Illustration, Example]**

&#10230; [Type de RNN, Illustration, Exemple]

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

&#10230; [Un à un, Un à plusieurs, Plusieurs à un, Plusieurs à plusieurs]

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

&#10230; [Réseau de neurones traditionnel, Géneration de musique, Classification de sentiment, Reconnaissance d'entité, Traduction machine]

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

&#10230; Fonction de loss ― Dans le contexte des réseaux de neurones récurrents, la fonction de loss L prend en compte le loss à chaque temps T de la manière suivante :

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

&#10230; Backpropagation temporelle ― L'étape de backpropagation est appliquée dans la dimension temporelle. À l'instant T, la dérivée du loss L par rapport à la matrice de coefficients W est donnée par :

<br>


**24. Handling long term dependencies**

&#10230; Dépendances à long terme

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

&#10230; Fonctions d'activation communément utilisées ― Les fonctions d'activation les plus utilisées dans les RNNs sont décrits ci-dessous :

<br>


**26. [Sigmoid, Tanh, RELU]**

&#10230; [Sigmoïde, Tanh, RELU]

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

&#10230; Gradient qui disparait/explose ― Les phénomènes de gradient qui disparait et qui explose (en anglais <i>vanishing gradient</i> et <i>exploding gradient</i>) sont souvent rencontrés dans le contexte des RNNs. Ceci est dû au fait qu'il est difficile de capturer des dépendances à long terme à cause du gradient multiplicatif qui peut décroître/croître de manière exponentielle en fonction du nombre de couches.

<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

&#10230; Coupure de gradient ― Cette technique est utilisée pour atténuer le phénomène de gradient qui explose qui peut être rencontré lors de l'étape de backpropagation. En plafonnant la valeur qui peut être prise par le gradient, ce phénomène est maîtrisé en pratique.

<br>


**29. clipped**

&#10230; coupé

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

&#10230; Types de porte ― Pour remédier au problème du gradient qui disparait, certains types de porte sont spécifiquement utilisés dans des variantes de RNNs et ont un but bien défini. Les portes sont souvent notées Γ et sont telles que :

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

&#10230; où W,U,b sont des coefficients spécifiques à la porte et σ est une sigmoïde. Les portes à retenir sont récapitulées dans le tableau ci-dessous :

<br>


**32. [Type of gate, Role, Used in]**

&#10230; [Type de porte, Rôle, Utilisée dans]

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

&#10230; [Porte d'actualisation, Porte de pertinence, Porte d'oubli, Porte de sortie]

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

&#10230; [Dans quelle mesure le passé devrait être important ?, Enlever les informations précédentes ?, Enlever une cellule ?, Combien devrait-on révéler d'une cellule ?]

<br>


**35. [LSTM, GRU]**

&#10230; [LSTM, GRU]

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

&#10230; GRU/LSTM ― Les unités de porte récurrente (en anglais <i>Gated Recurrent Unit</i>) (GRU) et les unités de mémoire à long/court terme (en anglais <i>Long Short-Term Memory units</i>) (LSTM) appaisent le problème du gradient qui disparait rencontré par les RNNs traditionnels, où le LSTM peut être vu comme étant une généralisation du GRU. Le tableau ci-dessous résume les équations caractéristiques de chacune de ces architectures :

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

&#10230; [Caractérisation, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dépendances]

<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

&#10230; Remarque : le signe ⋆ dénote le produit de Hadamard entre deux vecteurs.

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

&#10230; Variantes des RNNs ― Le tableau ci-dessous récapitule les autres architectures RNN commumément utilisées :

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

&#10230; [Bi-directionnel (BRNN), Profond (DRNN)]

<br>


**41. Learning word representation**

&#10230; Apprentissage de la représentation de mots

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

&#10230; Dans cette section, on note V le vocabulaire et |V| sa taille.

<br>


**43. Motivation and notations**

&#10230; Motivation et notations

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

&#10230; Techniques de représentation ― Les deux manières principales de représenter des mots sont décrits dans le tableau suivant :

<br>


**45. [1-hot representation, Word embedding]**

&#10230; [Représentation binaire, Représentation du mot]

<br>


**46. [teddy bear, book, soft]**

&#10230; [ours en peluche, livre, doux]

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

&#10230; [Noté ow, Approche naïve, pas d'information de similarité, Noté ew, Prend en compte la similarité des mots]

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

&#10230; Matrice de représentation ― Pour un mot donné w, la matrice de représentation (en anglais <i>embedding matrix</i>) E est une matrice qui relie une représentation binaire ow à sa représentation correspondante ew de la manière suivante :

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

&#10230; Remarque : l'apprentissage d'une matrice de représentation peut être effectuée en utilisant des modèles probabilistiques de cible/contexte.

<br>


**50. Word embeddings**

&#10230; Représentation de mots

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

&#10230; Word2vec ― Word2vec est un ensemble de techniques visant à apprendre comment représenter les mots en estimant la probabilité qu'un mot donné a d'être entouré par d'autres mots. Le skip-gram, l'échantillonnage négatif et le CBOW font parti des modèles les plus populaires.

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

&#10230; [Un ours en peluche mignon est en train de lire, ours en peluche, doux, poésie persane, art]

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

&#10230; [Entraîner le réseau, Extraire une représentation globale, Calculer une représentation des mots]

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

&#10230; Skip-gram ― Le skip-gram est un modèle de type supervisé qui apprend comment représenter les mots en évaluant la probabilité de chaque mot cible t donné dans un mot contexte c. En notant θt le paramètre associé à t, la probabilité P(t|c) est donnée par :

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

&#10230; Remarque : le fait d'additionner tout le vocabulaire dans le dénominateur du softmax rend le modèle coûteux en temps de calcul. CBOW est un autre modèle utilisant les mots avoisinants pour prédire un mot donné.

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

&#10230; Échantillonnage négatif ― Cette méthode utilise un ensemble de classifieurs binaires utilisant des régressions logistiques qui visent à évaluer dans quelle mesure des mots contexte et cible sont susceptible d'apparaître simultanément, avec des modèles étant entraînés sur des ensembles de k exemples négatifs et 1 exemple positif. Étant donnés un mot contexte c et un mot cible t, la prédiction est donnée par :

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

&#10230; Remarque : cette méthode est moins coûteuse en calcul par rapport au modèle skip-gram.

<br>


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

&#10230; GloVe ― Le modèle GloVe (en anglais <i>global vectors for word representation</i>) est une technique de représentation des mots qui utilise une matrice de co-occurrence X où chaque Xi,j correspond au nombre de fois qu'une cible i se produit avec un contexte j. Sa fonction de coût J est telle que :

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

&#10230; où f est une fonction à coefficients telle que Xi,j=0⟹f(Xi,j)=0.
Étant donné la symmétrie que e et θ ont dans un modèle, la représentation du mot final e(final)w est donnée par :

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

&#10230; Remarque : les composantes individuelles de la représentation d'un mot n'est pas nécessairement facilement interprétable.

<br>


**60. Comparing words**

&#10230; Comparaison de mots

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

&#10230; Similarité cosinus ― La similarité cosinus (en anglais <i>cosine similarity</i>) entre les mots w1 et w2 est donnée par :

<br>


**62. Remark: θ is the angle between words w1 and w2.**

&#10230; Remarque : θ est l'angle entre les mots w1 et w2.

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

&#10230; t-SNE ― La méthode t-SNE (en anglais <i>t-distributed Stochastic Neighbor Embedding</i>) est une technique visant à réduire une représentation dans un espace de haute dimension en un espace de plus faible dimension. En pratique, on visualise les vecteur-mots dans un espace 2D.

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

&#10230; [littérature, art, livre, culture, poème, lecture, connaissance, divertissant, aimable, enfance, gentil, ours en peluche, doux, câlin, mignon, adorable]

<br>


**65. Language model**

&#10230; Modèle de langage

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

&#10230; Vue d'ensemble ― Un modèle de langage vise à estimer la probabilité d'une phrase P(y).

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

&#10230; Modèle n-gram ― Ce modèle consiste en une approche naïve qui vise à quantifier la probabilité qu'une expression apparaisse dans un corpus en comptabilisant le nombre de son apparition dans le training data.

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

&#10230; Perplexité ― Les modèles de langage sont communément évalués en utilisant la perplexité, aussi noté PP, qui peut être interprété comme étant la probabilité inverse des données normalisée par le nombre de mots T. La perplexité est telle que plus elle est faible, mieux c'est. Elle est définie de la manière suivante :

<br>


**69. Remark: PP is commonly used in t-SNE.**

&#10230; Remarque : PP est souvent utilisée dans le cadre du t-SNE.

<br>


**70. Machine translation**

&#10230; Traduction machine

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

&#10230; Vue d'ensemble ― Un modèle de traduction machine est similaire à un modèle de langage ayant un auto-encodeur placé en amont. Pour cette raison, ce modèle est souvent surnommé modèle conditionnel de langage. Le but est de trouver une phrase y telle que :

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

&#10230; Recherche en faisceau ― Cette technique (en anglais <i>beam search</i>) est un algorithme de recherche heuristique, utilisé dans le cadre de la traduction machine et de la reconnaissance vocale, qui vise à trouver la phrase la plus probable y sachant l'entrée x.

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

&#10230; [Étape 1 : Trouver les B mots les plus probables y<1>, Étape 2 : Calculer les probabilités conditionnelles y<k>|x,y<1>,...,y<k−1>, Étape 3 : Garder les B combinaisons les plus probables x,y<1>,...,y<k>, Arrêter la procédure à un mot stop]

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

&#10230; Remarque : si la largeur du faisceau est prise égale à 1, alors ceci est équivalent à un algorithme glouton.

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

&#10230; Largeur du faisceau ― La largeur du faisceau (en anglais <i>beam width</i>) B est un paramètre de la recherche en faisceau. De grandes valeurs de B conduisent à avoir de meilleurs résultats mais avec un coût de mémoire plus lourd et à un temps de calcul plus long. De faibles valeurs de B conduisent à de moins bons résultats mais avec un coût de calcul plus faible. Une valeur de B égale à 10 est standarde et est souvent utilisée.

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

&#10230; Normalisation de longueur ― Pour que la stabilité numérique puisse être améliorée, la recherche en faisceau utilise un objectif normalisé, souvent appelé l'objectif de log-probabilité normalisé, défini par :

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

&#10230; Remarque : le paramètre α est souvent comprise entre 0.5 et 1.

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

&#10230; Analyse d'erreur ― Lorsque l'on obtient une mauvaise traduction prédite ˆy, on peut se demander la raison pour laquelle l'algorithme n'a pas obtenu une bonne traduction y∗ en faisant une analyse d'erreur de la manière suivante :

<br>


**79. [Case, Root cause, Remedies]**

&#10230; [Cas, Cause, Remèdes]

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

&#10230; [Recherche en faisceau défectueuse, RNN défectueux, Augmenter la largeur du faisceau, Essayer une différente architecture, Régulariser, Obtenir plus de données]

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

&#10230; Score bleu ― Le score bleu (en anglais <i>bilingual evaluation understudy</i>) a pour but de quantifier à quel point une traduction est bonne en calculant un score de similarité basé sur une précision n-gram. Il est défini de la manière suivante :

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

&#10230; où pn est le score bleu uniqué basé sur les n-gram, défini par :

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

&#10230; Remarque : une pénalité de brièveté peut être appliquée aux traductions prédites courtes pour empêcher que le score bleu soit artificiellement haut.

<br>


**84. Attention**

&#10230; Attention

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

&#10230; Modèle d'attention ― Le modèle d'attention (en anglais <i>attention model</i>) permet au RNN de mettre en valeur des parties spécifiques de l'entrée qui peuvent être considérées comme étant importantes, ce qui améliore la performance du modèle final en pratique. En notant α<t,t′> la quantité d'attention que la sortie y<t> devrait porter à l'activation a<t′> et au contexte c<t> à l'instant t, on a :

<br>


**86. with**

&#10230; avec

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

&#10230; Remarque : les scores d'attention sont communément utilisés dans la génération de légende d'image ainsi que dans la traduction machine.

<br>


**88. A cute teddy bear is reading Persian literature.**

&#10230; Un ours en peluche mignon est en train de lire de la littérature persane.

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

&#10230; Coefficient d'attention ― La quantité d'attention que la sortie y<t> devrait porter à l'activation a<t′> est donné α<t,t′>, qui est calculé de la manière suivante :

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

&#10230; Remarque : la complexité de calcul est quadratique par rapport à Tx.

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'apprentissage profond sont maintenant disponibles en français.

<br>

**92. Original authors**

&#10230; Auteurs

<br>

**93. Translated by X, Y and Z**

&#10230; Traduit par X, Y et Z

<br>

**94. Reviewed by X, Y and Z**

&#10230; Relu par X, Y et Z

<br>

**95. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>

**96. By X and Y**

&#10230; Par X et Y

<br>
