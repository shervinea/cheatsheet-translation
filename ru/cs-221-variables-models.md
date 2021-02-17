**Variables-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models)

<br>

**1. Variables-based models with CSP and Bayesian networks**

&#10230; Модели на основе переменных с CSP и байесовскими сетями

<br>


**2. Constraint satisfaction problems**

&#10230; Задачи удовлетворения ограничений

<br>


**3. In this section, our objective is to find maximum weight assignments of variable-based models. One advantage compared to states-based models is that these algorithms are more convenient to encode problem-specific constraints.**

&#10230; В этом разделе наша цель - найти максимальное присвоение веса моделям на основе переменных. Одно из преимуществ по сравнению с моделями, основанными на состояниях, состоит в том, что эти алгоритмы более удобны для кодирования специфичных для задачи ограничений.

<br>


**4. Factor graphs**

&#10230; Факторные графы

<br>


**5. Definition ― A factor graph, also referred to as a Markov random field, is a set of variables X=(X1,...,Xn) where Xi∈Domaini and m factors f1,...,fm with each fj(X)⩾0.**

&#10230; Определение ― Фактор-граф, также называемый марковским случайным полем, представляет собой набор переменных X=(X1,...,Xn), где Xi∈Domaini и m факторов f1,...,fm с каждым fj(X)⩾0.

<br>


**6. Domain**

&#10230; Домен

<br>


**7. Scope and arity ― The scope of a factor fj is the set of variables it depends on. The size of this set is called the arity.**

&#10230; Объем и арность ― Объем фактора fj - это набор переменных, от которых он зависит. Размер этого набора называется арностью.

<br>


**8. Remark: factors of arity 1 and 2 are called unary and binary respectively.**

&#10230; Примечание: факторы арности 1 и 2 называются соответственно унарными и бинарными.

<br>


**9. Assignment weight ― Each assignment x=(x1,...,xn) yields a weight Weight(x) defined as being the product of all factors fj applied to that assignment. Its expression is given by:**

&#10230; Вес присвоения ― каждое присвоение x=(x1,...,xn) дает весовой коэффициент Weight(x), определяемый как произведение всех факторов fj, примененных к этому назначению. Его выражение опредляется следующим образом:

<br>


**10. Constraint satisfaction problem ― A constraint satisfaction problem (CSP) is a factor graph where all factors are binary; we call them to be constraints:**

&#10230; Задачи удовлетворения ограничений ― Сonstraint satisfaction problem (CSP) - факторный граф, в котором все факторы бинарны; мы называем их ограничениями:

<br>


**11. Here, the constraint j with assignment x is said to be satisfied if and only if fj(x)=1.**

&#10230; Здесь ограничение j с присвоением x считается выполненным тогда и только тогда, когда fj(x)=1.

<br>


**12. Consistent assignment ― An assignment x of a CSP is said to be consistent if and only if Weight(x)=1, i.e. all constraints are satisfied.**

&#10230; Непротиворечивое присвоение ― Присвоение x CSP называется непротиворечивым тогда и только тогда, когда Weight(x)=1, т.е. все ограничения выполняются.

<br>


**13. Dynamic ordering**

&#10230; Динамическое упорядочивание

<br>


**14. Dependent factors ― The set of dependent factors of variable Xi with partial assignment x is called D(x,Xi), and denotes the set of factors that link Xi to already assigned variables.**

&#10230; Зависимые факторы ― Набор зависимых факторов переменной Xi с частичным присвоением x называется D(x,Xi) и обозначает набор факторов, которые связывают Xi с уже присвоенными переменными.

<br>


**15. Backtracking search ― Backtracking search is an algorithm used to find maximum weight assignments of a factor graph. At each step, it chooses an unassigned variable and explores its values by recursion. Dynamic ordering (i.e. choice of variables and values) and lookahead (i.e. early elimination of inconsistent options) can be used to explore the graph more efficiently, although the worst-case runtime stays exponential: O(|Domain|n).**

&#10230; Поиск с возвратом ― Backtracking search - это алгоритм, используемый для нахождения максимального веса факторного графа. На каждом этапе он выбирает неназначенную переменную и исследует её значения с помощью рекурсии. Динамическое упорядочение (то есть выбор переменных и значений) и опережение (то есть раннее устранение несовместимых вариантов) могут использоваться для более эффективного изучения графика, хотя время выполнения в худшем случае остается экспоненциальным: O(|Domain|n).

<br>


**16. [Forward checking ― It is a one-step lookahead heuristic that preemptively removes inconsistent values from the domains of neighboring variables. It has the following characteristics:, After assigning a variable Xi, it eliminates inconsistent values from the domains of all its neighbors., If any of these domains becomes empty, we stop the local backtracking search., If we un-assign a variable Xi, we have to restore the domain of its neighbors.]**

&#10230; [Прямая проверка ― это эвристика упреждающего просмотра за один шаг. Она упреждающе удаляет несогласованные значения из доменов соседних переменных. Она имеет следующие характеристики:, После присвоения переменной Xi, она удаляет несовместимые значения из доменов всех своих соседей., Если какой-либо из этих доменов становится пустым, мы останавливаем локальный поиск с возвратом., Если мы отменим присвоение переменной Xi, мы должны восстановить домен её соседей.]

<br>


**17. Most constrained variable ― It is a variable-level ordering heuristic that selects the next unassigned variable that has the fewest consistent values. This has the effect of making inconsistent assignments to fail earlier in the search, which enables more efficient pruning.**

&#10230; Наиболее ограниченная переменная ― Это эвристика упорядочения на уровне переменных, которая выбирает следующую неназначенную переменную, которая имеет наименьшее количество согласованных значений. Это приводит к тому, что непоследовательные присвоения терпят неудачу раньше при поиске, что обеспечивает более эффективное сокращение.

<br>


**18. Least constrained value ― It is a value-level ordering heuristic that assigns the next value that yields the highest number of consistent values of neighboring variables. Intuitively, this procedure chooses first the values that are most likely to work.**

&#10230; Наименее ограниченное значение ― Это эвристика упорядочивания на уровне значений, которая назначает следующее значение, которое дает наибольшее количество согласованных значений соседних переменных. Интуитивно эта процедура сначала выбирает значения, которые с наибольшей вероятностью будут работать.

<br>


**19. Remark: in practice, this heuristic is useful when all factors are constraints.**

&#10230; Примечание: на практике эта эвристика полезна, когда все факторы являются ограничениями.

<br>


**20. The example above is an illustration of the 3-color problem with backtracking search coupled with most constrained variable exploration and least constrained value heuristic, as well as forward checking at each step.**

&#10230; Приведенный выше пример является иллюстрацией задачи о трех цветах с поиском с возвратом в сочетании с наиболее ограниченным исследованием переменных и эвристикой наименее ограниченного значения, а также прямой проверкой на каждом шаге.

<br>


**21. [Arc consistency ― We say that arc consistency of variable Xl with respect to Xk is enforced when for each xl∈Domainl:, unary factors of Xl are non-zero, there exists at least one xk∈Domaink such that any factor between Xl and Xk is non-zero.]**

&#10230; [Согласованность дуги ― Мы говорим о непротиворечивости дуги переменной Xl относительно Xk при обеспечении для каждого xl∈Domainl:, унарные множители Xl не равны нулю, существует по крайней мере один xk∈Domaink. Для него любой множитель между Xl и Xk отличен от нуля.]

<br>


**22. AC-3 ― The AC-3 algorithm is a multi-step lookahead heuristic that applies forward checking to all relevant variables. After a given assignment, it performs forward checking and then successively enforces arc consistency with respect to the neighbors of variables for which the domain change during the process.**

&#10230; AC-3 ― Алгоритм AC-3 - это многоэтапная эвристика упреждающего просмотра, которая применяет упреждающую проверку ко всем соответствующим переменным. После заданного присвоения она выполняет прямую проверку, а затем последовательно обеспечивает согласованность дуги по отношению к соседям переменных, для которых домен изменяется во время процесса.

<br>


**23. Remark: AC-3 can be implemented both iteratively and recursively.**

&#10230; Примечание: AC-3 может быть реализован как итеративно, так и рекурсивно.

<br>


**24. Approximate methods**

&#10230; Аппроксимационные методы

<br>


**25. Beam search ― Beam search is an approximate algorithm that extends partial assignments of n variables of branching factor b=|Domain| by exploring the K top paths at each step. The beam size K∈{1,...,bn} controls the tradeoff between efficiency and accuracy. This algorithm has a time complexity of O(n⋅Kblog(Kb)).**

&#10230; Лучевой поиск ― (примечание переводчика: оптимизированный Поиск по первому наилучшему совпадению) - это приближенный алгоритм, расширяющий частичные присвоения n переменных коэффициента ветвления b=|Domain| исследуя K лучших путей на каждом этапе. Размер луча K∈{1,...,bn} определяет компромисс между эффективностью и точностью. Этот алгоритм имеет временную сложность O(n⋅Kblog(Kb)).

<br>


**26. The example below illustrates a possible beam search of parameters K=2, b=3 and n=5.**

&#10230; Пример ниже иллюстрирует возможный лучевой поиск при параметрах K=2, b=3 и n=5.

<br>


**27. Remark: K=1 corresponds to greedy search whereas K→+∞ is equivalent to BFS tree search.**

&#10230; Примечание: K=1 соответствует жадному поиску, тогда как K→+∞ эквивалентно поиску по дереву BFS.

<br>


**28. Iterated conditional modes ― Iterated conditional modes (ICM) is an iterative approximate algorithm that modifies the assignment of a factor graph one variable at a time until convergence. At step i, we assign to Xi the value v that maximizes the product of all factors connected to that variable.**

&#10230; Итерированные условные режимы ― Iterated conditional modes (ICM) представляет собой итерационный приближенный алгоритм, который изменяет присвоение факторному графу одной переменной за раз до сходимости. На шаге i мы присваиваем Xi значение v, которое максимизирует произведение всех факторов, связанных с этой переменной.

<br>


**29. Remark: ICM may get stuck in local minima.**

&#10230; Примечание: ICM может застрять в локальных минимумах.

<br>


**30. [Gibbs sampling ― Gibbs sampling is an iterative approximate method that modifies the assignment of a factor graph one variable at a time until convergence. At step i:, we assign to each element u∈Domaini a weight w(u) that is the product of all factors connected to that variable, we sample v from the probability distribution induced by w and assign it to Xi.]**

&#10230; [Выборка по Гиббсу ― это итеративный приближенный метод; который изменяет присвоение факторному графу одной переменной за раз до сходимости. На шаге i:, мы присваиваем каждому элементу u∈Domaini вес w(u). Вес является произведением всех связанных с этой переменной факторов, мы выбираем v из индуцированного w распределения вероятностей и присваиваем его Xi.]

<br>


**31. Remark: Gibbs sampling can be seen as the probabilistic counterpart of ICM. It has the advantage to be able to escape local minima in most cases.**

&#10230; Примечание: Выборка по Гиббсу можно рассматривать как вероятностный аналог ICM. Преимущество этого метода заключается в возможности избежать локальных минимумов в большинстве случаев.

<br>


**32. Factor graph transformations**

&#10230; Преобразования факторных графов

<br>


**33. Independence ― Let A,B be a partitioning of the variables X. We say that A and B are independent if there are no edges between A and B and we write:**

&#10230; Независимость ― Пусть A,B - разбиение переменных X. Мы говорим, что A и B независимы, если между A и B нет ребер, и пишем:

<br>


**34. Remark: independence is the key property that allows us to solve subproblems in parallel.**

&#10230; Примечание: независимость - ключевое свойство, которое позволяет нам решать подзадачи параллельно.

<br>


**35. Conditional independence ― We say that A and B are conditionally independent given C if conditioning on C produces a graph in which A and B are independent. In this case, it is written:**

&#10230; Условная независимость ― Мы говорим, что A и B условно независимы для данного C, если принятое условие C дает граф, в котором A и B независимы. В этом случае пишется:

<br>


**36. [Conditioning ― Conditioning is a transformation aiming at making variables independent that breaks up a factor graph into smaller pieces that can be solved in parallel and can use backtracking. In order to condition on a variable Xi=v, we do as follows:, Consider all factors f1,...,fk that depend on Xi, Remove Xi and f1,...,fk, Add gj(x) for j∈{1,...,k} defined as:]**

&#10230; [Кондиционирование ― это преобразование переменных в независимые. Факторный граф разбивается на более мелкие части. Их можно решать параллельно и использовать обратный поиск. Чтобы поставить условие на переменную Xi=v, делать так:, Рассмотреть все зависящие от Xi множители f1,...,fk, Удалить Xi и f1,...,fk, Добавить gj(x) для j∈{1,...,k} по определению:]

<br>


**37. Markov blanket ― Let A⊆X be a subset of variables. We define MarkovBlanket(A) to be the neighbors of A that are not in A.**

&#10230; Марковское ограждение ― Пусть A⊆X - подмножество переменных. Мы определяем MarkovBlanket(A) как соседей A, которые не находятся в A.

<br>


**38. Proposition ― Let C=MarkovBlanket(A) and B=X∖(A∪C). Then we have:**

&#10230; Утверждение ― Пусть C=MarkovBlanket(A) и B=X∖(A∪C). Тогда у нас есть:

<br>


**39. [Elimination ― Elimination is a factor graph transformation that removes Xi from the graph and solves a small subproblem conditioned on its Markov blanket as follows:, Consider all factors fi,1,...,fi,k that depend on Xi, Remove Xi and fi,1,...,fi,k, Add fnew,i(x) defined as:]**

&#10230; [Устранение ― Исключение - это преобразование факторного графа с удалением Xi из графа и решением небольшой подзадачи с условием марковского ограждения:, Рассмотреть все факторы fi,1,...,fi,k с зависимостью от Xi, Удалить Xi и fi,1,...,fi,k, Добавить fnew,i(x) по определению:]

<br>


**40. Treewidth ― The treewidth of a factor graph is the maximum arity of any factor created by variable elimination with the best variable ordering. In other words,**

&#10230; Ширина дерева ― Древовидная ширина факторного графа - это максимальная арность любого фактора, созданного путем исключения переменных с наилучшим порядком переменных. Другими словами,

<br>


**41. The example below illustrates the case of a factor graph of treewidth 3.**

&#10230; Пример ниже иллюстрирует случай факторного графа с шириной дерева 3.

<br>


**42. Remark: finding the best variable ordering is a NP-hard problem.**

&#10230; Примечание: поиск лучшего порядка переменных - это NP-сложная задача.

<br>


**43. Bayesian networks**

&#10230; Байесовские сети

<br>


**44. In this section, our goal will be to compute conditional probabilities. What is the probability of a query given evidence?**

&#10230; В этом разделе нашей целью будет вычисление условных вероятностей. Какова условная вероятность запроса при данных наблюдениях?

<br>


**45. Introduction**

&#10230; Введение

<br>


**46. Explaining away ― Suppose causes C1 and C2 influence an effect E. Conditioning on the effect E and on one of the causes (say C1) changes the probability of the other cause (say C2). In this case, we say that C1 has explained away C2.**

&#10230; Объяснение ― Предположим, что причины C1 и C2 влияют на эффект E. Обусловление эффекта E и одной из причин (скажем, C1) изменяет вероятность другой причины (скажем, C2). В этом случае мы говорим, что C1 объяснил C2.

<br>


**47. Directed acyclic graph ― A directed acyclic graph (DAG) is a finite directed graph with no directed cycles.**

&#10230; Направленный ациклический граф ― Directed acyclic graph (DAG) - это конечный ориентированный граф без ориентированных циклов.

<br>


**48. Bayesian network ― A Bayesian network is a directed acyclic graph (DAG) that specifies a joint distribution over random variables X=(X1,...,Xn) as a product of local conditional distributions, one for each node:**

&#10230; Байесовская сеть ― Bayesian network - это ориентированный ациклический граф (DAG), который определяет совместное распределение по случайным величинам X=(X1,...,Xn) как произведение локальных условных распределений, по одному для каждого узла:

<br>


**49. Remark: Bayesian networks are factor graphs imbued with the language of probability.**

&#10230; Примечание: Байесовские сети - это фактор-графы, пропитанные языком вероятностей.

<br>


**50. Locally normalized ― For each xParents(i), all factors are local conditional distributions. Hence they have to satisfy:**

&#10230; Локально нормализованные - для каждого xParents(i) все факторы являются локальными условными распределениями. Следовательно, они должны удовлетворить:

<br>


**51. As a result, sub-Bayesian networks and conditional distributions are consistent.**

&#10230; В результате суббайесовские сети и условные распределения согласованы.

<br>


**52. Remark: local conditional distributions are the true conditional distributions.**

&#10230; Примечание: локальные условные распределения являются истинными условными распределениями.

<br>


**53. Marginalization ― The marginalization of a leaf node yields a Bayesian network without that node.**

&#10230; Маргинализация ― маргинализация листового узла приводит к байесовской сети без этого узла.

<br>


**54. Probabilistic programs**

&#10230; Вероятностные программы

<br>


**55. Concept ― A probabilistic program randomizes variables assignment. That way, we can write down complex Bayesian networks that generate assignments without us having to explicitly specify associated probabilities.**

&#10230; Концепция ― вероятностная программа случайным образом присваивает переменные. Таким образом, мы можем записывать сложные байесовские сети, которые генерируют назначения, без необходимости явно указывать связанные вероятности.

<br>


**56. Remark: examples of probabilistic programs include Hidden Markov model (HMM), factorial HMM, naive Bayes, latent Dirichlet allocation, diseases and symptoms and stochastic block models.**

&#10230; Примечание: примеры вероятностных программ включают скрытую марковскую модель (Hidden Markov model, HMM), факторную HMM, наивный байесовский алгоритм, скрытое распределение Дирихле, болезни и симптомы (SBD-LDA) и стохастические блочные модели.

<br>


**57. Summary ― The table below summarizes the common probabilistic programs as well as their applications:**

&#10230; Сводка ― В таблице ниже приведены общие вероятностные программы, а также их приложения:

<br>


**58. [Program, Algorithm, Illustration, Example]**

&#10230; [Программа, Алгоритм, Иллюстрация, Пример]

<br>


**59. [Markov Model, Hidden Markov Model (HMM), Factorial HMM, Naive Bayes, Latent Dirichlet Allocation (LDA)]**

&#10230; [Markov Model, Hidden Markov Model (HMM), Factorial HMM, Naive Bayes, Latent Dirichlet Allocation (LDA)]

<br>


**60. [Generate, distribution]**

&#10230; [Создает, распределение]

<br>


**61. [Language modeling, Object tracking, Multiple object tracking, Document classification, Topic modeling]**

&#10230; [Языковое моделирование, Трекинг объектов, Трекинг нескольких объектов, Классификация документов, Тематическое моделирование]

<br>


**62. Inference**

&#10230; Байесовский вывод

<br>


**63. [General probabilistic inference strategy ― The strategy to compute the probability P(Q|E=e) of query Q given evidence E=e is as follows:, Step 1: Remove variables that are not ancestors of the query Q or the evidence E by marginalization, Step 2: Convert Bayesian network to factor graph, Step 3: Condition on the evidence E=e, Step 4: Remove nodes disconnected from the query Q by marginalization, Step 5: Run a probabilistic inference algorithm (manual, variable elimination, Gibbs sampling, particle filtering)]**

&#10230; [Общая стратегия вероятностного вывода ― Стратегия вычисления вероятности P(Q|E=e) запроса Q при наличии наблюдаемых значений E=e следующая:, Шаг 1. Удалить не являющиеся предками запроса Q или наблюдения E переменные путем маргинализации, Шаг 2: Преобразовать байесовскую сеть в факторный граф, Шаг 3: Улучшить состояние наблюдения E=e, Шаг 4: Удалить отключенные от запроса Q узлы путем маргинализации, Шаг 5: Запустить вероятностный алгоритм вывода (вручную, исключение переменных, выборка по Гиббсу, фильтрация частиц)]

<br>


**64. Forward-backward algorithm ― This algorithm computes the exact value of P(H=hk|E=e) (smoothing query) for any k∈{1,...,L} in the case of an HMM of size L. To do so, we proceed in 3 steps:**

&#10230; Алгоритм прямого-обратного хода - этот алгоритм вычисляет точное значение P(H=hk|E=e) (запрос сглаживания) для любого k∈{1,...,L} в случае HMM размера L. Чтобы сделать так, мы выполняем 3 шага:

<br>


**65. Step 1: for ..., compute ...**

&#10230; Шаг 1: для ..., вычислить ...

<br>


**66. with the convention F0=BL+1=1. From this procedure and these notations, we get that**

&#10230; с условием F0=BL+1=1. Из этой процедуры и этих обозначений мы получаем это

<br>


**67. Remark: this algorithm interprets each assignment to be a path where each edge hi−1→hi is of weight p(hi|hi−1)p(ei|hi).**

&#10230; Примечание: этот алгоритм интерпретирует каждое присвоение как путь, в котором каждое ребро hi−1→hi имеет вес p(hi|hi−1)p(ei|hi).

<br>


**68. [Gibbs sampling ― This algorithm is an iterative approximate method that uses a small set of assignments (particles) to represent a large probability distribution. From a random assignment x, Gibbs sampling performs the following steps for i∈{1,...,n} until convergence:, For all u∈Domaini, compute the weight w(u) of assignment x where Xi=u, Sample v from the probability distribution induced by w: v∼P(Xi=v|X−i=x−i), Set Xi=v]**

&#10230; [Выборка по Гиббсу ― Этот алгоритм представляет собой итеративный приближенный метод. Он использует небольшой набор присвоений (частиц) для представления большого распределения вероятностей. Из случайного присвоения x, Выборка по Гиббсу выполняет следующие шаги для i∈{1,...,n} до сходимости:, Для всех u∈Domaini, вычислить вес w(u) присвоения x; где Xi=u, Произвести выборку v из индуцированного w распределения вероятностей: v∼P(Xi=v|X−i=x−i), Установить Xi=v]

<br>


**69. Remark: X−i denotes X∖{Xi} and x−i represents the corresponding assignment.**

&#10230; Примечание: X−i обозначает X∖{Xi}, а x−i представляет соответствующее присвоение.

<br>


**70. [Particle filtering ― This algorithm approximates the posterior density of state variables given the evidence of observation variables by keeping track of K particles at a time. Starting from a set of particles C of size K, we run the following 3 steps iteratively:, Step 1: proposal - For each old particle xt−1∈C, sample x from the transition probability distribution p(x|xt−1) and add x to a set C′., Step 2: weighting - Weigh each x of the set C′ by w(x)=p(et|x), where et is the evidence observed at time t., Step 3: resampling - Sample K elements from the set C′ using the probability distribution induced by w and store them in C: these are the current particles xt.]**

&#10230; [Фильтрация частиц ― Этот алгоритм аппроксимирует апостериорную плотность переменных состояния с учетом данных наблюдений путем отслеживания K частиц за раз. Начиная с набора частиц C размера K, мы итеративно выполняем следующие 3 шага:, Шаг 1: предложение - Для каждой старой частицы xt−1∈C, выбрать x из распределения вероятности перехода p(x|xt−1) и добавить x к множеству C′., Шаг 2: взвешивание - Взвесить каждый x из множества C′ как w(x)=p(et|x), где et - свидетельство в момент времени t., Шаг 3: повторная выборка - выборка K элементов из множества C′ с использованием индуцированного w распределения вероятностей и сохранение их в C: это текущие частицы xt.]

<br>


**71. Remark: a more expensive version of this algorithm also keeps track of past particles in the proposal step.**

&#10230; Примечание: более дорогая версия этого алгоритма также отслеживает прошлые частицы на этапе предложения.

<br>


**72. Maximum likelihood ― If we don't know the local conditional distributions, we can learn them using maximum likelihood.**

&#10230; Максимальное правдоподобие ― Если мы не знаем локальных условных распределений, мы можем изучить их, используя максимальное правдоподобие.

<br>


**73. Laplace smoothing ― For each distribution d and partial assignment (xParents(i),xi), add λ to countd(xParents(i),xi), then normalize to get probability estimates.**

&#10230; Сглаживание Лапласа ― для каждого распределения d и частичного присвоения (xParents(i),xi) добавьте λ к countd(xParents(i),xi), затем нормализуйте, чтобы получить оценки вероятности.

<br>


**74. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Алгоритм Expectation-Maximization (EM) ― алгоритм максимизации ожидания дает эффективный метод оценки параметра θ посредством оценки максимального правдоподобия путем многократного построения нижней границы правдоподобия (E-шаг) и оптимизации этой нижней границы (M-шаг) следующим образом:

<br>


**75. [E-step: Evaluate the posterior probability q(h) that each data point e came from a particular cluster h as follows:, M-step: Use the posterior probabilities q(h) as cluster specific weights on data points e to determine θ through maximum likelihood.]**

&#10230; [Шаг E: Оценить апостериорную вероятность q(h) принадлежности каждой точки данных e из определенного кластера h следующим образом:, M-шаг: использовать апостериорные вероятности q(h) в качестве весовых коэффициентов для конкретных кластеров точек данных e и определить θ через максимальное правдоподобие.]

<br>


**76. [Factor graphs, Arity, Assignment weight, Constraint satisfaction problem, Consistent assignment]**

&#10230; [Факторные графы, Арность, Вес присвоения, Задачи удовлетворения ограничений, Непротиворечивое присвоение]

<br>


**77. [Dynamic ordering, Dependent factors, Backtracking search, Forward checking, Most constrained variable, Least constrained value]**

&#10230; [Динамическое упорядочивание, Зависимые факторы, Поиск с возвратом, Прямая проверка, Наиболее ограниченная переменная, Наименьшее ограниченное значение]

<br>


**78. [Approximate methods, Beam search, Iterated conditional modes, Gibbs sampling]**

&#10230; [Аппроксимационные методы, Лучевой поиск, Итерированные условные режимы, Выборка по Гиббсу]

<br>


**79. [Factor graph transformations, Conditioning, Elimination]**

&#10230; [Преобразования факторных графов, Кондиционирование, Устранение]

<br>


**80. [Bayesian networks, Definition, Locally normalized, Marginalization]**

&#10230; [Байесовские сети, Определение, Локально нормализованные, Маргинализация]

<br>


**81. [Probabilistic program, Concept, Summary]**

&#10230; [Вероятностная программа, Концепция, Сводка]

<br>


**82. [Inference, Forward-backward algorithm, Gibbs sampling, Laplace smoothing]**

&#10230; [Байесовский вывод, Алгоритм прямого-обратного хода, Выборка по Гиббсу, Сглаживание Лапласа]

<br>


**83. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**84. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>


**85. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>


**86. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>


**87. By X and Y**

&#10230; По X и Y

<br>


**88. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Шпаргалки по искусственному интеллекту теперь доступны на русском языке.
