**States-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-states-models)

<br>

**1. States-based models with search optimization and MDP**

&#10230; Модели на основе состояний с оптимизацией поиска и Марковским процессом принятия решений (MDP)

<br>


**2. Search optimization**

&#10230; Оптимизация поиска

<br>


**3. In this section, we assume that by accomplishing action a from state s, we deterministically arrive in state Succ(s,a). The goal here is to determine a sequence of actions (a1,a2,a3,a4,...) that starts from an initial state and leads to an end state. In order to solve this kind of problem, our objective will be to find the minimum cost path by using states-based models.**

&#10230; В этом разделе мы предполагаем, что, выполняя действие a из состояния s, мы детерминированно приходим в состояние Succ(s,a). Здесь цель - определить последовательность действий (a1,a2,a3,a4,...) которая начинается из начального состояния и приводит к конечному состоянию. Для решения задачи нам надо найти путь с минимальными затратами, с помощью моделей на основе состояний.

<br>


**4. Tree search**

&#10230; Поиск по дереву

<br>


**5. This category of states-based algorithms explores all possible states and actions. It is quite memory efficient, and is suitable for huge state spaces but the runtime can become exponential in the worst cases.**

&#10230; Эта категория алгоритмов на основе состояний исследует все возможные состояния и действия. Он довольно эффективен с точки зрения памяти и подходит для огромных пространств состояний, но в худших случаях время выполнения может стать экспоненциальным.

<br>


**6. [Self-loop, More than a parent, Cycle, More than a root, Valid tree]**

&#10230; [Переход в себя, Несколько родителей, Цикл, Несколько корней, Дерево]

<br>


**7. [Search problem ― A search problem is defined with:, a starting state sstart, possible actions Actions(s) from state s, action cost Cost(s,a) from state s with action a, successor Succ(s,a) of state s after action a, whether an end state was reached IsEnd(s)]**

&#10230; [Задача поиска ― определяется с помощью:, начальное состояние sstart, возможные действия Actions(s) из состояния s, стоимость действия Cost(s,a) из состояния s с действием a, преемник Succ(s,a) состояния s после действия a, было ли достигнуто конечное состояние IsEnd(s)]

<br>


**8. The objective is to find a path that minimizes the cost.**

&#10230; Цель состоит в том, чтобы найти путь, который минимизирует затраты.

<br>


**9. Backtracking search ― Backtracking search is a naive recursive algorithm that tries all possibilities to find the minimum cost path. Here, action costs can be either positive or negative.**

&#10230; Backtracking search ― Поиск с возвратом - это наивный рекурсивный алгоритм, который пробует все возможности, чтобы найти путь с минимальной стоимостью. Здесь затраты на действия могут быть как положительными, так и отрицательными.

<br>


**10. Breadth-first search (BFS) ― Breadth-first search is a graph search algorithm that does a level-by-level traversal. We can implement it iteratively with the help of a queue that stores at each step future nodes to be visited. For this algorithm, we can assume action costs to be equal to a constant c⩾0.**

&#10230; Breadth-first search (BFS) ― Поиск в ширину - это алгоритм поиска по графу, который выполняет обход уровня за уровнем. Мы можем реализовать это итеративно с помощью очереди, в которой на каждом шаге хранятся будущие узлы, которые нужно посетить. Для этого алгоритма можно считать, что затраты на действия равны константе c⩾0.

<br>


**11. Depth-first search (DFS) ― Depth-first search is a search algorithm that traverses a graph by following each path as deep as it can. We can implement it recursively, or iteratively with the help of a stack that stores at each step future nodes to be visited. For this algorithm, action costs are assumed to be equal to 0.**

&#10230; Depth-first search (DFS) ― Поиск в глубину - это алгоритм поиска, который обходит граф, прослеживая каждый путь как можно глубже. Мы можем реализовать это рекурсивно или итеративно с помощью стека, в котором на каждом шаге хранятся будущие узлы, которые необходимо посетить. Для этого алгоритма предполагается, что затраты на действия равны 0.

<br>


**12. Iterative deepening ― The iterative deepening trick is a modification of the depth-first search algorithm so that it stops after reaching a certain depth, which guarantees optimality when all action costs are equal. Here, we assume that action costs are equal to a constant c⩾0.**

&#10230; Iterative deepening ― Поиск в глубину с итеративным углублением - это модификация алгоритма поиска в глубину, так что он останавливается после достижения определенной глубины, что гарантирует оптимальность, когда все затраты на действия равны. Здесь мы предполагаем, что затраты на действия равны константе c⩾0.

<br>


**13. Tree search algorithms summary ― By noting b the number of actions per state, d the solution depth, and D the maximum depth, we have:**

&#10230; Сводка алгоритмов поиска по дереву ― отметив b количество действий на состояние, d глубину решения и D максимальную глубину, которую мы имеем:

<br>


**14. [Algorithm, Action costs, Space, Time]**

&#10230; [Алгоритм, Стоимость действий, Пространство, Время]

<br>


**15. [Backtracking search, any, Breadth-first search, Depth-first search, DFS-Iterative deepening]**

&#10230; [Поиск с возвратом, любая, Поиск в ширину, Поиск в глубину, Поиск в глубину с итеративным углублением]

<br>


**16. Graph search**

&#10230; Поиск по графу

<br>


**17. This category of states-based algorithms aims at constructing optimal paths, enabling exponential savings. In this section, we will focus on dynamic programming and uniform cost search.**

&#10230; Эта категория алгоритмов на основе состояний направлена на построение оптимальных путей, обеспечивающих экспоненциальную экономию. В этом разделе мы сосредоточимся на динамическом программировании и поиске по единой стоимости.

<br>


**18. Graph ― A graph is comprised of a set of vertices V (also called nodes) as well as a set of edges E (also called links).**

&#10230; Граф ― граф состоит из набора вершин V (также называемых узлами) и набора ребер E (также называемых связями).

<br>


**19. Remark: a graph is said to be acylic when there is no cycle.**

&#10230; Примечание: граф называется ациклическим, когда нет цикла.

<br>


**20. State ― A state is a summary of all past actions sufficient to choose future actions optimally.**

&#10230; Состояние ― это совокупность всех прошлых действий, достаточная для оптимального выбора будущих действий.

<br>


**21. Dynamic programming ― Dynamic programming (DP) is a backtracking search algorithm with memoization (i.e. partial results are saved) whose goal is to find a minimum cost path from state s to an end state send. It can potentially have exponential savings compared to traditional graph search algorithms, and has the property to only work for acyclic graphs. For any given state s, the future cost is computed as follows:**

&#10230; Динамическое программирование (DP) ― это алгоритм поиска с возвратом и запоминанием (то есть частичные результаты сохраняются), цель которого - найти путь с минимальной стоимостью от состояния s в конечное состояние send. Он потенциально может иметь экспоненциальную экономию по сравнению с традиционными алгоритмами поиска по графу и имеет свойство работать только для ациклических графов. Для любого заданного состояния s будущая стоимость рассчитывается следующим образом:

<br>


**22. [if, otherwise]**

&#10230; [если, иначе]

<br>


**23. Remark: the figure above illustrates a bottom-to-top approach whereas the formula provides the intuition of a top-to-bottom problem resolution.**

&#10230; Примечание: на рисунке выше показан подход снизу вверх, тогда как формула дает интуитивное представление о решении проблемы сверху вниз.

<br>


**24. Types of states ― The table below presents the terminology when it comes to states in the context of uniform cost search:**

&#10230; Типы состояний ― В таблице ниже представлена терминология, когда речь идет о состояниях в контексте поиска по единой стоимости:

<br>


**25. [State, Explanation]**

&#10230; [Состояние, Объяснение]

<br>


**26. [Explored, Frontier, Unexplored]**

&#10230; [Исследовано, Граница, Неизведано]

<br>


**27. [States for which the optimal path has already been found, States seen for which we are still figuring out how to get there with the cheapest cost, States not seen yet]**

&#10230; [Состояния c уже найденным оптимальным путем, Видны состояния (нам неизвестен путь к ним с самой низкой ценой), Состояний пока не видно]

<br>


**28. Uniform cost search ― Uniform cost search (UCS) is a search algorithm that aims at finding the shortest path from a state sstart to an end state send. It explores states s in increasing order of PastCost(s) and relies on the fact that all action costs are non-negative.**

&#10230; Uniform cost search (UCS) ― Поиск по единой стоимости - это алгоритм поиска, который направлен на поиск кратчайшего пути от состояния sstart до конечного состояния send. Он исследует состояния в порядке возрастания PastCost(s) и полагается на тот факт, что все затраты на действия неотрицательны.

<br>


**29. Remark 1: the UCS algorithm is logically equivalent to Dijkstra's algorithm.**

&#10230; Замечание 1: алгоритм UCS логически эквивалентен алгоритму Дейкстры.

<br>


**30. Remark 2: the algorithm would not work for a problem with negative action costs, and adding a positive constant to make them non-negative would not solve the problem since this would end up being a different problem.**

&#10230; Замечание 2: алгоритм не будет работать для задачи с отрицательными затратами действий, и добавление положительной константы, чтобы сделать их неотрицательными, не решит проблему, так как в конечном итоге это будет другой проблемой.

<br>


**31. Correctness theorem ― When a state s is popped from the frontier F and moved to explored set E, its priority is equal to PastCost(s) which is the minimum cost path from sstart to s.**

&#10230; Теорема корректности ― Когда состояние s выталкивается из границы F и перемещается в исследуемый набор E, его приоритет равен PastCost(s), который представляет собой путь с минимальной стоимостью от sstart до s.

<br>


**32. Graph search algorithms summary ― By noting N the number of total states, n of which are explored before the end state send, we have:**

&#10230; Сводка алгоритмов поиска по графу ― Обозначим N общее количество состояний, n из которых исследуются до конечного состояния send, у нас есть:

<br>


**33. [Algorithm, Acyclicity, Costs, Time/space]**

&#10230; [Алгоритм, Ацикличность, Стоимость, Время/пространство]

<br>


**34. [Dynamic programming, Uniform cost search]**

&#10230; [Динамическое программирование, Поиск по единой стоимости]

<br>


**35. Remark: the complexity countdown supposes the number of possible actions per state to be constant.**

&#10230; Примечание: обратный отсчет сложности предполагает, что количество возможных действий для каждого состояния будет постоянным.

<br>


**36. Learning costs**

&#10230; Стоимость обучения

<br>


**37. Suppose we are not given the values of Cost(s,a), we want to estimate these quantities from a training set of minimizing-cost-path sequence of actions (a1,a2,...,ak).**

&#10230; Предположим, нам не даны значения Cost(s,a), мы хотим оценить эти количества из обучающего набора  последовательности действий (a1,a2,...,ak) с минимизацией затрат пути.

<br>


**38. [Structured perceptron ― The structured perceptron is an algorithm aiming at iteratively learning the cost of each state-action pair. At each step, it:, decreases the estimated cost of each state-action of the true minimizing path y given by the training data, increases the estimated cost of each state-action of the current predicted path y' inferred from the learned weights.]**

&#10230; [Структурированный перцептрон ― Структурированный перцептрон - это алгоритм итеративного изучения стоимости каждой пары состояние-действие. На каждом шагу, он:, уменьшает оценочную стоимость каждого действия состояния истинного пути минимизации y (заданного обучающими данными), увеличивает оценочную стоимость каждого действия состояния текущего прогнозируемого пути y' (выведенную из изученных весов).]

<br>


**39. Remark: there are several versions of the algorithm, one of which simplifies the problem to only learning the cost of each action a, and the other parametrizes Cost(s,a) to a feature vector of learnable weights.**

&#10230; Примечание: существует несколько версий алгоритма, одна из которых упрощает задачу до изучения только стоимости каждого действия a, другая параметризует Cost(s,a) вектором характеристик обучаемых весов.

<br>


**40. A* search**

&#10230; A* поиск

<br>


**41. Heuristic function ― A heuristic is a function h over states s, where each h(s) aims at estimating FutureCost(s), the cost of the path from s to send.**

&#10230; Эвристическая функция ― эвристика - это функция h по состояниям s, где каждый h(s) направлен на оценку FutureCost(s), стоимость пути от s до send.

<br>


**42. Algorithm ― A∗ is a search algorithm that aims at finding the shortest path from a state s to an end state send. It explores states s in increasing order of PastCost(s)+h(s). It is equivalent to a uniform cost search with edge costs Cost′(s,a) given by:**

&#10230; Алгоритм ― A∗ - это алгоритм поиска, цель которого - найти кратчайший путь от состояния s до конечного состояния send. Он исследует состояния в порядке возрастания PastCost(s)+h(s). Это эквивалентно поиску по единой стоимости с краевыми затратами Cost′(s,a), задаваемыми выражением:

<br>


**43. Remark: this algorithm can be seen as a biased version of UCS exploring states estimated to be closer to the end state.**

&#10230; Примечание: этот алгоритм можно рассматривать как предвзятую версию исследования состояний UCS, которые оцениваются как более близкие к конечному состоянию.

<br>


**44. [Consistency ― A heuristic h is said to be consistent if it satisfies the two following properties:, For all states s and actions a, The end state verifies the following:]**

&#10230; [Консистентность ― Эвристика h называется согласованной (консистентной) при удовлетворении двух следующих свойств:, Для всех состояний s и действий a, Конечное состояние подтверждает следующее:]

<br>


**45. Correctness ― If h is consistent, then A∗ returns the minimum cost path.**

&#10230; Корректность ― Если h согласован, то A∗ возвращает путь с минимальной стоимостью.

<br>


**46. Admissibility ― A heuristic h is said to be admissible if we have:**

&#10230; Допустимость ― Говорят, что эвристика h допустима, если у нас есть:

<br>


**47. Theorem ― Let h(s) be a given heuristic. We have:**

&#10230; Теорема ― Пусть h(s) - заданная эвристика. У нас есть:

<br>


**48. [consistent, admissible]**

&#10230; [консистентна, допустима]

<br>


**49. Efficiency ― A* explores all states s satisfying the following equation:**

&#10230; Эффективность ― A* исследует все состояния s, удовлетворяющие следующему уравнению:

<br>


**50. Remark: larger values of h(s) is better as this equation shows it will restrict the set of states s going to be explored.**

&#10230; Примечание: большие значения h(s) лучше, поскольку это уравнение показывает, что оно ограничивает набор состояний s, которые будут исследованы.

<br>


**51. Relaxation**

&#10230; Ослабление ограничений

<br>


**52. It is a framework for producing consistent heuristics. The idea is to find closed-form reduced costs by removing constraints and use them as heuristics.**

&#10230; Это основа для создания последовательной эвристики. Идея состоит в том, чтобы найти сокращенные затраты в аналитическом виде, удалив ограничения и используя их в качестве эвристики.

<br>


**53. Relaxed search problem ― The relaxation of search problem P with costs Cost is noted Prel with costs Costrel, and satisfies the identity:**

&#10230; Задача с ослабленными ограничениями поиска ― Ослабление задачи поиска P с затратами Cost обозначается Prel с затратами Costrel и удовлетворяет тождеству:

<br>


**54. Relaxed heuristic ― Given a relaxed search problem Prel, we define the relaxed heuristic h(s)=FutureCostrel(s) as the minimum cost path from s to an end state in the graph of costs Costrel(s,a).**

&#10230; Ослабленная эвристика ― Задана упрощенная задача поиска Prel, мы определяем ослабленную эвристику h(s)=FutureCostrel(s) как путь с минимальной стоимостью от s до конечного состояния в графе затрат Costrel(s,a).

<br>


**55. Consistency of relaxed heuristics ― Let Prel be a given relaxed problem. By theorem, we have:**

&#10230; Консистентность ослабленной эвристики ― Пусть Prel - заданная ослабленная задача. По теореме у нас есть:

<br>


**56. consistent**

&#10230; консистентна

<br>


**57. [Tradeoff when choosing heuristic ― We have to balance two aspects in choosing a heuristic:, Computational efficiency: h(s)=FutureCostrel(s) must be easy to compute. It has to produce a closed form, easier search and independent subproblems., Good enough approximation: the heuristic h(s) should be close to FutureCost(s) and we have thus to not remove too many constraints.]**

&#10230; [Компромисс при выборе эвристики ― При выборе эвристики необходимо уравновесить два аспекта:, Вычислительная эффективность: h(s)=FutureCostrel(s) должно быть легко вычислить. Она должна создать аналитический вид, более легкий поиск и независимые подзадачи., Достаточно хорошее приближение: эвристика h(s) должна быть близка к FutureCost(s) ; и мы не должны удалять слишком много ограничений.]

<br>


**58. Max heuristic ― Let h1(s), h2(s) be two heuristics. We have the following property:**

&#10230; Max эвристика ― пусть h1(s), h2(s) - две эвристики. У нас есть следующее свойство:

<br>


**59. Markov decision processes**

&#10230; Марковские процессы принятия решений

<br>


**60. In this section, we assume that performing action a from state s can lead to several states s′1,s′2,... in a probabilistic manner. In order to find our way between an initial state and an end state, our objective will be to find the maximum value policy by using Markov decision processes that help us cope with randomness and uncertainty.**

&#10230; В этом разделе мы предполагаем, что выполнение действия a из состояния s может привести к нескольким состояниям s′1,s′2,... вероятностным образом. Чтобы найти путь между начальным и конечным состояниями, наша цель будет заключаться в том, чтобы найти политику максимальной ценности с помощью марковских процессов принятия решений, которые помогают нам справляться со случайностью и неопределенностью.

<br>


**61. Notations**

&#10230; Обозначения

<br>


**62. [Definition ― The objective of a Markov decision process is to maximize rewards. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, transition probabilities T(s,a,s′) from s to s′ with action a, rewards Reward(s,a,s′) from s to s′ with action a, whether an end state was reached IsEnd(s), a discount factor 0⩽γ⩽1]**

&#10230; [Определение ― Цель марковского процесса принятия решений - максимизировать вознаграждение. Он определяется с помощью:, начальное состояние sstart, возможные действия Actions(s) из состояния s, вероятности перехода T(s,a,s′) из s в s′ с действием a, награды Reward(s,a,s′) из s в s′ с действием a, было ли достигнуто конечное состояние IsEnd(s), коэффициент дисконтирования 0⩽γ⩽1]

<br>


**63. Transition probabilities ― The transition probability T(s,a,s′) specifies the probability of going to state s′ after action a is taken in state s. Each s′↦T(s,a,s′) is a probability distribution, which means that:**

&#10230; Вероятности перехода ― Вероятность перехода T(s,a,s′) определяет вероятность перехода в состояние s' после того, как действие a будет выполнено в состоянии s. Каждый s′↦T(s,a,s′) представляет собой распределение вероятностей, что означает, что:

<br>


**64. states**

&#10230; состояния

<br>


**65. Policy ― A policy π is a function that maps each state s to an action a, i.e.**

&#10230; Политика ― Политика π - это функция, которая сопоставляет каждое состояние s с действием a, то есть

<br>


**66. Utility ― The utility of a path (s0,...,sk) is the discounted sum of the rewards on that path. In other words,**

&#10230; Полезность ― Полезность пути (s0,...,sk) - это дисконтированная сумма вознаграждений на этом пути. Другими словами,

<br>


**67. The figure above is an illustration of the case k=4.**

&#10230; На рисунке выше показан случай k=4.

<br>


**68. Q-value ― The Q-value of a policy π at state s with action a, also noted Qπ(s,a), is the expected utility from state s after taking action a and then following policy π. It is defined as follows:**

&#10230; Q-ценность - Q-ценность политики π в состоянии s с действием a, также отмеченное Qπ(s,a), является ожидаемой полезностью из состояния s после выполнения действия a и последующего следования политике π. Это определяется следующим образом:

<br>


**69. Value of a policy ― The value of a policy π from state s, also noted Vπ(s), is the expected utility by following policy π from state s over random paths. It is defined as follows:**

&#10230; Ценность политики ― Ценность политики π из состояния s, также обозначаемое как Vπ(s), является ожидаемой полезностью при следовании политике π из состояния s по случайным путям. Это определяется следующим образом:

<br>


**70. Remark: Vπ(s) is equal to 0 if s is an end state.**

&#10230; Примечание: Vπ(s) равно 0, если s - конечное состояние.

<br>


**71. Applications**

&#10230; Приложения

<br>


**72. [Policy evaluation ― Given a policy π, policy evaluation is an iterative algorithm that aims at estimating Vπ. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TPE, we have, with]**

&#10230; [Оценка политики ― Задана политика π, оценка политики представляет собой итеративный алгоритм. Он нацелен на оценку Vπ. Делается это следующим образом:, Инициализация: для всех состояний s, имеем:, Итерация: для t от 1 до TPE, имеем, с]

<br>


**73. Remark: by noting S the number of states, A the number of actions per state, S′ the number of successors and T the number of iterations, then the time complexity is of O(TPESS′).**

&#10230; Примечание: обозначим S - количество состояний, A - количество действий на состояние, S′ - количество преемников и T - количество итераций, то временная сложность равна O(TPESS′).

<br>


**74. Optimal Q-value ― The optimal Q-value Qopt(s,a) of state s with action a is defined to be the maximum Q-value attained by any policy starting. It is computed as follows:**

&#10230; Оптимальная Q-ценность ― Оптимальная Q-ценность Qopt(s,a) состояния s с действием a определяется как максимальная Q-ценность, достигаемая при запуске любой стратегии. Она рассчитывается следующим образом:

<br>


**75. Optimal value ― The optimal value Vopt(s) of state s is defined as being the maximum value attained by any policy. It is computed as follows:**

&#10230; Оптимальная ценность ― Оптимальная ценность Vopt(s) состояния s определяется как максимальное ценность, достигаемая любой политикой. Она рассчитывается следующим образом:

<br>


**76. actions**

&#10230; действия

<br>


**77. Optimal policy ― The optimal policy πopt is defined as being the policy that leads to the optimal values. It is defined by:**

&#10230; Оптимальная политика ― Оптимальная политика πopt определяется как политика, которая приводит к оптимальным ценностям. Она определяется:

<br>


**78. [Value iteration ― Value iteration is an algorithm that finds the optimal value Vopt as well as the optimal policy πopt. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TVI, we have:, with]**

&#10230; [Итерация ценности ― Итерация ценности - это ищущий оптимальное значение Vopt и оптимальную политику πopt алгоритм. Делается это следующим образом:, Инициализация: для всех состояний s, имеем:, Итерация: для t от 1 до TVI, имеем:, с]

<br>


**79. Remark: if we have either γ<1 or the MDP graph being acyclic, then the value iteration algorithm is guaranteed to converge to the correct answer.**

&#10230; Примечание: если у нас либо γ<1, либо граф MDP ацикличен, то алгоритм итерации значений гарантированно сходится к правильному ответу.

<br>


**80. When unknown transitions and rewards**

&#10230; Когда неизвестны переходы и награды

<br>


**81. Now, let's assume that the transition probabilities and the rewards are unknown.**

&#10230; Теперь предположим, что вероятности перехода и награды неизвестны.

<br>


**82. Model-based Monte Carlo ― The model-based Monte Carlo method aims at estimating T(s,a,s′) and Reward(s,a,s′) using Monte Carlo simulation with: **

&#10230; Основанный на модели Монте-Карло ― Основанный на модели метод Монте-Карло направлен на оценку T(s,a,s′) и Reward(s,a,s′) с использованием моделирования Монте-Карло с: 

<br>


**83. [# times (s,a,s′) occurs, and]**

&#10230; [# раз (s,a,s′) происходит, и]

<br>


**84. These estimations will be then used to deduce Q-values, including Qπ and Qopt.**

&#10230; Эти оценки затем будут использоваться для вывода Q-ценностей, включая Qπ и Qopt.

<br>


**85. Remark: model-based Monte Carlo is said to be off-policy, because the estimation does not depend on the exact policy.**

&#10230; Примечание: Основанный на модели Монте-Карло считается вне политики, потому что оценка не зависит от конкретной политики.

<br>


**86. Model-free Monte Carlo ― The model-free Monte Carlo method aims at directly estimating Qπ, as follows:**

&#10230; Безмодельный Монте-Карло ― Безмодельный метод Монте-Карло направлен на прямую оценку Qπ следующим образом:

<br>


**87. Qπ(s,a)=average of ut where st−1=s,at=a**

&#10230; Qπ(s,a)=среднее значение ut, где st−1=s,at=a

<br>


**88. where ut denotes the utility starting at step t of a given episode.**

&#10230; где ut обозначает полезность, начиная с шага t данного эпизода.

<br>


**89. Remark: model-free Monte Carlo is said to be on-policy, because the estimated value is dependent on the policy π used to generate the data.**

&#10230; Примечание: Безмодельный Монте-Карло считается действующим в соответствии с политикой, поскольку оценочное значение зависит от политики π, используемой для генерации данных.

<br>


**90. Equivalent formulation - By introducing the constant η=11+(#updates to (s,a)) and for each (s,a,u) of the training set, the update rule of model-free Monte Carlo has a convex combination formulation:**

&#10230; Эквивалентная формулировка ― Путем введения константы η=11+(#updates to (s,a)) и для каждого (s,a,u) обучающего набора, правило обновления безмодельного Монте-Карло имеет формулировку выпуклой комбинации:

<br>


**91. as well as a stochastic gradient formulation:**

&#10230; а также формулировку стохастического градиента:

<br>


**92. SARSA ― State-action-reward-state-action (SARSA) is a boostrapping method estimating Qπ by using both raw data and estimates as part of the update rule. For each (s,a,r,s′,a′), we have:**

&#10230; State-action-reward-state-action (SARSA) ― Состояние-действие-награда-состояние-действие - это оценка Qπ метода начальной загрузки с использованием как необработанных данных, так и оценок как части правила обновления. Для каждого (s,a,r,s′,a′), у нас есть:

<br>


**93. Remark: the SARSA estimate is updated on the fly as opposed to the model-free Monte Carlo one where the estimate can only be updated at the end of the episode.**

&#10230; Примечание: оценка SARSA обновляется "на лету", в отличие от оценки Монте-Карло без использования модели, где оценка может быть обновлена только в конце эпизода.

<br>


**94. Q-learning ― Q-learning is an off-policy algorithm that produces an estimate for Qopt. On each (s,a,r,s′,a′), we have:**

&#10230; Q-обучение ― Q-обучение - это алгоритм вне политики, который производит оценку Qopt. На каждой (s,a,r,s′,a′), у нас есть:

<br>


**95. Epsilon-greedy ― The epsilon-greedy policy is an algorithm that balances exploration with probability ϵ and exploitation with probability 1−ϵ. For a given state s, the policy πact is computed as follows:**

&#10230; Эпсилон-жадная ― Эпсилон-жадная политика - это алгоритм, который уравновешивает исследование с вероятностью ϵ и использование с вероятностью 1−ϵ. Для заданного состояния s политика πact вычисляется следующим образом:

<br>


**96. [with probability, random from Actions(s)]**

&#10230; [с вероятностью, случайно из Actions(s)]

<br>


**97. Game playing**

&#10230; Игровой процесс

<br>


**98. In games (e.g. chess, backgammon, Go), other agents are present and need to be taken into account when constructing our policy.**

&#10230; В играх (например, шахматы, нарды, го) присутствуют другие агенты, которые необходимо учитывать при построении нашей политики.

<br>


**99. Game tree ― A game tree is a tree that describes the possibilities of a game. In particular, each node is a decision point for a player and each root-to-leaf path is a possible outcome of the game.**

&#10230; Дерево игры ― Дерево игры - это дерево, которое описывает возможности игры. В частности, каждый узел является точкой принятия решения для игрока, а каждый путь от корня к листу - это возможный результат игры.

<br>


**100. [Two-player zero-sum game ― It is a game where each state is fully observed and such that players take turns. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, successors Succ(s,a) from states s with actions a, whether an end state was reached IsEnd(s), the agent's utility Utility(s) at end state s, the player Player(s) who controls state s]**

&#10230; [Игра для двух игроков с нулевой суммой ― это игра; в которой полностью соблюдается каждое состояние и игроки ходят по очереди. Она определяется с помощью:, начальное состояние sstart, возможные действия Actions(s) из состояния s, преемники Succ(s,a) из состояния s с действиями a, было ли достигнуто конечное состояние IsEnd(s), полезность агента Utility(s) в конечном состоянии s, игрок Player(s) контролирует состояние s]

<br>


**101. Remark: we will assume that the utility of the agent has the opposite sign of the one of the opponent.**

&#10230; Примечание: предположим, что полезность агента противоположна полезности оппонента.

<br>


**102. [Types of policies ― There are two types of policies:, Deterministic policies, noted πp(s), which are actions that player p takes in state s., Stochastic policies, noted πp(s,a)∈[0,1], which are probabilities that player p takes action a in state s.]**

&#10230; [Типы политик ― Есть два типа политик:, Детерминистские политики, обозначаются πp(s), действия игрока p в состоянии s., Стохастические политики, обозначаются πp(s,a)∈[0,1], вероятности совершения игроком p действия a в состоянии s.]

<br>


**103. Expectimax ― For a given state s, the expectimax value Vexptmax(s) is the maximum expected utility of any agent policy when playing with respect to a fixed and known opponent policy πopp. It is computed as follows:**

&#10230; Expectimax ― Для данного состояния s значение expectimax Vexptmax(s) является максимальной ожидаемой полезностью любой политики агента при игре относительно фиксированной и известной политики противника πopp. Он рассчитывается следующим образом:

<br>


**104. Remark: expectimax is the analog of value iteration for MDPs.**

&#10230; Примечание: expectimax - аналог итерации значений для MDP.

<br>


**105. Minimax ― The goal of minimax policies is to find an optimal policy against an adversary by assuming the worst case, i.e. that the opponent is doing everything to minimize the agent's utility. It is done as follows:**

&#10230; Minimax ― Цель минимаксных политик - найти оптимальную политику против противника, предполагая наихудший случай, то есть то, что противник делает все, чтобы минимизировать полезность агента. Делается это следующим образом:

<br>


**106. Remark: we can extract πmax and πmin from the minimax value Vminimax.**

&#10230; Примечание: мы можем извлечь πmax и πmin из минимаксного значения Vminimax.

<br>


**107. Minimax properties ― By noting V the value function, there are 3 properties around minimax to have in mind:**

&#10230; Свойства minimax ― Обозначим V как функцию ценности, необходимо иметь в виду 3 свойства вокруг минимакса:

<br>


**108. Property 1: if the agent were to change its policy to any πagent, then the agent would be no better off.**

&#10230; Свойство 1: если агент изменит свою политику на какой-либо πagent, то ему не будет лучше.

<br>


**109. Property 2: if the opponent changes its policy from πmin to πopp, then he will be no better off.**

&#10230; Свойство 2: если противник изменит свою политику с πmin на πopp, то ему не станет лучше.

<br>


**110. Property 3: if the opponent is known to be not playing the adversarial policy, then the minimax policy might not be optimal for the agent.**

&#10230; Свойство 3: если известно, что противник не использует состязательную политику, то минимаксная политика может быть неоптимальной для агента.

<br>


**111. In the end, we have the following relationship:**

&#10230; В итоге имеем следующие отношения:

<br>


**112. Speeding up minimax**

&#10230; Ускорение minimax

<br>


**113. Evaluation function ― An evaluation function is a domain-specific and approximate estimate of the value Vminimax(s). It is noted Eval(s).**

&#10230; Функция оценки ― Функция оценки - это зависящая от предметной области и приблизительная оценка значения Vminimax(s). Обозначается Eval(s).

<br>


**114. Remark: FutureCost(s) is an analogy for search problems.**

&#10230; Примечание: FutureCost(s) - аналог задачи поиска.

<br>


**115. Alpha-beta pruning ― Alpha-beta pruning is a domain-general exact method optimizing the minimax algorithm by avoiding the unnecessary exploration of parts of the game tree. To do so, each player keeps track of the best value they can hope for (stored in α for the maximizing player and in β for the minimizing player). At a given step, the condition β<α means that the optimal path is not going to be in the current branch as the earlier player had a better option at their disposal.**

&#10230; Альфа-бета обрезка ― Alpha-beta pruning - это общий для предметной области точный метод, оптимизирующий алгоритм минимакса, избегая ненужного исследования частей игрового дерева. Для этого каждый игрок отслеживает наилучшее значение, на которое он может надеяться (сохраняется в α для максимизирующего игрока и в β для минимизирующего игрока). На данном шаге условие β<α означает, что оптимальный путь не будет в текущей ветви, поскольку у более раннего игрока был лучший вариант в своем распоряжении.

<br>


**116. TD learning ― Temporal difference (TD) learning is used when we don't know the transitions/rewards. The value is based on exploration policy. To be able to use it, we need to know rules of the game Succ(s,a). For each (s,a,r,s′), the update is done as follows:**

&#10230; Обучение временной разнице ― Обучение Temporal difference (TD) используется, когда мы не знаем переходов/наград. Значение основано на политике разведки. Чтобы использовать его, нам необходимо знать правила игры Succ(s,a). Для каждого (s,a,r,s′) обновление выполняется следующим образом:

<br>


**117. Simultaneous games**

&#10230; Одновременные игры

<br>


**118. This is the contrary of turn-based games, where there is no ordering on the player's moves.**

&#10230; Это полная противоположность пошаговым играм, где нет упорядочивания ходов игрока.

<br>


**119. Single-move simultaneous game ― Let there be two players A and B, with given possible actions. We note V(a,b) to be A's utility if A chooses action a, B chooses action b. V is called the payoff matrix.**

&#10230; Одновременная игра с одним ходом ― пусть есть два игрока A и B с заданными возможными действиями. Мы обозначаем, что V(a,b) является полезностью A, если A выбирает действие a, B выбирает действие b. V называется матрицей выигрыша.

<br>


**120. [Strategies ― There are two main types of strategies:, A pure strategy is a single action:, A mixed strategy is a probability distribution over actions:]**

&#10230; [Стратегии ― Есть два основных типа стратегий:, Чистая стратегия - это одно действие:, Смешанная стратегия - это распределение вероятностей по действиям:]

<br>


**121. Game evaluation ― The value of the game V(πA,πB) when player A follows πA and player B follows πB is such that:**

&#10230; Оценка игры ― Ценность игры V(πA,πB), когда игрок A следует за πA, а игрок B следует за πB, такова, что:

<br>


**122. Minimax theorem ― By noting πA,πB ranging over mixed strategies, for every simultaneous two-player zero-sum game with a finite number of actions, we have:**

&#10230; Minimax теорема ― Положим, что πA,πB пробегают по смешанным стратегиям для любой одновременной игры двух игроков с нулевой суммой и конечного числа действий, у нас есть:

<br>


**123. Non-zero-sum games**

&#10230; Игры с ненулевой суммой

<br>


**124. Payoff matrix ― We define Vp(πA,πB) to be the utility for player p.**

&#10230; Матрица выигрыша ― Мы определяем Vp(πA,πB) как полезность для игрока p.

<br>


**125. Nash equilibrium ― A Nash equilibrium is (π∗A,π∗B) such that no player has an incentive to change its strategy. We have:**

&#10230; Nash equilibrium ― Равновесие по Нэшу - это (π∗A,π∗B) такое, что ни у одного игрока нет стимула изменить свою стратегию, у нас есть:

<br>


**126. and**

&#10230; и

<br>


**127. Remark: in any finite-player game with finite number of actions, there exists at least one Nash equilibrium.**

&#10230; Примечание: в любой игре с конечным числом игроков и конечным числом действий существует хотя бы одно равновесие по Нэшу.

<br>


**128. [Tree search, Backtracking search, Breadth-first search, Depth-first search, Iterative deepening]**

&#10230; [Поиск по дереву, Поиск с возвратом, Поиск в ширину, Поиск в глубину, С итеративным углублением]

<br>


**129. [Graph search, Dynamic programming, Uniform cost search]**

&#10230; [Поиск по графу, Динамическое программирование, Поиск по единой стоимости]

<br>


**130. [Learning costs, Structured perceptron]**

&#10230; [Стоимость обучения, Структурированный перцептрон]

<br>


**131. [A star search, Heuristic function, Algorithm, Consistency, correctness, Admissibility, efficiency]**

&#10230; [Алгоритм поиска A*, Эвристическая функция, Алгоритм, Консистентность, Корректность, Допустимость, Эффективность]

<br>


**132. [Relaxation, Relaxed search problem, Relaxed heuristic, Max heuristic]**

&#10230; [Ослабление, Задача с ослабленными ограничениями поиска, Ослабленная эвристика, Max эвристика]

<br>


**133. [Markov decision processes, Overview, Policy evaluation, Value iteration, Transitions, rewards]**

&#10230; [Марковские процессы принятия решений, Обзор, Оценка политики, Итерация ценности, Переходы, Награды]

<br>


**134. [Game playing, Expectimax, Minimax, Speeding up minimax, Simultaneous games, Non-zero-sum games]**

&#10230; [Игровой процесс, Expectimax, Minimax, Ускорение minimax, Одновременные игры, Игры с ненулевой суммой]

<br>


**135. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**136. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/shervinea

<br>


**137. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>


**138. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>


**139. By X and Y**

&#10230; По X и Y

<br>


**140. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Шпаргалки по искусственному интеллекту теперь доступны на русском языке.
