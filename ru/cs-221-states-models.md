**States-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-states-models)

<br>

**1. States-based models with search optimization and MDP**

&#10230; Модели на основе состояний с поисковой оптимизацией и Марковским процессом принятия решений (MDP)

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

&#10230; [Самопереход, Много родителей, Цикл, Много корней, Правильное дерево]

<br>


**7. [Search problem ― A search problem is defined with:, a starting state sstart, possible actions Actions(s) from state s, action cost Cost(s,a) from state s with action a, successor Succ(s,a) of state s after action a, whether an end state was reached IsEnd(s)]**

&#10230; [Проблема поиска ― Задача поиска определяется с помощью:, начальное состояние sstart, возможные действия Actions(s) из состояния s, стоимость действия Cost(s,a) из состояния s с действием a, преемник Succ(s,a) состояния s после действия a, было ли достигнуто конечное состояние IsEnd(s)]

<br>


**8. The objective is to find a path that minimizes the cost.**

&#10230; Цель состоит в том, чтобы найти путь, который минимизирует затраты.

<br>


**9. Backtracking search ― Backtracking search is a naive recursive algorithm that tries all possibilities to find the minimum cost path. Here, action costs can be either positive or negative.**

&#10230; Backtracking ― [Поиск с возвратом](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%B8%D1%81%D0%BA_%D1%81_%D0%B2%D0%BE%D0%B7%D0%B2%D1%80%D0%B0%D1%82%D0%BE%D0%BC) - это наивный рекурсивный алгоритм, который пробует все возможности, чтобы найти путь с минимальной стоимостью. Здесь затраты на действия могут быть как положительными, так и отрицательными.

<br>


**10. Breadth-first search (BFS) ― Breadth-first search is a graph search algorithm that does a level-by-level traversal. We can implement it iteratively with the help of a queue that stores at each step future nodes to be visited. For this algorithm, we can assume action costs to be equal to a constant c⩾0.**

&#10230; Breadth-first search (BFS) ― [Поиск в ширину](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%B8%D1%81%D0%BA_%D0%B2_%D1%88%D0%B8%D1%80%D0%B8%D0%BD%D1%83) - это алгоритм поиска по графу, который выполняет обход уровня за уровнем. Мы можем реализовать это итеративно с помощью очереди, в которой на каждом шаге хранятся будущие узлы, которые нужно посетить. Для этого алгоритма можно считать, что затраты на действия равны константе c⩾0.

<br>


**11. Depth-first search (DFS) ― Depth-first search is a search algorithm that traverses a graph by following each path as deep as it can. We can implement it recursively, or iteratively with the help of a stack that stores at each step future nodes to be visited. For this algorithm, action costs are assumed to be equal to 0.**

&#10230; [Depth-first search (DFS)](https://en.wikipedia.org/wiki/Depth-first_search) ― [Поиск в глубину](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%B8%D1%81%D0%BA_%D0%B2_%D0%B3%D0%BB%D1%83%D0%B1%D0%B8%D0%BD%D1%83) - это алгоритм поиска, который просматривает граф, прослеживая каждый путь как можно глубже. Мы можем реализовать это рекурсивно или итеративно с помощью стека, в котором на каждом шаге хранятся будущие узлы, которые необходимо посетить. Для этого алгоритма предполагается, что затраты на действия равны 0.

<br>


**12. Iterative deepening ― The iterative deepening trick is a modification of the depth-first search algorithm so that it stops after reaching a certain depth, which guarantees optimality when all action costs are equal. Here, we assume that action costs are equal to a constant c⩾0.**

&#10230; [Iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) ― [Поиск в глубину с итеративным углублением](https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D0%BE%D0%B8%D1%81%D0%BA%D0%B0#IDDFS) - это модификация алгоритма поиска в глубину, так что он останавливается после достижения определенной глубины, что гарантирует оптимальность, когда все затраты на действия равны. Здесь мы предполагаем, что затраты на действия равны константе c⩾0.

<br>


**13. Tree search algorithms summary ― By noting b the number of actions per state, d the solution depth, and D the maximum depth, we have:**

&#10230; Сводка алгоритмов поиска по дереву - отметив b количество действий на состояние, d глубину решения и D максимальную глубину, которую мы имеем:

<br>


**14. [Algorithm, Action costs, Space, Time]**

&#10230; [Алгоритм, Стоимость действий, Пространство, Время]

<br>


**15. [Backtracking search, any, Breadth-first search, Depth-first search, DFS-Iterative deepening]**

&#10230; [Backtracking search, any, Breadth-first search, Depth-first search, DFS-Iterative deepening]

<br>


**16. Graph search**

&#10230; Поиск графа

<br>


**17. This category of states-based algorithms aims at constructing optimal paths, enabling exponential savings. In this section, we will focus on dynamic programming and uniform cost search.**

&#10230; Эта категория алгоритмов на основе состояний направлена на построение оптимальных путей, обеспечивающих экспоненциальную экономию. В этом разделе мы сосредоточимся на динамическом программировании и поиске единой стоимости.

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

&#10230; Динамическое программирование (DP) ― это алгоритм поиска с возвратом и запоминанием (то есть частичные результаты сохраняются), цель которого - найти путь с минимальной стоимостью от состояния s к отправке конечного состояния. It can potentially have exponential savings compared to traditional graph search algorithms, and has the property to only work for acyclic graphs. For any given state s, the future cost is computed as follows:

<br>


**22. [if, otherwise]**

&#10230; [if, otherwise]

<br>


**23. Remark: the figure above illustrates a bottom-to-top approach whereas the formula provides the intuition of a top-to-bottom problem resolution.**

&#10230; Примечание: the figure above illustrates a bottom-to-top approach whereas the formula provides the intuition of a top-to-bottom problem resolution.

<br>


**24. Types of states ― The table below presents the terminology when it comes to states in the context of uniform cost search:**

&#10230; Types of states ― The table below presents the terminology when it comes to states in the context of uniform cost search:

<br>


**25. [State, Explanation]**

&#10230; [State, Explanation]

<br>


**26. [Explored, Frontier, Unexplored]**

&#10230; [Explored, Frontier, Unexplored]

<br>


**27. [States for which the optimal path has already been found, States seen for which we are still figuring out how to get there with the cheapest cost, States not seen yet]**

&#10230; [States for which the optimal path has already been found, States seen for which we are still figuring out how to get there with the cheapest cost, States not seen yet]

<br>


**28. Uniform cost search ― Uniform cost search (UCS) is a search algorithm that aims at finding the shortest path from a state sstart to an end state send. It explores states s in increasing order of PastCost(s) and relies on the fact that all action costs are non-negative.**

&#10230; Uniform cost search ― Uniform cost search (UCS) is a search algorithm that aims at finding the shortest path from a state sstart to an end state send. It explores states s in increasing order of PastCost(s) and relies on the fact that all action costs are non-negative.

<br>


**29. Remark 1: the UCS algorithm is logically equivalent to Dijkstra's algorithm.**

&#10230; Remark 1: the UCS algorithm is logically equivalent to Dijkstra's algorithm.

<br>


**30. Remark 2: the algorithm would not work for a problem with negative action costs, and adding a positive constant to make them non-negative would not solve the problem since this would end up being a different problem.**

&#10230; Remark 2: the algorithm would not work for a problem with negative action costs, and adding a positive constant to make them non-negative would not solve the problem since this would end up being a different problem.

<br>


**31. Correctness theorem ― When a state s is popped from the frontier F and moved to explored set E, its priority is equal to PastCost(s) which is the minimum cost path from sstart to s.**

&#10230; Correctness theorem ― When a state s is popped from the frontier F and moved to explored set E, its priority is equal to PastCost(s) which is the minimum cost path from sstart to s.

<br>


**32. Graph search algorithms summary ― By noting N the number of total states, n of which are explored before the end state send, we have:**

&#10230; Graph search algorithms summary ― By noting N the number of total states, n of which are explored before the end state send, we have:

<br>


**33. [Algorithm, Acyclicity, Costs, Time/space]**

&#10230; [Algorithm, Acyclicity, Costs, Time/space]

<br>


**34. [Dynamic programming, Uniform cost search]**

&#10230; [Dynamic programming, Uniform cost search]

<br>


**35. Remark: the complexity countdown supposes the number of possible actions per state to be constant.**

&#10230; Примечание: the complexity countdown supposes the number of possible actions per state to be constant.

<br>


**36. Learning costs**

&#10230; Learning costs

<br>


**37. Suppose we are not given the values of Cost(s,a), we want to estimate these quantities from a training set of minimizing-cost-path sequence of actions (a1,a2,...,ak).**

&#10230; Suppose we are not given the values of Cost(s,a), we want to estimate these quantities from a training set of minimizing-cost-path sequence of actions (a1,a2,...,ak).

<br>


**38. [Structured perceptron ― The structured perceptron is an algorithm aiming at iteratively learning the cost of each state-action pair. At each step, it:, decreases the estimated cost of each state-action of the true minimizing path y given by the training data, increases the estimated cost of each state-action of the current predicted path y' inferred from the learned weights.]**

&#10230; [Structured perceptron ― The structured perceptron is an algorithm aiming at iteratively learning the cost of each state-action pair. At each step, it:, decreases the estimated cost of each state-action of the true minimizing path y given by the training data, increases the estimated cost of each state-action of the current predicted path y' inferred from the learned weights.]

<br>


**39. Remark: there are several versions of the algorithm, one of which simplifies the problem to only learning the cost of each action a, and the other parametrizes Cost(s,a) to a feature vector of learnable weights.**

&#10230; Примечание: there are several versions of the algorithm, one of which simplifies the problem to only learning the cost of each action a, and the other parametrizes Cost(s,a) to a feature vector of learnable weights.

<br>


**40. A* search**

&#10230; A* search

<br>


**41. Heuristic function ― A heuristic is a function h over states s, where each h(s) aims at estimating FutureCost(s), the cost of the path from s to send.**

&#10230; Heuristic function ― A heuristic is a function h over states s, where each h(s) aims at estimating FutureCost(s), the cost of the path from s to send.

<br>


**42. Algorithm ― A∗ is a search algorithm that aims at finding the shortest path from a state s to an end state send. It explores states s in increasing order of PastCost(s)+h(s). It is equivalent to a uniform cost search with edge costs Cost′(s,a) given by:**

&#10230; Algorithm ― A∗ is a search algorithm that aims at finding the shortest path from a state s to an end state send. It explores states s in increasing order of PastCost(s)+h(s). It is equivalent to a uniform cost search with edge costs Cost′(s,a) given by:

<br>


**43. Remark: this algorithm can be seen as a biased version of UCS exploring states estimated to be closer to the end state.**

&#10230; Примечание: this algorithm can be seen as a biased version of UCS exploring states estimated to be closer to the end state.

<br>


**44. [Consistency ― A heuristic h is said to be consistent if it satisfies the two following properties:, For all states s and actions a, The end state verifies the following:]**

&#10230; [Consistency ― A heuristic h is said to be consistent if it satisfies the two following properties:, For all states s and actions a, The end state verifies the following:]

<br>


**45. Correctness ― If h is consistent, then A∗ returns the minimum cost path.**

&#10230; Correctness ― If h is consistent, then A∗ returns the minimum cost path.

<br>


**46. Admissibility ― A heuristic h is said to be admissible if we have:**

&#10230; Admissibility ― A heuristic h is said to be admissible if we have:

<br>


**47. Theorem ― Let h(s) be a given heuristic. We have:**

&#10230; Theorem ― Let h(s) be a given heuristic. We have:

<br>


**48. [consistent, admissible]**

&#10230; [consistent, admissible]

<br>


**49. Efficiency ― A* explores all states s satisfying the following equation:**

&#10230; Efficiency ― A* explores all states s satisfying the following equation:

<br>


**50. Remark: larger values of h(s) is better as this equation shows it will restrict the set of states s going to be explored.**

&#10230; Примечание: larger values of h(s) is better as this equation shows it will restrict the set of states s going to be explored.

<br>


**51. Relaxation**

&#10230; Relaxation

<br>


**52. It is a framework for producing consistent heuristics. The idea is to find closed-form reduced costs by removing constraints and use them as heuristics.**

&#10230; It is a framework for producing consistent heuristics. The idea is to find closed-form reduced costs by removing constraints and use them as heuristics.

<br>


**53. Relaxed search problem ― The relaxation of search problem P with costs Cost is noted Prel with costs Costrel, and satisfies the identity:**

&#10230; Relaxed search problem ― The relaxation of search problem P with costs Cost is noted Prel with costs Costrel, and satisfies the identity:

<br>


**54. Relaxed heuristic ― Given a relaxed search problem Prel, we define the relaxed heuristic h(s)=FutureCostrel(s) as the minimum cost path from s to an end state in the graph of costs Costrel(s,a).**

&#10230; Relaxed heuristic ― Given a relaxed search problem Prel, we define the relaxed heuristic h(s)=FutureCostrel(s) as the minimum cost path from s to an end state in the graph of costs Costrel(s,a).

<br>


**55. Consistency of relaxed heuristics ― Let Prel be a given relaxed problem. By theorem, we have:**

&#10230; Consistency of relaxed heuristics ― Let Prel be a given relaxed problem. By theorem, we have:

<br>


**56. consistent**

&#10230; consistent

<br>


**57. [Tradeoff when choosing heuristic ― We have to balance two aspects in choosing a heuristic:, Computational efficiency: h(s)=FutureCostrel(s) must be easy to compute. It has to produce a closed form, easier search and independent subproblems., Good enough approximation: the heuristic h(s) should be close to FutureCost(s) and we have thus to not remove too many constraints.]**

&#10230; [Tradeoff when choosing heuristic ― We have to balance two aspects in choosing a heuristic:, Computational efficiency: h(s)=FutureCostrel(s) must be easy to compute. It has to produce a closed form, easier search and independent subproblems., Good enough approximation: the heuristic h(s) should be close to FutureCost(s) and we have thus to not remove too many constraints.]

<br>


**58. Max heuristic ― Let h1(s), h2(s) be two heuristics. We have the following property:**

&#10230; Max heuristic ― Let h1(s), h2(s) be two heuristics. We have the following property:

<br>


**59. Markov decision processes**

&#10230; Markov decision processes

<br>


**60. In this section, we assume that performing action a from state s can lead to several states s′1,s′2,... in a probabilistic manner. In order to find our way between an initial state and an end state, our objective will be to find the maximum value policy by using Markov decision processes that help us cope with randomness and uncertainty.**

&#10230; In this section, we assume that performing action a from state s can lead to several states s′1,s′2,... in a probabilistic manner. In order to find our way between an initial state and an end state, our objective will be to find the maximum value policy by using Markov decision processes that help us cope with randomness and uncertainty.

<br>


**61. Notations**

&#10230; Notations

<br>


**62. [Definition ― The objective of a Markov decision process is to maximize rewards. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, transition probabilities T(s,a,s′) from s to s′ with action a, rewards Reward(s,a,s′) from s to s′ with action a, whether an end state was reached IsEnd(s), a discount factor 0⩽γ⩽1]**

&#10230; [Definition ― The objective of a Markov decision process is to maximize rewards. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, transition probabilities T(s,a,s′) from s to s′ with action a, rewards Reward(s,a,s′) from s to s′ with action a, whether an end state was reached IsEnd(s), a discount factor 0⩽γ⩽1]

<br>


**63. Transition probabilities ― The transition probability T(s,a,s′) specifies the probability of going to state s′ after action a is taken in state s. Each s′↦T(s,a,s′) is a probability distribution, which means that:**

&#10230; Transition probabilities ― The transition probability T(s,a,s′) specifies the probability of going to state s′ after action a is taken in state s. Each s′↦T(s,a,s′) is a probability distribution, which means that:

<br>


**64. states**

&#10230; states

<br>


**65. Policy ― A policy π is a function that maps each state s to an action a, i.e.**

&#10230; Policy ― A policy π is a function that maps each state s to an action a, i.e.

<br>


**66. Utility ― The utility of a path (s0,...,sk) is the discounted sum of the rewards on that path. In other words,**

&#10230; Utility ― The utility of a path (s0,...,sk) is the discounted sum of the rewards on that path. In other words,

<br>


**67. The figure above is an illustration of the case k=4.**

&#10230; The figure above is an illustration of the case k=4.

<br>


**68. Q-value ― The Q-value of a policy π at state s with action a, also noted Qπ(s,a), is the expected utility from state s after taking action a and then following policy π. It is defined as follows:**

&#10230; Q-value ― The Q-value of a policy π at state s with action a, also noted Qπ(s,a), is the expected utility from state s after taking action a and then following policy π. It is defined as follows:

<br>


**69. Value of a policy ― The value of a policy π from state s, also noted Vπ(s), is the expected utility by following policy π from state s over random paths. It is defined as follows:**

&#10230; Value of a policy ― The value of a policy π from state s, also noted Vπ(s), is the expected utility by following policy π from state s over random paths. It is defined as follows:

<br>


**70. Remark: Vπ(s) is equal to 0 if s is an end state.**

&#10230; Примечание: Vπ(s) is equal to 0 if s is an end state.

<br>


**71. Applications**

&#10230; Applications

<br>


**72. [Policy evaluation ― Given a policy π, policy evaluation is an iterative algorithm that aims at estimating Vπ. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TPE, we have, with]**

&#10230; [Policy evaluation ― Given a policy π, policy evaluation is an iterative algorithm that aims at estimating Vπ. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TPE, we have, with]

<br>


**73. Remark: by noting S the number of states, A the number of actions per state, S′ the number of successors and T the number of iterations, then the time complexity is of O(TPESS′).**

&#10230; Примечание: by noting S the number of states, A the number of actions per state, S′ the number of successors and T the number of iterations, then the time complexity is of O(TPESS′).

<br>


**74. Optimal Q-value ― The optimal Q-value Qopt(s,a) of state s with action a is defined to be the maximum Q-value attained by any policy starting. It is computed as follows:**

&#10230; Optimal Q-value ― The optimal Q-value Qopt(s,a) of state s with action a is defined to be the maximum Q-value attained by any policy starting. It is computed as follows:

<br>


**75. Optimal value ― The optimal value Vopt(s) of state s is defined as being the maximum value attained by any policy. It is computed as follows:**

&#10230; Optimal value ― The optimal value Vopt(s) of state s is defined as being the maximum value attained by any policy. It is computed as follows:

<br>


**76. actions**

&#10230; actions

<br>


**77. Optimal policy ― The optimal policy πopt is defined as being the policy that leads to the optimal values. It is defined by:**

&#10230; Optimal policy ― The optimal policy πopt is defined as being the policy that leads to the optimal values. It is defined by:

<br>


**78. [Value iteration ― Value iteration is an algorithm that finds the optimal value Vopt as well as the optimal policy πopt. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TVI, we have:, with]**

&#10230; [Value iteration ― Value iteration is an algorithm that finds the optimal value Vopt as well as the optimal policy πopt. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TVI, we have:, with]

<br>


**79. Remark: if we have either γ<1 or the MDP graph being acyclic, then the value iteration algorithm is guaranteed to converge to the correct answer.**

&#10230; Примечание: if we have either γ<1 or the MDP graph being acyclic, then the value iteration algorithm is guaranteed to converge to the correct answer.

<br>


**80. When unknown transitions and rewards**

&#10230; When unknown transitions and rewards

<br>


**81. Now, let's assume that the transition probabilities and the rewards are unknown.**

&#10230; Now, let's assume that the transition probabilities and the rewards are unknown.

<br>


**82. Model-based Monte Carlo ― The model-based Monte Carlo method aims at estimating T(s,a,s′) and Reward(s,a,s′) using Monte Carlo simulation with: **

&#10230; Model-based Monte Carlo ― The model-based Monte Carlo method aims at estimating T(s,a,s′) and Reward(s,a,s′) using Monte Carlo simulation with: 

<br>


**83. [# times (s,a,s′) occurs, and]**

&#10230; [# times (s,a,s′) occurs, and]

<br>


**84. These estimations will be then used to deduce Q-values, including Qπ and Qopt.**

&#10230; These estimations will be then used to deduce Q-values, including Qπ and Qopt.

<br>


**85. Remark: model-based Monte Carlo is said to be off-policy, because the estimation does not depend on the exact policy.**

&#10230; Примечание: model-based Monte Carlo is said to be off-policy, because the estimation does not depend on the exact policy.

<br>


**86. Model-free Monte Carlo ― The model-free Monte Carlo method aims at directly estimating Qπ, as follows:**

&#10230; Model-free Monte Carlo ― The model-free Monte Carlo method aims at directly estimating Qπ, as follows:

<br>


**87. Qπ(s,a)=average of ut where st−1=s,at=a**

&#10230; Qπ(s,a)=average of ut where st−1=s,at=a

<br>


**88. where ut denotes the utility starting at step t of a given episode.**

&#10230; where ut denotes the utility starting at step t of a given episode.

<br>


**89. Remark: model-free Monte Carlo is said to be on-policy, because the estimated value is dependent on the policy π used to generate the data.**

&#10230; Примечание: model-free Monte Carlo is said to be on-policy, because the estimated value is dependent on the policy π used to generate the data.

<br>


**90. Equivalent formulation - By introducing the constant η=11+(#updates to (s,a)) and for each (s,a,u) of the training set, the update rule of model-free Monte Carlo has a convex combination formulation:**

&#10230; Equivalent formulation - By introducing the constant η=11+(#updates to (s,a)) and for each (s,a,u) of the training set, the update rule of model-free Monte Carlo has a convex combination formulation:

<br>


**91. as well as a stochastic gradient formulation:**

&#10230; as well as a stochastic gradient formulation:

<br>


**92. SARSA ― State-action-reward-state-action (SARSA) is a boostrapping method estimating Qπ by using both raw data and estimates as part of the update rule. For each (s,a,r,s′,a′), we have:**

&#10230; SARSA ― State-action-reward-state-action (SARSA) is a boostrapping method estimating Qπ by using both raw data and estimates as part of the update rule. For each (s,a,r,s′,a′), we have:

<br>


**93. Remark: the SARSA estimate is updated on the fly as opposed to the model-free Monte Carlo one where the estimate can only be updated at the end of the episode.**

&#10230; Примечание: the SARSA estimate is updated on the fly as opposed to the model-free Monte Carlo one where the estimate can only be updated at the end of the episode.

<br>


**94. Q-learning ― Q-learning is an off-policy algorithm that produces an estimate for Qopt. On each (s,a,r,s′,a′), we have:**

&#10230; Q-learning ― Q-learning is an off-policy algorithm that produces an estimate for Qopt. On each (s,a,r,s′,a′), we have:

<br>


**95. Epsilon-greedy ― The epsilon-greedy policy is an algorithm that balances exploration with probability ϵ and exploitation with probability 1−ϵ. For a given state s, the policy πact is computed as follows:**

&#10230; Epsilon-greedy ― The epsilon-greedy policy is an algorithm that balances exploration with probability ϵ and exploitation with probability 1−ϵ. For a given state s, the policy πact is computed as follows:

<br>


**96. [with probability, random from Actions(s)]**

&#10230; [with probability, random from Actions(s)]

<br>


**97. Game playing**

&#10230; Game playing

<br>


**98. In games (e.g. chess, backgammon, Go), other agents are present and need to be taken into account when constructing our policy.**

&#10230; In games (e.g. chess, backgammon, Go), other agents are present and need to be taken into account when constructing our policy.

<br>


**99. Game tree ― A game tree is a tree that describes the possibilities of a game. In particular, each node is a decision point for a player and each root-to-leaf path is a possible outcome of the game.**

&#10230; Game tree ― A game tree is a tree that describes the possibilities of a game. In particular, each node is a decision point for a player and each root-to-leaf path is a possible outcome of the game.

<br>


**100. [Two-player zero-sum game ― It is a game where each state is fully observed and such that players take turns. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, successors Succ(s,a) from states s with actions a, whether an end state was reached IsEnd(s), the agent's utility Utility(s) at end state s, the player Player(s) who controls state s]**

&#10230; [Two-player zero-sum game ― It is a game where each state is fully observed and such that players take turns. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, successors Succ(s,a) from states s with actions a, whether an end state was reached IsEnd(s), the agent's utility Utility(s) at end state s, the player Player(s) who controls state s]

<br>


**101. Remark: we will assume that the utility of the agent has the opposite sign of the one of the opponent.**

&#10230; Примечание: we will assume that the utility of the agent has the opposite sign of the one of the opponent.

<br>


**102. [Types of policies ― There are two types of policies:, Deterministic policies, noted πp(s), which are actions that player p takes in state s., Stochastic policies, noted πp(s,a)∈[0,1], which are probabilities that player p takes action a in state s.]**

&#10230; [Types of policies ― There are two types of policies:, Deterministic policies, noted πp(s), which are actions that player p takes in state s., Stochastic policies, noted πp(s,a)∈[0,1], which are probabilities that player p takes action a in state s.]

<br>


**103. Expectimax ― For a given state s, the expectimax value Vexptmax(s) is the maximum expected utility of any agent policy when playing with respect to a fixed and known opponent policy πopp. It is computed as follows:**

&#10230; Expectimax ― For a given state s, the expectimax value Vexptmax(s) is the maximum expected utility of any agent policy when playing with respect to a fixed and known opponent policy πopp. It is computed as follows:

<br>


**104. Remark: expectimax is the analog of value iteration for MDPs.**

&#10230; Примечание: expectimax is the analog of value iteration for MDPs.

<br>


**105. Minimax ― The goal of minimax policies is to find an optimal policy against an adversary by assuming the worst case, i.e. that the opponent is doing everything to minimize the agent's utility. It is done as follows:**

&#10230; Minimax ― The goal of minimax policies is to find an optimal policy against an adversary by assuming the worst case, i.e. that the opponent is doing everything to minimize the agent's utility. It is done as follows:

<br>


**106. Remark: we can extract πmax and πmin from the minimax value Vminimax.**

&#10230; Примечание: we can extract πmax and πmin from the minimax value Vminimax.

<br>


**107. Minimax properties ― By noting V the value function, there are 3 properties around minimax to have in mind:**

&#10230; Minimax properties ― By noting V the value function, there are 3 properties around minimax to have in mind:

<br>


**108. Property 1: if the agent were to change its policy to any πagent, then the agent would be no better off.**

&#10230; Property 1: if the agent were to change its policy to any πagent, then the agent would be no better off.

<br>


**109. Property 2: if the opponent changes its policy from πmin to πopp, then he will be no better off.**

&#10230; Property 2: if the opponent changes its policy from πmin to πopp, then he will be no better off.

<br>


**110. Property 3: if the opponent is known to be not playing the adversarial policy, then the minimax policy might not be optimal for the agent.**

&#10230; Property 3: if the opponent is known to be not playing the adversarial policy, then the minimax policy might not be optimal for the agent.

<br>


**111. In the end, we have the following relationship:**

&#10230; In the end, we have the following relationship:

<br>


**112. Speeding up minimax**

&#10230; Speeding up minimax

<br>


**113. Evaluation function ― An evaluation function is a domain-specific and approximate estimate of the value Vminimax(s). It is noted Eval(s).**

&#10230; Evaluation function ― An evaluation function is a domain-specific and approximate estimate of the value Vminimax(s). It is noted Eval(s).

<br>


**114. Remark: FutureCost(s) is an analogy for search problems.**

&#10230; Примечание: FutureCost(s) is an analogy for search problems.

<br>


**115. Alpha-beta pruning ― Alpha-beta pruning is a domain-general exact method optimizing the minimax algorithm by avoiding the unnecessary exploration of parts of the game tree. To do so, each player keeps track of the best value they can hope for (stored in α for the maximizing player and in β for the minimizing player). At a given step, the condition β<α means that the optimal path is not going to be in the current branch as the earlier player had a better option at their disposal.**

&#10230; Alpha-beta pruning ― Alpha-beta pruning is a domain-general exact method optimizing the minimax algorithm by avoiding the unnecessary exploration of parts of the game tree. To do so, each player keeps track of the best value they can hope for (stored in α for the maximizing player and in β for the minimizing player). At a given step, the condition β<α means that the optimal path is not going to be in the current branch as the earlier player had a better option at their disposal.

<br>


**116. TD learning ― Temporal difference (TD) learning is used when we don't know the transitions/rewards. The value is based on exploration policy. To be able to use it, we need to know rules of the game Succ(s,a). For each (s,a,r,s′), the update is done as follows:**

&#10230; TD learning ― Temporal difference (TD) learning is used when we don't know the transitions/rewards. The value is based on exploration policy. To be able to use it, we need to know rules of the game Succ(s,a). For each (s,a,r,s′), the update is done as follows:

<br>


**117. Simultaneous games**

&#10230; Simultaneous games

<br>


**118. This is the contrary of turn-based games, where there is no ordering on the player's moves.**

&#10230; This is the contrary of turn-based games, where there is no ordering on the player's moves.

<br>


**119. Single-move simultaneous game ― Let there be two players A and B, with given possible actions. We note V(a,b) to be A's utility if A chooses action a, B chooses action b. V is called the payoff matrix.**

&#10230; Single-move simultaneous game ― Let there be two players A and B, with given possible actions. We note V(a,b) to be A's utility if A chooses action a, B chooses action b. V is called the payoff matrix.

<br>


**120. [Strategies ― There are two main types of strategies:, A pure strategy is a single action:, A mixed strategy is a probability distribution over actions:]**

&#10230; [Strategies ― There are two main types of strategies:, A pure strategy is a single action:, A mixed strategy is a probability distribution over actions:]

<br>


**121. Game evaluation ― The value of the game V(πA,πB) when player A follows πA and player B follows πB is such that:**

&#10230; Game evaluation ― The value of the game V(πA,πB) when player A follows πA and player B follows πB is such that:

<br>


**122. Minimax theorem ― By noting πA,πB ranging over mixed strategies, for every simultaneous two-player zero-sum game with a finite number of actions, we have:**

&#10230; Minimax theorem ― By noting πA,πB ranging over mixed strategies, for every simultaneous two-player zero-sum game with a finite number of actions, we have:

<br>


**123. Non-zero-sum games**

&#10230; Non-zero-sum games

<br>


**124. Payoff matrix ― We define Vp(πA,πB) to be the utility for player p.**

&#10230; Payoff matrix ― We define Vp(πA,πB) to be the utility for player p.

<br>


**125. Nash equilibrium ― A Nash equilibrium is (π∗A,π∗B) such that no player has an incentive to change its strategy. We have:**

&#10230; Nash equilibrium ― A Nash equilibrium is (π∗A,π∗B) such that no player has an incentive to change its strategy. We have:

<br>


**126. and**

&#10230; and

<br>


**127. Remark: in any finite-player game with finite number of actions, there exists at least one Nash equilibrium.**

&#10230; Примечание: in any finite-player game with finite number of actions, there exists at least one Nash equilibrium.

<br>


**128. [Tree search, Backtracking search, Breadth-first search, Depth-first search, Iterative deepening]**

&#10230; [Tree search, Backtracking search, Breadth-first search, Depth-first search, Iterative deepening]

<br>


**129. [Graph search, Dynamic programming, Uniform cost search]**

&#10230; [Graph search, Dynamic programming, Uniform cost search]

<br>


**130. [Learning costs, Structured perceptron]**

&#10230; [Learning costs, Structured perceptron]

<br>


**131. [A star search, Heuristic function, Algorithm, Consistency, correctness, Admissibility, efficiency]**

&#10230; [A star search, Heuristic function, Algorithm, Consistency, correctness, Admissibility, efficiency]

<br>


**132. [Relaxation, Relaxed search problem, Relaxed heuristic, Max heuristic]**

&#10230; [Relaxation, Relaxed search problem, Relaxed heuristic, Max heuristic]

<br>


**133. [Markov decision processes, Overview, Policy evaluation, Value iteration, Transitions, rewards]**

&#10230; [Markov decision processes, Overview, Policy evaluation, Value iteration, Transitions, rewards]

<br>


**134. [Game playing, Expectimax, Minimax, Speeding up minimax, Simultaneous games, Non-zero-sum games]**

&#10230; [Game playing, Expectimax, Minimax, Speeding up minimax, Simultaneous games, Non-zero-sum games]

<br>


**135. View PDF version on GitHub**

&#10230; View PDF version on GitHub

<br>


**136. Original authors**

&#10230; Original authors

<br>


**137. Translated by X, Y and Z**

&#10230; Translated by X, Y and Z

<br>


**138. Reviewed by X, Y and Z**

&#10230; Reviewed by X, Y and Z

<br>


**139. By X and Y**

&#10230; By X and Y

<br>


**140. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; The Artificial Intelligence cheatsheets are now available in [target language].
