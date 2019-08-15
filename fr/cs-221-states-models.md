**States-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-states-models)

<br>

**1. States-based models with search optimization and MDP**

&#10230; Modèles basés sur les états, utilisés pour optimiser le parcours et les MDPs

<br>


**2. Search optimization**

&#10230; Optimisation de parcours

<br>


**3. In this section, we assume that by accomplishing action a from state s, we deterministically arrive in state Succ(s,a). The goal here is to determine a sequence of actions (a1,a2,a3,a4,...) that starts from an initial state and leads to an end state. In order to solve this kind of problem, our objective will be to find the minimum cost path by using states-based models.**

&#10230; Dans cette section, nous supposons qu'en effectuant une action a à partir d'un état s, on arrive de manière déterministe à l'état Succ(s,a). Le but de cette étude est de déterminer une séquence d'actions (a1,a2,a3,a4,...) démarrant d'un état initial et aboutissant à un état final. Pour y parvenir, notre objectif est de minimiser le coût associés à ces actions à l'aide de modèles basés sur les états (state-based model en anglais).

<br>


**4. Tree search**

&#10230; Parcours d'arbre

<br>


**5. This category of states-based algorithms explores all possible states and actions. It is quite memory efficient, and is suitable for huge state spaces but the runtime can become exponential in the worst cases.**

&#10230; Cette catégorie d'algorithmes explore tous les états et actions possibles. Même si leur consommation en mémoire est raisonnable et peut supporter des espaces d'états de taille très grande, ce type d'algorithmes est néanmoins susceptible d'engendrer des complexités en temps exponentielles dans le pire des cas.

<br>


**6. [Self-loop, More than a parent, Cycle, More than a root, Valid tree]**

&#10230; [Boucle, Plus d'un parent, Cycle, Plus d'une racine, Arbre valide]

<br>


**7. [Search problem ― A search problem is defined with:, a starting state sstart, possible actions Actions(s) from state s, action cost Cost(s,a) from state s with action a, successor Succ(s,a) of state s after action a, whether an end state was reached IsEnd(s)]**

&#10230; [Problème de recherche - Un problème de recherche est défini par :, un état de départ sstart, des actions Actions(s) pouvant être effectuées depuis l'état s, le coût de l'action Cost(s,a) depuis l'état s pour effectuer l'action a, le successeur Succ(s,a) de l'état s après avoir effectué l'action a, la connaissance d'avoir atteint ou non un état final IsEnd(s)]

<br>


**8. The objective is to find a path that minimizes the cost.**

&#10230; L'objectif est de trouver un chemin minimisant le coût total des actions utilisées.

<br>


**9. Backtracking search ― Backtracking search is a naive recursive algorithm that tries all possibilities to find the minimum cost path. Here, action costs can be either positive or negative.**

&#10230; Retour sur trace - L'algorithme de retour sur trace (en anglais backtracking search) est un algorithme récursif explorant naïvement toutes les possibilités jusqu'à trouver le chemin de coût minimal.

<br>


**10. Breadth-first search (BFS) ― Breadth-first search is a graph search algorithm that does a level-by-level traversal. We can implement it iteratively with the help of a queue that stores at each step future nodes to be visited. For this algorithm, we can assume action costs to be equal to a constant c⩾0.**

&#10230; Parcours en largeur (BFS) - L'algorithme de parcours en largeur (en anglais breadth-first search ou BFS) est un algorithme de parcours de graphe traversant chaque niveau de manière successive. On peut le coder de manière itérative à l'aide d'une queue stockant à chaque étape les prochains nœuds à visiter. Cet algorithme suppose que le coût de toutes les actions est égal à une constante c⩾0.

<br>


**11. Depth-first search (DFS) ― Depth-first search is a search algorithm that traverses a graph by following each path as deep as it can. We can implement it recursively, or iteratively with the help of a stack that stores at each step future nodes to be visited. For this algorithm, action costs are assumed to be equal to 0.**

&#10230; Parcours en profondeur (DFS) - L'algorithme de parcours en profondeur (en anglais depth-first search ou DFS) est un algorithme de parcours de graphe traversant chaque chemin qu'il emprunte aussi loin que possible. On peut le coder de manière récursive, ou itérative à l'aide d'une pile qui stocke à chaque étape les prochains nœuds à visiter. Cet algorithme suppose que le coût de toutes les actions est égal à 0.

<br>


**12. Iterative deepening ― The iterative deepening trick is a modification of the depth-first search algorithm so that it stops after reaching a certain depth, which guarantees optimality when all action costs are equal. Here, we assume that action costs are equal to a constant c⩾0.**

&#10230; Approfondissement itératif - L'astuce de l'approfondissement itératif (en anglais iterative deepening) est une modification de l'algorithme de DFS qui l'arrête après avoir atteint une certaine profondeur, garantissant l'optimalité de la solution trouvée quand toutes les actions ont un même coût constant c⩾0.

<br>


**13. Tree search algorithms summary ― By noting b the number of actions per state, d the solution depth, and D the maximum depth, we have:**

&#10230; Récapitulatif des algorithmes de parcours d'arbre - En notant b le nombre d'actions par état, d la profondeur de la solution et D la profondeur maximale, on a :

<br>


**14. [Algorithm, Action costs, Space, Time]**

&#10230; [Algorithme, Coût des actions, Espace, Temps]

<br>


**15. [Backtracking search, any, Breadth-first search, Depth-first search, DFS-Iterative deepening]**

&#10230; [Retour sur trace, peu importe, Parcours en largeur, Parcours en profondeur, DFS-approfondissement itératif]

<br>


**16. Graph search**

&#10230; Parcours de graphe

<br>


**17. This category of states-based algorithms aims at constructing optimal paths, enabling exponential savings. In this section, we will focus on dynamic programming and uniform cost search.**

&#10230; Cette catégorie d'algorithmes basés sur les états vise à trouver des chemins optimaux avec une complexité moins grande qu'exponentielle. Dans cette section, nous allons nous concentrer sur la programmation dynamique et la recherche à coût uniforme.

<br>


**18. Graph ― A graph is comprised of a set of vertices V (also called nodes) as well as a set of edges E (also called links).**

&#10230; Graphe - Un graphe se compose d'un ensemble de sommets V (aussi appelés noeuds) et d'arêtes E (appelés arcs lorsque le graphe est orienté).

<br>


**19. Remark: a graph is said to be acylic when there is no cycle.**

&#10230; Remarque : un graphe est dit être acyclique lorsqu'il ne contient pas de cycle.

<br>


**20. State ― A state is a summary of all past actions sufficient to choose future actions optimally.**

&#10230; État - Un état contient le résumé des actions passées suffisant pour choisir les actions futures de manière optimale.

<br>


**21. Dynamic programming ― Dynamic programming (DP) is a backtracking search algorithm with memoization (i.e. partial results are saved) whose goal is to find a minimum cost path from state s to an end state send. It can potentially have exponential savings compared to traditional graph search algorithms, and has the property to only work for acyclic graphs. For any given state s, the future cost is computed as follows:**

&#10230; Programmation dynamique - La programmation dynamique (en anglais dynamic programming ou DP) est un algorithme de recherche de type retour sur trace qui utilise le principe de mémoïsation (i.e. les résultats intermédiaires sont enregistrés) et ayant pour but de trouver le chemin à coût minimal allant de l'état s à l'état final send. Cette procédure peut potentiellement engendrer des économies exponentielles si on la compare aux algorithmes de parcours de graphe traditionnels, et a la propriété de ne marcher que dans le cas de graphes acycliques. Pour un état s donné, le coût futur est calculé de la manière suivante :

<br>


**22. [if, otherwise]**

&#10230; [si, sinon]

<br>


**23. Remark: the figure above illustrates a bottom-to-top approach whereas the formula provides the intuition of a top-to-bottom problem resolution.**

&#10230; Remarque : la figure ci-dessus illustre une approche ascendante alors que la formule nous donne l'intuition d'une résolution avec une approche descendante.

<br>


**24. Types of states ― The table below presents the terminology when it comes to states in the context of uniform cost search:**

&#10230; Types d'états - La table ci-dessous présente la terminologie relative aux états dans le contexte de la recherche à coût uniforme :

<br>


**25. [State, Explanation]**

&#10230; [État, Explication]

<br>


**26. [Explored, Frontier, Unexplored]**

&#10230; [Exploré, Frontière, Inexploré]

<br>


**27. [States for which the optimal path has already been found, States seen for which we are still figuring out how to get there with the cheapest cost, States not seen yet]**

&#10230; [États pour lesquels le chemin optimal a déjà été trouvé, États rencontrés mais pour lesquels on se demande toujours comment s'y rendre avec un coût minimal, États non rencontrés jusqu'à présent]

<br>


**28. Uniform cost search ― Uniform cost search (UCS) is a search algorithm that aims at finding the shortest path from a state sstart to an end state send. It explores states s in increasing order of PastCost(s) and relies on the fact that all action costs are non-negative.**

&#10230; Recherche à coût uniforme - La recherche à coût uniforme (uniform cost search ou UCS en anglais) est un algorithme de recherche qui a pour but de trouver le chemin le plus court entre les états sstart et send. Celui-ci explore les états s en les triant par coût croissant de PastCost(s) et repose sur le fait que toutes les actions ont un coût non négatif.

<br>


**29. Remark 1: the UCS algorithm is logically equivalent to Dijkstra's algorithm.**

&#10230; Remarque 1 : UCS fonctionne de la même manière que l'algorithme de Dijkstra.

<br>


**30. Remark 2: the algorithm would not work for a problem with negative action costs, and adding a positive constant to make them non-negative would not solve the problem since this would end up being a different problem.**

&#10230; Remarque 2 : cet algorithme ne marche pas sur une configuration contenant des actions à coût négatif. Quelqu'un pourrait penser à ajouter une constante positive à tous les coûts, mais cela ne résoudrait rien puisque le problème résultant serait différent.

<br>


**31. Correctness theorem ― When a state s is popped from the frontier F and moved to explored set E, its priority is equal to PastCost(s) which is the minimum cost path from sstart to s.**

&#10230; Théorème de correction - Lorsqu'un état s passe de la frontière F à l'ensemble exploré E, sa priorité est égale à PastCost(s), représentant le chemin de coût minimal allant de sstart à s.

<br>


**32. Graph search algorithms summary ― By noting N the number of total states, n of which are explored before the end state send, we have:**

&#10230; Récapitulatif des algorithmes de parcours de graphe - En notant N le nombre total d'états dont n sont explorés avant l'état final send, on a :

<br>


**33. [Algorithm, Acyclicity, Costs, Time/space]**

&#10230; [Algorithme, Acyclicité, Coûts, Temps/Espace]

<br>


**34. [Dynamic programming, Uniform cost search]**

&#10230; [Programmation dynamique, Recherche à coût uniforme]

<br>


**35. Remark: the complexity countdown supposes the number of possible actions per state to be constant.**

&#10230; Remarque : ce décompte de la complexité suppose que le nombre d'actions possibles à partir de chaque état est constant.

<br>


**36. Learning costs**

&#10230; Apprendre les coûts

<br>


**37. Suppose we are not given the values of Cost(s,a), we want to estimate these quantities from a training set of minimizing-cost-path sequence of actions (a1,a2,...,ak).**

&#10230; Supposons que nous ne sommes pas donnés les valeurs de Cost(s,a). Nous souhaitons estimer ces quantités à partir d'un ensemble d'apprentissage de chemins à coût minimaux d'actions (a1,a2,...,ak).

<br>


**38. [Structured perceptron ― The structured perceptron is an algorithm aiming at iteratively learning the cost of each state-action pair. At each step, it:, decreases the estimated cost of each state-action of the true minimizing path y given by the training data, increases the estimated cost of each state-action of the current predicted path y' inferred from the learned weights.]**

&#10230; [Perceptron structuré - L'algorithme du perceptron structuré vise à apprendre de manière itérative les coûts des paires état-action. À chaque étape, il :, fait décroître le coût estimé de chaque état-action du vrai chemin minimisant y donné par la base d'apprentissage, fait croître le coût estimé de chaque état-action du chemin y' prédit comme étant minimisant par les paramètres appris par l'algorithme.]

<br>


**39. Remark: there are several versions of the algorithm, one of which simplifies the problem to only learning the cost of each action a, and the other parametrizes Cost(s,a) to a feature vector of learnable weights.**

&#10230; Remarque : plusieurs versions de cette algorithme existent, l'une d'elles réduisant ce problème à l'apprentissage du coût de chaque action a et l'autre paramétrisant chaque Cost(s,a) à un vecteur de paramètres pouvant être appris.

<br>


**40. A* search**

&#10230; Algorithme A*

<br>


**41. Heuristic function ― A heuristic is a function h over states s, where each h(s) aims at estimating FutureCost(s), the cost of the path from s to send.**

&#10230; Fonction heuristique - Une heuristique est une fonction h opérant sur les états s, où chaque h(s) vise à estimer FutureCost(s), le coût du chemin optimal allant de s à send.

<br>


**42. Algorithm ― A∗ is a search algorithm that aims at finding the shortest path from a state s to an end state send. It explores states s in increasing order of PastCost(s)+h(s). It is equivalent to a uniform cost search with edge costs Cost′(s,a) given by:**

&#10230; Algorithme - A* est un algorithme de recherche visant à trouver le chemin le plus court entre un état s et un état final send. Il le fait en explorant les états s triés par ordre croissant de PastCost(s)+h(s). Cela revient à utiliser l'algorithme UCS où chaque arête est associée au coût Cost′(s,a) donné par :

<br>


**43. Remark: this algorithm can be seen as a biased version of UCS exploring states estimated to be closer to the end state.**

&#10230; Remarque : cet algorithme peut être vu comme une version biaisée de UCS explorant les états estimés comme étant plus proches de l'état final.

<br>


**44. [Consistency ― A heuristic h is said to be consistent if it satisfies the two following properties:, For all states s and actions a, The end state verifies the following:]**

&#10230; [Consistance - Une heuristique h est dite consistante si elle satisfait les deux propriétés suivantes :, Pour tous états s et actions a, L'état final vérifie la propriété :]

<br>


**45. Correctness ― If h is consistent, then A∗ returns the minimum cost path.**

&#10230; Correction - Si h est consistante, alors A* renvoie le chemin de coût minimal.

<br>


**46. Admissibility ― A heuristic h is said to be admissible if we have:**

&#10230; Admissibilité - Une heuristique est dite admissible si l'on a :

<br>


**47. Theorem ― Let h(s) be a given heuristic. We have:**

&#10230; Théorème - Soit h(s) une heuristique. On a :

<br>


**48. [consistent, admissible]**

&#10230; [consistante, admissible]

<br>


**49. Efficiency ― A* explores all states s satisfying the following equation:**

&#10230; Efficacité - A* explore les états s satisfaisant l'équation :

<br>


**50. Remark: larger values of h(s) is better as this equation shows it will restrict the set of states s going to be explored.**

&#10230; Remarque : avoir h(s) élevé est préférable puisque cette équation montre que le nombre d'états s à explorer est alors réduit.

<br>


**51. Relaxation**

&#10230; Relaxation

<br>


**52. It is a framework for producing consistent heuristics. The idea is to find closed-form reduced costs by removing constraints and use them as heuristics.**

&#10230; C'est un type de procédure permettant de produire des heuristiques consistantes. L'idée est de trouver une fonction de coût facile à exprimer en enlevant des contraintes au problème, et ensuite l'utiliser en tant qu'heuristique.

<br>


**53. Relaxed search problem ― The relaxation of search problem P with costs Cost is noted Prel with costs Costrel, and satisfies the identity:**

&#10230; Relaxation d'un problème de recherche - La relaxation d'un problème de recherche P aux coûts Cost est noté Prel avec coûts Costrel, et vérifie la relation :

<br>


**54. Relaxed heuristic ― Given a relaxed search problem Prel, we define the relaxed heuristic h(s)=FutureCostrel(s) as the minimum cost path from s to an end state in the graph of costs Costrel(s,a).**

&#10230; Relaxation d'une heuristique - Étant donné la relaxation d'un problème de recherche Prel, on définit l'heuristique relaxée h(s)=FutureCostrel(s) comme étant le chemin de coût minimal allant de s à un état final dans le graphe de fonction de coût Costrel(s,a).

<br>


**55. Consistency of relaxed heuristics ― Let Prel be a given relaxed problem. By theorem, we have:**

&#10230; Consistance de la relaxation d'heuristiques - Soit Prel une relaxation d'un problème de recherche. Par théorème, on a :

<br>


**56. consistent**

&#10230; consistante

<br>


**57. [Tradeoff when choosing heuristic ― We have to balance two aspects in choosing a heuristic:, Computational efficiency: h(s)=FutureCostrel(s) must be easy to compute. It has to produce a closed form, easier search and independent subproblems., Good enough approximation: the heuristic h(s) should be close to FutureCost(s) and we have thus to not remove too many constraints.]**

&#10230; [Compromis lors du choix d'heuristique - Le choix d'heuristique se repose sur un compromis entre :, Complexité de calcul : h(s)=FutureCostrel(s) doit être facile à calculer. De manière préférable, cette fonction peut s'exprimer de manière explicite et elle permet de diviser le problème en sous-parties indépendantes.]

<br>


**58. Max heuristic ― Let h1(s), h2(s) be two heuristics. We have the following property:**

&#10230; Heuristique max - Soient h1(s) et h2(s) deux heuristiques. On a la propriété suivante :

<br>


**59. Markov decision processes**

&#10230; Processus de décision markovien

<br>


**60. In this section, we assume that performing action a from state s can lead to several states s′1,s′2,... in a probabilistic manner. In order to find our way between an initial state and an end state, our objective will be to find the maximum value policy by using Markov decision processes that help us cope with randomness and uncertainty.**

&#10230; Dans cette section, on suppose qu'effectuer l'action a à partir de l'état s peut mener de manière probabiliste à plusieurs états s′1,s′2,... Dans le but de trouver ce qu'il faudrait faire entre un état initial et un état final, on souhaite trouver une stratégie maximisant la quantité des récompenses en utilisant un outil adapté à l'imprévisibilité et l'incertitude : les processus de décision markoviens.

<br>


**61. Notations**

&#10230; Notations

<br>


**62. [Definition ― The objective of a Markov decision process is to maximize rewards. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, transition probabilities T(s,a,s′) from s to s′ with action a, rewards Reward(s,a,s′) from s to s′ with action a, whether an end state was reached IsEnd(s), a discount factor 0⩽γ⩽1]**

&#10230; [Définition - L'objectif d'un processus de décision markovien (en anglais Markov decision process ou MDP) est de maximiser la quantité de récompenses. Un tel problème est défini par :, un état de départ sstart, l'ensemble des actions Actions(s) pouvant être effectuées à partir de l'état s, la probabilité de transition T(s,a,s′) de l'état s vers l'état s' après avoir pris l'action a, la récompense Reward(s,a,s′) pour être passé de l'état s à l'état s' après avoir pris l'action a, la connaissance d'avoir atteint ou non un état final IsEnd(s), un facteur de dévaluation 0⩽γ⩽1]

<br>


**63. Transition probabilities ― The transition probability T(s,a,s′) specifies the probability of going to state s′ after action a is taken in state s. Each s′↦T(s,a,s′) is a probability distribution, which means that:**

&#10230; Probabilités de transitions - La probabilité de transition T(s,a,s′) représente la probabilité de transitionner vers l'état s' après avoir effectué l'action a en étant dans l'état s. Chaque s′↦T(s,a,s′) est une loi de probabilité :

<br>


**64. states**

&#10230; états

<br>


**65. Policy ― A policy π is a function that maps each state s to an action a, i.e.**

&#10230; Politique - Une politique π est une fonction liant chaque état s à une action a, i.e. :

<br>


**66. Utility ― The utility of a path (s0,...,sk) is the discounted sum of the rewards on that path. In other words,**

&#10230; Utilité - L'utilité d'un chemin (s0,...,sk) est la somme des récompenses dévaluées récoltées sur ce chemin. En d'autres termes,

<br>


**67. The figure above is an illustration of the case k=4.**

&#10230; La figure ci-dessus illustre le cas k=4.

<br>


**68. Q-value ― The Q-value of a policy π at state s with action a, also noted Qπ(s,a), is the expected utility from state s after taking action a and then following policy π. It is defined as follows:**

&#10230; Q-value - La fonction de valeur des états-actions (Q-value en anglais) d'une politique π évaluée à l'état s avec l'action a, aussi notée Qπ(s,a), est l'espérance de l'utilité partant de l'état s avec l'action a et adoptant ensuite la politique π. Cette fonction est définie par :

<br>


**69. Value of a policy ― The value of a policy π from state s, also noted Vπ(s), is the expected utility by following policy π from state s over random paths. It is defined as follows:**

&#10230; Fonction de valeur des états d'une politique - La fonction de valeur des états d'une politique π évaluée à l'état s, aussi notée Vπ(s), est l'espérance de l'utilité partant de l'état s et adoptant ensuite la politique π. Cette fonction est définie par :

<br>


**70. Remark: Vπ(s) is equal to 0 if s is an end state.**

&#10230; Remarque : Vπ(s) vaut 0 si s est un état final.

<br>


**71. Applications**

&#10230; Applications

<br>


**72. [Policy evaluation ― Given a policy π, policy evaluation is an iterative algorithm that aims at estimating Vπ. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TPE, we have, with]**

&#10230; [Évaluation d'une politique - Étant donnée une politique π, on peut utiliser l'algorithme itératif d'évaluation de politiques (en anglais policy evaluation) pour estimer Vπ :, Initialisation : pour tous les états s, on a, Itération : pour t allant de 1 à TPE, on a, avec]

<br>


**73. Remark: by noting S the number of states, A the number of actions per state, S′ the number of successors and T the number of iterations, then the time complexity is of O(TPESS′).**

&#10230; Remarque : en notant S le nombre d'états, A le nombre d'actions par états, S' le nombre de successeurs et T le nombre d'itérations, la complexité en temps est alors de O(TPESS′).

<br>


**74. Optimal Q-value ― The optimal Q-value Qopt(s,a) of state s with action a is defined to be the maximum Q-value attained by any policy starting. It is computed as follows:**

&#10230; Q-value optimale - La Q-value optimale Qopt(s,a) d'un état s avec l'action a est définie comme étant la Q-value maximale atteinte avec n'importe quelle politique. Elle est calculée avec la formule :

<br>


**75. Optimal value ― The optimal value Vopt(s) of state s is defined as being the maximum value attained by any policy. It is computed as follows:**

&#10230; Valeur optimale - La valeur optimale Vopt(s) d'un état s est définie comme étant la valeur maximum atteinte par n'importe quelle politique. Elle est calculée avec la formule :

<br>


**76. actions**

&#10230; actions

<br>


**77. Optimal policy ― The optimal policy πopt is defined as being the policy that leads to the optimal values. It is defined by:**

&#10230; Politique optimale - La politique optimale πopt est définie comme étant la politique liée aux valeurs optimales. Elle est définie par :

<br>


**78. [Value iteration ― Value iteration is an algorithm that finds the optimal value Vopt as well as the optimal policy πopt. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TVI, we have:, with]**

&#10230; [Itération sur la valeur - L'algorithme d'itération sur la valeur (en anglais value iteration) vise à trouver la valeur optimale Vopt ainsi que la politique optimale πopt en deux temps :, Initialisation : pour tout état s, on a, Itération : pour t allant de 1 à TVI, on a, avec]

<br>


**79. Remark: if we have either γ<1 or the MDP graph being acyclic, then the value iteration algorithm is guaranteed to converge to the correct answer.**

&#10230; Remarque : si γ<1 ou si le graphe associé au processus de décision markovien est acyclique, alors l'algorithme d'itération sur la valeur est garanti de converger vers la bonne solution.

<br>


**80. When unknown transitions and rewards**

&#10230; Cas des transitions et récompenses inconnues

<br>


**81. Now, let's assume that the transition probabilities and the rewards are unknown.**

&#10230; On suppose maintenant que les probabilités de transition et les récompenses sont inconnues.

<br>


**82. Model-based Monte Carlo ― The model-based Monte Carlo method aims at estimating T(s,a,s′) and Reward(s,a,s′) using Monte Carlo simulation with:**

&#10230; Monte-Carlo basé sur modèle - La méthode de Monte-Carlo basée sur modèle (en anglais model-based Monte Carlo) vise à estimer T(s,a,s′) et Reward(s,a,s′) en utilisant des simulations de Monte-Carlo avec :

<br>


**83. [# times (s,a,s′) occurs, and]**

&#10230; [# de fois où (s,a,s') se produit]

<br>


**84. These estimations will be then used to deduce Q-values, including Qπ and Qopt.**

&#10230; Ces estimations sont ensuite utilisées pour trouver les Q-values, ainsi que Qπ et Qopt.

<br>


**85. Remark: model-based Monte Carlo is said to be off-policy, because the estimation does not depend on the exact policy.**

&#10230; Remarque : la méthode de Monte-Carlo basée sur modèle est dite "hors politique" (en anglais "off-policy") car l'estimation produite ne dépend pas de la politique utilisée.

<br>


**86. Model-free Monte Carlo ― The model-free Monte Carlo method aims at directly estimating Qπ, as follows:**

&#10230; Monte-Carlo sans modèle - La méthode de Monte-Carlo sans modèle (en anglais model-free Monte Carlo) vise à directement estimer Qπ de la manière suivante :

<br>


**87. Qπ(s,a)=average of ut where st−1=s,at=a**

&#10230; Qπ(s,a)=moyenne de ut où st−1=s,at=a

<br>


**88. where ut denotes the utility starting at step t of a given episode.**

&#10230; où ut désigne l'utilité à partir de l'étape t d'un épisode donné.

<br>


**89. Remark: model-free Monte Carlo is said to be on-policy, because the estimated value is dependent on the policy π used to generate the data.**

&#10230; Remarque : la méthode de Monte-Carlo sans modèle est dite "sur politique" (en anglais "on-policy") car l'estimation produite dépend de la politique π utilisée pour générer les données.

<br>


**90. Equivalent formulation - By introducing the constant η=11+(#updates to (s,a)) and for each (s,a,u) of the training set, the update rule of model-free Monte Carlo has a convex combination formulation:**

&#10230; Formulation équivalente - En introduisant la constante η=11+(#mises à jour à (s,a)) et pour chaque triplet (s,a,u) de la base d'apprentissage, la formule de récurrence de la méthode de Monte-Carlo sans modèle s'écrit à l'aide de la combinaison convexe :

<br>


**91. as well as a stochastic gradient formulation:**

&#10230; ainsi qu'une formulation mettant en valeur une sorte de gradient :

<br>


**92. SARSA ― State-action-reward-state-action (SARSA) is a boostrapping method estimating Qπ by using both raw data and estimates as part of the update rule. For each (s,a,r,s′,a′), we have:**

&#10230; SARSA - État-action-récompense-état-action (en anglais state-action-reward-state-action ou SARSA) est une méthode de bootstrap qui estime Qπ en utilisant à la fois des données réelles et estimées dans sa formule de mise à jour. Pour chaque (s,a,r,s′,a′), on a :

<br>


**93. Remark: the SARSA estimate is updated on the fly as opposed to the model-free Monte Carlo one where the estimate can only be updated at the end of the episode.**

&#10230; Remarque : l'estimation donnée par SARSA est mise à jour à la volée contrairement à celle donnée par la méthode de Monte-Carlo sans modèle où la mise à jour est uniquement effectuée à la fin de l'épisode.

<br>


**94. Q-learning ― Q-learning is an off-policy algorithm that produces an estimate for Qopt. On each (s,a,r,s′,a′), we have:**

&#10230; Q-learning - Le Q-apprentissage (en anglais Q-learning) est un algorithme hors politique (en anglais off-policy) donnant une estimation de Qopt. Pour chaque (s,a,r,s′,a′), on a :

<br>


**95. Epsilon-greedy ― The epsilon-greedy policy is an algorithm that balances exploration with probability ϵ and exploitation with probability 1−ϵ. For a given state s, the policy πact is computed as follows:**

&#10230; Epsilon-glouton - La politique epsilon-gloutonne (en anglais epsilon-greedy) est un algorithme essayant de trouver un compromis entre l'exploration avec probabilité ϵ et l'exploitation avec probabilité 1-ϵ. Pour un état s, la politique πact est calculée par :

<br>


**96. [with probability, random from Actions(s)]**

&#10230; [avec probabilité, aléatoire venant d'Actions(s)]

<br>


**97. Game playing**

&#10230; Jeux

<br>


**98. In games (e.g. chess, backgammon, Go), other agents are present and need to be taken into account when constructing our policy.**

&#10230; Dans les jeux (e.g. échecs, backgammon, Go), d'autres agents sont présents et doivent être pris en compte au moment d'élaborer une politique.

<br>


**99. Game tree ― A game tree is a tree that describes the possibilities of a game. In particular, each node is a decision point for a player and each root-to-leaf path is a possible outcome of the game.**

&#10230; Arbre de jeu - Un arbre de jeu est un arbre détaillant toutes les issues possibles d'un jeu. En particulier, chaque noeud représente un point de décision pour un joueur et chaque chemin liant la racine à une des feuilles traduit une possible instance du jeu.

<br>


**100. [Two-player zero-sum game ― It is a game where each state is fully observed and such that players take turns. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, successors Succ(s,a) from states s with actions a, whether an end state was reached IsEnd(s), the agent's utility Utility(s) at end state s, the player Player(s) who controls state s]**

&#10230; [Jeu à somme nulle à deux joueurs - C'est un type jeu où chaque état est entièrement observé et où les joueurs jouent de manière successive. On le définit par :, un état de départ sstart, de possibles actions Actions(s) partant de l'état s, du successeur Succ(s,a) l'état s après avoir effectué l'action a, la connaissance d'avoir atteint ou non un état final IsEnd(s), l'utilité de l'agent Utility(s) à l'état final s, le joueur Player(s) qui contrôle l'état s]

<br>


**101. Remark: we will assume that the utility of the agent has the opposite sign of the one of the opponent.**

&#10230; Remarque : nous assumerons que l'utilité de l'agent a le signe opposé de celui de son adversaire.

<br>


**102. [Types of policies ― There are two types of policies:, Deterministic policies, noted πp(s), which are actions that player p takes in state s., Stochastic policies, noted πp(s,a)∈[0,1], which are probabilities that player p takes action a in state s.]**

&#10230; [Types de politiques - Il y a deux types de politiques :, Les politiques déterministes, notées πp(s), qui représentent pour tout s l'action que le joueur p prend dans l'état s., Les politiques stochastiques, notées πp(s,a)∈[0,1], qui sont décrites pour tout s et a par la probabilité que le joueur p prenne l'action a dans l'état s.]

<br>


**103. Expectimax ― For a given state s, the expectimax value Vexptmax(s) is the maximum expected utility of any agent policy when playing with respect to a fixed and known opponent policy πopp. It is computed as follows:**

&#10230; Expectimax - Pour un état donné s, la valeur d'expectimax Vexptmax(s) est l'utilité maximum sur l'ensemble des politiques utilisées par l'agent lorsque celui-ci joue avec un adversaire de politique connue πopp. Cette valeur est calculée de la manière suivante :

<br>


**104. Remark: expectimax is the analog of value iteration for MDPs.**

&#10230; Remarque : expectimax est l'analogue de l'algorithme d'itération sur la valeur pour les MDPs.

<br>


**105. Minimax ― The goal of minimax policies is to find an optimal policy against an adversary by assuming the worst case, i.e. that the opponent is doing everything to minimize the agent's utility. It is done as follows:**

&#10230; Minimax - Le but des politiques minimax est de trouver une politique optimale contre un adversaire que l'on assume effectuer toutes les pires actions, i.e. toutes celles qui minimisent l'utilité de l'agent. La valeur correspondante est calculée par :

<br>


**106. Remark: we can extract πmax and πmin from the minimax value Vminimax.**

&#10230; Remarque : on peut déduire πmax et πmin à partir de la valeur minimax Vminimax.

<br>


**107. Minimax properties ― By noting V the value function, there are 3 properties around minimax to have in mind:**

&#10230; Propriétés de minimax - En notant V la fonction de valeur, il y a 3 propriétés sur minimax qu'il faut avoir à l'esprit :

<br>


**108. Property 1: if the agent were to change its policy to any πagent, then the agent would be no better off.**

&#10230; Propriété 1 : si l'agent changeait sa politique en un quelconque πagent, alors il ne s'en sortirait pas mieux.

<br>


**109. Property 2: if the opponent changes its policy from πmin to πopp, then he will be no better off.**

&#10230; Propriété 2 : si son adversaire change sa politique de πmin à πopp, alors il ne s'en sortira pas mieux.

<br>


**110. Property 3: if the opponent is known to be not playing the adversarial policy, then the minimax policy might not be optimal for the agent.**

&#10230; Propriété 3 : si l'on sait que son adversaire ne joue pas les pires actions possibles, alors la politique minimax peut ne pas être optimale pour l'agent.

<br>


**111. In the end, we have the following relationship:**

&#10230; À la fin, on a la relation suivante :

<br>


**112. Speeding up minimax**

&#10230; Accélération de minimax

<br>


**113. Evaluation function ― An evaluation function is a domain-specific and approximate estimate of the value Vminimax(s). It is noted Eval(s).**

&#10230; Fonction d'évaluation - Une fonction d'évaluation estime de manière approximative la valeur Vminimax(s) selon les paramètres du problème. Elle est notée Eval(s).

<br>


**114. Remark: FutureCost(s) is an analogy for search problems.**

&#10230; Remarque : l'analogue de cette fonction utilisé dans les problèmes de recherche est FutureCost(s).

<br>


**115. Alpha-beta pruning ― Alpha-beta pruning is a domain-general exact method optimizing the minimax algorithm by avoiding the unnecessary exploration of parts of the game tree. To do so, each player keeps track of the best value they can hope for (stored in α for the maximizing player and in β for the minimizing player). At a given step, the condition β<α means that the optimal path is not going to be in the current branch as the earlier player had a better option at their disposal.**

&#10230; Élagage alpha-bêta - L'élagage alpha-bêta (en anglais alpha-beta pruning) est une méthode exacte d'optimisation employée sur l'algorithme de minimax et a pour but d'éviter l'exploration de parties inutiles de l'arbre de jeu. Pour ce faire, chaque joueur garde en mémoire la meilleure valeur qu'il puisse espérer (appelée α chez le joueur maximisant et β chez le joueur minimisant). À une étape donnée, la condition β<α signifie que le chemin optimal ne peut pas passer par la branche actuelle puisque le joueur qui précédait avait une meilleure option à sa disposition.

<br>


**116. TD learning ― Temporal difference (TD) learning is used when we don't know the transitions/rewards. The value is based on exploration policy. To be able to use it, we need to know rules of the game Succ(s,a). For each (s,a,r,s′), the update is done as follows:**

&#10230; TD learning - L'apprentissage par différence de temps (en anglais temporal difference learning ou TD learning) est une méthode utilisée lorsque l'on ne connait pas les transitions/récompenses. La valeur et alors basée sur la politique d'exploration. Pour pouvoir l'utiliser, on a besoin de connaître les règles du jeu Succ(s,a). Pour chaque (s,a,r,s′), la mise à jour des coefficients est faite de la manière suivante :

<br>


**117. Simultaneous games**

&#10230; Jeux simultanés

<br>


**118. This is the contrary of turn-based games, where there is no ordering on the player's moves.**

&#10230; Ce cas est opposé aux jeux joués tour à tour. Il n'y a pas d'ordre prédéterminé sur le mouvement du joueur.

<br>


**119. Single-move simultaneous game ― Let there be two players A and B, with given possible actions. We note V(a,b) to be A's utility if A chooses action a, B chooses action b. V is called the payoff matrix.**

&#10230; Jeu simultané à un mouvement - Soient deux joueurs A et B, munis de possibles actions. On note V(a,b) l'utilité de A si A choisit l'action a et B l'action b. V est appelée la matrice de profit (en anglais payoff matrix).

<br>


**120. [Strategies ― There are two main types of strategies:, A pure strategy is a single action:, A mixed strategy is a probability distribution over actions:]**

&#10230; [Stratégies - Il y a principalement deux types de stratégies :, Une stratégie pure est une seule action, Une stratégie mixte est une loi de probabilité sur les actions :]

<br>


**121. Game evaluation ― The value of the game V(πA,πB) when player A follows πA and player B follows πB is such that:**

&#10230; Évaluation de jeu - La valeur d'un jeu V(πA,πB) quand le joueur A suit πA et le joueur B suit πB est telle que :

<br>


**122. Minimax theorem ― By noting πA,πB ranging over mixed strategies, for every simultaneous two-player zero-sum game with a finite number of actions, we have:**

&#10230; Théorème Minimax - Soient πA et πB des stratégies mixtes. Pour chaque jeu à somme nulle à deux joueurs ayant un nombre fini d'actions, on a :

<br>


**123. Non-zero-sum games**

&#10230; Jeux à somme non nulle

<br>


**124. Payoff matrix ― We define Vp(πA,πB) to be the utility for player p.**

&#10230; Matrice de profit - On définit Vp(πA,πB) l'utilité du joueur p.

<br>


**125. Nash equilibrium ― A Nash equilibrium is (π∗A,π∗B) such that no player has an incentive to change its strategy. We have:**

&#10230; Équilibre de Nash - Un équilibre de Nash est défini par (π∗A,π∗B) tel qu'aucun joueur n'a d'intérêt de changer sa stratégie. On a :

<br>


**126. and**

&#10230; et

<br>


**127. Remark: in any finite-player game with finite number of actions, there exists at least one Nash equilibrium.**

&#10230; Remarque : dans un jeu à nombre de joueurs et d'actions finis, il existe au moins un équilibre de Nash.

<br>


**128. [Tree search, Backtracking search, Breadth-first search, Depth-first search, Iterative deepening]**

&#10230; [Parcours d'arbre, Retour sur trace, Parcours en largeur, Parcours en profondeur, Approfondissement itératif]

<br>


**129. [Graph search, Dynamic programming, Uniform cost search]**

&#10230; [Parcours de graphe, Programmation dynamique, Recherche à coût uniforme]

<br>


**130. [Learning costs, Structured perceptron]**

&#10230; [Apprendre les coûts, Perceptron structuré]

<br>


**131. [A star search, Heuristic function, Algorithm, Consistency, correctness, Admissibility, efficiency]**

&#10230; [A étoile, Fonction heuristique, Algorithme, Consistance, Correction, Admissibilité, Efficacité]

<br>


**132. [Relaxation, Relaxed search problem, Relaxed heuristic, Max heuristic]**

&#10230; [Relaxation, Relaxation d'un problème de recherche, Relaxation d'une heuristique, Heuristique max]

<br>


**133. [Markov decision processes, Overview, Policy evaluation, Value iteration, Transitions, rewards]**

&#10230; [Processus de décision markovien, Aperçu, Évaluation d'une politique, Itération sur la valeur, Transitions, Récompenses]

<br>


**134. [Game playing, Expectimax, Minimax, Speeding up minimax, Simultaneous games, Non-zero-sum games]**

&#10230; [Jeux, Expectimax, Minimax, Accélération de minimax, Jeux simultanés, Jeux à somme non nulle]

<br>


**135. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub.

<br>


**136. Original authors**

&#10230; Auteurs d'origine.

<br>


**137. Translated by X, Y and Z**

&#10230; Traduit de l'anglais par X, Y et Z.

<br>


**138. Reviewed by X, Y and Z**

&#10230; Revu par X, Y et Z.

<br>


**139. By X and Y**

&#10230; De X et Y.

<br>


**140. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Les pense-bête d'intelligence artificielle sont maintenant disponibles en français !
