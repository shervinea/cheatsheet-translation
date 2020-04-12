**Variables-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models)

<br>

**1. Variables-based models with CSP and Bayesian networks**

&#10230; Modèles basés sur les variables : CSP et réseaux bayésiens

<br>


**2. Constraint satisfaction problems**

&#10230; Problèmes de satisfaction de contraintes

<br>


**3. In this section, our objective is to find maximum weight assignments of variable-based models. One advantage compared to states-based models is that these algorithms are more convenient to encode problem-specific constraints.**

&#10230; Dans cette section, notre but est de trouver des affectations de poids maximisants dans des problèmes impliquant des modèles basés sur les variables. Un avantage comparé aux modèles basés sur les états est que ces algorithmes sont plus commodes lorsqu'il s'agit de transcrire des contraintes spécifiques à certains problèmes.

<br>


**4. Factor graphs**

&#10230; Graphes de facteurs

<br>


**5. Definition ― A factor graph, also referred to as a Markov random field, is a set of variables X=(X1,...,Xn) where Xi∈Domaini and m factors f1,...,fm with each fj(X)⩾0.**

&#10230; Définition - Un graphe de facteurs, aussi appelé champ aléatoire de Markov, est un ensemble de variables X=(X1,...,Xn) où Xi∈Domaini muni de m facteurs f1,...,fm où chaque fj(X)⩾0.

<br>


**6. Domain**

&#10230; Domaine

<br>


**7. Scope and arity ― The scope of a factor fj is the set of variables it depends on. The size of this set is called the arity.**

&#10230; Arité - Le nombre de variables dépendant d'un facteur fj est appelé son arité.

<br>


**8. Remark: factors of arity 1 and 2 are called unary and binary respectively.**

&#10230; Remarque : les facteurs d'arité 1 et 2 sont respectivement appelés unaire et binaire.

<br>


**9. Assignment weight ― Each assignment x=(x1,...,xn) yields a weight Weight(x) defined as being the product of all factors fj applied to that assignment. Its expression is given by:**

&#10230; Affectation de poids - Chaque affectation x=(x1,...,xn) donne un poids Weight(x) défini comme étant le produit de tous les facteurs fj appliqués à cette affectation. Son expression est donnée par :

<br>


**10. Constraint satisfaction problem ― A constraint satisfaction problem (CSP) is a factor graph where all factors are binary; we call them to be constraints:**

&#10230; Problème de satisfaction de contraintes - Un problème de satisfaction de contraintes (en anglais constraint satisfaction problem ou CSP) est un graphe de facteurs où tous les facteurs sont binaires ; on les appelle "contraintes".

<br>


**11. Here, the constraint j with assignment x is said to be satisfied if and only if fj(x)=1.**

&#10230; Ici, on dit que l'affectation x satisfait la contrainte j si et seulement si fj(x)=1.

<br>


**12. Consistent assignment ― An assignment x of a CSP is said to be consistent if and only if Weight(x)=1, i.e. all constraints are satisfied.**

&#10230; Affectation consistante - Une affectation x d'un CSP est dite consistante si et seulement si Weight(x)=1, i.e. toutes les contraintes sont satisfaites.

<br>


**13. Dynamic ordering**

&#10230; Mise en ordre dynamique

<br>


**14. Dependent factors ― The set of dependent factors of variable Xi with partial assignment x is called D(x,Xi), and denotes the set of factors that link Xi to already assigned variables.**

&#10230; Facteurs dépendants - L'ensemble des facteurs dépendants de la variable Xi dont l'affectation partielle est x est appelé D(x,Xi) et désigne l'ensemble des facteurs liant Xi à des variables déjà affectées.

<br>


**15. Backtracking search ― Backtracking search is an algorithm used to find maximum weight assignments of a factor graph. At each step, it chooses an unassigned variable and explores its values by recursion. Dynamic ordering (i.e. choice of variables and values) and lookahead (i.e. early elimination of inconsistent options) can be used to explore the graph more efficiently, although the worst-case runtime stays exponential: O(|Domain|n).**

&#10230; Recherche avec retour sur trace - L'algorithme de recherche avec retour sur trace (en anglais backtracking search) est utilisé pour trouver l'affectation de poids maximum d'un graphe de facteurs. À chaque étape, une variable non assignée est choisie et ses valeurs sont explorées par récursivité. On peut utiliser un processus de mise en ordre dynamique sur le choix des variables et valeurs et/ou d'anticipation (i.e. élimination précoce d'options non consistantes) pour explorer le graphe de manière plus efficace. La complexité temporelle dans tous les cas reste néanmoins exponentielle : O(|Domaine|n).

<br>


**16. [Forward checking ― It is a one-step lookahead heuristic that preemptively removes inconsistent values from the domains of neighboring variables. It has the following characteristics:, After assigning a variable Xi, it eliminates inconsistent values from the domains of all its neighbors., If any of these domains becomes empty, we stop the local backtracking search., If we un-assign a variable Xi, we have to restore the domain of its neighbors.]**

&#10230; [Vérification en avant - La vérification en avant (forward checking en anglais) est une heuristique d'anticipation à une étape qui enlève des variables voisines les valeurs impossibles de manière préemptive. Cette méthode a les caractéristiques suivantes :, Après l'affectation d'une variable Xi, les valeurs non consistantes sont éliminées du domaine de tous ses voisins., Si l'un de ces domaines devient vide, la recherche locale s'arrête., Si l'on enlève l'affectation d'une valeur Xi, on doit restaurer le domaine de ses voisins.]

<br>


**17. Most constrained variable ― It is a variable-level ordering heuristic that selects the next unassigned variable that has the fewest consistent values. This has the effect of making inconsistent assignments to fail earlier in the search, which enables more efficient pruning.**

&#10230; Variable la plus contrainte - L'heuristique de la variable la plus contrainte (en anglais most constrained variable ou MCV) sélectionne la prochaine variable sans affectation ayant le moins de valeurs consistantes. Cette procédure a pour effet de faire échouer les affectations impossibles plus tôt dans la recherche, permettant un élagage plus efficace.

<br>


**18. Least constrained value ― It is a value-level ordering heuristic that assigns the next value that yields the highest number of consistent values of neighboring variables. Intuitively, this procedure chooses first the values that are most likely to work.**

&#10230; Valeur la moins contraignante - L'heuristique de la valeur la moins contraignante (en anglais least constrained value ou LCV) sélectionne pour une variable donnée la prochaine valeur maximisant le nombre de valeurs consistantes chez les variables voisines. De manière intuitive, on peut dire que cette procédure choisit en premier les valeurs qui sont le plus susceptible de marcher.

<br>


**19. Remark: in practice, this heuristic is useful when all factors are constraints.**

&#10230; Remarque : en pratique, cette heuristique est utile quand tous les facteurs sont des contraintes.

<br>


**20. The example above is an illustration of the 3-color problem with backtracking search coupled with most constrained variable exploration and least constrained value heuristic, as well as forward checking at each step.**

&#10230; L'exemple ci-dessus est une illustration du problème de coloration de graphe à 3 couleurs en utilisant l'algorithme de recherche avec retour sur trace couplé avec les heuristiques de MCV, de LCV ainsi que de vérification en avant à chaque étape.

<br>


**21. [Arc consistency ― We say that arc consistency of variable Xl with respect to Xk is enforced when for each xl∈Domainl:, unary factors of Xl are non-zero, there exists at least one xk∈Domaink such that any factor between Xl and Xk is non-zero.]**

&#10230; [Arc-consistance - On dit que l'arc-consistance de la variable Xl par rapport à Xk est vérifiée lorsque pour tout xl∈Domainl :, les facteurs unaires de Xl sont non-nuls, il existe au moins un xk∈Domaink tel que n'importe quel facteur entre Xl et Xk est non nul.]

<br>


**22. AC-3 ― The AC-3 algorithm is a multi-step lookahead heuristic that applies forward checking to all relevant variables. After a given assignment, it performs forward checking and then successively enforces arc consistency with respect to the neighbors of variables for which the domain change during the process.**

&#10230; AC-3 - L'algorithme d'AC-3 est une heuristique qui applique le principe de vérification en avant à toutes les variables susceptibles d'être concernées. Après l'affectation d'une variable, cet algorithme effectue une vérification en avant et applique successivement l'arc-consistance avec tous les voisins de variables pour lesquels le domaine change.

<br>


**23. Remark: AC-3 can be implemented both iteratively and recursively.**

&#10230; Remarque : AC-3 peut être codé de manière itérative ou récursive.

<br>


**24. Approximate methods**

&#10230; Méthodes approximatives

<br>


**25. Beam search ― Beam search is an approximate algorithm that extends partial assignments of n variables of branching factor b=|Domain| by exploring the K top paths at each step. The beam size K∈{1,...,bn} controls the tradeoff between efficiency and accuracy. This algorithm has a time complexity of O(n⋅Kblog(Kb)).**

&#10230; Recherche en faisceau - L'algorithme de recherche en faisceau (en anglais beam search) est une technique approximative qui étend les affectations partielles de n variables de facteur de branchement b=|Domain| en explorant les K meilleurs chemins qui s'offrent à chaque étape. La largeur du faisceau K∈{1,...,bn} détermine la balance entre efficacité et précision de l'algorithme. Sa complexité en temps est de O(n⋅Kblog(Kb)).

<br>


**26. The example below illustrates a possible beam search of parameters K=2, b=3 and n=5.**

&#10230; L'exemple ci-dessous illustre une recherche en faisceau de paramètres K=2, b=3 et n=5.

<br>


**27. Remark: K=1 corresponds to greedy search whereas K→+∞ is equivalent to BFS tree search.**

&#10230; Remarque : K=1 correspond à la recherche gloutonne alors que K→+∞ est équivalent à effectuer un parcours en largeur.

<br>


**28. Iterated conditional modes ― Iterated conditional modes (ICM) is an iterative approximate algorithm that modifies the assignment of a factor graph one variable at a time until convergence. At step i, we assign to Xi the value v that maximizes the product of all factors connected to that variable.**

&#10230; Modes conditionnels itérés - L'algorithme des modes conditionnels itérés (en anglais iterated conditional modes ou ICM) est une technique itérative et approximative qui modifie l'affectation d'un graphe de facteurs une variable à la fois jusqu'à convergence. À l'étape i, Xi prend la valeur v qui maximise le produit de tous les facteurs connectés à cette variable.

<br>


**29. Remark: ICM may get stuck in local minima.**

&#10230; Remarque : il est possible qu'ICM reste bloqué dans un minimum local.

<br>


**30. [Gibbs sampling ― Gibbs sampling is an iterative approximate method that modifies the assignment of a factor graph one variable at a time until convergence. At step i:, we assign to each element u∈Domaini a weight w(u) that is the product of all factors connected to that variable, we sample v from the probability distribution induced by w and assign it to Xi.]**

&#10230; [Échantillonnage de Gibbs - La méthode d'échantillonnage de Gibbs (en anglais Gibbs sampling) est une technique itérative et approximative qui modifie les affectations d'un graphe de facteurs une variable à la fois jusqu'à convergence. À l'étape i :, on assigne à chaque élément u∈Domaini un poids w(u) qui est le produit de tous les facteurs connectés à cette variable, on échantillonne v de la loi de probabilité engendrée par w et on l'associe à Xi.]

<br>


**31. Remark: Gibbs sampling can be seen as the probabilistic counterpart of ICM. It has the advantage to be able to escape local minima in most cases.**

&#10230; Remarque : la méthode d'échantillonnage de Gibbs peut être vue comme étant la version probabiliste de ICM. Cette méthode a l'avantage de pouvoir échapper aux potentiels minimum locaux dans la plupart des situations.

<br>


**32. Factor graph transformations**

&#10230; Transformations sur les graphes de facteurs

<br>


**33. Independence ― Let A,B be a partitioning of the variables X. We say that A and B are independent if there are no edges between A and B and we write:**

&#10230; Indépendance - Soit A, B une partition des variables X. On dit que A et B sont indépendants s'il n'y a pas d'arête connectant A et B et on écrit :

<br>


**34. Remark: independence is the key property that allows us to solve subproblems in parallel.**

&#10230; Remarque : l'indépendance est une propriété importante car elle nous permet de décomposer la situation en sous-problèmes que l'on peut résoudre en parallèle.

<br>


**35. Conditional independence ― We say that A and B are conditionally independent given C if conditioning on C produces a graph in which A and B are independent. In this case, it is written:**

&#10230; Indépendance conditionnelle - On dit que A et B sont conditionnellement indépendants par rapport à C si le fait de conditionner sur C produit un graphe dans lequel A et B sont indépendants. Dans ce cas, on écrit :

<br>


**36. [Conditioning ― Conditioning is a transformation aiming at making variables independent that breaks up a factor graph into smaller pieces that can be solved in parallel and can use backtracking. In order to condition on a variable Xi=v, we do as follows:, Consider all factors f1,...,fk that depend on Xi, Remove Xi and f1,...,fk, Add gj(x) for j∈{1,...,k} defined as:]**

&#10230; [Conditionnement - Le conditionnement est une transformation visant à rendre des variables indépendantes et ainsi diviser un graphe de facteurs en pièces plus petites qui peuvent être traitées en parallèle et utiliser le retour sur trace. Pour conditionner par rapport à une variable Xi=v, on :, considère toues les facteurs f1,...,fk qui dépendent de Xi, enlève Xi et f1,...,fk, ajoute gj(x) pour j∈{1,...,k} défini par :]

<br>


**37. Markov blanket ― Let A⊆X be a subset of variables. We define MarkovBlanket(A) to be the neighbors of A that are not in A.**

&#10230; Couverture de Markov - Soit A⊆X une partie des variables. On définit MarkovBlanket(A) comme étant les voisins de A qui ne sont pas dans A.

<br>


**38. Proposition ― Let C=MarkovBlanket(A) and B=X∖(A∪C). Then we have:**

&#10230; Proposition - Soit C=MarkovBlanket(A) et B=X∖(A∪C). On a alors :

<br>


**39. [Elimination ― Elimination is a factor graph transformation that removes Xi from the graph and solves a small subproblem conditioned on its Markov blanket as follows:, Consider all factors fi,1,...,fi,k that depend on Xi, Remove Xi
and fi,1,...,fi,k, Add fnew,i(x) defined as:]**

&#10230; [Élimination - L'élimination est une transformation consistant à enlever Xi d'un graphe de facteurs pour ensuite résoudre un sous-problème conditionné sur sa couverture de Markov où l'on :, considère tous les facteurs fi,1,...,fi,k qui dépendent de Xi, enlève Xi et fi,1,...,fi,k, ajoute fnew,i(x) défini par :]

<br>


**40. Treewidth ― The treewidth of a factor graph is the maximum arity of any factor created by variable elimination with the best variable ordering. In other words,**

&#10230; Largeur arborescente - La largeur arborescente (en anglais treewidth) d'un graphe de facteurs est l'arité maximum de n'importe quel facteur créé par élimination avec le meilleur ordre de variable. En d'autres termes,

<br>


**41. The example below illustrates the case of a factor graph of treewidth 3.**

&#10230; L'exemple ci-dessous illustre le cas d'un graphe de facteurs ayant une largeur arborescente égale à 3.

<br>


**42. Remark: finding the best variable ordering is a NP-hard problem.**

&#10230; Remarque : trouver le meilleur ordre de variable est un problème NP-difficile.

<br>


**43. Bayesian networks**

&#10230; Réseaux bayésiens

<br>


**44. In this section, our goal will be to compute conditional probabilities. What is the probability of a query given evidence?**

&#10230; Dans cette section, notre but est de calculer des probabilités conditionnelles. Quelle est la probabilité d'un événement étant donné des observations ?

<br>


**45. Introduction**

&#10230; Introduction

<br>


**46. Explaining away ― Suppose causes C1 and C2 influence an effect E. Conditioning on the effect E and on one of the causes (say C1) changes the probability of the other cause (say C2). In this case, we say that C1 has explained away C2.**

&#10230; Explication - Supposons que les causes C1 et C2 influencent un effet E. Le conditionnement sur l'effet E et une des causes (disons C1) change la probabilité de l'autre cause (disons C2). Dans ce cas, on dit que C1 a expliqué C2.

<br>


**47. Directed acyclic graph ― A directed acyclic graph (DAG) is a finite directed graph with no directed cycles.**

&#10230; Graphe orienté acyclique - Un graphe orienté acyclique (en anglais directed acyclic graph ou DAG) est un graphe orienté fini sans cycle orienté.

<br>


**48. Bayesian network ― A Bayesian network is a directed acyclic graph (DAG) that specifies a joint distribution over random variables X=(X1,...,Xn) as a product of local conditional distributions, one for each node:**

&#10230; Réseau bayésien - Un réseau bayésien (en anglais Bayesian network) est un DAG qui définit une loi de probabilité jointe sur les variables aléatoires X=(X1,...,Xn) comme étant le produit des lois de probabilités conditionnelles locales (une pour chaque noeud) :

<br>


**49. Remark: Bayesian networks are factor graphs imbued with the language of probability.**

&#10230; Remarque : les réseaux bayésiens sont des graphes de facteurs imprégnés de concepts de probabilité.

<br>


**50. Locally normalized ― For each xParents(i), all factors are local conditional distributions. Hence they have to satisfy:**

&#10230; Normalisation locale - Pour chaque xParents(i), tous les facteurs sont localement des lois de probabilité conditionnelles. Elles doivent donc vérifier :

<br>


**51. As a result, sub-Bayesian networks and conditional distributions are consistent.**

&#10230; De ce fait, les sous-réseaux bayésiens et les distributions conditionnelles sont consistants.

<br>


**52. Remark: local conditional distributions are the true conditional distributions.**

&#10230; Remarque : les lois locales de probabilité conditionnelles sont de vraies lois de probabilité conditionnelles.

<br>


**53. Marginalization ― The marginalization of a leaf node yields a Bayesian network without that node.**

&#10230; Marginalisation - La marginalisation d'un noeud sans enfant entraine un réseau bayésian sans ce noeud.

<br>


**54. Probabilistic programs**

&#10230; Programmes probabilistes

<br>


**55. Concept ― A probabilistic program randomizes variables assignment. That way, we can write down complex Bayesian networks that generate assignments without us having to explicitly specify associated probabilities.**

&#10230; Concept - Un programme probabiliste rend aléatoire l'affectation de variables. De ce fait, on peut imaginer des réseaux bayésiens compliqués pour la génération d'affectations sans avoir à écrire de manière explicite les probabilités associées.

<br>


**56. Remark: examples of probabilistic programs include Hidden Markov model (HMM), factorial HMM, naive Bayes, latent Dirichlet allocation, diseases and symptoms and stochastic block models.**

&#10230; Remarque : quelques exemples de programmes probabilistes incluent parmi d'autres le modèle de Markov caché (en anglais hidden Markov model ou HMM), HMM factoriel, le modèle bayésien naïf (en anglais naive Bayes), l'allocation de Dirichlet latente (en anglais latent Dirichlet allocation ou LDA), le modèle à blocs stochastiques (en anglais stochastic block model).

<br>


**57. Summary ― The table below summarizes the common probabilistic programs as well as their applications:**

&#10230; Récapitulatif - La table ci-dessous résume les programmes probabilistes les plus fréquents ainsi que leur champ d'application associé :

<br>


**58. [Program, Algorithm, Illustration, Example]**

&#10230; [Programme, Algorithme, Illustration, Exemple]

<br>


**59. [Markov Model, Hidden Markov Model (HMM), Factorial HMM, Naive Bayes, Latent Dirichlet Allocation (LDA)]**

&#10230; [Modèle de Markov, Modèle de Markov caché (HMM), HMM factoriel, Bayésien naïf, Allocation de Dirichlet latente (LDA)]

<br>


**60. [Generate, distribution]**

&#10230; [Génère, distribution]

<br>


**61. [Language modeling, Object tracking, Multiple object tracking, Document classification, Topic modeling]**

&#10230; [Modélisation du langage, Suivi d'objet, Suivi de plusieurs objets, Classification de document, Modélisation de sujet]

<br>


**62. Inference**

&#10230; Inférence

<br>


**63. [General probabilistic inference strategy ― The strategy to compute the probability P(Q|E=e) of query Q given evidence E=e is as follows:, Step 1: Remove variables that are not ancestors of the query Q or the evidence E by marginalization, Step 2: Convert Bayesian network to factor graph, Step 3: Condition on the evidence E=e, Step 4: Remove nodes disconnected from the query Q by marginalization, Step 5: Run a probabilistic inference algorithm (manual, variable elimination, Gibbs sampling, particle filtering)]**

&#10230; [Stratégie générale pour l'inférence probabiliste - La stratégie que l'on utilise pour calculer la probabilité P(Q|E=e) d'une requête Q étant donnée l'observation E=e est la suivante :, Étape 1 : on enlève les variables qui ne sont pas les ancêtres de la requête Q ou de l'observation E par marginalisation, Étape 2 : on convertit le réseau bayésien en un graphe de facteurs, Étape 3 : on conditionne sur l'observation E=e, Étape 4 : on enlève les noeuds déconnectés de la requête Q par marginalisation, Étape 5 : on lance un algorithme d'inférence probabiliste (manuel, élimination de variables, échantillonnage de Gibbs, filtrage particulaire)]

<br>


**64. Forward-backward algorithm ― This algorithm computes the exact value of P(H=hk|E=e) (smoothing query) for any k∈{1,...,L} in the case of an HMM of size L. To do so, we proceed in 3 steps:**

&#10230; Algorithme progressif-rétrogressif - L'algorithme progressif-rétrogressif (en anglais forward-backward) calcule la valeur exacte de P(H=hk|E=e) pour chaque k∈{1,...,L} dans le cas d'un HMM de taille L. Pour ce faire, on procède en 3 étapes :

<br>


**65. Step 1: for ..., compute ...**

&#10230; Étape 1 : pour ..., calculer ...

<br>


**66. with the convention F0=BL+1=1. From this procedure and these notations, we get that**

&#10230; avec la convention F0=BL+1=1. À partir de cette procédure et avec ces notations, on obtient

<br>


**67. Remark: this algorithm interprets each assignment to be a path where each edge hi−1→hi is of weight p(hi|hi−1)p(ei|hi).**

&#10230; Remarque : cet algorithme interprète une affectation comme étant un chemin où chaque arête hi−1→hi a un poids p(hi|hi−1)p(ei|hi).

<br>


**68. [Gibbs sampling ― This algorithm is an iterative approximate method that uses a small set of assignments (particles) to represent a large probability distribution. From a random assignment x, Gibbs sampling performs the following steps for i∈{1,...,n} until convergence:, For all u∈Domaini, compute the weight w(u) of assignment x where Xi=u, Sample v from the probability distribution induced by w: v∼P(Xi=v|X−i=x−i), Set Xi=v]**

&#10230; [Échantillonnage de Gibbs - L'algorithme d'échantillonnage de Gibbs (en anglais Gibbs sampling) est une méthode itérative et approximative qui utilise un petit ensemble d'affectations (particules) pour représenter une loi de probabilité. Pour une affectation aléatoire x, l'échantillonnage de Gibbs effectue les étapes suivantes pour i∈{1,...,n} jusqu'à convergence :, Pour tout u∈Domaini, on calcule le poids w(u) de l'affectation x où Xi=u, On échantillonne v de la loi de probabilité engendrée par w : v∼P(Xi=v|X−i=x−i), On pose Xi=v]

<br>


**69. Remark: X−i denotes X∖{Xi} and x−i represents the corresponding assignment.**

&#10230; Remarque X-i veut dire X∖{Xi} et x−i représente l'affectation correspondante.

<br>


**70. [Particle filtering ― This algorithm approximates the posterior density of state variables given the evidence of observation variables by keeping track of K particles at a time. Starting from a set of particles C of size K, we run the following 3 steps iteratively:, Step 1: proposal - For each old particle xt−1∈C, sample x from the transition probability distribution p(x|xt−1) and add x to a set C′., Step 2: weighting - Weigh each x of the set C′ by w(x)=p(et|x), where et is the evidence observed at time t., Step 3: resampling - Sample K elements from the set C′ using the probability distribution induced by w and store them in C: these are the current particles xt.]**

&#10230; [Filtrage particulaire - L'algorithme de filtrage particulaire (en anglais particle filtering) approxime la densité postérieure de variables d'états à partir des variables observées en suivant K particules à la fois. En commençant avec un ensemble de particules C de taille K, on répète les 3 étapes suivantes :, Étape 1 : proposition - Pour chaque particule xt−1∈C, on échantillonne x avec loi de probabilité p(x|xt−1) et on ajoute x à un ensemble C′., Étape 2 : pondération - On associe chaque x de l'ensemble C′ au poids w(x)=p(et|x), où et est l'observation vue à l'instant t. Étape 3 : ré-échantillonnage - On échantillonne K éléments de l'ensemble C´ en utilisant la loi de probabilité engendrée par w et on les met dans C : ce sont les particules actuelles xt.]

<br>


**71. Remark: a more expensive version of this algorithm also keeps track of past particles in the proposal step.**

&#10230; Remarque : une version plus coûteuse de cet algorithme tient aussi compte des particules passée à l'étape de proposition.

<br>


**72. Maximum likelihood ― If we don't know the local conditional distributions, we can learn them using maximum likelihood.**

&#10230; Maximum de vraisemblance - Si l'on ne connaît pas les lois de probabilité locales, on peut les trouver en utilisant le maximum de vraisemblance.

<br>


**73. Laplace smoothing ― For each distribution d and partial assignment (xParents(i),xi), add λ to countd(xParents(i),xi), then normalize to get probability estimates.**

&#10230; Lissage de Laplace - Pour chaque loi de probabilité d et affectation partielle (xParents(i),xi), on ajoute λ à countd(xParents(i),xi) et on normalise ensuite pour obtenir des probabilités.

<br>


**74. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Espérance-maximisation - L'algorithme d'espérance-maximisation (en anglais expectation-maximization ou EM) est une méthode efficace utilisée pour estimer le paramètre θ via l'estimation du maximum de vraisemblance en construisant de manière répétée une borne inférieure de la vraisemblance (étape E) et en optimisant cette borne inférieure (étape M) :

<br>


**75. [E-step: Evaluate the posterior probability q(h) that each data point e came from a particular cluster h as follows:, M-step: Use the posterior probabilities q(h) as cluster specific weights on data points e to determine θ through maximum likelihood.]**

&#10230; [Étape E : on évalue la probabilité postérieure q(h) que chaque point e vienne d'une partition particulière h avec :, Étape M : on utilise la probabilité postérieure q(h) en tant que poids de la partition h sur les points e pour déterminer θ via maximum de vraisemblance]

<br>


**76. [Factor graphs, Arity, Assignment weight, Constraint satisfaction problem, Consistent assignment]**

&#10230; [Graphe de facteurs, Arité, Poids, Satisfaction de contraintes, Affectation consistante]

<br>


**77. [Dynamic ordering, Dependent factors, Backtracking search, Forward checking, Most constrained variable, Least constrained value]**

&#10230; [Mise en ordre dynamique, Facteurs dépendants, Retour sur trace, Vérification en avant, Variable la plus contrainte, Valeur la moins contraignante]

<br>


**78. [Approximate methods, Beam search, Iterated conditional modes, Gibbs sampling]**

&#10230; [Méthodes approximatives, Recherche en faisceau, Modes conditionnels itérés, Échantillonnage de Gibbs]

<br>


**79. [Factor graph transformations, Conditioning, Elimination]**

&#10230; [Transformations de graphes de facteurs, Conditionnement, Élimination]

<br>


**80. [Bayesian networks, Definition, Locally normalized, Marginalization]**

&#10230; [Réseaux bayésiens, Définition, Normalisé localement, Marginalisation]

<br>


**81. [Probabilistic program, Concept, Summary]**

&#10230; [Programme probabiliste, Concept, Récapitulatif]

<br>


**82. [Inference, Forward-backward algorithm, Gibbs sampling, Laplace smoothing]**

&#10230; [Inférence, Algorithme progressif-rétrogressif, Échantillonnage de Gibbs, Lissage de Laplace]

<br>


**83. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub.

<br>


**84. Original authors**

&#10230; Auteurs d'origine.

<br>


**85. Translated by X, Y and Z**

&#10230; Traduit de l'anglais par X, Y et Z.

<br>


**86. Reviewed by X, Y and Z**

&#10230; Revu par X, Y et Z.

<br>


**87. By X and Y**

&#10230; De X et Y.

<br>


**88. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'intelligence artificielle sont maintenant disponibles en français !
