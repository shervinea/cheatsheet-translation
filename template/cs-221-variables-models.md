**Variables-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models)

<br>

**1. Variables-based models with CSP and Bayesian networks**

&#10230;

<br>


**2. Constraint satisfaction problems**

&#10230;

<br>


**3. In this section, our objective is to find maximum weight assignments of variable-based models. One advantage compared to states-based models is that these algorithms are more convenient to encode problem-specific constraints.**

&#10230;

<br>


**4. Factor graphs**

&#10230;

<br>


**5. Definition ― A factor graph, also referred to as a Markov random field, is a set of variables X=(X1,...,Xn) where Xi∈Domaini and m factors f1,...,fm with each fj(X)⩾0.**

&#10230;

<br>


**6. Domain**

&#10230;

<br>


**7. Scope and arity ― The scope of a factor fj is the set of variables it depends on. The size of this set is called the arity.**

&#10230;

<br>


**8. Remark: factors of arity 1 and 2 are called unary and binary respectively.**

&#10230;

<br>


**9. Assignment weight ― Each assignment x=(x1,...,xn) yields a weight Weight(x) defined as being the product of all factors fj applied to that assignment. Its expression is given by:**

&#10230;

<br>


**10. Constraint satisfaction problem ― A constraint satisfaction problem (CSP) is a factor graph where all factors are binary; we call them to be constraints:**

&#10230;

<br>


**11. Here, the constraint j with assignment x is said to be satisfied if and only if fj(x)=1.**

&#10230;

<br>


**12. Consistent assignment ― An assignment x of a CSP is said to be consistent if and only if Weight(x)=1, i.e. all constraints are satisfied.**

&#10230;

<br>


**13. Dynamic ordering**

&#10230;

<br>


**14. Dependent factors ― The set of dependent factors of variable Xi with partial assignment x is called D(x,Xi), and denotes the set of factors that link Xi to already assigned variables.**

&#10230;

<br>


**15. Backtracking search ― Backtracking search is an algorithm used to find maximum weight assignments of a factor graph. At each step, it chooses an unassigned variable and explores its values by recursion. Dynamic ordering (i.e. choice of variables and values) and lookahead (i.e. early elimination of inconsistent options) can be used to explore the graph more efficiently, although the worst-case runtime stays exponential: O(|Domain|n).**

&#10230;

<br>


**16. [Forward checking ― It is a one-step lookahead heuristic that preemptively removes inconsistent values from the domains of neighboring variables. It has the following characteristics:, After assigning a variable Xi, it eliminates inconsistent values from the domains of all its neighbors., If any of these domains becomes empty, we stop the local backtracking search., If we un-assign a variable Xi, we have to restore the domain of its neighbors.]**

&#10230;

<br>


**17. Most constrained variable ― It is a variable-level ordering heuristic that selects the next unassigned variable that has the fewest consistent values. This has the effect of making inconsistent assignments to fail earlier in the search, which enables more efficient pruning.**

&#10230;

<br>


**18. Least constrained value ― It is a value-level ordering heuristic that assigns the next value that yields the highest number of consistent values of neighboring variables. Intuitively, this procedure chooses first the values that are most likely to work.**

&#10230;

<br>


**19. Remark: in practice, this heuristic is useful when all factors are constraints.**

&#10230;

<br>


**20. The example above is an illustration of the 3-color problem with backtracking search coupled with most constrained variable exploration and least constrained value heuristic, as well as forward checking at each step.**

&#10230;

<br>


**21. [Arc consistency ― We say that arc consistency of variable Xl with respect to Xk is enforced when for each xl∈Domainl:, unary factors of Xl are non-zero, there exists at least one xk∈Domaink such that any factor between Xl and Xk is non-zero.]**

&#10230;

<br>


**22. AC-3 ― The AC-3 algorithm is a multi-step lookahead heuristic that applies forward checking to all relevant variables. After a given assignment, it performs forward checking and then successively enforces arc consistency with respect to the neighbors of variables for which the domain change during the process.**

&#10230;

<br>


**23. Remark: AC-3 can be implemented both iteratively and recursively.**

&#10230;

<br>


**24. Approximate methods**

&#10230;

<br>


**25. Beam search ― Beam search is an approximate algorithm that extends partial assignments of n variables of branching factor b=|Domain| by exploring the K top paths at each step. The beam size K∈{1,...,bn} controls the tradeoff between efficiency and accuracy. This algorithm has a time complexity of O(n⋅Kblog(Kb)).**

&#10230;

<br>


**26. The example below illustrates a possible beam search of parameters K=2, b=3 and n=5.**

&#10230;

<br>


**27. Remark: K=1 corresponds to greedy search whereas K→+∞ is equivalent to BFS tree search.**

&#10230;

<br>


**28. Iterated conditional modes ― Iterated conditional modes (ICM) is an iterative approximate algorithm that modifies the assignment of a factor graph one variable at a time until convergence. At step i, we assign to Xi the value v that maximizes the product of all factors connected to that variable.**

&#10230;

<br>


**29. Remark: ICM may get stuck in local minima.**

&#10230;

<br>


**30. [Gibbs sampling ― Gibbs sampling is an iterative approximate method that modifies the assignment of a factor graph one variable at a time until convergence. At step i:, we assign to each element u∈Domaini a weight w(u) that is the product of all factors connected to that variable, we sample v from the probability distribution induced by w and assign it to Xi.]**

&#10230;

<br>


**31. Remark: Gibbs sampling can be seen as the probabilistic counterpart of ICM. It has the advantage to be able to escape local minima in most cases.**

&#10230;

<br>


**32. Factor graph transformations**

&#10230;

<br>


**33. Independence ― Let A,B be a partitioning of the variables X. We say that A and B are independent if there are no edges between A and B and we write:**

&#10230;

<br>


**34. Remark: independence is the key property that allows us to solve subproblems in parallel.**

&#10230;

<br>


**35. Conditional independence ― We say that A and B are conditionally independent given C if conditioning on C produces a graph in which A and B are independent. In this case, it is written:**

&#10230;

<br>


**36. [Conditioning ― Conditioning is a transformation aiming at making variables independent that breaks up a factor graph into smaller pieces that can be solved in parallel and can use backtracking. In order to condition on a variable Xi=v, we do as follows:, Consider all factors f1,...,fk that depend on Xi, Remove Xi and f1,...,fk, Add gj(x) for j∈{1,...,k} defined as:]**

&#10230;

<br>


**37. Markov blanket ― Let A⊆X be a subset of variables. We define MarkovBlanket(A) to be the neighbors of A that are not in A.**

&#10230;

<br>


**38. Proposition ― Let C=MarkovBlanket(A) and B=X∖(A∪C). Then we have:**

&#10230;

<br>


**39. [Elimination ― Elimination is a factor graph transformation that removes Xi from the graph and solves a small subproblem conditioned on its Markov blanket as follows:, Consider all factors fi,1,...,fi,k that depend on Xi, Remove Xi
and fi,1,...,fi,k, Add fnew,i(x) defined as:]**

&#10230;

<br>


**40. Treewidth ― The treewidth of a factor graph is the maximum arity of any factor created by variable elimination with the best variable ordering. In other words,**

&#10230;

<br>


**41. The example below illustrates the case of a factor graph of treewidth 3.**

&#10230;

<br>


**42. Remark: finding the best variable ordering is a NP-hard problem.**

&#10230;

<br>


**43. Bayesian networks**

&#10230;

<br>


**44. In this section, our goal will be to compute conditional probabilities. What is the probability of a query given evidence?**

&#10230;

<br>


**45. Introduction**

&#10230;

<br>


**46. Explaining away ― Suppose causes C1 and C2 influence an effect E. Conditioning on the effect E and on one of the causes (say C1) changes the probability of the other cause (say C2). In this case, we say that C1 has explained away C2.**

&#10230;

<br>


**47. Directed acyclic graph ― A directed acyclic graph (DAG) is a finite directed graph with no directed cycles.**

&#10230;

<br>


**48. Bayesian network ― A Bayesian network is a directed acyclic graph (DAG) that specifies a joint distribution over random variables X=(X1,...,Xn) as a product of local conditional distributions, one for each node:**

&#10230;

<br>


**49. Remark: Bayesian networks are factor graphs imbued with the language of probability.**

&#10230;

<br>


**50. Locally normalized ― For each xParents(i), all factors are local conditional distributions. Hence they have to satisfy:**

&#10230;

<br>


**51. As a result, sub-Bayesian networks and conditional distributions are consistent.**

&#10230;

<br>


**52. Remark: local conditional distributions are the true conditional distributions.**

&#10230;

<br>


**53. Marginalization ― The marginalization of a leaf node yields a Bayesian network without that node.**

&#10230;

<br>


**54. Probabilistic programs**

&#10230;

<br>


**55. Concept ― A probabilistic program randomizes variables assignment. That way, we can write down complex Bayesian networks that generate assignments without us having to explicitly specify associated probabilities.**

&#10230;

<br>


**56. Remark: examples of probabilistic programs include Hidden Markov model (HMM), factorial HMM, naive Bayes, latent Dirichlet allocation, diseases and symptoms and stochastic block models.**

&#10230;

<br>


**57. Summary ― The table below summarizes the common probabilistic programs as well as their applications:**

&#10230;

<br>


**58. [Program, Algorithm, Illustration, Example]**

&#10230;

<br>


**59. [Markov Model, Hidden Markov Model (HMM), Factorial HMM, Naive Bayes, Latent Dirichlet Allocation (LDA)]**

&#10230;

<br>


**60. [Generate, distribution]**

&#10230;

<br>


**61. [Language modeling, Object tracking, Multiple object tracking, Document classification, Topic modeling]**

&#10230;

<br>


**62. Inference**

&#10230;

<br>


**63. [General probabilistic inference strategy ― The strategy to compute the probability P(Q|E=e) of query Q given evidence E=e is as follows:, Step 1: Remove variables that are not ancestors of the query Q or the evidence E by marginalization, Step 2: Convert Bayesian network to factor graph, Step 3: Condition on the evidence E=e, Step 4: Remove nodes disconnected from the query Q by marginalization, Step 5: Run a probabilistic inference algorithm (manual, variable elimination, Gibbs sampling, particle filtering)]**

&#10230;

<br>


**64. Forward-backward algorithm ― This algorithm computes the exact value of P(H=hk|E=e) (smoothing query) for any k∈{1,...,L} in the case of an HMM of size L. To do so, we proceed in 3 steps:**

&#10230;

<br>


**65. Step 1: for ..., compute ...**

&#10230;

<br>


**66. with the convention F0=BL+1=1. From this procedure and these notations, we get that**

&#10230;

<br>


**67. Remark: this algorithm interprets each assignment to be a path where each edge hi−1→hi is of weight p(hi|hi−1)p(ei|hi).**

&#10230;

<br>


**68. [Gibbs sampling ― This algorithm is an iterative approximate method that uses a small set of assignments (particles) to represent a large probability distribution. From a random assignment x, Gibbs sampling performs the following steps for i∈{1,...,n} until convergence:, For all u∈Domaini, compute the weight w(u) of assignment x where Xi=u, Sample v from the probability distribution induced by w: v∼P(Xi=v|X−i=x−i), Set Xi=v]**

&#10230;

<br>


**69. Remark: X−i denotes X∖{Xi} and x−i represents the corresponding assignment.**

&#10230;

<br>


**70. [Particle filtering ― This algorithm approximates the posterior density of state variables given the evidence of observation variables by keeping track of K particles at a time. Starting from a set of particles C of size K, we run the following 3 steps iteratively:, Step 1: proposal - For each old particle xt−1∈C, sample x from the transition probability distribution p(x|xt−1) and add x to a set C′., Step 2: weighting - Weigh each x of the set C′ by w(x)=p(et|x), where et is the evidence observed at time t., Step 3: resampling - Sample K elements from the set C′ using the probability distribution induced by w and store them in C: these are the current particles xt.]**

&#10230;

<br>


**71. Remark: a more expensive version of this algorithm also keeps track of past particles in the proposal step.**

&#10230;

<br>


**72. Maximum likelihood ― If we don't know the local conditional distributions, we can learn them using maximum likelihood.**

&#10230;

<br>


**73. Laplace smoothing ― For each distribution d and partial assignment (xParents(i),xi), add λ to countd(xParents(i),xi), then normalize to get probability estimates.**

&#10230;

<br>


**74. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230;

<br>


**75. [E-step: Evaluate the posterior probability q(h) that each data point e came from a particular cluster h as follows:, M-step: Use the posterior probabilities q(h) as cluster specific weights on data points e to determine θ through maximum likelihood.]**

&#10230;

<br>


**76. [Factor graphs, Arity, Assignment weight, Constraint satisfaction problem, Consistent assignment]**

&#10230;

<br>


**77. [Dynamic ordering, Dependent factors, Backtracking search, Forward checking, Most constrained variable, Least constrained value]**

&#10230;

<br>


**78. [Approximate methods, Beam search, Iterated conditional modes, Gibbs sampling]**

&#10230;

<br>


**79. [Factor graph transformations, Conditioning, Elimination]**

&#10230;

<br>


**80. [Bayesian networks, Definition, Locally normalized, Marginalization]**

&#10230;

<br>


**81. [Probabilistic program, Concept, Summary]**

&#10230;

<br>


**82. [Inference, Forward-backward algorithm, Gibbs sampling, Laplace smoothing]**

&#10230;

<br>


**83. View PDF version on GitHub**

&#10230;

<br>


**84. Original authors**

&#10230;

<br>


**85. Translated by X, Y and Z**

&#10230;

<br>


**86. Reviewed by X, Y and Z**

&#10230;

<br>


**87. By X and Y**

&#10230;

<br>


**88. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230;
