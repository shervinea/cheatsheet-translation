**Logic-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-logic-models)

<br>

**1. Logic-based models with propositional and first-order logic**

&#10230;

<br>


**2. Basics**

&#10230;

<br>


**3. Syntax of propositional logic ― By noting f,g formulas, and ¬,∧,∨,→,↔ connectives, we can write the following logical expressions:**

&#10230;

<br>


**4. [Name, Symbol, Meaning, Illustration]**

&#10230;

<br>


**5. [Affirmation, Negation, Conjunction, Disjunction, Implication, Biconditional]**

&#10230;

<br>


**6. [not f, f and g, f or g, if f then g, f, that is to say g]**

&#10230;

<br>


**7. Remark: formulas can be built up recursively out of these connectives.**

&#10230;

<br>


**8. Model ― A model w denotes an assignment of binary weights to propositional symbols.**

&#10230;

<br>


**9. Example: the set of truth values w={A:0,B:1,C:0} is one possible model to the propositional symbols A, B and C.**

&#10230;

<br>


**10. Interpretation function ― The interpretation function I(f,w) outputs whether model w satisfies formula f:**

&#10230;

<br>


**11. Set of models ― M(f) denotes the set of models w that satisfy formula f. Mathematically speaking, we define it as follows:**

&#10230;

<br>


**12. Knowledge base**

&#10230;

<br>


**13. Definition ― The knowledge base KB is the conjunction of all formulas that have been considered so far. The set of models of the knowledge base is the intersection of the set of models that satisfy each formula. In other words:**

&#10230;

<br>


**14. Probabilistic interpretation ― The probability that query f is evaluated to 1 can be seen as the proportion of models w of the knowledge base KB that satisfy f, i.e.:**

&#10230;

<br>


**15. Satisfiability ― The knowledge base KB is said to be satisfiable if at least one model w satisfies all its constraints. In other words:**

&#10230;

<br>


**16. satisfiable**

&#10230;

<br>


**17. Remark: M(KB) denotes the set of models compatible with all the constraints of the knowledge base.**

&#10230;

<br>


**18. Relation between formulas and knowledge base - We define the following properties between the knowledge base KB and a new formula f:**

&#10230;

<br>


**19. [Name, Mathematical formulation, Illustration, Notes]**

&#10230;

<br>


**20. [KB entails f, KB contradicts f, f contingent to KB]**

&#10230;

<br>


**21. [f does not bring any new information, Also written KB⊨f, No model satisfies the constraints after adding f, Equivalent to KB⊨¬f, f does not contradict KB, f adds a non-trivial amount of information to KB]**

&#10230;

<br>


**22. Model checking ― A model checking algorithm takes as input a knowledge base KB and outputs whether it is satisfiable or not.**

&#10230;

<br>


**23. Remark: popular model checking algorithms include DPLL and WalkSat.**

&#10230;

<br>


**24. Inference rule ― An inference rule of premises f1,...,fk and conclusion g is written:**

&#10230;

<br>


**25. Forward inference algorithm ― From a set of inference rules Rules, this algorithm goes through all possible f1,...,fk and adds g to the knowledge base KB if a matching rule exists. This process is repeated until no more additions can be made to KB.**

&#10230;

<br>


**26. Derivation ― We say that KB derives f (written KB⊢f) with rules Rules if f already is in KB or gets added during the forward inference algorithm using the set of rules Rules.**

&#10230;

<br>


**27. Properties of inference rules ― A set of inference rules Rules can have the following properties:**

&#10230;

<br>


**28. [Name, Mathematical formulation, Notes]**

&#10230;

<br>


**29. [Soundness, Completeness]**

&#10230;

<br>


**30. [Inferred formulas are entailed by KB, Can be checked one rule at a time, "Nothing but the truth", Formulas entailing KB are either already in the knowledge base or inferred from it, "The whole truth"]**

&#10230;

<br>


**31. Propositional logic**

&#10230;

<br>


**32. In this section, we will go through logic-based models that use logical formulas and inference rules. The idea here is to balance expressivity and computational efficiency.**

&#10230;

<br>


**33. Horn clause ― By noting p1,...,pk and q propositional symbols, a Horn clause has the form:**

&#10230;

<br>


**34. Remark: when q=false, it is called a "goal clause", otherwise we denote it as a "definite clause".**

&#10230;

<br>


**35. Modus ponens ― For propositional symbols f1,...,fk and p, the modus ponens rule is written:**

&#10230;

<br>


**36. Remark: it takes linear time to apply this rule, as each application generate a clause that contains a single propositional symbol.**

&#10230;

<br>


**37. Completeness ― Modus ponens is complete with respect to Horn clauses if we suppose that KB contains only Horn clauses and p is an entailed propositional symbol. Applying modus ponens will then derive p.**

&#10230;

<br>


**38. Conjunctive normal form ― A conjunctive normal form (CNF) formula is a conjunction of clauses, where each clause is a disjunction of atomic formulas.**

&#10230;

<br>


**39. Remark: in other words, CNFs are ∧ of ∨.**

&#10230;

<br>


**40. Equivalent representation ― Every formula in propositional logic can be written into an equivalent CNF formula. The table below presents general conversion properties:**

&#10230;

<br>


**41. [Rule name, Initial, Converted, Eliminate, Distribute, over]**

&#10230;

<br>


**42. Resolution rule ― For propositional symbols f1,...,fn, and g1,...,gm as well as p, the resolution rule is written:**

&#10230;

<br>


**43. Remark: it can take exponential time to apply this rule, as each application generates a clause that has a subset of the propositional symbols.**

&#10230;

<br>


**44. [Resolution-based inference ― The resolution-based inference algorithm follows the following steps:, Step 1: Convert all formulas into CNF, Step 2: Repeatedly apply resolution rule, Step 3: Return unsatisfiable if and only if False, is derived]**

&#10230;

<br>


**45. First-order logic**

&#10230;

<br>


**46. The idea here is to use variables to yield more compact knowledge representations.**

&#10230;

<br>


**47. [Model ― A model w in first-order logic maps:, constant symbols to objects, predicate symbols to tuple of objects]**

&#10230;

<br>


**48. Horn clause ― By noting x1,...,xn variables and a1,...,ak,b atomic formulas, the first-order logic version of a horn clause has the form:**

&#10230;

<br>


**49. Substitution ― A substitution θ maps variables to terms and Subst[θ,f] denotes the result of substitution θ on f.**

&#10230;

<br>


**50. Unification ― Unification takes two formulas f and g and returns the most general substitution θ that makes them equal:**

&#10230;

<br>


**51. such that**

&#10230;

<br>


**52. Note: Unify[f,g] returns Fail if no such θ exists.**

&#10230;

<br>


**53. Modus ponens ― By noting x1,...,xn variables, a1,...,ak and a′1,...,a′k atomic formulas and by calling θ=Unify(a′1∧...∧a′k,a1∧...∧ak) the first-order logic version of modus ponens can be written:**

&#10230;

<br>


**54. Completeness ― Modus ponens is complete for first-order logic with only Horn clauses.**

&#10230;

<br>


**55. Resolution rule ― By noting f1,...,fn, g1,...,gm, p, q formulas and by calling θ=Unify(p,q), the first-order logic version of the resolution rule can be written:**

&#10230;

<br>


**56. [Semi-decidability ― First-order logic, even restricted to only Horn clauses, is semi-decidable., if KB⊨f, forward inference on complete inference rules will prove f in finite time, if KB⊭f, no algorithm can show this in finite time]**

&#10230;

<br>


**57. [Basics, Notations, Model, Interpretation function, Set of models]**

&#10230;

<br>


**58. [Knowledge base, Definition, Probabilistic interpretation, Satisfiability, Relationship with formulas, Forward inference, Rule properties]**

&#10230;

<br>


**59. [Propositional logic, Clauses, Modus ponens, Conjunctive normal form, Representation equivalence, Resolution]**

&#10230;

<br>


**60. [First-order logic, Substitution, Unification, Resolution rule, Modus ponens, Resolution, Semi-decidability]**

&#10230;

<br>


**61. View PDF version on GitHub**

&#10230;

<br>


**62. Original authors**

&#10230;

<br>


**63. Translated by X, Y and Z**

&#10230;

<br>


**64. Reviewed by X, Y and Z**

&#10230;

<br>


**65. By X and Y**

&#10230;

<br>


**66. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230;
