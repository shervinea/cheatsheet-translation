**Logic-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-logic-models)

<br>

**1. Logic-based models with propositional and first-order logic**

&#10230; Modèles logiques propositionnels et calcul des prédicats du premier ordre

<br>


**2. Basics**

&#10230; Bases

<br>


**3. Syntax of propositional logic ― By noting f,g formulas, and ¬,∧,∨,→,↔ connectives, we can write the following logical expressions:**

&#10230; Syntaxe de la logique propositionnelle - En notant f et g formules et ¬,∧,∨,→,↔ opérateurs, on peut écrire les expressions logiques suivantes :

<br>


**4. [Name, Symbol, Meaning, Illustration]**

&#10230; [Nom, Symbole, Signification, Illustration]

<br>


**5. [Affirmation, Negation, Conjunction, Disjunction, Implication, Biconditional]**

&#10230; [Affirmation, Négation, Conjonction, Disjonction, Implication, Biconditionnel]

<br>


**6. [not f, f and g, f or g, if f then g, f, that is to say g]**

&#10230; [non f, f et g, f ou g, si f alors g, f, c'est à dire g]

<br>


**7. Remark: formulas can be built up recursively out of these connectives.**

&#10230; Remarque : n'importe quelle formule peut être construite de manière récursive à partir de ces opérateurs.

<br>


**8. Model ― A model w denotes an assignment of binary weights to propositional symbols.**

&#10230; [Modèle - Un modèle w dénote une combinaison de valeurs binaires liées à des symboles propositionnels]

<br>


**9. Example: the set of truth values w={A:0,B:1,C:0} is one possible model to the propositional symbols A, B and C.**

&#10230; Exemple : l'ensemble de valeurs de vérité w={A:0,B:1,C:0} est un modèle possible pour les symboles propositionnels A, B et C.

<br>


**10. Interpretation function ― The interpretation function I(f,w) outputs whether model w satisfies formula f:**

&#10230; Interprétation - L'interprétation I(f,w) nous renseigne si le modèle w satisfait la formule f :

<br>


**11. Set of models ― M(f) denotes the set of models w that satisfy formula f. Mathematically speaking, we define it as follows:**

&#10230; Ensemble de modèles - M(f) dénote l'ensemble des modèles w qui satisfont la formule f. Sa définition mathématique est donnée par :

<br>


**12. Knowledge base**

&#10230; Base de connaissance

<br>


**13. Definition ― The knowledge base KB is the conjunction of all formulas that have been considered so far. The set of models of the knowledge base is the intersection of the set of models that satisfy each formula. In other words:**

&#10230; Définition - La base de connaissance KB est la conjonction de toutes les formules considérées jusqu'à présent. L'ensemble des modèles de la base de connaissance est l'intersection de l'ensemble des modèles satisfaisant chaque formule. En d'autres termes :

<br>


**14. Probabilistic interpretation ― The probability that query f is evaluated to 1 can be seen as the proportion of models w of the knowledge base KB that satisfy f, i.e.:**

&#10230; Interprétation en termes de probabilités - La probabilté que la requête f soit évaluée à 1 peut être vue comme la proportion des modèles w de la base de connaissance KB qui satisfait f, i.e. :

<br>


**15. Satisfiability ― The knowledge base KB is said to be satisfiable if at least one model w satisfies all its constraints. In other words:**

&#10230; Satisfaisabilité - La base de connaissance KB est dite satisfaisable si au moins un modèle w satisfait toutes ses contraintes. En d'autres termes :

<br>


**16. satisfiable**

&#10230; satisfaisable

<br>


**17. Remark: M(KB) denotes the set of models compatible with all the constraints of the knowledge base.**

&#10230; Remarque : M(KB) dénote l'ensemble des modèles compatibles avec toutes les contraintes de la base de connaissance.

<br>


**18. Relation between formulas and knowledge base - We define the following properties between the knowledge base KB and a new formula f:**

&#10230; Relation entre formules et base de connaissance - On définit les propriétés suivantes entre la base de connaissance KB et une nouvelle formule f :

<br>


**19. [Name, Mathematical formulation, Illustration, Notes]**

&#10230; [Nom, Formulation mathématique, Illustration, Notes]

<br>


**20. [KB entails f, KB contradicts f, f contingent to KB]**

&#10230; [KB déduit f, KB contredit f, f est contingent à KB]

<br>


**21. [f does not bring any new information, Also written KB⊨f, No model satisfies the constraints after adding f, Equivalent to KB⊨¬f, f does not contradict KB, f adds a non-trivial amount of information to KB]**

&#10230; [f n'apporte aucune nouvelle information, Aussi écrit KB⊨f, Aucun modèle ne satisfait les contraintes après l'ajout de f, Équivalent à KB⊨¬f, f ne contredit pas KB, f ajoute une quantité non-triviale d'information à KB]

<br>


**22. Model checking ― A model checking algorithm takes as input a knowledge base KB and outputs whether it is satisfiable or not.**

&#10230; Vérification de modèles - Un algorithme de vérification de modèles (model checking en anglais) prend comme argument une base de connaissance KB et nous renseigne si celle-ci est satisfaisable ou pas.

<br>


**23. Remark: popular model checking algorithms include DPLL and WalkSat.**

&#10230; Remarque : DPLL et WalkSat sont des exemples populaires d'algorithmes de vérification de modèles.

<br>


**24. Inference rule ― An inference rule of premises f1,...,fk and conclusion g is written:**

&#10230;  Règle d'inférence - Une règle d'inférence de prémisses f1,...,fk et de conclusion g s'écrit :

<br>


**25. Forward inference algorithm ― From a set of inference rules Rules, this algorithm goes through all possible f1,...,fk and adds g to the knowledge base KB if a matching rule exists. This process is repeated until no more additions can be made to KB.**

&#10230; Algorithme de chaînage avant (forward inference algorithm) - Partant d'un ensemble de règles d'inférence Rules, cet algorithme parcourt tous les f1,...,fk et ajoute g à la base de connaissance KB si une règle parvient à une telle conclusion. Cette démarche est répétée jusqu'à ce qu'aucun autre ajout ne puisse être fait à KB.

<br>


**26. Derivation ― We say that KB derives f (written KB⊢f) with rules Rules if f already is in KB or gets added during the forward inference algorithm using the set of rules Rules.**

&#10230; Dérivation - On dit que KB dérive f (noté KB⊢f) par le biais des règles Rules soit si f est déjà dans KB ou si elle se fait ajouter pendant l'application du chaînage avant utilisant les règles Rules.

<br>


**27. Properties of inference rules ― A set of inference rules Rules can have the following properties:**

&#10230; Propriétés des règles d'inférence - Un ensemble de règles d'inférence Rules peut avoir les propriétés suivantes :

<br>


**28. [Name, Mathematical formulation, Notes]**

&#10230; [Nom, Formulation mathématique, Notes]

<br>


**29. [Soundness, Completeness]**

&#10230; [Correction, Complétude]

<br>


**30. [Inferred formulas are entailed by KB, Can be checked one rule at a time, "Nothing but the truth", Formulas entailing KB are either already in the knowledge base or inferred from it, "The whole truth"]**

&#10230; [Les formules inférées sont déduites par KB, Peut être vérifiée un règle à la fois, "Rien que la vérité", Les formules déduites par KB sont soit déjà dans la base de connaissance, soit inférées de celle-ci, "La vérité dans sa totalité"]

<br>


**31. Propositional logic**

&#10230; Logique propositionnelle

<br>


**32. In this section, we will go through logic-based models that use logical formulas and inference rules. The idea here is to balance expressivity and computational efficiency.**

&#10230; Dans cette section, nous allons parcourir les modèles logiques utilisant des formules logiques et des règles d'inférence. L'idée est de trouver le juste milieu entre expressivité et efficacité en termes de calculs.

<br>


**33. Horn clause ― By noting p1,...,pk and q propositional symbols, a Horn clause has the form:**

&#10230; Clause de Horn - En notant p1,...,pk et q des symboles propositionnels, une clause de Horn s'écrit :

<br>


**34. Remark: when q=false, it is called a "goal clause", otherwise we denote it as a "definite clause".**

&#10230; Remarque : quand q=false, cette clause de Horn est "négative", autrement elle est appelée "stricte".

<br>


**35. Modus ponens ― For propositional symbols f1,...,fk and p, the modus ponens rule is written:**

&#10230; Modus ponens - Sur les symboles propositionnels f1,...,fk et p, la règle de modus ponens est écrite :

<br>


**36. Remark: it takes linear time to apply this rule, as each application generate a clause that contains a single propositional symbol.**

&#10230; Remarque : l'application de cette règle se fait en temps linéaire, puisque chaque exécution génère une clause contenant un symbole propositionnel.

<br>


**37. Completeness ― Modus ponens is complete with respect to Horn clauses if we suppose that KB contains only Horn clauses and p is an entailed propositional symbol. Applying modus ponens will then derive p.**

&#10230; Complétude - Modus ponens est complet lorsqu'on le munit des clauses de Horn si l'on suppose que KB contient uniquement des clauses de Horn et que p est un symbole propositionnel qui est déduit. L'application de modus ponens dérivera alors p.

<br>


**38. Conjunctive normal form ― A conjunctive normal form (CNF) formula is a conjunction of clauses, where each clause is a disjunction of atomic formulas.**

&#10230; Forme normale conjonctive - La forme normale conjonctive (en anglais conjunctive normal form ou CNF) d'une formule est une conjonction de clauses, chacune d'entre elles étant une dijonction de formules atomiques.

<br>


**39. Remark: in other words, CNFs are ∧ of ∨.**

&#10230; Remarque : en d'autres termes, les CNFs sont des ∧ de ∨.

<br>


**40. Equivalent representation ― Every formula in propositional logic can be written into an equivalent CNF formula. The table below presents general conversion properties:**

&#10230; Représentation équivalente - Chaque formule en logique propositionnelle peut être écrite de manière équivalente sous la forme d'une formule CNF. Le tableau ci-dessous présente les propriétés principales permettant une telle conversion :

<br>


**41. [Rule name, Initial, Converted, Eliminate, Distribute, over]**

&#10230; [Nom de la règle, Début, Résultat, Élimine, Distribue, sur]

<br>


**42. Resolution rule ― For propositional symbols f1,...,fn, and g1,...,gm as well as p, the resolution rule is written:**

&#10230; Règle de résolution - Pour des symboles propositionnels f1,...,fn, et g1,...,gm ainsi que p, la règle de résolution s'écrit :

<br>


**43. Remark: it can take exponential time to apply this rule, as each application generates a clause that has a subset of the propositional symbols.**

&#10230; Remarque : l'application de cette règle peut prendre un temps exponentiel, vu que chaque itération génère une clause constituée d'une partie des symboles propositionnels.

<br>


**44. [Resolution-based inference ― The resolution-based inference algorithm follows the following steps:, Step 1: Convert all formulas into CNF, Step 2: Repeatedly apply resolution rule, Step 3: Return unsatisfiable if and only if False, is derived]**

&#10230; [Inférence basée sur la règle de résolution - L'algorithme d'inférence basée sur la règle de résolution se déroule en plusieurs étapes :, Étape 1 : Conversion de toutes les formules vers leur forme CNF, Étape 2 : Application répétée de la règle de résolution, Étape 3 : Renvoyer "non satisfaisable" si et seulement si False est dérivé]

<br>


**45. First-order logic**

&#10230; Calcul des prédicats du premier ordre

<br>


**46. The idea here is to use variables to yield more compact knowledge representations.**

&#10230; L'idée ici est d'utiliser des variables et ainsi permettre une représentation des connaissances plus compacte.

<br>


**47. [Model ― A model w in first-order logic maps:, constant symbols to objects, predicate symbols to tuple of objects]**

&#10230; [Modèle - Un modèle w en calcul des prédicats du premier ordre lie :, des symboles constants à des objets, des prédicats à n-uplets d'objets]

<br>


**48. Horn clause ― By noting x1,...,xn variables and a1,...,ak,b atomic formulas, the first-order logic version of a horn clause has the form:**

&#10230; Clause de Horn - En notant x1,...,xn variables et a1,...,ak,b formules atomiques, une clause de Horn pour le calcul des prédicats du premier ordre a la forme :

<br>


**49. Substitution ― A substitution θ maps variables to terms and Subst[θ,f] denotes the result of substitution θ on f.**

&#10230; Substitution - Une substitution θ lie les variables aux termes et Subst[θ,f] désigne le résultat de la substitution θ sur f.

<br>


**50. Unification ― Unification takes two formulas f and g and returns the most general substitution θ that makes them equal:**

&#10230; Unification - Une unification prend deux formules f et g et renvoie la substitution θ la plus générale les rendant égales :

<br>


**51. such that**

&#10230; tel que

<br>


**52. Note: Unify[f,g] returns Fail if no such θ exists.**

&#10230; Note : Unify[f,g] renvoie Fail si un tel θ n'existe pas.

<br>


**53. Modus ponens ― By noting x1,...,xn variables, a1,...,ak and a′1,...,a′k atomic formulas and by calling θ=Unify(a′1∧...∧a′k,a1∧...∧ak) the first-order logic version of modus ponens can be written:**

&#10230; Modus ponens - En notant x1,...,xn variables, a1,...,ak et a′1,...,a′k formules atomiques et en notant θ=Unify(a′1∧...∧a′k,a1∧...∧ak), modus ponens pour le calcul des prédicats du premier ordre s'écrit :

<br>


**54. Completeness ― Modus ponens is complete for first-order logic with only Horn clauses.**

&#10230; Complétude - Modus ponens est complet pour le calcul des prédicats du premier ordre lorsqu'il agit uniquement sur les clauses de Horn.

<br>


**55. Resolution rule ― By noting f1,...,fn, g1,...,gm, p, q formulas and by calling θ=Unify(p,q), the first-order logic version of the resolution rule can be written:**

&#10230; Règle de résolution - En notant f1,...,fn, g1,...,gm, p, q formules et en posant θ=Unify(p,q), le règle de résolution pour le calcul des prédicats du premier ordre s'écrit :

<br>


**56. [Semi-decidability ― First-order logic, even restricted to only Horn clauses, is semi-decidable., if KB⊨f, forward inference on complete inference rules will prove f in finite time, if KB⊭f, no algorithm can show this in finite time]**

&#10230; [Semi-décidabilité - Le calcul des prédicats du premier ordre, même restreint aux clauses de Horn, n'est que semi-décidable., si KB⊨f, l'algorithme de chaînage avant sur des règles d'inférence complètes prouvera f en temps fini, si KB⊭f, aucun algorithme ne peut le prouver en temps fini]

<br>


**57. [Basics, Notations, Model, Interpretation function, Set of models]**

&#10230; [Bases, Notations, Modèle, Interprétation, Ensemble de modèles]

<br>


**58. [Knowledge base, Definition, Probabilistic interpretation, Satisfiability, Relationship with formulas, Forward inference, Rule properties]**

&#10230; [Base de connaissance, Définition, Interprétation en termes de probabilité, Satisfaisabilité, Lien avec les formules, Chaînage en avant, Propriétés des règles]

<br>


**59. [Propositional logic, Clauses, Modus ponens, Conjunctive normal form, Representation equivalence, Resolution]**

&#10230; [Logique propositionnelle, Clauses, Modus ponens, Forme normale conjonctive, Représentation équivalente, Résolution]

<br>


**60. [First-order logic, Substitution, Unification, Resolution rule, Modus ponens, Resolution, Semi-decidability]**

&#10230; [Calcul des prédicats du premier ordre, Substitution, Unification, Règle de résolution, Modus ponens, Résolution, Semi-décidabilité]

<br>


**61. View PDF version on GitHub**

&#10230; Voir la version PDF sur GitHub

<br>


**62. Original authors**

&#10230; Auteurs originaux.

<br>


**63. Translated by X, Y and Z**

&#10230; Traduit par X, Y et Z.

<br>


**64. Reviewed by X, Y and Z**

&#10230; Revu par X, Y et Z.

<br>


**65. By X and Y**

&#10230; Par X et Y.

<br>


**66. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Les pense-bêtes d'intelligence artificielle sont maintenant disponibles en français.
