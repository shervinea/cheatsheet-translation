**Logic-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-logic-models)

<br>

**1. Logic-based models with propositional and first-order logic**

&#10230; Модели логик с логикой высказываний и логикой первого порядка

<br>


**2. Basics**

&#10230; Основы

<br>


**3. Syntax of propositional logic ― By noting f,g formulas, and ¬,∧,∨,→,↔ connectives, we can write the following logical expressions:**

&#10230; Синтаксис логики высказываний ― Обозначим формулы f,g и ¬,∧,∨,→,↔ связки, мы можем написать следующие логические выражения:

<br>


**4. [Name, Symbol, Meaning, Illustration]**

&#10230; [Название, Символ, Значение, Иллюстрация]

<br>


**5. [Affirmation, Negation, Conjunction, Disjunction, Implication, Biconditional]**

&#10230; [Утверждение, Отрицание, Конъюнкция, Дизъюнкция, Импликация, Biconditional (Эквивалентность)]

<br>


**6. [not f, f and g, f or g, if f then g, f, that is to say g]**

&#10230; [не f, f и g, f или g, если f то g, f означает g]

<br>


**7. Remark: formulas can be built up recursively out of these connectives.**

&#10230; Примечание: формулы могут быть построены рекурсивно из этих связок.

<br>


**8. Model ― A model w denotes an assignment of binary weights to propositional symbols.**

&#10230; Модель ― Модель w обозначает присвоение бинарных весов символам высказываний.

<br>


**9. Example: the set of truth values w={A:0,B:1,C:0} is one possible model to the propositional symbols A, B and C.**

&#10230; Пример: набор значений истинности w={A:0,B:1,C:0} является одной из возможных моделей символов высказываний A, B и C.

<br>


**10. Interpretation function ― The interpretation function I(f,w) outputs whether model w satisfies formula f:**

&#10230; Функция интерпретации ― Функция интерпретации I(f,w) выводит, удовлетворяет ли модель w формуле f:

<br>


**11. Set of models ― M(f) denotes the set of models w that satisfy formula f. Mathematically speaking, we define it as follows:**

&#10230; Набор моделей ― M(f) обозначает множество моделей w, удовлетворяющих формуле f. С математической точки зрения мы определяем это следующим образом:

<br>


**12. Knowledge base**

&#10230; База знаний

<br>


**13. Definition ― The knowledge base KB is the conjunction of all formulas that have been considered so far. The set of models of the knowledge base is the intersection of the set of models that satisfy each formula. In other words:**

&#10230; Определение ― Knowledge base (KB) - База знаний - это совокупность всех рассмотренных формул. Набор моделей базы знаний - это пересечение набора моделей, удовлетворяющих каждой формуле. Другими словами:

<br>


**14. Probabilistic interpretation ― The probability that query f is evaluated to 1 can be seen as the proportion of models w of the knowledge base KB that satisfy f, i.e.:**

&#10230; Вероятностная интерпретация ― Вероятность того, что запрос f будет оценен как 1, можно рассматривать как долю моделей w в базе знаний KB, которые удовлетворяют f, то есть:

<br>


**15. Satisfiability ― The knowledge base KB is said to be satisfiable if at least one model w satisfies all its constraints. In other words:**

&#10230; Выполнимость ― База знаний KB называется выполнимой, если хотя бы одна модель w удовлетворяет всем её ограничениям. Другими словами:

<br>


**16. satisfiable**

&#10230; удовлетворительно

<br>


**17. Remark: M(KB) denotes the set of models compatible with all the constraints of the knowledge base.**

&#10230; Примечание: M(KB) обозначает набор моделей, совместимых со всеми ограничениями базы знаний.

<br>


**18. Relation between formulas and knowledge base - We define the following properties between the knowledge base KB and a new formula f:**

&#10230; Связь между формулами и базой знаний. Мы определяем следующие свойства между базой знаний KB и новой формулой f:

<br>


**19. [Name, Mathematical formulation, Illustration, Notes]**

&#10230; [Название, Математическая формулировка, Иллюстрация, Примечания]

<br>


**20. [KB entails f, KB contradicts f, f contingent to KB]**

&#10230; [KB влечет за собой f, KB противоречит f, f зависит от KB]

<br>


**21. [f does not bring any new information, Also written KB⊨f, No model satisfies the constraints after adding f, Equivalent to KB⊨¬f, f does not contradict KB, f adds a non-trivial amount of information to KB]**

&#10230; [f не несет никакой новой информации, Также пишут KB⊨f, После добавления f ни одна модель не удовлетворяет ограничениям, Эквивалентно KB⊨¬f, f не противоречит KB, f добавляет в КБ нетривиальное количество информации]

<br>


**22. Model checking ― A model checking algorithm takes as input a knowledge base KB and outputs whether it is satisfiable or not.**

&#10230; Проверка модели ― Алгоритм проверки модели принимает в качестве входных данных базу знаний KB и выводит, выполнима она или нет.

<br>


**23. Remark: popular model checking algorithms include DPLL and WalkSat.**

&#10230; Примечание: популярные алгоритмы проверки моделей включают DPLL и WalkSat.

<br>


**24. Inference rule ― An inference rule of premises f1,...,fk and conclusion g is written:**

&#10230; Правило вывода ― Записывается правило вывода посылок f1,...,fk и заключения g:

<br>


**25. Forward inference algorithm ― From a set of inference rules Rules, this algorithm goes through all possible f1,...,fk and adds g to the knowledge base KB if a matching rule exists. This process is repeated until no more additions can be made to KB.**

&#10230; Алгоритм прямого вывода ― Из набора правил вывода Rules этот алгоритм перебирает все возможные f1,...,fk и добавляет g в базу знаний KB, если существует правило сопоставления. Этот процесс повторяется до тех пор, пока не перестанут поступать дополнения в базу знаний.

<br>


**26. Derivation ― We say that KB derives f (written KB⊢f) with rules Rules if f already is in KB or gets added during the forward inference algorithm using the set of rules Rules.**

&#10230; Вывод ― Мы говорим, что KB выводит f (записывается как KB⊢f) с помощью правил Rules, если f уже находится в KB или добавляется во время алгоритма прямого вывода с использованием набора правил Rules.

<br>


**27. Properties of inference rules ― A set of inference rules Rules can have the following properties:**

&#10230; Свойства правил вывода ― Набор правил вывода. Правила могут иметь следующие свойства:

<br>


**28. [Name, Mathematical formulation, Notes]**

&#10230; [Название, Математическая формулировка, Примечания]

<br>


**29. [Soundness, Completeness]**

&#10230; [Обоснованность, Полнота]

<br>


**30. [Inferred formulas are entailed by KB, Can be checked one rule at a time, "Nothing but the truth", Formulas entailing KB are either already in the knowledge base or inferred from it, "The whole truth"]**

&#10230; [Предполагаемые формулы вытекают из KB, Можно проверить одно правило за раз, "Ничего кроме правды", Влекущие за собой КБ формулы уже есть в базе знаний либо выведены из неё, "Полная правда"]

<br>


**31. Propositional logic**

&#10230; Логика высказываний

<br>


**32. In this section, we will go through logic-based models that use logical formulas and inference rules. The idea here is to balance expressivity and computational efficiency.**

&#10230; В этом разделе мы рассмотрим модели на основе логики, в которых используются логические формулы и правила вывода. Идея состоит в том, чтобы сбалансировать выразительность и вычислительную эффективность.

<br>


**33. Horn clause ― By noting p1,...,pk and q propositional symbols, a Horn clause has the form:**

&#10230; Хорновский дизъюнкт ― Обозначим p1,...,pk и q пропозициональные символы, Хорновский дизъюнкт имеет вид:

<br>


**34. Remark: when q=false, it is called a "goal clause", otherwise we denote it as a "definite clause".**

&#10230; Примечание: когда q=false, это называется "целевой дизъюнкт", в противном случае обозначим его как "определенный дизъюнкт".

<br>


**35. Modus ponens ― For propositional symbols f1,...,fk and p, the modus ponens rule is written:**

&#10230; Modus ponens ― Правило вывода - Для пропозициональных символов f1,...,fk и p записывается правило вывода:

<br>


**36. Remark: it takes linear time to apply this rule, as each application generate a clause that contains a single propositional symbol.**

&#10230; Примечание: для применения этого правила требуется линейное время, так как каждое приложение генерирует предложение, содержащее один пропозициональный символ.

<br>


**37. Completeness ― Modus ponens is complete with respect to Horn clauses if we suppose that KB contains only Horn clauses and p is an entailed propositional symbol. Applying modus ponens will then derive p.**

&#10230; Полнота ― Modus ponens полон относительно Хорновских дизъюнктов, если предположить, что КБ содержит только Хорновские дизъюнкты а p - подразумеваемый пропозициональный символ. После применения modus ponens будет получено p.

<br>


**38. Conjunctive normal form ― A conjunctive normal form (CNF) formula is a conjunction of clauses, where each clause is a disjunction of atomic formulas.**

&#10230; Конъюнктивная нормальная форма (CNF) ― Формула конъюнктивной нормальной формы - это конъюнкция предложений, где каждое предложение является дизъюнкцией атомарных формул.

<br>


**39. Remark: in other words, CNFs are ∧ of ∨.**

&#10230; Примечание: другими словами, CNF-ы составляют ∧ из ∨.

<br>


**40. Equivalent representation ― Every formula in propositional logic can be written into an equivalent CNF formula. The table below presents general conversion properties:**

&#10230; Эквивалентное представление ― Каждую формулу логики высказываний можно записать в эквивалентную формулу CNF. В таблице ниже представлены общие свойства преобразования:

<br>


**41. [Rule name, Initial, Converted, Eliminate, Distribute, over]**

&#10230; [Имя правила, Начальное, Преобразованное, Исключить, Распределить, над]

<br>


**42. Resolution rule ― For propositional symbols f1,...,fn, and g1,...,gm as well as p, the resolution rule is written:**

&#10230; Правило разрешения ― Для пропозициональных символов f1,...,fn, и g1,...,gm так же, как и p, правило разрешения записывается:

<br>


**43. Remark: it can take exponential time to apply this rule, as each application generates a clause that has a subset of the propositional symbols.**

&#10230; Примечание: для применения этого правила может потребоваться экспоненциальное время, поскольку каждое приложение генерирует предложение, которое имеет подмножество пропозициональных символов.

<br>


**44. [Resolution-based inference ― The resolution-based inference algorithm follows the following steps:, Step 1: Convert all formulas into CNF, Step 2: Repeatedly apply resolution rule, Step 3: Return unsatisfiable if and only if False, is derived]**

&#10230; [Вывод на основе разрешения ― Алгоритм вывода на основе разрешении. Включает следующие шаги:, Шаг 1: Преобразовать все формулы в CNF, Шаг 2. Повторно применить правило разрешения, Шаг 3: Вернуть неудовлетворительно; если и только если False, выводится]

<br>


**45. First-order logic**

&#10230; Логика первого порядка

<br>


**46. The idea here is to use variables to yield more compact knowledge representations.**

&#10230; Идея здесь состоит в том, чтобы использовать переменные для получения более компактных представлений знаний.

<br>


**47. [Model ― A model w in first-order logic maps:, constant symbols to objects, predicate symbols to tuple of objects]**

&#10230; [Модель ― Модель w в логике первого порядка отображает:, константные символы на объекты, предикатные символы на кортеж объектов]

<br>


**48. Horn clause ― By noting x1,...,xn variables and a1,...,ak,b atomic formulas, the first-order logic version of a horn clause has the form:**

&#10230; Хорновский дизъюнкт ― Обозначим переменные x1,...,xn и атомарные формулы a1,...,ak,b, логическая версия Хорновского дизъюнкт первого порядка имеет вид:

<br>


**49. Substitution ― A substitution θ maps variables to terms and Subst[θ,f] denotes the result of substitution θ on f.**

&#10230; Подстановка ― Замена θ отображает переменные в составляющие формул, а Subst[θ,f] обозначает результат замены θ на f.

<br>


**50. Unification ― Unification takes two formulas f and g and returns the most general substitution θ that makes them equal:**

&#10230; Объединение ― Объединение берет две формулы f и g и возвращает наиболее общую замену θ, которая уравнивает их:

<br>


**51. such that**

&#10230; такой что

<br>


**52. Note: Unify[f,g] returns Fail if no such θ exists.**

&#10230; Примечание: Unify[f,g] возвращает Fail, если такой θ не существует.

<br>


**53. Modus ponens ― By noting x1,...,xn variables, a1,...,ak and a′1,...,a′k atomic formulas and by calling θ=Unify(a′1∧...∧a′k,a1∧...∧ak) the first-order logic version of modus ponens can be written:**

&#10230; Modus ponens ― Обозначим переменные x1,...,xn, атомарные формулы a1,...,ak и a′1,...,a′k и вызвав θ=Unify(a′1∧...∧a′k,a1∧...∧ak) логическая версия modus ponens первого порядка может быть записана:

<br>


**54. Completeness ― Modus ponens is complete for first-order logic with only Horn clauses.**

&#10230; Полнота ― Modus ponens полон для логики первого порядка только с Хорновскими дизъюнктами.

<br>


**55. Resolution rule ― By noting f1,...,fn, g1,...,gm, p, q formulas and by calling θ=Unify(p,q), the first-order logic version of the resolution rule can be written:**

&#10230; Правило разрешения ― Обозначим формулы f1,...,fn, g1,...,gm, p, q и вызовем θ=Unify(p,q), логическая версия первого порядка правила разрешения может быть записана:

<br>


**56. [Semi-decidability ― First-order logic, even restricted to only Horn clauses, is semi-decidable., if KB⊨f, forward inference on complete inference rules will prove f in finite time, if KB⊭f, no algorithm can show this in finite time]**

&#10230; [Полуразрешимость ― Логика первого порядка, даже ограниченная только Хорновскими дизъюнктами, полуразрешима., если KB⊨f, прямой вывод по правилам полного вывода докажет f за конечное время, если KB⊭f, ни один алгоритм не может показать это за конечное время]

<br>


**57. [Basics, Notations, Model, Interpretation function, Set of models]**

&#10230; [Основы, Обозначения, Модель, Функция интерпретации, Набор моделей]

<br>


**58. [Knowledge base, Definition, Probabilistic interpretation, Satisfiability, Relationship with formulas, Forward inference, Rule properties]**

&#10230; [База знаний, Определение, Вероятностная интерпретация, Выполнимость, Связь с формулами, Прямой вывод, Свойства правил]

<br>


**59. [Propositional logic, Clauses, Modus ponens, Conjunctive normal form, Representation equivalence, Resolution]**

&#10230; [Логика высказываний, Дизъюнкты, Modus ponens, Конъюнктивная нормальная форма, Эквивалентность представлений, Разрешение]

<br>


**60. [First-order logic, Substitution, Unification, Resolution rule, Modus ponens, Resolution, Semi-decidability]**

&#10230; [Логика первого порядка, Подстановка, Объединение, Правило разрешения, Modus ponens, Разрешение, Полуразрешимость]

<br>


**61. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**62. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>


**63. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>


**64. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>


**65. By X and Y**

&#10230; По X и Y

<br>


**66. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Шпаргалки по искусственному интеллекту теперь доступны на русском языке.
