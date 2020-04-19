**Logic-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-logic-models)

<br>

**1. Logic-based models with propositional and first-order logic**

&#10230; Các mô hình dựa trên logic với logic mệnh đề và logic bậc nhất

<br>


**2. Basics**

&#10230; Cơ bản

<br>


**3. Syntax of propositional logic ― By noting f,g formulas, and ¬,∧,∨,→,↔ connectives, we can write the following logical expressions:**

&#10230; Cú pháp của logic mệnh đề ― Kí hiệu f,g là các công thức, và ¬,∧,∨,→,↔ các kết nối, chúng ta có thể viết các biểu thức logic sau:

<br>


**4. [Name, Symbol, Meaning, Illustration]**

&#10230; [Tên, Kí hiệu, Ý nghĩa, Miêu tả]

<br>


**5. [Affirmation, Negation, Conjunction, Disjunction, Implication, Biconditional]**

&#10230; [Khẳng định, phủ định, kết hợp, phân ly, hàm ý, nhị phân]

<br>


**6. [not f, f and g, f or g, if f then g, f, that is to say g]**

&#10230; [phủ định f, f và g, f hoặc g, nếu f thì g, f, đó là nói g]

<br>


**7. Remark: formulas can be built up recursively out of these connectives.**

&#10230; Ghi chú: công thức có thể được xây dựng đệ quy từ các kết nối này.

<br>


**8. Model ― A model w denotes an assignment of binary weights to propositional symbols.**

&#10230; Mô hình - Một mô hình w biểu thị việc gán trọng số nhị phân cho các ký hiệu mệnh đề.

<br>


**9. Example: the set of truth values w={A:0,B:1,C:0} is one possible model to the propositional symbols A, B and C.**

&#10230; Ví dụ: tập hợp các giá trị chân lý w ={A:0,B:1,C:0} là một mô hình có thể có cho các ký hiệu mệnh đề A, B và C.

<br>


**10. Interpretation function ― The interpretation function I(f,w) outputs whether model w satisfies formula f:**

&#10230; Hàm giải thích - Hàm giải thích I (f, w) đưa ra liệu mô hình w có thỏa mãn công thức f:

<br>


**11. Set of models ― M(f) denotes the set of models w that satisfy formula f. Mathematically speaking, we define it as follows:**

&#10230; Tập hợp các mô hình - M(f) biểu thị tập hợp các mô hình w thỏa mãn công thức f. Về mặt toán học, chúng ta định nghĩa nó như sau:

<br>


**12. Knowledge base**

&#10230; Cơ sở tri thức

<br>


**13. Definition ― The knowledge base KB is the conjunction of all formulas that have been considered so far. The set of models of the knowledge base is the intersection of the set of models that satisfy each formula. In other words:**

&#10230; Định nghĩa - Cơ sở tri thức KB là sự kết hợp của tất cả các công thức đã được xem xét cho đến nay. Tập hợp các mô hình của cơ sở tri thức là tập giao của tập hợp các mô hình thỏa mãn từng công thức. Nói cách khác:

<br>


**14. Probabilistic interpretation ― The probability that query f is evaluated to 1 can be seen as the proportion of models w of the knowledge base KB that satisfy f, i.e.:**

&#10230; Giải thích xác suất - Xác suất mà truy vấn f được ước tính là 1 có thể được xem là tỷ lệ của các mô hình w của cơ sở tri thức KB thỏa mãn f, tức là:

<br>


**15. Satisfiability ― The knowledge base KB is said to be satisfiable if at least one model w satisfies all its constraints. In other words:**

&#10230; Mức độ thỏa mãn - Cơ sở tri thức KB được cho là thỏa đáng nếu có ít nhất một mô hình w thỏa mãn tất cả các ràng buộc của nó. Nói cách khác:

<br>


**16. satisfiable**

&#10230; thỏa đáng

<br>


**17. Remark: M(KB) denotes the set of models compatible with all the constraints of the knowledge base.**

&#10230; Ghi chú: M(KB) biểu thị tập hợp các mô hình tương thích với tất cả các ràng buộc của cơ sở tri thức.

<br>


**18. Relation between formulas and knowledge base - We define the following properties between the knowledge base KB and a new formula f:**

&#10230; Mối liên hệ giữa công thức và cơ sở tri thức - Chúng tôi định nghĩa các thuộc tính sau giữa KB cơ sở tri thức và công thức mới f:

<br>


**19. [Name, Mathematical formulation, Illustration, Notes]**

&#10230; [Tên, Công thức toán học, Minh họa, Ghi chú]

<br>


**20. [KB entails f, KB contradicts f, f contingent to KB]**

&#10230; [KB suy luận (kết thừa) từ f, KB mâu thuẫn với f, f phụ thuộc vào KB]

<br>


**21. [f does not bring any new information, Also written KB⊨f, No model satisfies the constraints after adding f, Equivalent to KB⊨¬f, f does not contradict KB, f adds a non-trivial amount of information to KB]**

&#10230; [f không mang lại bất kỳ thông tin mới nào, KB writtenf cũng được viết, Không có mô hình nào thỏa mãn các ràng buộc sau khi thêm f, Tương đương với KB⊨¬f, f không mâu thuẫn với KB, f thêm một lượng thông tin không tầm thường vào KB]

<br>


**22. Model checking ― A model checking algorithm takes as input a knowledge base KB and outputs whether it is satisfiable or not.**

&#10230; Kiểm tra mô hình - Thuật toán kiểm tra mô hình lấy đầu vào là KB cơ sở tri thức và đưa ra liệu nó có thỏa đáng hay không.

<br>


**23. Remark: popular model checking algorithms include DPLL and WalkSat.**

&#10230; Ghi chú: các thuật toán kiểm tra mô hình phổ biến bao gồm DPLL và WalkSat.

<br>


**24. Inference rule ― An inference rule of premises f1,...,fk and conclusion g is written:**

&#10230; Quy tắc suy luận - Một quy tắc suy luận của các cơ sở f1,...,fk và kết luận g được viết:

<br>


**25. Forward inference algorithm ― From a set of inference rules Rules, this algorithm goes through all possible f1,...,fk and adds g to the knowledge base KB if a matching rule exists. This process is repeated until no more additions can be made to KB.**

&#10230; Thuật toán suy luận chuyển tiếp - Từ một tập hợp các quy tắc suy luận Quy tắc, thuật toán này sẽ đi qua tất cả các F1, ..., fk và thêm g vào cơ sở kiến ​​thức KB nếu tồn tại quy tắc phù hợp. Quá trình này được lặp lại cho đến khi không thể bổ sung thêm vào KB.

<br>


**26. Derivation ― We say that KB derives f (written KB⊢f) with rules Rules if f already is in KB or gets added during the forward inference algorithm using the set of rules Rules.**

&#10230; Đạo hàm - Chúng ta nói rằng KB xuất phát f (viết KB⊢f) với các quy tắc Quy tắc nếu f đã có trong KB hoặc được thêm vào trong thuật toán suy luận chuyển tiếp bằng cách sử dụng bộ quy tắc Quy tắc.

<br>


**27. Properties of inference rules ― A set of inference rules Rules can have the following properties:**

&#10230; Thuộc tính của quy tắc suy luận - Một tập hợp các quy tắc suy luận Quy tắc có thể có các thuộc tính sau:

<br>


**28. [Name, Mathematical formulation, Notes]**

&#10230; [Tên, Công thức toán học, Ghi chú]

<br>


**29. [Soundness, Completeness]**

&#10230; [Âm thanh, Hoàn chỉnh]

<br>


**30. [Inferred formulas are entailed by KB, Can be checked one rule at a time, "Nothing but the truth", Formulas entailing KB are either already in the knowledge base or inferred from it, "The whole truth"]**

&#10230; [Các công thức được suy luận được KB yêu cầu, Có thể kiểm tra một quy tắc tại một thời điểm, "Không có gì ngoài sự thật", Các công thức đòi hỏi KB đã có trong cơ sở tri thức hoặc được suy ra từ đó, "Toàn bộ sự thật"]

<br>


**31. Propositional logic**

&#10230; Logic mệnh đề

<br>


**32. In this section, we will go through logic-based models that use logical formulas and inference rules. The idea here is to balance expressivity and computational efficiency.**

&#10230; Trong phần này, chúng ta sẽ đi qua các mô hình dựa trên logic sử dụng các công thức logic và quy tắc suy luận. Ý tưởng ở đây là để cân bằng giữa tính biểu thức và hiệu quả tính toán.

<br>


**33. Horn clause ― By noting p1,...,pk and q propositional symbols, a Horn clause has the form:**

&#10230; Mệnh đề sừng - Bằng cách lưu ý các ký hiệu mệnh đề p1,...,pk và q, mệnh đề Sừng có dạng:

<br>


**34. Remark: when q=false, it is called a "goal clause", otherwise we denote it as a "definite clause".**

&#10230; Ghi chú: khi q = false, nó được gọi là "mệnh đề mục tiêu", nếu không, chúng ta biểu thị nó là "mệnh đề xác định".

<br>


**35. Modus ponens ― For propositional symbols f1,...,fk and p, the modus ponens rule is written:**

&#10230; Modus ponens - Đối với các ký hiệu mệnh đề F1, ..., fk và p, quy tắc modus ponens được viết:

<br>


**36. Remark: it takes linear time to apply this rule, as each application generate a clause that contains a single propositional symbol.**

&#10230; Lưu ý: phải mất thời gian tuyến tính để áp dụng quy tắc này, vì mỗi ứng dụng tạo ra một mệnh đề có chứa một ký hiệu mệnh đề duy nhất.

<br>


**37. Completeness ― Modus ponens is complete with respect to Horn clauses if we suppose that KB contains only Horn clauses and p is an entailed propositional symbol. Applying modus ponens will then derive p.**

&#10230; Tính đầy đủ - Modus ponens hoàn thành đối với các mệnh đề Sừng nếu chúng ta cho rằng KB chỉ chứa các mệnh đề Sừng và p là một biểu tượng mệnh đề bắt buộc. Áp dụng modus ponens sau đó sẽ lấy được p.

<br>


**38. Conjunctive normal form ― A conjunctive normal form (CNF) formula is a conjunction of clauses, where each clause is a disjunction of atomic formulas.**

&#10230; Dạng bình thường kết hợp - Một công thức dạng thường kết hợp (CNF) là một sự kết hợp của các mệnh đề, trong đó mỗi mệnh đề là một sự tách rời của các công thức nguyên tử.

<br>


**39. Remark: in other words, CNFs are ∧ of ∨.**

&#10230; Ghi chú: nói cách khác, CNF là ∧ của ∨.

<br>


**40. Equivalent representation ― Every formula in propositional logic can be written into an equivalent CNF formula. The table below presents general conversion properties:**

&#10230; Biểu diễn tương đương - Mọi công thức trong logic mệnh đề có thể được viết thành một công thức CNF tương đương. Bảng dưới đây trình bày các thuộc tính chuyển đổi chung:

<br>


**41. [Rule name, Initial, Converted, Eliminate, Distribute, over]**

&#10230; [Tên quy tắc, Ban đầu, Chuyển đổi, Loại bỏ, Phân phối, kết thúc]

<br>


**42. Resolution rule ― For propositional symbols f1,...,fn, and g1,...,gm as well as p, the resolution rule is written:**

&#10230; Quy tắc phân giải - Đối với các ký hiệu mệnh đề F1, ..., fn và g1, ..., gm cũng như p, quy tắc phân giải được viết:

<br>


**43. Remark: it can take exponential time to apply this rule, as each application generates a clause that has a subset of the propositional symbols.**

&#10230; Lưu ý: có thể mất thời gian theo cấp số nhân để áp dụng quy tắc này, vì mỗi ứng dụng tạo ra một mệnh đề có tập hợp con của các ký hiệu mệnh đề.

<br>


**44. [Resolution-based inference ― The resolution-based inference algorithm follows the following steps:, Step 1: Convert all formulas into CNF, Step 2: Repeatedly apply resolution rule, Step 3: Return unsatisfiable if and only if False, is derived]**

&#10230; [Suy luận dựa trên độ phân giải - Thuật toán suy luận dựa trên độ phân giải tuân theo các bước sau:, Bước 1: Chuyển đổi tất cả các công thức thành CNF, Bước 2: Áp dụng lại quy tắc độ phân giải, Bước 3: Trả về không thỏa đáng khi và chỉ khi Sai, được dẫn xuất]

<br>


**45. First-order logic**

&#10230; Logic bậc nhất

<br>


**46. The idea here is to use variables to yield more compact knowledge representations.**

&#10230; Ý tưởng ở đây là sử dụng các biến để mang lại các biểu diễn tri thức nhỏ gọn hơn.

<br>


**47. [Model ― A model w in first-order logic maps:, constant symbols to objects, predicate symbols to tuple of objects]**

&#10230; [Mô hình - Một mô hình w trong các ánh xạ logic bậc nhất:, các ký hiệu không đổi cho các đối tượng, các ký hiệu vị ngữ cho đến các đối tượng]

<br>


**48. Horn clause ― By noting x1,...,xn variables and a1,...,ak,b atomic formulas, the first-order logic version of a horn clause has the form:**

&#10230; Mệnh đề sừng - Bằng cách lưu ý các biến x1,...,xn và a1,...,ak,b công thức nguyên tử, phiên bản logic thứ nhất của mệnh đề sừng có dạng:

<br>


**49. Substitution ― A substitution θ maps variables to terms and Subst[θ,f] denotes the result of substitution θ on f.**

&#10230; Thay thế - Một thay thế θ ánh xạ các biến thành các thuật ngữ và Subst [θ, f] biểu thị kết quả của sự thay thế θ trên f.

<br>


**50. Unification ― Unification takes two formulas f and g and returns the most general substitution θ that makes them equal:**

&#10230; Hợp nhất - Hợp nhất có hai công thức f và g và trả về sự thay thế chung nhất làm cho chúng bằng nhau:

<br>


**51. such that**

&#10230; sao cho

<br>


**52. Note: Unify[f,g] returns Fail if no such θ exists.**

&#10230; Lưu ý: Thống nhất [f, g] trả về Fail nếu không tồn tại θ.

<br>


**53. Modus ponens ― By noting x1,...,xn variables, a1,...,ak and a′1,...,a′k atomic formulas and by calling θ=Unify(a′1∧...∧a′k,a1∧...∧ak) the first-order logic version of modus ponens can be written:**

&#10230; Modus ponens - Bằng cách lưu ý các biến x1, ..., xn, a1, ..., ak và a′1, ..., a′k công thức nguyên tử và bằng cách gọi θ=Unify(a′1∧... ∧a′k,a1∧...ak) phiên bản logic bậc nhất của modus ponens có thể được viết:

<br>


**54. Completeness ― Modus ponens is complete for first-order logic with only Horn clauses.**

&#10230; Tính đầy đủ - Modus ponens hoàn thành cho logic thứ nhất chỉ với các mệnh đề Horn.

<br>


**55. Resolution rule ― By noting f1,...,fn, g1,...,gm, p, q formulas and by calling θ=Unify(p,q), the first-order logic version of the resolution rule can be written:**

&#10230; Quy tắc phân giải - Bằng cách lưu ý các công thức f1,...,fn,g1,...,gm,p,q và bằng cách gọi θ=Unify(p,q), có thể viết phiên bản logic bậc nhất của quy tắc phân giải :

<br>


**56. [Semi-decidability ― First-order logic, even restricted to only Horn clauses, is semi-decidable., if KB⊨f, forward inference on complete inference rules will prove f in finite time, if KB⊭f, no algorithm can show this in finite time]**

&#10230; [Độ phân giải bán - Logic bậc một, thậm chí chỉ giới hạn ở các mệnh đề Sừng, là bán có thể quyết định., Nếu KB⊨f, suy luận về các quy tắc suy luận hoàn chỉnh sẽ chứng minh f trong thời gian hữu hạn, nếu KB⊭f, không thuật toán nào có thể hiển thị Điều này trong thời gian hữu hạn]

<br>


**57. [Basics, Notations, Model, Interpretation function, Set of models]**

&#10230; [Khái niệm cơ bản, Ký hiệu, Mô hình, Hàm diễn giải, Bộ mô hình]

<br>


**58. [Knowledge base, Definition, Probabilistic interpretation, Satisfiability, Relationship with formulas, Forward inference, Rule properties]**

&#10230; [Cơ sở tri thức, Định nghĩa, Giải thích xác suất, Sự thỏa mãn, Mối quan hệ với các công thức, Suy luận chuyển tiếp, Thuộc tính quy tắc]

<br>


**59. [Propositional logic, Clauses, Modus ponens, Conjunctive normal form, Representation equivalence, Resolution]**

&#10230; [Logic đề xuất, Mệnh đề, Modus ponens, Hình thức bình thường kết hợp, Tương đương đại diện, Độ phân giải]

<br>


**60. [First-order logic, Substitution, Unification, Resolution rule, Modus ponens, Resolution, Semi-decidability]**

&#10230; [Logic thứ nhất, Thay thế, Thống nhất, Quy tắc giải quyết, Modus ponens, Độ phân giải, Bán quyết định]

<br>


**61. View PDF version on GitHub**

&#10230; Xem bản PDF trên GitHub

<br>


**62. Original authors**

&#10230; Các tác giả

<br>


**63. Translated by X, Y and Z**

&#10230; Dịch bởi X, Y và Z

<br>


**64. Reviewed by X, Y and Z**

&#10230; Đánh giá bới X, Y và Z

<br>


**65. By X and Y**

&#10230; Bởi X và Y

<br>


**66. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Trí tuệ nhân tạo cheatsheats hiện đã có với ngôn ngữ [Tiếng Việt]
