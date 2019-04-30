**1. Linear Algebra and Calculus refresher**

<div dir="rtl">
ملخص الجبر الخطي و التفاضل و التكامل
</div>
<br>

**2. General notations**
<div dir="rtl">
الرموز العامة 
</div> 

<br>

**3. Definitions**

<div dir="rtl">
التعريفات  
</div>

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**
<div dir="rtl">
  متجه (vector) - نرمز ل $x \in \mathbb{R^n}$ متجه يحتوي على $n$ مدخلات، حيث $x_i \in \mathbb{R}$  يعتبر المدخل رقم $i$ . 
</div>
<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

<div dir="rtl">
 مصفوفة (Matrix) - نرمز ل ${A \in \mathbb{R}^{m\times n$ مصفوفة تحتوي على $m$ صفوف و $n$ أعمدة، حيث $A_{i,j}$  يرمز للمدخل في الصف$ i$ و العمود $j$  
</div>

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**
<div dir="rtl">
ملاحظة : المتجه $x$ المعرف مسبقا يمكن اعتباره مصفوفة من الشكل $n \times 1$ والذي يسمى ب مصفوفة من عمود واحد.
</div>

<br>

**7. Main matrices**

<div dir="rtl">
المصفوفات الأساسية
</div>
<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**
<div dir="rtl">
  مصفوفة الوحدة (Identity) - مصفوفة الوحدة $I \in \mathbb{R^{n\times n}$ تعتبر مصفوفة مربعة تحتوي على المدخل 1 في قطر المصفوفة و 0 في بقية المدخلات:

</div>
<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

<div dir="rtl">
ملاحظة : جميع المصفوفات من الشكل $A \in \mathbb{R^}{n\times n}$  فإن $A \times I = I \times A = A$.
</div>
<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**
<div dir="rtl">
مصفوفة قطرية (diagonal) - المصفوفة القطرية هي مصفوفة من الشكل
 $D \in \mathbb{R}^{n\times n}$  حيث أن جميع العناصر الواقعة خارج القطر الرئيسي تساوي الصفر والعناصر على القطر الرئيسي تحتوي أعداد لاتساوي الصفر.   
</div>
<br>

**11. Remark: we also note D as diag(d1,...,dn).**

<div dir="rtl">
ملاحظة: نرمز كذلك ل $D$ ب $text{diag}(d_1, \dots, d_n)\$.
</div>
<br>

**12. Matrix operations**

<div dir="rtl">
 عمليات المصفوفات
</div>

<br>

**13. Multiplication**

<div dir="rtl">
  الضرب
</div>

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

<div dir="rtl">
  ضرب المتجهات - توجد طريقتين لضرب متجه بمتجه : 
</div>

<br>

**15. inner product: for x,y∈Rn, we have:**

<div dir="rtl">
  ضرب داخلي (inner product): ل $x,y \in \mathbb{R}^n$ نستنتج :
</div>

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

<div dir="rtl">
  ضرب خارجي (outer product):  ل $x \in \mathbb{m}, y \in \mathbb{R}^n$ نستنتج : 
</div>

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

<div dir="rtl">
  مصفوفة - متجه : ضرب المصفوفة $A \in \mathbb{R}^{n\times m}$ والمتجه $x \in \mathbb{R}^n$ ينتجه متجه من الشكل $x \in \mathbb{R}^n$ حيث : 
</div>
<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

<div dir="rtl">
  حيث $a^{T}_{r,i}$ يعتبر متجه الصفوف و $a_{c,j}$ يعتبر متجه الأعمدة ل $A$ كذلك $x_i$ يرمز لعناصر $x$.
</div>

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

<div dir="rtl">
  ضرب مصفوفة ومصفوفة - ضرب المصفوفة $A \in \mathbb{R}^{n \times m}$ و $A \in \mathbb{R}^{n \times p}$ ينتجه عنه المصفوفة $A \in \mathbb{R}^{n \times p}$ حيث أن : 
</div>

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

<div dir="rtl">
حيث $a^T_{r, i}$ و $b^T_{r, i}$ يعتبر متجه الصفوف $a_{c, j}$ و b_{c, j}$ متجه الأعمدة ل $A$ و $B$ على التوالي.
</div>

<br>

**21. Other operations**

<div dir="rtl">
  عمليات أخرى
</div>

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

<div dir="rtl">
  المنقول (Transpose) - منقول المصفوفة$A \in \mathbb{R}^{m \times n}$ يرمز له ب $A^T$ حيث الصفوف يتم تبديلها مع الأعمدة : 
</div>

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

<div dir="rtl">
   ملاحظة: لأي مصفوفتين $A$ و $B$، نستنتج $(AB)^T = B^T A^T$. 
</div>
<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

<div dir="rtl">
   المعكوس (Inverse)- معكوس أي مصفوفة $A$ قابلة للعكس (Invertible) يرمز له ب $A^{-1}$ ويعتبر المعكوس المصفوفة الوحيدة التي لديها الخاصية التالية :
</div>
<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

<div dir="rtl">
ملاحظة: ليس جميع المصفوفات يمكن إيجاد معكوس لها. كذلك لأي مصفوفتين $A$ و $B$ نستنتج $(AB)^{-1} = B^{-1} A^{-1}$.
</div>

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

<div dir="rtl">
أثر المصفوفة (Trace) - أثر أي مصفوفة مربعة $A$ يرمز له ب $tr(A)$ يعتبر مجموع العناصر التي في القطر:
</div>
<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

<div dir="rtl">
 ملاحظة : لأي مصفوفتين $A$ و $B$ لدينا $tr(A^T) = tr(A)$ و $tr(AB) = tr(BA)$. 
</div>
<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

<div dir="rtl">
المحدد (Determinant) - المحدد لأي مصفوفة مربعة من الشكل $A \in \mathbb{R}^{n \times n}$ يرمز له ب $|A|$ او $det(A)$يتم تعريفه بإستخدام $ِA_{\\i,\\j}$ والذي يعتبر المصفوفة $A$ مع حذف الصف $i$ والعمود $j$ كالتالي : 
</div>
<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

<div dir="rtl">
 ملاحظة: $A$ يكون لديه معكوذ إذا وفقط إذا $\neq 0 |A|$. كذلك $|A B| = |A| |B|$ و $|A^T| = |A|$. 
</div>
<br>

**30. Matrix properties**

<div dir="rtl">
خواص المصفوفات
</div>
<br>

**31. Definitions**

<div dir="rtl">
التعريفات
</div>
<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

<div dir="rtl">
  التفكيك المتماثل (Symmetric Decomposition)- المصفوفة $A$ يمكن التعبير عنها بإستخدام جزئين مثماثل (Symmetric) وغير متماثل(Antisymmetric) كالتالي : 
</div>
<br>

**33. [Symmetric, Antisymmetric]**

<div dir="rtl">
[متماثل، غير متماثل]
</div>

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

<div dir="rtl">
المعيار (Norm) - المعيار يعتبر دالة $N: V \to [0, +\infity)$ حيث $V$ يعتبر فضاء متجه (Vector Space)، حيث أن لكل $x,y \in V$ لدينا :
</div>
<br>

**35. N(ax)=|a|N(x) for a scalar**

<div dir="rtl">
لأي عدد $a$ فإن $N(ax) = |a| N(x)$
</div>
<br>

**36. if N(x)=0, then x=0**

<div dir="rtl">
$N(x) =0 \implies x = 0$
</div>
<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

<div dir="rtl">
لأي $x \in V$ المعايير الأكثر إستخداماً ملخصة في الجدول التالي:
</div>
<br>

**38. [Norm, Notation, Definition, Use case]**

<div dir="rtl">
[المعيار، الرمز، التعريف، مثال للإستخدام]
</div>
<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

<div dir="rtl">
 الارتباط الخطي (Linear Dependence): مجموعة المتجهات تعتبر تابعة خطياً إذا وفقط إذا كل متجه يمكن كتابته بشكل خطي بإسخدام مجموعة من المتجهات الأخرى. 
</div>
<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

<div dir="rtl">
ملاحظة: إذا لم يتحقق هذا الشرط فإنها تسمى مستقلة خطياً . 
</div>
<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

<div dir="rtl">
 رتبة المصفوفة (Rank) - رتبة المصفوفة $A$ يرمز له ب $text{rank}(A)\$ وهو يصف حجم الفضاء المتجهي الذي نتج من أعمدة المصفوفة. يمكن وصفه كذلك بأقصى عدد من أعمدة المصفوفة $A$ التي تمتلك خاصية أنها مستقلة خطياً. 
</div>
<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

<div dir="rtl">
  مصفوفة شبه معرفة موجبة (Positive semi-definite) - المصفوفة  $A \in \mathbb{R}^{n \times n}$ تعتبر مصفوفة شبه معرفة موجبة (PSD) ويرمز لها بالرمز  $A \succed 0  $ إذا : 
</div>
<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

<div dir="rtl">
  ملاحظة: المصفوفة $A$ تعتبر مصفوفة معرفة موجبة إذا $A \succ 0  $  وهي تعتبر مصفوفة (PSD) والتي تستوفي الشرط : لكل متجه غير الصفر $x$ حيث $x^TAx>0 $.
</div>
<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

<div dir="rtl">
  القيم الذايتة (eigenvalue), المتجه الذاتي (eigenvector) - إذا كان لدينا مصفوفة $A \in \mathbb{R}^{n \times n}$، القيمة $\lambda$  تعتبر قيمة ذاتية للمصفوفة $A$ إذا وجد متجه $z \in \mathbb{R}^n \\ \{0\}$ يسمى متجه ذاتي حيث أن : 
</div>
<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

<div dir="rtl">
  النظرية الطيفية (spectral theorem) - نفرض $A \in \mathbb{R}^{n \times n}$ إذا كانت المصفوفة $A$ متماثلة فإن $A$ تعتبر مصفوفة قطرية بإستخدام مصفوفة  متعامدة (orthogonal) $U \in \mathbb{R} ^{n \times n}$ ويرمز لها بالرمز  $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$ حيث أن:
</div>
<br>

**46. diagonal**

<div dir="rtl">
  قطرية 
</div>
<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

<div dir="rtl">
  مجزئ القيمة المفرده (singular value decomposition) : لأي مصفوفة $A$ من الشكل $n\times m$ ، تفكيك القيمة المنفردة (SVD) يعتبر طريقة تحليل تضمن وجود $U \in \mathbb{R}^{m \times m}$ , مصفوفة قطرية  $\Sigma \in \mathbb{R}^{m \times n}$ و $V \in \mathbb{R}^{n \times n}$ حيث أن : 
</div>
<br>

**48. Matrix calculus**

<div dir="rtl">
  حساب المصفوفات 
</div>
<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

<div dir="rtl">
   المشتقة في فضاءات عالية (gradient) - افترض $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ تعتبر دالة و $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ تعتبر مصفوفة. المشتقة العليا ل $f$ بالنسبة ل $A$  يعتبر مصفوفة $n\times m$ يرمز له $nabla_A f(A)\$ حيث أن:
</div>
<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

<div dir="rtl">
ملاحظة : المشتقة العليا معرفة فقط إذا كانت الدالة $f$ لديها مدى ضمن الأعداد الحقيقية.
</div>
<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

<div dir="rtl">
هيشيان (Hessian) - افترض $f: \mathbb{R}^n \rightarrow \mathbb{R}$ تعتبر دالة و $x \in \mathbb{R}^n$ يعتبر متجه. الهيشيان ل $f$ بالنسبة ل $x$ تعتبر مصفوفة متماثلة من الشكل $n \times n$ يرمز لها بالرمز $nabla^2_x f(x)\$ حيثب أن : 
</div>
<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

<div dir="rtl">
  ملاحظة : الهيشيان معرفة فقط إذا كانت الدالة $f$ لديها مدى ضمن الأعداد الحقيقية.

</div>
<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

<div dir="rtl">
  الحساب في مشتقة الفضاءات العالية- لأي مصفوفات $A,B,C$ فإن الخواص التالية مهمة : 

</div>
<br>

**54. [General notations, Definitions, Main matrices]**

<div dir="rtl">
    [الرموز العامة، التعاريف، المصفوفات الرئيسية]
</div>

<br>

**55. [Matrix operations, Multiplication, Other operations]**

<div dir="rtl">
  [عمليات المصفوفات، الضرب، عمليات أخرى]
</div>
<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

<div dir="rtl">
  [خواص المصفوفات، المعيار، قيمة ذاتية/متجه ذاتي، تفكيك القيمة المنفردة]
</div>
<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

<div dir="rtl">
  [حساب المصفوفات، مشتقة الفضاءات العالية، الهيشيان، العمليات]
</div>
