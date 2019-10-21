**1. Linear Algebra and Calculus refresher**

&#10230;
線形代数と微積分の復習
<br>

**2. General notations**

&#10230;
一般表記
<br>

**3. Definitions**

&#10230;
定義
<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230;
ベクトル - x∈Rn は n個の要素を持つベクトルを表し、xi∈Rはi番目の要素を表します。
<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230;
行列 - m行n列の行列をA∈Rm×nと表記し、Ai、j∈Rは i行目のj列目の要素を指します。
<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230;
備考：上記で定義されたベクトルxはn×1の行列と見なすことができ、列ベクトルと呼ばれます。
<br>

**7. Main matrices**

&#10230;
主な行列の種類
<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230;
単位行列 - 単位行列I∈Rn×nは、対角成分に 1 が並び、他は全て 0 となる正方行列です。
<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230;
備考：すべての行列A∈Rn×nに対して、A×I = I×A = Aとなります。
<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230;
対角行列 - 対角行列D∈Rn×nは、対角成分の値がゼロ以外で、それ以外はゼロである正方行列です。
<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230;
備考：Dをdiag（d 1、...、d n）とも表記します。
<br>

**12. Matrix operations**

&#10230;
行列演算
<br>

**13. Multiplication**

&#10230;
行列乗算
<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230;
ベクトル-ベクトル - ベクトル-ベクトル積には2種類あります。
<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230;
内積: x、y∈Rnに対して、内積の定義は下記の通りです:
<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230;
外積: x∈Rm,y∈Rnに対して、外積の定義は下記の通りです:
<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230;
行列-ベクトル - 行列A∈Rm×nとベクトルx∈Rnの積は以下の条件を満たすようなサイズRnのベクトルです。
<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230;
上記 aTr、iはAの行ベクトルで、ac、jはAの列ベクトルです。 xiはxの要素です。
<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230;
行列-行列 - 行列A∈Rm×nとB∈Rn×pの積は以下の条件を満たすようなサイズRm×pの行列です。 (There is a typo in the original: Rn×p)
<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230;
aTr,i、bTr,iはAとBの行ベクトルで　ac,j、bc,jはAとBの列ベクトルです。
<br>

**21. Other operations**

&#10230;
その他の演算
<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230;
転置 ― A∈Rm×nの転置行列はATと表記し、Aの行列要素が交換した行列です。
<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230;
備考： 行列AとBの場合、（AB）T = BTAT** となります。
<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230;
逆行列 ― 可逆正方行列Ａの逆行列はＡ − １と表記し、 以下の条件を満たす唯一の行列です。
<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230;
備考： すべての正方行列が可逆とは限りません。　行列A、Bについては、(AB)−1=B−1A−1
<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230;
跡 - 正方行列Aの跡は、tr(A)と表記し、その対角成分の要素の和です。
<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230;
備考： 行列A、Bの場合：　tr(AT)=tr(A)とtr(AB)=tr(BA)となります。
<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230;
行列式 ― 正方行列A∈Rn×nの行列式は|A| または det(A) と表記し、以下のように i番目の行とj番目の列を抜いたA, Aijによって再帰的に表現されます。
 それはi番目の行とj番目の列のない行列Aです。 次のように：
<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230;
備考： |A|≠0の場合に限り、行列は可逆行列です。また|AB|=|A||B| と |AT|=|A|。
<br>

**30. Matrix properties**

&#10230;
行列の性質
<br>

**31. Definitions**

&#10230;
定義
<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230;
対称分解 ― 行列Aは次のように対称および反対称的な部分で表現できます。
<br>

**33. [Symmetric, Antisymmetric]**

&#10230;
[対称、反対称]
<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230;
ノルムは関数N:V⟶[0,+∞[　Vはすべてのx、y∈Vに対して、以下の条件を満たすようなベクトル空間です。
]]
<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230;
スカラー a に対して N(ax)=|a|N(x) 
<br>

**36. if N(x)=0, then x=0**

&#10230;
N（x）= 0ならば x = 0
<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230;
x∈Vに対して、最も多用されているノルムは、以下の表にまとめられています。
<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230;
[ノルム、表記法、定義、使用事例]
<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230;
線形従属 ― ベクトルの集合に対して、少なくともどれか一つのベクトルを他のベクトルの線形結合として定義できる場合、その集合が線形従属であるといいます。
<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230;
備考：この方法でベクトルを書くことができない場合、ベクトルは線形独立していると言われます。
<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230;
行列の階数　―　行列Aの階数は rank（A）と表記し、列空間の次元を表します。これは、Aの線形独立の列の最大数に相当します。
<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230;
半正定値行列 ― 行列 A, A∈Rn×nに対して、以下の式が成り立つならば、 Aを半正定値(PSD)といい、A⪰0と表記します。
<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230;
備考：　同様に、全ての非ゼロベクトルx, xTAx>0に対して条件を満たすような行列Aは正定値行列といい、A≻0と表記します。
<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230;
固有値、固有ベクトル　―　行列 A, A∈Rn×nに対して、以下の条件を満たすようなベクトルz, z∈Rn∖{0}が存在するならば、λは固有値といい、z は固有ベクトルといいます。
<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230;
スペクトル定理 ― A∈Rn×nとします。　Aが対称ならば、Aは実直交行列U∈Rn×nによって対角化可能です。Λ=diag(λ1,...,λn)と表記すると、次のように表現できます。
<br>

**46. diagonal**

&#10230;
対角
<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230;
特異値分解 ― Aをm×nの行列とします。特異値分解（SVD）は、ユニタリ行列Ｕ ｍ×ｍ、Σ ｍ×ｎの対角行列、およびユニタリ行列Ｖ ｎ×ｎの存在を保証する因数分解手法で、以下の条件を満たします。
<br>

**48. Matrix calculus**

&#10230;
行列微積分
<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230;
勾配 ― f:Rm×n→Rを関数とし、A∈Rm×nを行列とします。 Aに対するfの勾配はm×n行列で、∇Af（A）と表記し、次の条件を満たします。
<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230;
備考：　fの勾配は、fがスカラーを返す関数であるときに限り存在します。
<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230;
ヘッセ行列 ― f：Rn→Rを関数とし、x∈Rnをベクトルとします。 xに対するfのヘッセ行列は、n×n対称行列で∇2xf（x）と表記し、以下の条件を満たします。
<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230;
備考：　fのヘッセ行列は、fがスカラーを返す関数である場合に限り存在します。
<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230;
勾配演算 ― 行列A、B、Cの場合、特に以下の勾配の性質を意識する甲斐があります。
<br>

**54. [General notations, Definitions, Main matrices]**

&#10230;
[表記, 定義, 主な行列の種類]
<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230;
[行列演算, 乗算, その他の演算]
<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230;
[行列特性, 行列ノルム, 固有値/固有ベクトル, 特異値分解]
<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230;
[行列微積分, 勾配, ヘッセ行列, 演算]
