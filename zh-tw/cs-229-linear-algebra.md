1. **Linear Algebra and Calculus refresher**

&#10230;
線性代數與微積分回顧
<br>

2. **General notations**

&#10230;
通用符號
<br>

3. **Definitions**

&#10230;
定義
<br>

4. **Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230;
向量 - 我們定義 x∈Rn 是一個向量，包含 n 維元素，xi∈R 是第 i 維元素：
<br>

5. **Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230;
矩陣 - 我們定義 A∈Rm×n 是一個 m 列 n 行的矩陣，Ai,j∈R 代表位在第 i 列第 j 行的元素：
<br>

6. **Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230;
注意：上述定義的向量 x 可以視為 nx1 的矩陣，或是更常被稱為行向量
<br>

7. **Main matrices**

&#10230;
主要的矩陣
<br>

8. **Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230;
單位矩陣 - 單位矩陣 I∈Rn×n 是一個方陣，其主對角線皆為 1，其餘皆為 0
<br>

9. **Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230;
注意：對於所有矩陣 A∈Rn×n，我們有 A×I=I×A=A
<br>

10. **Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230;
對角矩陣 - 對角矩陣 D∈Rn×n 是一個方陣，其主對角線為非 0，其餘皆為 0
<br>

11. **Remark: we also note D as diag(d1,...,dn).**

&#10230;
注意：我們令 D 為 diag(d1,...,dn)
<br>

12. **Matrix operations**

&#10230;
矩陣運算
<br>

13. **Multiplication**

&#10230;
乘法
<br>

14. **Vector-vector ― There are two types of vector-vector products:**

&#10230;
向量-向量 - 有兩種類型的向量-向量相乘：
<br>

15. **inner product: for x,y∈Rn, we have:**

&#10230;
內積：對於 x,y∈Rn，我們可以得到：
<br>

16. **outer product: for x∈Rm,y∈Rn, we have:**

&#10230;
外積：對於 x∈Rm,y∈Rn，我們可以得到：
<br>

17. **Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230;
矩陣-向量 - 矩陣 A∈Rm×n 和向量 x∈Rn 的乘積是一個大小為 Rm 的向量，使得：
<br>

18. **where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230;
其中 aTr,i 是 A 的列向量、ac,j 是 A 的行向量、xi 是 x 的元素
<br>

19. **Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230;
矩陣-矩陣：矩陣 A∈Rm×n 和 B∈Rn×p 的乘積為一個大小 Rm×p 的矩陣，使得：
<br>

20. **where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230;
其中，aTr,i,bTr,i 和 ac,j,bc,j 分別是 A 和 B 的列向量與行向量
<br>

21. **Other operations**

&#10230;
其他操作
<br>

22. **Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230;
轉置 - 一個矩陣的轉置矩陣 A∈Rm×n，記作 AT，指的是其中元素的翻轉：
<br>

23. **Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230;
注意：對於矩陣 A、B，我們有 (AB)T=BTAT
<br>

24. **Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230;
可逆 - 一個可逆矩陣 A 記作 A−1，存在唯一的矩陣，使得：
<br>

25. **Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230;
注意：並非所有的方陣都是可逆的。同樣的，對於矩陣 A、B 來說，我們有 (AB)−1=B−1A−1
<br>

26. **Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230;
跡 - 一個方陣 A 的跡，記作 tr(A)，指的是主對角線元素之合：
<br>

27. **Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230;
注意：對於矩陣 A、B 來說，我們有 tr(AT)=tr(A) 及 tr(AB)=tr(BA)
<br>

28. **Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230;
行列式 - 一個方陣 A∈Rn×n 的行列式，記作|A| 或 det(A)，可以透過 A∖i,∖j 來遞迴表示，它是一個沒有第 i 列和第 j 行的矩陣 A：
<br>

29. **Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230;
注意：A 是一個可逆矩陣，若且唯若 |A|≠0。同樣的，|AB|=|A||B| 且 |AT|=|A|
<br>

30. **Matrix properties**

&#10230;
矩陣的性質
<br>

31. **Definitions**

&#10230;
定義
<br>

32. **Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230;
對稱分解 - 給定一個矩陣 A，它可以透過其對稱和反對稱的部分表示如下：
<br>

33. **[Symmetric, Antisymmetric]**

&#10230;
[對稱, 反對稱]
<br>

34. **Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230;
範數 - 範數指的是一個函式 N:V⟶[0,+∞[，其中 V 是一個向量空間，且對於所有 x,y∈V，我們有：
<br>

35. **N(ax)=|a|N(x) for a scalar**

&#10230;
對一個純量來說，我們有 N(ax)=|a|N(x)
<br>

36. **if N(x)=0, then x=0**

&#10230;
若 N(x)=0 時，則 x=0
<br>

37. **For x∈V, the most commonly used norms are summed up in the table below:**

&#10230;
對於 x∈V，最常用的範數總結如下表：
<br>

38. **[Norm, Notation, Definition, Use case]**

&#10230;
[範數, 表示法, 定義, 使用情境]
<br>

39. **Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230;
線性相關 - 當集合中的一個向量可以用被定義為集合中其他向量的線性組合時，則則稱此集合的向量為線性相關
<br>

40. **Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230;
注意：如果沒有向量可以如上表示時，則稱此集合的向量彼此為線性獨立
<br>

41. **Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230;
矩陣的秩 - 一個矩陣 A 的秩記作 rank(A)，指的是其列向量空間所產生的維度，等價於 A 的線性獨立的最大最大行向量
<br>

42. **Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230;
半正定矩陣 - 當以下成立時，一個矩陣 A∈Rn×n 是半正定矩陣 (PSD)，且記作A⪰0：
<br>

43. **Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230;
注意：同樣的，一個矩陣 A 是一個半正定矩陣 (PSD)，且滿足所有非零向量 x，xTAx>0 時，稱之為正定矩陣，記作 A≻0
<br>

44. **Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230;
特徵值、特徵向量 - 給定一個矩陣 A∈Rn×n，當存在一個向量 z∈Rn∖{0} 時，此向量被稱為特徵向量，λ 稱之為 A 的特徵值，且滿足：
<br>

45. **Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230;
譜分解 - 令 A∈Rn×n，如果 A 是對稱的，則 A 可以被一個實數正交矩陣 U∈Rn×n 給對角化。令 Λ=diag(λ1,...,λn)，我們得到：
<br>

46. **diagonal**

&#10230;
對角線
<br>

47. **Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230;
奇異值分解 - 對於給定維度為 mxn 的矩陣 A，其奇異值分解指的是一種因子分解技巧，保證存在 mxm 的單式矩陣 U、對角線矩陣 Σ m×n 和 nxn 的單式矩陣 V，滿足：
<br>

48. **Matrix calculus**

&#10230;
矩陣導數
<br>

49. **Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230;
梯度 - 令 f:Rm×n→R 是一個函式，且 A∈Rm×n 是一個矩陣。f 相對於 A 的梯度是一個 mxn 的矩陣，記作 ∇Af(A)，滿足：
<br>

50. **Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230;
注意：f 的梯度僅在 f 為一個函數且該函數回傳一個純量時有效
<br>

51. **Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230;
海森 - 令 f:Rn→R 是一個函式，且 x∈Rn 是一個向量，則一個 f 的海森對於向量 x 是一個 nxn 的對稱矩陣，記作 ∇2xf(x)，滿足：
<br>

52. **Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230;
注意：f 的海森僅在 f 為一個函數且該函數回傳一個純量時有效
<br>

53. **Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**
梯度運算 - 對於矩陣 A、B、C，下列的梯度性質值得牢牢記住：
&#10230;

54. **[General notations, Definitions, Main matrices]**

&#10230;
[通用符號, 定義, 主要矩陣]
<br>

55. **[Matrix operations, Multiplication, Other operations]**

&#10230;
[矩陣運算, 矩陣乘法, 其他運算]
<br>

56. **[Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230;
[矩陣性質, 範數, 特徵值/特徵向量, 奇異值分解]
<br>

57. **[Matrix calculus, Gradient, Hessian, Operations]**

&#10230;
[矩陣導數, 梯度, 海森, 運算]