1. **Linear Algebra and Calculus refresher**

&#10230; 线性代数和微积分回顾

<br>

2. **General notations**

&#10230; 通用符号

<br>

3. **Definitions**

&#10230; 定义

<br>

4. **Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; 向量 - 我们记 为一个 n 维的向量，其中 xi∈R 是第 i 维的元素：

<br>

5. **Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; 矩阵 - 我们记 A∈Rm×n 为一个 m 行 n 列的矩阵，其中 Ai,j∈R 是第 i 行 j 列的元素：

<br>

6. **Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; 注意：如上定义的向量 x 可以被看做是一个 nx1 的矩阵，常被称为一个列向量。

<br>

7. **Main matrices**

&#10230; 主要的矩阵

<br>

8. **Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; 单位矩阵 - 单位矩阵 I∈Rn×n 是一个方阵其对角线上均是 1 其余位置均为 0。

<br>

9. **Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; 注：对所有矩阵 A∈Rn×n，我们有 A×I=I×A=A。

<br>

10. **Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; 对角阵 - 对角阵 D∈Rn×n 是一个方阵其对角线上元素均是非零值，其余位置均为 0。

<br>

11. **Remark: we also note D as diag(d1,...,dn).**

&#10230; 注：我们记 D 为 diag(d1,...,dn)。

<br>

12. **Matrix operations**

&#10230; 矩阵运算

<br>

13. **Multiplication**

&#10230; 乘法

<br>

14. **Vector-vector ― There are two types of vector-vector products:**

&#10230; 向量-向量 - 存在两种类型的向量-向量乘积：

<br>

15. **inner product: for x,y∈Rn, we have:**

&#10230; 内积：对 x,y∈Rn，我们有：

<br>

16. **outer product: for x∈Rm,y∈Rn, we have:**

&#10230; 外积： 对 x∈Rm,y∈Rn，我们有：

<br>

17. **Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; 矩阵-向量 - 矩阵 A∈Rm×n 和向量 x∈Rn 的乘积是一个大小为 Rn 的向量，满足：

<br>

18. **where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; 其中 aTr,i 是行向量，ac,j 是 A 的列向量，xi 是 x 的元素。

<br>

19. **Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; 矩阵-矩阵 - 矩阵 A∈Rm×n 和 B∈Rn×p 的乘积是一个大小为 Rn×p 的矩阵，满足：

<br>

20. **where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; 其中 aTr,i,bTr,i 是行向量，ac,j,bc,j 分别是 A 和 B 的列向量

<br>

21. **Other operations**

&#10230; 其他操作

<br>

22. **Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; 转置 - 矩阵 A∈Rm×n 的转置，记作 AT， 是其中元素的翻转

<br>

23. **Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; 注：对矩阵 A,B，我们有 (AB)T=BTAT

<br>

24. **Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; 逆 - 可逆方阵 A 的逆记作 A-1 和唯一满足下列要求的矩阵：

<br>

25. **Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; 注：不是所有方阵都是可逆的。同样，对矩阵 A,B，我们有 (AB)−1=B−1A−1

<br>

26. **Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; 迹 - 方阵 A 的迹，记作 tr(A)，是对角线元素的和：

<br>

27. **Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; 注：对矩阵 A,B，我们有 tr(AT)=tr(A) 和 tr(AB)=tr(BA)

<br>

28. **Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; 行列式 - 方阵 的行列式，记作 |A| 或者 det(A) 采用去掉第 i 行 j 列的矩阵 A\i,\j 递归表达为如下形式：

<br>

29. **Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; 注：A 可逆当且仅当 |A|≠0。同样，有 |AB|=|A||B| 和 |AT|=|A|。

<br>

30. **Matrix properties**

&#10230; 矩阵的性质

<br>

31. **Definitions**

&#10230; 定义

<br>

32. **Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; 对称分解 - 一个给定矩阵 A 可以用其对阵和反对称部分进行表示：

<br>

33. **[Symmetric, Antisymmetric]**

&#10230; [对称，反对阵]

<br>

34. **Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; 范数 - 一个范数是一个函数 N:V⟶[0,+∞[ 其中 V 是一个向量空间，满足对所有 x,y∈V，有：

<br>

35. **N(ax)=|a|N(x) for a scalar**

&#10230; 对一个标量 a，有 N(ax)=|a|N(x)

<br>

36. **if N(x)=0, then x=0**

&#10230; 若 N(x)=0，则 x=0

<br>

37. **For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; 对 x∈V，最常用的范数列在下表中：

<br>

38. **[Norm, Notation, Definition, Use case]**

&#10230; [范数，符号，定义，用例]

<br>

39. **Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; 线性相关 - 向量集合被称作线性相关的当其中一个向量可以被定义为其他向量的线性组合。

<br>

40. **Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; 注：若无向量可以按照此法表示，则这些向量被称为线性无关。

<br>

41. **Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; 矩阵的秩 - 给定矩阵 A 的秩记作 rank(A) 是由列向量生成的向量空间的维度。这等价于 A 的线性无关列向量的最大数目。

<br>

42. **Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; 半正定矩阵 - 矩阵 A∈Rn×n 是半正定矩阵（PSD），记作 A⪰0，当我们有：

<br>

43. **Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; 注：类似地，矩阵 A 被称作正定，记作 A≻0，当它是一个 PSD 矩阵且满足所有非零向量 x，xTAx>0。

<br>

44. **Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; 特征值，特征向量 - 给定矩阵 A∈Rn×n，λ 被称作 A 的一个特征值当存在一个向量 z∈Rn 称作特征向量，满足：

<br>

45. **Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; 谱定理 - 令 A∈Rn×n，若 A 是对称的，则 A 可以被一个实正交矩阵 U∈Rn×n 对角化。记 Λ=diag(λ1,...,λn)，我们有：

<br>

46. **diagonal**

&#10230; 对角阵

<br>

47. **Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; 奇异值分解 - 对一个给定矩阵 A，其维度为 mxn，奇异值分解（SVD）是一个因子分解机巧，能保证存在酉矩阵 U mxm，对角阵 Σ m×n 和酉矩阵 V n×n，满足：

<br>

48. **Matrix calculus**

&#10230; 矩阵的微积分

<br>

49. **Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; 梯度 - 令 f:Rm×n→R 一个函数 A∈Rm×n 一个矩阵。f 关于 A 的梯度是一个 mxn 的矩阵，记作 Af(A)，满足：

<br>

50. **Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; 注：f 的梯度仅当 f 是返回一个标量的函数时有定义。

<br>

51. **Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessian - 令 f:Rn→R 一个函数，x∈Rn 一个向量。f 的关于 x 的 Hessian 是一个 nxn 的对称阵，记作 2xf(x)，满足：

<br>

52. **Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; 注：f 的 Hessian 仅当 f 是一个返回标量的函数时有定义。

<br>

53. **Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; 梯度运算 - 对矩阵 A,B,C，下列梯度性质值得记住：

<br>

54. **[General notations, Definitions, Main matrices]**

&#10230; [通用符号，定义，主要的矩阵]

<br>

55. **[Matrix operations, Multiplication, Other operations]**

&#10230; [矩阵运算，乘法，其他运算]

<br>

56. **[Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [矩阵性质， 范数，特征值/特征向量，奇异值分解]

<br>

57. **[Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [矩阵微积分，梯度，Hessian，运算]
