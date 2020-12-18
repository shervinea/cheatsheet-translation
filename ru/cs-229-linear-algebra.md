**Linear Algebra and Calculus translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)

<br>

**1. Linear Algebra and Calculus refresher**

&#10230; Linear Algebra and Calculus refresher

<br>

**2. General notations**

&#10230; General notations

<br>

**3. Definitions**

&#10230; Definitions

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Примечание: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.

<br>

**7. Main matrices**

&#10230; Main matrices

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Примечание: for all matrices A∈Rn×n, we have A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Примечание: we also note D as diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Matrix operations

<br>

**13. Multiplication**

&#10230; Multiplication

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vector-vector ― There are two types of vector-vector products:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; inner product: for x,y∈Rn, we have:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; outer product: for x∈Rm,y∈Rn, we have:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively

<br>

**21. Other operations**

&#10230; Other operations

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Примечание: for matrices A,B, we have (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Примечание: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Примечание: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Примечание: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Matrix properties

<br>

**31. Definitions**

&#10230; Definitions

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Symmetric, Antisymmetric]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) for a scalar

<br>

**36. if N(x)=0, then x=0**

&#10230; if N(x)=0, then x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; For x∈V, the most commonly used norms are summed up in the table below:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norm, Notation, Definition, Use case]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Примечание: if no vector can be written this way, then the vectors are said to be linearly independent

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Примечание: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:

<br>

**46. diagonal**

&#10230; diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:

<br>

**48. Matrix calculus**

&#10230; Matrix calculus

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Примечание: the gradient of f is only defined when f is a function that returns a scalar.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Примечание: the hessian of f is only defined when f is a function that returns a scalar

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [General notations, Definitions, Main matrices]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Matrix operations, Multiplication, Other operations]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Matrix calculus, Gradient, Hessian, Operations]
