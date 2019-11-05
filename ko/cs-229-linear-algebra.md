**1. Linear Algebra and Calculus refresher**

&#10230; 선형대수와 미적분학 복습

<br>

**2. General notations**

&#10230; 일반적인 표기법

<br>

**3. Definitions**

&#10230; 정의

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; 벡터 - x∈Rn는 n개의 요소를 가진 벡터이고, xi∈R는 i번째 요소이다.

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; 행렬 - A∈Rm×n는 m개의 행과 n개의 열을 가진 행렬이고, Ai,j∈R는 i번째 행, j번째 열에 있는 원소이다.

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; 비고 : 위에서 정의된 벡터 x는 n×1행렬로 볼 수 있으며, 열벡터라고도 불린다.

<br>

**7. Main matrices**

&#10230; 주요 행렬

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; 단위행렬 - 단위행렬 I∈Rn×n는 대각성분이 모두 1이고 대각성분이 아닌 성분은 모두 0인 정사각행렬이다.

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; 비고 : 모든 행렬 A∈Rn×n에 대하여, A×I=I×A=A를 만족한다.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; 대각행렬 - 대각행렬 D∈Rn×n는 대각성분은 모두 0이 아니고, 대각성분이 아닌 성분은 모두 0인 정사각행렬이다.

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; 비고 : D를 diag(d1,...,dn)라고도 표시한다.

<br>

**12. Matrix operations**

&#10230; 행렬 연산

<br>

**13. Multiplication**

&#10230; 곱셈

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; 벡터-벡터 – 벡터 간 연산에는 두 가지 종류가 있다.

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; 내적 : x,y∈Rn에 대하여, 

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; 외적 : x∈Rm,y∈Rn에 대하여, 

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; 행렬-벡터 - 행렬 A∈Rm×n와 벡터 x∈Rn의 곱은 다음을 만족하는 Rn크기의 벡터이다.

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; aTr,i는 A의 벡터행, ac,j는 A의 벡터열, xi는 x의 성분이다.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; 행렬-행렬 - 행렬 A∈Rm×n와 행렬 B∈Rn×p의 곱은 다음을 만족하는 Rn×p크기의 행렬이다.

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; aTr,i,bTr,i는 A,B의 벡터행, ac,j,bc,j는 A,B의 벡터열이다.

<br>

**21. Other operations**

&#10230; 그 외 연산

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; 전치 - 행렬 A∈Rm×n의 전치 AT는 모든 성분을 뒤집은 것이다.

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; 비고 - 행렬 A,B에 대하여, (AB)T=BTAT가 성립힌다.

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; 역행렬 - 가역행렬 A의 역행렬은 A-1로 표기하며, 유일하다.

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; 모든 정사각행렬이 역행렬을 갖는 것은 아니다. 그리고, 행렬 A,B에 대하여 (AB)−1=B−1A−1가 성립힌다.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; 대각합 – 정사각행렬 A의 대각합 tr(A)는 대각성분의 합이다.

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; 비고 : 행렬 A,B에 대하여, tr(AT)=tr(A)와 tr(AB)=tr(BA)가 성립힌다.

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; 행렬식 - 정사각행렬 A∈Rn×n의 행렬식 |A| 또는 det(A)는 i번째 행과 j번째 열이 없는 행렬 A인 A∖i,∖j에 대해 재귀적으로 표현된다.

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; 비고 : A가 가역일 필요충분조건은 |A|≠0이다. 또한 |AB|=|A||B|와 |AT|=|A|도 그렇다.

<br>

**30. Matrix properties**

&#10230; 행렬의 성질

<br>

**31. Definitions**

&#10230; 정의

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; 대칭 분해 - 주어진 행렬 A는 다음과 같이 대칭과 비대칭 부분으로 표현될 수 있다.

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [대칭, 비대칭]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞] where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; 노름 – V는 벡터공간일 때, 노름은 모든 x,y∈V에 대해 다음을 만족하는 함수 N:V⟶[0,+∞]이다.

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; scalar a에 대해서 N(ax)=|a|N(x)를 만족한다.

<br>

**36. if N(x)=0, then x=0**

&#10230; N(x)=0이면 x=0이다.

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; x∈V에 대해, 가장 일반적으로 사용되는 규범이 아래 표에 요약되어 있다.

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [규범, 표기법, 정의, 유스케이스]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; 일차 종속 - 집합 내의 벡터 중 하나가 다른 벡터들의 선형결합으로 정의될 수 있으면, 그 벡터 집합은 일차 종속이라고 한다.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; 비고 : 어느 벡터도 이런 방식으로 표현될 수 없다면, 그 벡터들은 일차 독립이라고 한다.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; 행렬 랭크 - 주어진 행렬 A의 랭크는 열에 의해 생성된 벡터공간의 차원이고, rank(A)라고 쓴다. 이는 A의 선형독립인 열의 최대 수와 동일하다.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; 양의 준정부호 행렬 – 행렬 A∈Rn×n는 다음을 만족하면 양의 준정부호(PSD)라고 하고 A⪰0라고 쓴다.

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; 비고 : 마찬가지로 PSD 행렬이 모든 0이 아닌 벡터 x에 대하여 xTAx>0를 만족하면 행렬 A를 양의 정부호라고 말하고 A≻0라고 쓴다.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; 고유값, 고유벡터 - 주어진 행렬 A∈Rn×n에 대하여, 다음을 만족하는 벡터 z∈Rn∖{0}가 존재하면, z를 고유벡터라고 부르고, λ를 A의 고유값이라고 부른다.

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; 스펙트럼 정리 – A∈Rn×n라고 하자. A가 대칭이면, A는 실수 직교행렬 U∈Rn×n에 의해 대각화 가능하다. Λ=diag(λ1,...,λn)인 것에 주목하면, 다음을 만족한다.

<br>

**46. diagonal**

&#10230; 대각

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; 특이값 분해 – 주어진 m×n차원 행렬 A에 대하여, 특이값 분해(SVD)는 다음과 같이 U m×m 유니터리와 Σ m×n 대각 및 V n×n 유니터리 행렬의 존재를 보증하는 인수분해 기술이다.

<br>

**48. Matrix calculus**

&#10230; 행렬 미적분

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; 그라디언트 – f:Rm×n→R는 함수이고 A∈Rm×n는 행렬이라 하자. A에 대한 f의 그라디언트 ∇Af(A)는 다음을 만족하는 m×n 행렬이다.

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.** 비고 : f의 그라디언트는 f가 스칼라를 반환하는 함수일 때만 정의된다. 

&#10230;

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; 헤시안 – f:Rn→R는 함수이고 x∈Rn는 벡터라고 하자. x에 대한 f의 헤시안 ∇2xf(x)는 다음을 만족하는 n×n 대칭행렬이다.

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar** 

&#10230; 비고 : f의 헤시안은 f가 스칼라를 반환하는 함수일 때만 정의된다.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; 그라디언트 연산 – 행렬 A,B,C에 대하여, 다음 그라디언트 성질을 염두해두는 것이 좋다.

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [일반적인 표기법, 정의, 주요 행렬]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [행렬 연산, 곱셈, 다른 연산]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [행렬 성질, 노름, 고유값/고유벡터, 특이값 분해]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [행렬 미적분, 그라디언트, 헤시안, 연산]

