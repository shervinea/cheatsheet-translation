**1. Linear Algebra and Calculus refresher**

&#10230; Revisão de Cálculo e Álgebra Linear

<br>

**2. General notations**

&#10230; Notações gerais

<br>

**3. Definitions**

&#10230; Definições

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vetor - Indicamos por x∈Rn um vetor com n elementos, onde xi∈R é o i-ésimo elemento:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matriz - Indicamos por A∈Rm×n uma matriz com m linhas e n colunas, onde Ai,j∈R é o elementos localizado na i-ésima linha e j-ésima coluna:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Observação: o vetor x defindo acima pode ser visto como uma matriz nx1 e é mais particularmente chamado de vetor coluna. 

<br>

**7. Main matrices**

&#10230; Matrizes principais

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Matriz identidade - A matriz identidade é uma matriz quadrada com uns na sua diagonal e zeros nas demais posições:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Observação: para todas as matrizes A∈Rn×n, nós temos A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Matriz diagonal - Uma matriz diagonal D∈Rn×n é uma matriz quadrada com valores não nulos na sua diagonal e zeros nas demais posições:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Observação: nós também indicamos D como diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Operações de matriz

<br>

**13. Multiplication**

&#10230; Multiplicação

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vetor-vetor - Há dois tipos de produtos vetoriais:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; produto interno: para x,y∈Rn, temos:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; produto tensorial: para x∈Rm,y∈Rn, temos:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matriz-vetor - O produto de uma matriz A∈Rm×n e um vetor x∈Rn é um vetor de tamanho Rn, de tal modo que:

<br> 

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; onde aTr,i são vetores linhas e ac,j vetores colunas de A, e xi são os elementos de x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matriz-matriz - O produto das matrizes A∈Rm×n e B∈Rn×p é uma matriz de tamanho Rn×p, de tal modo que:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; onde aTr,i,bTr,i são vetores linhas e ac,j,bc,j vetores colunas de A e B respectivamente

<br>

**21. Other operations**

&#10230; Outras operações

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transposta - A transposta de uma matriz A∈Rm×n, indicada por AT, é tal que suas linhas são trocadas por suas colunas: 

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Observação: para matrizes A,B, temos (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inversa - A inversa de uma matriz quadrada inversível A é indicada por A-1 e é uma matriz única de tal modo que:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230;

<br> Observação: nem todas as matrizes quadrada são inversíveis. Também, para matrizes A,B, temos (AB)−1=B−1A−1

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Traço - O traço de uma matriz quadrada A, indicado por tr(A), é a soma dos elementos de sua diagonal:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Observação: para matrizes A,B, temos tr(AT)=tr(A) e tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinante - A determinante de uma matriz quadrada A∈Rn×n, indicada por |A| ou det(A) é expressa recursivamente em  termos de A∖i,∖j, a qual é a matriz A sem a sua i-ésima linha e j-ésima coluna, como se segue:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Observação: A é inversível se e somente se |A|≠0. Além disso, |AB|=|A||B| e |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Propriedades da matriz

<br>

**31. Definitions**

&#10230; Definições

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Decomposição simétrica - Uma dada matriz A pode ser expressa em termos de suas partes simétricas e assimétricas como a seguir:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Simétrica, Assimétrica]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norma - Uma norma é uma função N:V⟶[0,+∞[ onde V é um vetor espaço, e de tal modo que para todo x,y∈V, nós temos;

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) para um escalar

<br>

**36. if N(x)=0, then x=0**

&#10230; se N(x)=0, então x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Para x∈V, as mais comumente utilizadas normas estão resumidas na tabela abaixo:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norma, Notação, Definição, Caso de uso]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Dependência linear - Um conjunto de vetores é dito ser linearmente dependete se um dos vetores no conjunto puder ser definido como uma combinação linear dos demais.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Observação: se nenhum vetor puder ser escrito dessa maneira, então os vetores são ditos serem linearmente independentes

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Posto da matriz - O posto de uma dada matriz A é indicada por rank(A) e é a dimensão do vetor espaço gerado por suas colunas. Isso é equivalente ao número máximo de colunas linearmente independentes de A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Matriz positiva semi-definida - Uma matriz A∈Rn×n é positiva semi-definida (PSD) e é indicada por A⪰0 se tivermos:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Observação: de forma similar, uma matriz A é dita ser positiva definida, e é indicada por A≻0, se ela é uma matriz (PSD) que satisfaz todo vetor x não nulo, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalor, autovetor - Dada uma matriz A∈Rn×n, λ é dita ser um autovalor de A se existe um vetor z∈Rn∖{0}, chamado autovetor, nós temos:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema spectral - Seja A∈Rn×n. Se A é simétrica, então A é diagonalizável por uma matriz ortogonal U∈Rn×n. Indicando Λ=diag(λ1,...,λn), nós temos:

<br>

**46. diagonal**

&#10230; diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Decomposição em valor singular - Para uma dada matriz A de dimensões mxn, a decomposição em valor singular (SVD) é uma técnica de fatorização que garante a existência de matrizes unitária U m×m, diagonal Σ m×n e unitária V n×n, de tal modo que:

<br>

**48. Matrix calculus**

&#10230; Cálculo com matriz

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradiente -  Seja f:Rm×n→R uma função e A∈Rm×n uma matriz. O gradiente de f a respeito a A é a matriz Mxn, indicada por ∇Af(A), de tal modo que:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Observação: o gradiente de f é somente definido quando f é uma função que retorna um escalar.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessiano - Seja f:Rn→R uma função e x∈Rn um vetor. O hessiano de f a respeito a x é uma matriz simétrica nxn, indicada por ∇2xf(x), de tal modo que:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Observação: o hessiano de f é somente definifo quando f é uma função que retorna um escalar

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Operações com gradiente - Para matrizes A,B,C, as seguintes propriedade de gradiente valem a pena ter em mente:
