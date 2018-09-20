**1. Linear Algebra and Calculus refresher**

&#10230; Hoja de referencia de Álgebra lineal y cálculo

<br> 

**2. General notations**

&#10230; Notaciones Generales

<br> 

**3. Definitions**

&#10230; Definiciones

<br> 

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry: **

&#10230; Vector ― Sea x∈Rn un vector con n entradas, donde xi∈R es la n-ésima entrada:

<br> 

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matriz ― Sea A∈Rm×n una matriz con n filas y m columnas; donde Ai, j∈R es el valor alocado en la i-ésima fila y la n-ésima columna:   

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Nota: el vector x definido arriba puede ser visto como una matriz de n×1 y es particularmente llamado vector-columna.

<br>

**7. Main matrices**

&#10230; Matrices principales

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Matriz identidad - La matriz identidad I∈Rn×n es una matriz cuadrada con valor 1 en su diagonal y cero en el resto:

<br>

**9. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Nota: para todas las matrices A∈Rn×n, tenemos A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Matriz diagonal ― Una matriz diagonal D∈Rn×n es una matriz cuadrada con valores diferentes de zero en su diagonal y cero en el resto:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Nota: Sea D una diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Operaciones de matrices

<br>

**13. Multiplication**

&#10230; Multiplicación

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vector-vector ― Hay dos tipos de multiplicaciones vector-vector:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; producto interno: para x,y∈Rn, se tiene:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; producto externo: para x∈Rm,y∈Rn, se tiene:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that**

&#10230; Matriz-vector ― El producto de la matriz A∈Rm×n y el vector x∈Rn, es un vector de tamaño Rn; tal que:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; donde aTr,i son las filas del vector and ac,j son las columnas del vector A, y xi son las entradas de x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matriz-matriz ― El producto de las matrices A∈Rm×n y B∈Rn×p es una matriz de tamaño Rn×p, tal que:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; donde aTr,i,bTr,i son las filas del vector and ac,j,bc,j las columnas de A y B respectivamente

<br>

**21. Other operations**

&#10230; Otras operaciones

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transpuesta ― La transpuesta de la matriz A∈Rm×n, con notación AT, es tal que sus entradas son volteadas:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Nota: para matrices A,B, se tiene (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inversa ― La inversa de una matriz cuadrada invertible A, llamada A−1 y es la única matriz tal que:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Nota: no todas las matrices se pueden invertir. Además, para las matrices A,B, se tiene que (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Traza ― La traza de una matriz cuadrada A, tr(A), es la suma de sus elementos en la diagonal:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Nota: para matrices A,B, se tiene tr(AT)=tr(A) y tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinanate ― El determinante de una matriz cuadrada A∈Rn×n, llamado |A| or det(A) es expresado recursivamente en términos de A∖i,∖j, que es la matriz A en su i-ésima fila y j-ésima columna, como se muestra:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Nota: A es tiene inversa si y solo si |A|≠0. Además, |AB|=|A||B| y |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Propiedades de matrices

<br>

**31. Definitions**

&#10230; Definiciones

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Descomposición Simétrica ― Una matriz A puede ser expresada en términos de sus partes simétricas y asimetricas, como se muestra:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Simétrica, Asimétrica]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norma ― La norma o módulo es una función N:V⟶[0,+∞[ donde V es un vector espacial, y tal que para todos los x,y∈V, se tiene:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) para un escalar

<br>

**36. if N(x)=0, then x=0**

&#10230; si N(x)=0, entonces x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Para x∈V, los módulos o normas más comúnmente usadas están descritas en la tabla de abajo:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norma, Notación, Definición, Caso de uso]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Dependencia Lineal ― Un conjunto de vectores se dice que es linealmente dependiente si uno de los vectores en el grupo puede ser definido como una combinación lineal de los otros.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Nota: si no se puede escribir el vector de esta manera, entonces el vector se dice que es linealmente independiente

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Rango matricial ― El rango de una matriz A, nombrado rank(A), que es la dimensión del vector espacial generado por sus columnas. Lo que es equivalente al número máximo de columnas linealmente independientes de A.
 
<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Matriz semi-definida positiva ― Una matriz A∈Rn×n es semi-defininda positivamente (PSD) y se tiene que A⪰0 si:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Nota: de igual forma, una matriz A se dice positiva y definida, A≻0, si esa una matriz PSD que satisface para todos los vectores diferentes de cero x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenvalor, eigenvector ― Dado una matriz A∈Rn×n, λ se dice que es el eigenvalor de A si existe un vector z∈Rn∖{0}, llamado eigenvector, tal que se tiene:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema espectral ― Sea A∈Rn×n. si A es simétrica, entonces A es diagonalizable por una matriz real ortogonal U∈Rn×n.
Notando Λ=diag(λ1,...,λn), se tiene que:

<br>

**46. diagonal**

&#10230; diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Descomposición de valores singulares ― Para una mtraiz A de dimensiones m×n, la descomposición en valores singulares (SVD) es una técnica de factorizacion que garantiza que existen matrices U m×m unitaria, Σ m×n diagonal y V n×n unitaria, tal que:

<br>

**48. Matrix calculus**
 
&#10230; Cálculo de matrices

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradiente ― Sea f:Rm×n→R una función y A∈Rm×n una matriz. El gradiente de f con respecto a A es una matriz de m×n, notando que ∇Af(A), tal que:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Nota: el gradiente de f esta solo definido cuando f es una función cuyo resultado es un escalar.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Matriz Hessiana ― Sea f:Rn→R una función y x∈Rn un vector. La matriz hessiana o hessiano de f con recpecto a x
es una matriz simétrica de n×n, para ∇2xf(x), tal que:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Nota: la matriz hessiana de f solo esta definida cuando f es una función que devuelve un escalar

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Operaciones de gradiente ― Para matrices A,B,C, las siguientes propiedades del gradiente vale la pena tener en cuenta:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Notaciones generales, Definiciones, Matrices principales]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Operaciones matriciales, Multiplicación, Otras operaciones]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Propiedades matriciales, Norma, Eigenvalor/Eigenvector, Descomposición de valores singulares]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Cálculo matricial, Gradiante, Matriz Hessiana, Operaciones]
