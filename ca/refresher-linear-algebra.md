**1. Linear Algebra and Calculus refresher**

&#10230; Repàs d'Àlgebra lineal i càlcul

<br> 

**2. General notations**

&#10230; Notacions Generals

<br> 

**3. Definitions**

&#10230; Definicions

<br> 

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vector ― Siga x∈Rn un vector amb n entrades, on xi∈R es la n-èsima entrada:

<br> 

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matriu ― Siga A∈Rm×n una matriu amb n filas i m columnes; on Ai, j∈R es el valor de la i-èsima fila i la n-èsima columna:   

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Nota: el vector x definit dalt pot ser vist com una matriu d'n×1 i es anomenat particularment vector-columna.

<br>

**7. Main matrices**

&#10230; Matrius principals

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Matriu identitat - La matriu identitat I∈Rn×n es una matriu quadrada amb valor 1 a la seva diagonal i cero a la resta:

<br>

**9. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Nota: per a totes les matrius A∈Rn×n, tenim A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Matriu diagonal ― Una matriu diagonal D∈Rn×n es una matriu quadrada amb valores diferentes de zero en la seva diagonal i cero a la resta:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Nota: També denotem D com diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Operacions de matrius

<br>

**13. Multiplication**

&#10230; Multiplicació

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vector-vector ― Hi ha dos tipus de multiplicacions vector-vector:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; producte intern: per a x,y∈Rn, es té:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; product extern: per a x∈Rm,y∈Rn, es té:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that**

&#10230; Matriu-vector ― El producte de la matriu A∈Rm×n i el vector x∈Rn, es un vector de tamany Rn; tal que:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; on aTr,i son les files del vector i ac,j son les columnes del vector A, i xi son les entrades d'x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matriu-matriu ― El productw de les matrius A∈Rm×n i B∈Rn×p es una matriu de tamany Rn×p, tal que:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; on aTr,i,bTr,i son les files del vector i ac,j,bc,j les columnes d'A i B respectivament

<br>

**21. Other operations**

&#10230; Altres operacions

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transposada ― La transposada de la matriu A∈Rm×n, denotada AT, es tal que les seves entrades estan voltejades:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Nota: per a matrius A,B, es té (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inversa ― La inversa d'una matriu quadrada invertible A, es denota per A−1 i es l'única matriu tal que:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Nota: no totes les matrius es poden invertir. A més, per a les matrius A,B, es té que (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Traça ― La traça d'una matriu cuadrada A, tr(A), es la suma dels seus elements de la diagonal:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Nota: per a matrius A,B, es té tr(AT)=tr(A) i tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinanat ― El determinant d'una matriu cuadrada A∈Rn×n, denotat per |A| o det(A) es expresat recursivament en termes d'A∖i,∖j, que es la matriu A en la seva i-èsima fila i j-èsima columna, com es mostra:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Nota: A té inversa si i sols si |A|≠0. A més, |AB|=|A||B| i |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Propietats de matrius

<br>

**31. Definitions**

&#10230; Definicions

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Descomposició simètrica ― Una matriu A pot ser expressada en termes de les seves parts simètriques i asimètriques, com es mostra:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Simètrica, Asimètrica]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norma ― La norma o mòdul es una funció N:V⟶[0,+∞[ on V es un espai vectorial, tal que para todos los x,y∈V, es té:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) per a un escalar

<br>

**36. if N(x)=0, then x=0**

&#10230; si N(x)=0, aleshores x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Per a x∈V, els mòduls o normes més comunment utilitzades estan descrites en la tabla inferior:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norma, Notació, Definició, Cas d'us]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Dependència lineal ― Un conjunt de vectors es diu que es linealmente dependent si un dels vectores en el conjunt pot ser definit com una combinació lineal dels altres.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Nota: si no es pot escriure el vector d'aquesta forma, aleshores el vector es diu que es linealmente independent

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Rang matricial ― El rang d'una matriu A, denotat per rank(A), es la dimensió de l'espai vectorial generat per les seves columnes. Es equivalent al nombre màxim de columnes linealmente independents d'A.
 
<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Matriz semi-definida positiva ― Una matriu A∈Rn×n es semi-defininda positiva (PSD) i es denota per A⪰0 si:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Nota: de igual forma, una matriu A es diu definida positiva, A≻0, si es una matriu PSD que satisfa per a tots els vectors diferents de zero x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenvalor, eigenvector ― Donada una matriu A∈Rn×n, λ es diu que es el eigenvalor de A si existeix un vector z∈Rn∖{0}, anomenat eigenvector, tal que es té:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema espectral ― Siga A∈Rn×n. si A es simètrica, aleshores A es diagonalitzable per una matriu real ortogonal U∈Rn×n.
Denotant Λ=diag(λ1,...,λn), es té que:

<br>

**46. diagonal**

&#10230; diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Descomposició de valors singulars ― Per a una matriu A de dimensions m×n, la descomposició en valors singulars (SVD) es una tècnica de factorització que garantitza que existeixen matrius U m×m unitaria, Σ m×n diagonal i V n×n unitària, tal que:

<br>

**48. Matrix calculus**
 
&#10230; Càlcul de matrius

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradient ― Siga f:Rm×n→R una funció i A∈Rm×n una matriu. El gradient d'f amb respecte a A es una matriu de m×n, denotat ∇Af(A), tal que:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Nota: el gradient d'f està sols definit quan f es una funció que retorna un escalar.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Matriu Hessiana ― Siga f:Rn→R una funció i x∈Rn un vector. La matriu hessiana o hessià d'f recpecte a x
es una matriu simètrica de n×n, denotada ∇2xf(x), tal que:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Nota: la matriu hessiana d'f sols està definida quan f es una funció que retorna un escalar

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Operacions de gradient ― Per a matrius A,B,C, val la pena tindre en compte les següents propietats del gradient:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Notacions generals, Definicions, Matrius principals]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Operacions matricials, Multiplicació, Altres operacions]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Propietats matricials, Norma, Eigenvalor/Eigenvector, Descomposició de valors singulars]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Càlcul matricial, Gradiant, Matriu Hessiana, Operacions]
