**Linear Algebra and Calculus translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)

&#10230; Algebra lineare e Analisi

<br>

**1. Linear Algebra and Calculus refresher**

&#10230; Ripasso di Algebra lineare e Analisi

<br>

**2. General notations**

&#10230; Notazione generale

<br>

**3. Definitions**

&#10230; Definizioni

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vettore ― Definiamo x∈Rn un vettore con n elementi, dove xi∈R è l'i-esimo elemento:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matrice ― Definiamo A∈Rm×n una matriche con m righe e n colonne, dove Ai,j∈R è l'elemento posizionato alla i-esima riga e j-esima colonna:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Osservazione: il vettore x, definito precedentemente, può essere visto come una matrice nx1 ed è chiamato, più particolarmente, un vettore colonna.

<br>

**7. Main matrices**

&#10230; Matrici principali

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; matrice identità ― La matrice identità I∈Rn×n è una matrice quadrata con tutti 1 sulla diagonale principale e 0 in tutte le altre posizioni:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Osservazione: per tutte le matrici A∈Rn×n, abbiamo che A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; matrice diagonale ― Una matrice diagonale D∈Rn×n è una matrice quadrata con valori diversi da zero sulla diagonale principale e zero in tutte le altre posizioni:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Osservazione: definiamo, inoltre, D come diag(d1,...,dn)

<br>

**12. Matrix operations**

&#10230; Operazioni sulle matrici

<br>

**13. Multiplication**

&#10230; Moltiplicazione

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vettore-vettore ― Ci sono due tipi di prodotto vettore-vettore:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; prodotto scalare: per x,y∈Rn, abbiamo che:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; prodotto vettoriale: per x∈Rm,y∈Rn, abbiamo che:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matrice-vettore ― Il prodotto di una matrice A∈Rm×n ed un vettore x∈Rn, è un vettore di dimensione Rn, tale che:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; dove aTr,i sono i vettori riga, ac,j sono i vettori colonna di A e xi sono gli elementi di x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matrice-matrice — Il prodotto di matrici A∈Rm×n e B∈Rn×p è una matriche di dimensione Rn×p, tale che:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; dove aTr,i,bTr,i sono i vettori riga e ac,j,bc,j sono i vettori colonna rispettivamente di A e di B

<br>

**21. Other operations**

&#10230; Altre operazioni

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Trasposta — La trasposta di una matrice A∈Rm×n, indicata con AT, è tale che i suoi elementi sono scambiati:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Osservazione: per le matrici A,B abbiamo che (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inversa — L'inversa di una matrice quadrata invertibile A è indicata con A-1 ed è l'unica matrice tale che:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Osservazione: non tutte le matrici quadrate sono invertibili. Inoltre, per le matrici A,B, abbiamo che (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Traccia — La traccia di una matrice quadrata A, indicata con tr(A), è la somma degli elementi sulla diagonale principale:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Osservazione: per le matrici A,C, abbiamo che tr(AT)=tr(A) e tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinante — Il determinante di una matrice quadrata A∈Rn×n, indicata con |A| o det(A) è espresso ricorsivamente rispetto a A\i,\j, che è la matrice A senza l'i-esima riga e la j-esima colonna, come segue:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Osservazione: A è invertibile se e solo se |A|≠0. Inoltre, |AB|=|A||B| e |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Proprietà delle matrici

<br>

**31. Definitions**

&#10230; Definizioni

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Decomposizione simmetrica — Una matrice A può essere espressa tramite la sua componente simmetrica ed antisimmetrica come segue:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Simmetrica, Antisimmetrica]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norma — La norma è una funzione N:V⟶[0,+∞[ dove V è uno spazio vettoriale, tale che per 
x,y∈V, abbiamo che:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) per uno scalare

<br>

**36. if N(x)=0, then x=0**

&#10230; if N(x)=0, allora x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Per x∈V, le norme più usate comunemente sono riassunte nella tabella seguente:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norma, Notazione, Definizione, Caso d'uso]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Linearmente dipendente — Un insieme di vettori è detto linearmente dipendente, se uno dei vettori dell'insieme può essere definito come combinazione lineare degli altri.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Osservazione: se nessun vettore può essere scritto in questo modo, allora i vettori sono detti linearmente indipendenti

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Rango di una matrice — Il rango di una data matrice A si indica rg(A) ed è la dimensione dello spazio vettoriale generato dalle sue colonne. Questo è equivalente al numero massimo di colonne linearmente indipendenti di A. 

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Matrice semidefinita positiva — Una matrice A∈Rn×n è semidefinita positiva (PSD) ed è indicata da A⪰0, se abbiamo che:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Osservazione: analogamente, una matrice A è detta definita positiva, ed è indicata con A≻0, se è una matrice PSD che soddisfa per ogni vettore x non nullo, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalore, autovettore — Data una matrice A∈Rn×n, si dice che λ è un autovalore di A, se esiste un vettore z∈Rn∖{0}, chiamato autovettore, tale che abbiamo:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema spettrale — Sia A∈Rn×n. Se A è simmetrico, allora A è diagonalizzabile da una matrice reale ortogonale U∈Rn×n. Chiamando Λ=diag(λ1,...,λn), abbiamo che: 

<br>

**46. diagonal**

&#10230; diagonale

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Decomposizione ai valori singolari — Per una data matrice A di dimensione m×n, la decomposizione ai valori singolari (SVD) è una tecnica di fattorizzazione che garantisce l'esistenza della matrice unitaria U m×m, della matrice diagonale Σ m×n e della matrice unitaria V n×n, tale che:

<br>

**48. Matrix calculus**

&#10230; Matrice

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradiente — Sia f:Rm×n→R una funzione e A∈Rm×n una matrice. Il gradiente di f in funzione di A è una matrice m×n, indicata con ∇Af(A), tale che:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Osservazione: il gradiente di f è definito solamente quando f è una funzione che restituisce uno scalare.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Matrice Hessiana — Sia f:Rn→R una funzione e x∈Rn un vettore. La matrice hessiana di f in funzione di x è una matrice simmetrica n×n, indicata con ∇2xf(x), tale che:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Osservazione: la matrice Hessiana di f è definita solamente quando f è una funzione che restituisce uno scalare

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Operazioni del gradiente — Per le matrici A,B,C, vale la pena ricordare le seguenti proprietà del gradiente:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Notazione generale, Definizioni, Matrici principali]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Operazioni tra matrici, Moltiplicazione, Altre operazioni]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Proprietà delle matrici, Norma, Autovalore/Autovettore, Decomposizione ai valori singolari]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Calcolo tra matrici, Gradiente, Matrice Hessiana, Operazioni]
