**1. Linear Algebra and Calculus refresher**

&#10230; Auffrischung Lineare Algebra und Differentialrechnung

<br>

**2. General notations**

&#10230; Allgemeine Notationen

<br>

**3. Definitions**

&#10230; Definitionen

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vektor - Sei x∈Rn ein Vektor mit n Einheiten und xi∈R der i-te Eintrag:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matrix - Sei A∈Rm×n eine Matrix mit m Zeilen und n Spalten und Ai,j∈R die Einheit in der i-ten Reihe und j-te Spalte:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Wichtig: Der Vektor x, wie oben erwähnt, ist eine n×1 Matrix und wird auch Spaltenvektor genannt.

<br>

**7. Main matrices**

&#10230; Spezielle Matrizen

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Identitätsmatrix - Die Identitätsmatrix I∈Rn×n ist eine quadratische Matrix welche 1er in der Diagonale enthält und sonst 0en.

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Wichtig: Für eine Matrix A∈Rn×n gilt: A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Diagonalmatrix - Eine Diagonalmatrix D∈Rn×n ist eine quadratische Matrix welche außerhalb der Diagonale 0 enthält. Die Einheiten in der Diagonale sind ungleich 0:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Wichtig: Die Matrix D wird auch diag(d1,...,dn) bezeichnet.

<br>

**12. Matrix operations**

&#10230; Rechenoperationen mit Matrizen

<br>

**13. Multiplication**

&#10230; Multiplikation

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vektor-Vektor - Es gibt zwei Arten des Vektor-Vektor Produkts:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; Innere Produkt: Auch Skalarprodukt, Es gilt x,y∈Rn:**

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; Äußere Produkt: Auch Kreuzprodukt, Es gilt x∈Rm,y∈Rn:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matrix-Vektor : Das Produkt einer Matrix A∈Rm×n und Vektor x∈Rn ist ein Vektor der Größe Rn, und es gilt:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; aTr,i sind die Zeilenvektoren und ac,j sind die Spaltenvektoren der Matrix A und xi sind die Einheiten des Vektors x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matrix-Matrix - Das Produkt einer Matrix A∈Rm×n und B∈Rn×p ist ebenfalls eine Matrix der Größe Rn×p und es gilt:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; aTr,i,bTr,i sind jeweils die Zeilenvektoren und ac,j,bc,j sind die Spaltenvektoren von A und B

<br>

**21. Other operations**

&#10230; Weitere Rechenoperationen

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transposition - Die Transposition oder Transponierung einer Matrix A∈Rm×n, geschrieben als AT, spiegelt die Einheiten der ursprünglichen Matrix entlang der Hauptdiagonale: 

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Wichtig: Für Matrix A und B gilt (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inverse - Das Inverse, oder Umkehrung, einer invertierbaren, quadratischen Matrix A wird auch als A-1 bezeichnet und ist die einzige Matrix für die gilt:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Wichtig: Nicht alle quadratischen Matrizen sind invertierbar. Weiteres gilt für Matrix A und B (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Spur - Die Spurabbildung einer quadratischen Matrix A, geschrieben als tr(A), ist die Summe der Diagonaleinheiten:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Wichtig: Für Matrix A und B gilt: tr(AT)=tr(A) und tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinante - Die Determinante einer quadratischen Matrix A∈Rn×n, auch |A| oder det(A) geschrieben, wird mit einer Matrix A\i,\j definiert welche die Matrix A ohne i-te Zeile und ohne j-te Spalte darstellt.

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Wichtig: A ist nur invertierbar falls |A|≠0. Weiteres gilt |AB|=|A||B| and |AT|=|A| 

<br>

**30. Matrix properties**

&#10230; Eigenschaften einer Matrix

<br>

**31. Definitions**

&#10230; Definitionen

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Symmetrische Zerlegung - Eine Matrix A kann durch dessen symmetrischen und schiefsymmetrischen Anteil wie folgt definiert werden:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Symmetrisch, Schiefsymmetrisch]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Matrixnorm - Die Matrixnorm is eine Funktion N:V⟶[0,+∞[ wobei V der Vektorraum ist und für alle x,y∈V gilt:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) wobei a ein Skalar ist

<br>

**36. if N(x)=0, then x=0**

&#10230; falls N(x)=0, folgt x=0**

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Sei x∈V, die häufigsten Matrixnormen werden in der folgenden Tabelle erläutert:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Matrixnorm, Notation, Definition, Verwendung]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Lineare Abhängigkeit - Vektoren eines Vektorraums sind linear abhängig falls sich ein Vektor als Linearkombination der anderen Vektoren darstellen lässt.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Wichtig: falls kein Vektor als Linearkombination der anderen gebildet werden kann, werden die Vektoren als linear unabhängig bezeichnet.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Rang - Der Rang einer Matrix A wird auch als rank(A) bezeichnet und wird durch die Dimension des Vektorraums, aufgespannt durch die Spaltenvektoren, definiert. Der Rang ist äquivalent mit der maximalen Anzahl an linear unabhängigen Spaltenvektoren von A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Positiv-Semidefinierte Matrix - Eine Matrix A∈Rn×n ist positiv semidefinit (PSD), A⪰0 geschrieben, falls folgendes gilt:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Wichtig: Ebenfalls gilt, Eine Matrix A ist positiv definit, A≻0, falls diese eine PSD Matrix ist und es gilt für alle Einheiten eines Vektors x ungleich 0: x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenwert, Eigenvektor - Sei A∈Rn×n eine Matrix und λ der Eigentwert von A, falls es einen Vektor z∈Rn∖{0} gibt, Eigentvektor genannt, gilt folgendes:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spektralsatz - Sei A∈Rn×n, falls A eine symmetrische Matrix, dann ist diese diagonalisierbar durch eine orthogonale Matrix U∈Rn×n. Durch Λ=diag(λ1,...,λn) gilt folgendes:

<br>

**46. diagonal**

&#10230; Diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Singulärwertzerlegung - Eine Matrix A mit den Dimensionen m×n kann durch Singulärwertzerlegung (SVD) als Produkt drei anderer dargestellt werden. Folgendes gilt, wobei U m×m eine unitäre Matrix, Σ m×n eine Diagonalmatrix und V n×n eine unitäre Matrix ist: 

<br>

**48. Matrix calculus**

&#10230; Matrix Differentialrechnung

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Ableitung - Sei f:Rm×n→R eine Funktion und A∈Rm×n eine Matrix. Die Ableitung von f nach A ist eine m×n Matrix, geschrieben ∇Af(A) für die gilt:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Wichtig: Die Ableitung von f ist nur definiert falls f ein Skalar als Ergebnis hat.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hesse-Matrix - Sei f:Rn→R eine Funktion und x∈Rn ein Vektor. Die Hesse-Matrix von f nach x ist eine symmetrische Matrix der Größe n×n.

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Wichtig: Die Hesse-Matrix von f ist nur definiert falls f ein Skalar als Ergebnis hat.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Eigenschaften der Ableitung - Für die Ableitungen der Matrizen A,B,C gelten folgende Eigenschaften:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Allgemeine Notationen, Definitionen, Spezielle Matrizen]

<br> 

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Rechenoperationen mit Matrizen, Multiplikation, Weitere Rechenoperationen]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Eigenschaften einer Matrix, Matrixnorm, Eigenwert/Eigenvektor, Singulärwertzerlegung]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Matrix Differentialrechnung, Ableitung, Hesse-Matrix, Rechenoperationen]
