**1. Linear Algebra and Calculus refresher**

&#10230; Lineáris algebra és analízis felfrissítés

<br>

**2. General notations**

&#10230; Általános jelölések

<br>

**3. Definitions**

&#10230; Definíciók

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vektor ― Az n komponensű x∈Rn vektort, melynek xi∈R az i-edik komponense, így jelöljük:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Mátrix ― Az m sorú és n oszlopú A∈Rm×n mátrixot, melynek Ai,j∈R az i-edik sorban és j-edik oszlopban található eleme, így jelöljük:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Megjegyzés: a fent definiált x vektor tekinthető egy n×1-es mátrixnak, és ekkor oszlopvektornak hívjuk.

<br>

**7. Main matrices**

&#10230; Főbb mátrixtípusok

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Egységmátrix ― Az I∈Rn×n egységmátrix olyan négyzetes mátrix, melynek a diagonálisában (főátlójában) 1-esek állnak és 0-k mindenhol máshol.

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Megjegyzés: bármely A∈Rn×n mátrix esetén igaz a következő: A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Diagonális mátrix ― A D∈Rn×n diagonális mátrix olyan négyzetes mátrix, melynek a diagonálisában (főátlójában) nemnulla elemek állnak és 0-k mindenhol máshol.

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Megjegyzés: D-t jelölhetjük így is: diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Mátrixműveletek

<br>

**13. Multiplication**

&#10230; Szorzás

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vektor-vektor ― Kétféle vektor-vektor szorzat létezik.

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; skaláris (vagy belső) szorzat: bármely x,y∈Rn esetén:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; külső szorzat: bármely x∈Rm,y∈Rn esetén:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; mátrix-vektor ― az A∈Rm×n mátrix és x∈Rn vektor szorzata az az Rn-beli vektor, melyre:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; ahol az aTr,i jelöli az A sorait és ac,j jelöli az A oszlopait, és xi az x vektor komponensei.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Mátrix-mátrix ― Az A∈Rm×n és B∈Rn×p mátrixok szorzatai az az Rn×p-beli mátrix, melyre:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; ahol aTr,i,bTr,i rendre az A és B mátrixok sorai és ac,j,bc,j az A és B mátrixok oszlopai.

<br>

**21. Other operations**

&#10230; Egyéb műveletek

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transzponálás ― Az A∈Rm×n mátrix transzponáltja (jel.: AT) alatt azt a mátrixot értjük, mely az A elemeinek főátlóra való tükrözésével keletkezik:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Megjegyzés: bármely A,B mátrix esetén (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inverzképzés (invertálás) ― Az A invertálható négyzetes mátrix inverzét A−1-vel jelöljük, és azt a mátrixot értjük alatta, melyre:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Megjegyzés: nem minden négyzetes mátrix invertálható. Ha viszont A,B mátrixok invertálhatóak, akkor AB is invertálható és (AB)−1=B−1A−1.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Nyom ― Az A négyzetes mátrix nyoma (jel.: tr(A)) alatt a főátlóbeli elemek összegét értjük.

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Megjegyzés: bármely A,B négyzetes mátrix esetén tr(AT)=tr(A) és tr(AB)=tr(BA).

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determináns ― Az A∈Rn×n négyzetes mátrix determinánsát (jel.: |A| vagy det(A)) rekurzívan, A∖i,∖j segítségével számolhatjuk ki, ahol A∖i,∖j az A mátrix azon részmátrixa, mely nem tartalmazza az A i-edik sorát és j-edik oszlopát.

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Megjegyzés: A mátrix akkor és csak akkor invertálható, ha |A|≠0. Továbbá |AB|=|A||B| és |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Mátrixtulajdonságok

<br>

**31. Definitions**

&#10230; Definíciók

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Szimmetrikus felbontás ― Egy adott A mátrix felírható szimmetrikus és antiszimmetrikus mátrixok összegeként az alábbi módon: 

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Szimmetrikus, Antiszimmetrikus]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norma ― Az N:V⟶[0,+∞[ függvényt normának nevezünk, ha V vektortér és minden x,y∈V esetén:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) ahol a egy skalár

<br>

**36. if N(x)=0, then x=0**

&#10230; ha N(x)=0, akkor x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Az alábbi táblázatban foglaljuk össze a leggyakrabban használt normákat (x∈V):

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norma, Jelölés, Definíció, Itt (is) használjuk]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Lineáris összefüggőség ― Egy vektorrendszert lineárisan összefüggőnek nevezünk, ha van olyan vektora, mely kifejezhető a többi vektor lineáris kombinációjaként.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Megjegyzés: ha egyetlen vektor sem fejezhető ki így, akkor azt mondjuk, hogy a vektorrendszer lineárisan független.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Mátrixrang ― Az A mátrix rangja (jel.: r(A)) alatt az oszlopai által generált altér dimenziója. Ekvivalensen: ha A oszlopai közt található r(A) lineárisan független, de több nem.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Pozitív szemidefinit mátrix ― Az A∈Rn×n pozitív szemidefinit (jel.: A⪰0), ha igazak az alábbiak:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Megjegyzés: hasonlóan, az A∈Rn×n pozitív definit (jel.: A≻0), ha igazak pozitív szemidefinit és minden nemnulla x vektorra xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Sajátérték, sajátvektor ― Legyen A∈Rn×n. Azt mondjuk, hogy λ sajátértéke az A-nak, ha létezik olyan z∈Rn∖{0} vektor (az ún. sajátvektorI, melyre:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spektráltétel ― Legyen A∈Rn×n. Ha A szimmetrikus, akkor A diagonalizálható egy U∈Rn×n valós ortogonális mátrixszal. Azaz ha Λ=diag(λ1,...,λn), akkor

<br>

**46. diagonal**

&#10230; diagonális

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Szinguláris felbontás ― Legyen m×n-es valós (komplex) A mátrix adott. Ekkor A szinguláris érték szerinti felbontása olyan faktorizációs technika, mely garantálja az U m×m-es ortogonális (unitér), Σ m×n-es diagonális és V n×n-es ortogonális (unitér) mátrixok létezését, melyekre:

<br>

**48. Matrix calculus**

&#10230; Mátrixanalízis

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradiens ― Legyen f:Rm×n→R függvény és A∈Rm×n mátrix. Az f gradiense az A-ra nézve az az m×n-es mátrix (jel.: ∇Af(A)), melyre:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Megjegyzés: az f gradiensét csak skalárértékű függvény esetén definiáljuk.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hesse-mátrix ― Legyen f:Rn→R függvény és x∈Rn vektor. Ekkor az f Hesse-mátrixa x-ben az az n×n-es mátrix (jel.: ∇2xf(x)), melyre:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Megjegyzés: Az f Hesse-mátrixát csak skalárértékű függvény esetén definiáljuk.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Műveletek a gradienssel ― Adott A,B,C mátrixok esetén érdemes a gradiens alábbi tulajdonságait megjegyeznünk:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Általános jelölések, Definíciók, Főbb mátrixtípusok]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Mátrixműveletek, Szorzás, Egyéb műveletek]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Mátrixtulajdonságok, Norma, Sajátérték/Sajátvektor, Szinguláris felbontás]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Mátrixanalízis, Gradiens, Hesse-mátrix, Műveletek]
