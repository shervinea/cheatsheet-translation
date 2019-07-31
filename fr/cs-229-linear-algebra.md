**1. Linear Algebra and Calculus refresher**

&#10230; Rappels d'algèbre lineaire et d'analyse

<br>

**2. General notations**

&#10230; Notations générales

<br>

**3. Definitions**

&#10230; Définitions

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vecteur ― On note x∈Rn un vecteur à n entrées, où xi∈R est la ième entrée

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matrice ― On note A∈Rm×n une matrice à n lignes et m colonnes, où Ai,j∈R est le coefficient située à la ième ligne et jième colonne :

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Remarque : le vecteur x défini ci-dessus peut être vu comme une matrice n×1 et est aussi appelé vecteur colonne.

<br>

**7. Main matrices**

&#10230; Matrices principales

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Matrice identité ― La matrice identité I∈Rn×n est une matrice carrée avec des 1 sur sa diagonale et des 0 partout ailleurs :

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Remarque : pour toute matrice A∈Rn×n, on a A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Matrice diagonale ― Une matrice diagonale D∈Rn×n est une matrice carrée avec des valeurs non nulles sur sa diagonale et des zéros partout ailleurs.

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Remarque : on note aussi D = diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Opérations matricielles

<br>

**13. Multiplication**

&#10230; Multiplication

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vecteur-vecteur ― Il y a deux types de multiplication vecteur-vecteur :

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; Produit scalaire : pour x,y∈Rn, on a :

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; Produit dyadique : pour x∈Rm,y∈Rn, on a :

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matrice-vecteur : Le product de la matrice A∈Rm×n et du vecteur x∈Rn est un vecteur de taille Rn, tel que :

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; où aTr,i sont les vecteurs-ligne et ac,j sont les vecteurs-colonne de A et xi sont les entrées de x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matrice-matrice ― Le produit des matrices A∈Rm×n et B∈Rn×p est une matrice de taille Rn×p, tel que :

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; où aTr,i,bTr,i sont des vecteurs-ligne et ac,j,bc,j sont des vecteurs-colonne de A et B respectivement.

<br>

**21. Other operations**

&#10230; Autres opérations

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Transposée ― La transposée est une matrice A∈Rm×n, notée AT, qui est telle que ses entrées sont renversées.

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Remarque : pour des matrices A, B, on a (AB)T=BTAT.

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Inverse ― L'inverse d'une matrice carrée inversible A est notée A−1 et est l'unique matrice telle que :

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Remarque : toutes les matricées carrés ne sont pas inversibles. Aussi, pour des matrices A,B, on a (AB)−1=B−1A−1.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Trace ― La trace d'une matrice carrée A, notée tr(A), est définie comme la somme de ses coefficients diagonaux:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Remarque : pour toutes matrices A, B, on a tr(AT)=tr(A) et tr(AB)=tr(BA).

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Déterminant ― Le déterminant d'une matrice carrée A∈Rn×n notée |A| ou det(A) est exprimée récursivement en termes de A∖i,∖j, qui est la matrice A sans sa ième ligne et jième colonne, de la manière suivante :

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Remarque : A est inversible si et seulement si |A|≠0. Aussi, |AB|=|A||B| et |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Propriétés matricielles

<br>

**31. Definitions**

&#10230; Définitions

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Décomposition symétrique ― Une matrice donnée A peut être exprimée en termes de ses parties symétrique et antisymétrique de la manière suivante :

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Symétrique, antisymétrique]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norme ― Une norme est une fonction N:V⟶[0,+∞[ où V est un espace vectoriel, et tel que pour tous x,y∈V, on a :

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) pour a scalaire

<br>

**36. if N(x)=0, then x=0**

&#10230; si N(x)=0, alors x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Pour x∈V, les normes les plus utilisées sont récapitulées dans le tableau ci-dessous :

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norme, Notation, Définition, Cas d'utilisation]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Dépendance linéaire ― Un ensemble de vecteurs est considéré comme étant linéairement dépendant si un des vecteurs dans l'ensemble peut être défini comme une combinaison linéaire des autres.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Remarque : si aucun vecteur ne peut être noté de cette manière, alors les vecteurs sont dits linéairement indépendants.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Rang d'une matrice ― Le rang d'une matrice donnée A est notée rang(A) et est la dimension de l'espace vectoriel généré par ses colonnes. Ceci est équivalent au nombre maximum de colonnes indépendantes de A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Matrice semi-définie positive ― Une matrice A∈Rn×n est semi-définie positive et est notée A⪰0 si l'on a :

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Remarque : de manière similaire, une matrice A est dite définie positive et est notée A≻0 si elle est semi-définie positive et que pour tout vecteur x non-nul, on a xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Valeur propre, vecteur propre ― Étant donné une matrice A∈Rn×n, λ est une valeur propre de A s'il existe un vecteur z∈Rn∖{0}, appelé vecteur propre, tel que :

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Théorème spectral ― Soit A∈Rn×n. Si A est symétrique, alors A est diagonalisable par une matrice orthogonale réelle U∈Rn×n. En notant Λ=diag(λ1,...,λn), on a :

<br>

**46. diagonal**

&#10230; diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Décomposition en valeurs singulières ― Pour une matrice A de dimensions m×n, la décomposition en valeurs singulières est une technique de factorisation qui garantit l'existence d'une matrice unitaire U m×m, d'une matrice diagonale Σ m×n et d'une matrice unitaire V n×n, tel que :

<br>

**48. Matrix calculus**

&#10230; Calcul matriciel

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradient ― Soit f:Rm×n→R une fonction et A∈Rm×n une matrice. Le gradient de f par rapport à A est une matrice de taille m×n, notée ∇Af(A), telle que :

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Remarque : le gradient de f est seulement défini lorsque f est une fonction donnant un scalaire.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessienne ― Soit f:Rn→R une fonction et x∈Rn un vecteur. La hessienne de f par rapport à x est une matrice symétrique n×n, notée ∇2xf(x), telle que :

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Remarque : la hessienne de f est seulement définie lorsque f est une fonction qui donne un scalaire.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Opérations de gradient ― Pour des matrices A,B,C, les propriétés de gradient suivants sont bons à savoir :
