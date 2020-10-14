**1. Linear Algebra and Calculus refresher**

&#10230;Ulasan Aljabar Linier dan Kalkulus

<br>

**2. General notations**

&#10230;Notasi-notasi umum

<br>

**3. Definitions**

&#10230;Definisi-definisi

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230;Vektor ― Kita mendefinisikan x∈Rn sebagai sebuah vektor yang memiliki n elemen, dengan xi∈R adalah elemen ke-i:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230;Matriks ― Kita mendefinisikan A∈Rm×n sebagai sebuah matriks yang memiliki baris sebanyak m dan kolom sebanyak n, dengan Ai,j∈R adalah sebuah elemen yang berada pada baris ke-i dan kolom ke-j:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230;Catatan: vektor x yang didefinisikan di atas dapat juga dianggap sebagai matriks n×1 dan lebih khususnya disebut vektor dengan satu kolom.

<br>

**7. Main matrices**

&#10230;Matriks-matriks utama

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230;Matriks identitas ― Matriks identitas I∈Rn×n adalah matriks bujursangkar dengan elemen diagonal bernilai 1 dan sisanya bernilai 0.

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230;Catatan: untuk semua matriks A∈Rn×n, kita memiliki A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230;Matriks diagonal ― Matriks diagonal D∈Rn×n adalah sebuah matriks bujursangkar yang nilainya tidak berjumlah nol pada diagonalnya dan sisanya bernilai nol.

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230;Catatan: kita juga melambangkan D sebagai diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230;Operasi matriks

<br>

**13. Multiplication**

&#10230; Perkalian

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230;Vektor-vektor ― Terdapat dua tipe perkalian dari vektor-vektor:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230;perkalian dalam: untuk x,y∈Rn, kita memiliki:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230;outer product: untuk x∈Rm,y∈Rn, kita memiliki:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230;Matriks-vektor - Produk dari matriks A∈Rm×n dan vektor x∈Rn adalah sebuah vektor dengan ukuran Rn, seperti:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230;di mana aTr,i adalah baris vektor dan ac,j adalah kolom vektor dari matriks A, dan xi adalah elemen dari x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230;Matriks-matriks - Produk dari matriks A∈Rm×n dan matriks B∈Rn×p adalah sebuah matriks dengan ukuran Rn×p, seperti:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230;di mana aTr,i,bTr,i adalah baris vektor dan ac,j,bc,j adalah kolom vektor dari masing-masing matriks A dan B

<br>

**21. Other operations**

&#10230;Operasi lainnya

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230;Transpos ― Transpos matriks A∈Rm×n, yang dilambangkan dengan AT, adalah matriks yang elemennya terbalik:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230;Catatan: untuk matriks A,B, disimbolkan (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230;Invers ― Invers matriks bujursangkar terbalikkan A dilambangkan dengan A-1 dan itu hanya terdapat di matrix tersebut:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230;Catatan: tidak semua matriks bujursangkar adalah terbalikkan. Dan juga, untuk matriks A,B, dapat ditandai dengan (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230;Runutan ― Runutan matriks bujursangkar A, dilambangkan dengan tr(A), adalah jumlah dari entri diagonal tersebut:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230;Catatan: untuk matriks A,B, kita memiliki tr(AT)=tr(A) dan tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230;Determinan - Determinan matriks bujursangkar A∈Rn×n, yang dilambangkan dengan |A| atau det(A) dinyatakan secara rekursif dalam konteks A∖i,∖j, yang mana matriks A tanpa baris ke-i dan kolom ke-j, sebagai berikut:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230;Catatan: A adalah terbalikkan jika dan hanya jika |A|≠0. Serta, |AB|=|A||B| dan |AT|=|A|.

<br>

**30. Matrix properties**

&#10230;Sifat-sifat matriks

<br>

**31. Definitions**

&#10230;Definisi

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230;Dekomposisi simetrik ― Sebuah matriks A dapat dinyatakan dari segi simetrik dan antisimetrik ke dalam bentuk sebagai berikut:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230;[Simetrik, Antisimetrik]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230;Norma ― Norma adalah sebuah fungsi N:V⟶[0,+∞[ yang mana V adalah sebuah vektor ruang, dan begitu pun untuk semua x,y∈V, kita mendapatkan:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230;N(ax)=|a|N(x) untuk sebuah skalar

<br>

**36. if N(x)=0, then x=0**

&#10230;Jika N(x)=0, maka x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230;Untuk x∈V, norma yang paling sering digunakan, dijabarkan pada tabel di bawah ini:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230;[Norma, Notasi, Definisi, Use case]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230;Tak-Bebas Linier - Sebuah himpunan vektor disebut tak-bebas linier jika salah satu vektor dalam himpunan vektor tersebut dapat didefinisikan sebagai kombinasi linier dari vektor-vektor lainnya.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230;Catatan: jika tidak ada vektor yang dapat ditulis seperti ini, maka vektor-vektor tersebut dinyatakan independen linear.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230;Matriks berpangkat: Pangkat matriks A dinotasikan dengan rank(A) dan merupakan dimensi dari ruang vektor yang dihasilkan oleh kolom matriks tersebut. Hal tersebut setara dengan jumlah maksimum dari kolom independen linear dari matriks A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230;Matriks semi definit positif ― Sebuah matriks A∈Rn×n adalah semi definit positif (PSD) dan dinotasikan dengan A⪰0 jika kita memiliki:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230;Catatan: demikian pula, sebuah matriks A dikategorikan sebagai matriks definit positif, dan dinotasikan dengan A≻0, jika matriks tersebut adalah matriks PSD yang memenuhi ketentuan untuk seluruh vektor non-zero x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230;Eigennilai, eigenvektor - Untuk sebuah matriks A∈Rn×n, λ dikategorikan sebagai sebuah eigennilai dari A jika terdapat sebuah vektor z∈Rn∖{0}, disebut eigenvektor, sehingga kita mendapatkan:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230;Teorama spektral ― A∈Rn×n. Jika A adalah simetris, maka A terdiagonalkan oleh sebuah matriks ortogonal U∈Rn×n. Dengan menotasikan Λ=diag(λ1,...,λn), kita mendapatkan:

<br>

**46. diagonal**

&#10230;diagonal

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230;Dekomposisi nilai tunggal ― Untuk sebuah matriks berdimensi mxn, dekomposisi nilai tunggal (SVD) adalah sebuah teknik pemfaktoran yang menghasilkan sebuah matriks uniter U m×m, matriks diagonal Σ m×n dan matriks uniter V n×n, sehingga:

<br>

**48. Matrix calculus**

&#10230;Kalkulus matriks

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230;Gradien - Diketahui f:Rm×n→R sebagai sebuah fungsi dan A∈Rm×n sebagai sebuah matriks. Gradien f terhadap A adalah matriks berdimensi m×n, dinotasikan ∇Af(A), sehingga:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230;Catatan: gradien f hanya dapat didefinisikan ketika f adalah sebuah fungsi yang mengembalikan sebuah skalar.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230;Hesse ― Diketahui f:Rn→R adalah sebuah fungsi dan x∈Rn adalah sebuah vektor. Matriks Hesse dari f terhadap x adalah sebuah matriks simetris n×n, dinotasikan ∇2xf(x), sehingga:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230;Catatan: matriks Hesse dari f hanya dapat didefinisikan ketika f adalah sebuah fungsi yang mengembalikan sebuah skalar

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230;Operasi gradien - Untuk matriks A, B, C, properti-properti gradient berikut layak untuk diingat:

<br>

**54. [General notations, Definitions, Main matrices]**
Multiplication
&#10230;[Notasi umum, Definisi, Matriks-matriks Utama]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230;[Sifat-sifat Matriks, Norm, Nilai eigen/vektor eigen, Dekomposisi nilai-singular]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230;[Properti matriks, Norm, Eigenvalue/Eigenvektor, Dekomposisi singular-value]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230;[Kalkulus matriks, Gradien, Hesse, Operasi]
