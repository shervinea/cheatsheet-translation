**1. Linear Algebra and Calculus refresher**

&#10230; Doğrusal Cebir ve Kalkülüs hatırlatması

<br>

**2. General notations**

&#10230; Genel notasyonlar

<br>

**3. Definitions**

&#10230; Tanımlar

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vektör - i-inci elemanı xi∈R olmak üzere n elemanlı bir vektör, x∈Rn:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Matris ― Ai,j∈R i-inci satır ve j-inci sütundaki elemanları olmak üzere m satırlı ve n sütunlu bir matris, A∈Rm×n:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Uyarı: Yukarıda tanımlanan x vektörü n×1 tipinde bir matris olarak ele alınabilir ve genellikle sütun vektörü olarak adlandırılır.

<br>

**7. Main matrices**

&#10230; Ana matrisler

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Birim matris ― Birim matris, köşegeni birlerden ve diğer tüm elemanları sıfırlardan oluşan karesel matris, I∈Rn×n:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Uyarı: Her A∈Rn×n matrisi için A×I=I×A=A eşitliği sağlanır.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Köşegen matris ― Bir köşegen matris, köşegenindeki elemanları sıfırdan farklı diğer tüm elemanları sıfır olan karesel matris, D∈Rn×n:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Uyarı: D matrisi diag(d1,...,dn) olarak da gösterilir.

<br>

**12. Matrix operations**

&#10230; Matris işlemleri

<br>

**13. Multiplication**

&#10230; Çarpma

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vektör-vektör ― İki çeşit vektör-vektör çarpımı vardır.

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; iç çarpım: x,y∈Rn için:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; dış çarpım: x∈Rm,y∈Rn için:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Matris-vektör ― A∈Rm×n matrisi ve x∈Rn vektörünün çarpımları Rn boyutunda bir vektördür:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; burada aTr,i A'nın vektör satırları ve ac,j A'nın vektör sütunları ve xi x vektörünün elemanlarıdır.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Matris-matris ― A∈Rm×n matrisi ve B∈Rn×p matrisinin çarpımları Rn×p boyutunda bir matristir:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; burada aTr,i,bTr,i sırasıyla A ve B'nin vektör satırları ve ac,j,bc,j sırasıyla A ve B'nin vektör sütunlarıdır.

<br>

**21. Other operations**

&#10230; Diğer işlemler

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Devrik (Transpoze) ― Bir A∈Rm×n matrisinin devriği, satır ve sütunların yer değiştirmesi ile elde edilir, ve AT ile gösterilir:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Uyarı: Her A,B için (AB)T=BTAT vardır.

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Ters ― Tersinir bir A karesel matrisinin tersi, aşağıdaki koşulu sağlayan matristir, ve A-1 ile gösterilir:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Uyarı: Her karesel matris tersinir değildir. Ayrıca, Her tersinir A,B matrisi için (AB)−1=B−1A−1 dir.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; İz ― Bir A karesel matrisinin izi, köşegenindeki elemanlarının toplamıdır, ve tr(A) ile gösterilir: 

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Uyarı: A,B matrisleri için tr(AT)=tr(A) ve tr(AB)=tr(BA) vardır.

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Determinant ― A∈Rn×n matrisinin determinantı, A∖i,∖j gösterimi i-inci satırsız ve j-inci sütunsuz şekilde A matrisi olmak üzere özyinelemeli olarak aşağıdaki gibi ifade edilir, ve |A| ya da det(A) ile gösterilir:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230;

<br> Uyarı: A tersinirdir ancak ve ancak |A|≠0. Ayrıca, |AB|=|A||B| ve |AT|=|A|.

**30. Matrix properties**

&#10230; Matris özellikleri

<br>

**31. Definitions**

&#10230; Tanımlar

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Simetrik ayrışım ― Verilen bir A matrisi simetrik ve ters simetrik parçalarının cinsinden aşağıdaki gibi ifade edilebilir: 

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Simetrik, Ters simetrik]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Norm ― V vektör uzayı ve her x,y∈V için aşağıdaki özellikleri sağlayan N:V⟶[0,+∞[ fonksiyonu bir normdur:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; Bir a sabiti için N(ax)=|a|N(x).

<br>

**36. if N(x)=0, then x=0**

&#10230; N(x)=0 ise x=0.

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; x∈V için en yaygın şekilde kullanılan normlar aşağıdaki tabloda verilmektedir.

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Norm, Notasyon, Tanım, Kullanım]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Doğrusal bağımlılık ― Bir vektör kümesinden bir vektör diğer vektörlerin doğrusal birleşimi (kombinasyonu) cinsinden yazılabiliyorsa bu vektör kümesine doğrusal bağımlı denir.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Uyarı: Eğer bu şekilde yazılabilen herhangi bir vektör yoksa bu vektörlere doğrusal bağımsız denir.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Matris rankı  ― Verilen bir A matrisinin rankı, ran(A), bu matrisinin sütunları tarafından üretilen vektör uzayının boyutudur. Bu ifade A matrisinin doğrusal bağımsız sütunlarının maksimum sayısına denktir.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Pozitif yarı-tanımlı matris ― Aşağıdaki koşulu sağlayan bir A∈Rn×n matrisi pozitif yarı-tanımlıdır ve A⪰0 ile gösterilir:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Uyarı: Benzer olarak, pozitif yarı-tanımlı bir A matrisi sıfırdan farklı her x vektörü için xTAx>0 koşulunu sağlıyorsa A matrisine pozitif tanımlı denir ve A≻0 ile gösterilir.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Özdeğer, özvektör  ― Verilen bir A∈Rn×n için aşağıdaki gibi bir z∈Rn∖{0} vektörü var ise buna özvektör, λ sayısına da A matrisinin öz değeri denir.

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spektral teorem  ― A∈Rn×n olsun. Eğer A simetrik ise, A matrisi gerçel ortogonal  U∈Rn×n matrisi ile köşegenleştirilebilir. Λ=diag(λ1,...,λn) olmak üzere:

<br>

**46. diagonal**

&#10230; köşegen

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Tekil-değer ayrışımı  ― m×n tipindeki bir A matrisi için tekil-değer ayrışımı; m×m tipinde bir üniter U, m×n tipinde bir köşegen Σ ve n×n tipinde bir üniter V matrislerinin varlığını garanti eden bir parçalama tekniğidir.

<br>

**48. Matrix calculus**

&#10230; Matris kalkülüsü

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradiyent ― f:Rm×n→R bir fonksiyon ve A∈Rm×n bir matris olsun. f nin A ya göre gradiyenti m×n tipinde bir matristir, ve ∇Af(A) ile gösterilir:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Uyarı: f fonksiyonunun gradiyenti yalnızca f skaler döndüren bir fonksiyon ise tanımlıdır.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessian ― f:Rn→R bir fonksiyon ve x∈Rn bir vektör olsun. f fonksiyonun x vektörüne göre Hessian’ı  n×n tipinde bir simetrik matristir, ve ∇2xf(x) ile gösterilir:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Uyarı: f fonksiyonunun Hessian’ı yalnızca f skaler döndüren bir fonksiyon ise tanımlıdır.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Gradiyent işlemleri ― A,B,C matrisleri için aşağıdaki işlemlerin akılda bulunmasında fayda vardır:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Genel notasyonlar, Tanımlar, Ana matrisler]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Matris işlemleri, Çarpma, Diğer işlemler]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Matris özellikleri, Norm, Özdeğer/Özvektör, Tekil-değer ayrışımı]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Matris kalkülüsü, Gradiyent, Hessian, İşlemler]
