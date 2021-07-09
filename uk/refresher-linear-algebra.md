**1. Linear Algebra and Calculus refresher**

&#10230; Повторення з лінійної алгебри та диференційного числення

<br>

**2. General notations**

&#10230; Загальні означення

<br>

**3. Definitions**

&#10230; Визначення

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Вектор - вектором x∈Rn з n елементів, де xi∈R є i-им елементом

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Матриця - матриця A∈Rm×n має n рядків і m стовпчиків, де Ai,j∈R є елементом на i-ому рядку і j-ому стовпчику.

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Примітка : вектор x, визначений вище, може бути розглянутий як матриця n×1 і називається вектор-стовпчиком.

<br>

**7. Main matrices**

&#10230; Найважливіші типи матриць

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Одинична матриця  ― одинична матриця I∈Rn×n з одиницями на головній діагоналі та нулями у всіх інших елементах:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Примітка: для всіх матриць A∈Rn×n, маємо A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Діагональна матриця ― діагональна матриця D∈Rn×n з ненульовими значеннями на головній діагоналі та нулями у всіх інших елементах.

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Примітка: D визначається як diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Матричні операції

<br>

**13. Multiplication**

&#10230; Множення

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Вектор-вектор - Існують два типи векторних добутків:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; Скалярний добуток: для x,y∈Rn, маємо :

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; Векторний добуток : для x∈Rm,y∈Rn, маємо :

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Матриця-вектор : добуток A∈Rm×n і вектора x∈Rn є вектор розміру Rn :

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; де aTr,i є вектор-рядками et ac,j є вектор-стовпчиками A та xi є елементами x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Матриця-матриця ― добуток матриць A∈Rm×n та B∈Rn×p є матриця розміру Rn×p :

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; де aTr,i,bTr,i є вектор-рядками та ac,j,bc,j є вектор-стовпчиками відповідно A і B.

<br>

**21. Other operations**

&#10230; Інші дії

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Транспонування ― транспонованою матрицею A∈Rm×n, визначеною AT, є матриця елементи якої є віддзеркаленими.

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Примітка : для матриць A, B, маємо (AB)T=BTAT.

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Обернення ― Обернення квадратної матриці A визначається A−1 і є єдиною матрицею що задовольняє наступне :

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Примітка : не всі квадратні матриці є оберненими. Також для матриць A,B, маємо (AB)−1=B−1A−1.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Слід матриці ― слід квадратної матриці A, визначений tr(A), є сумою її діагональних елементів:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Примітка : для матриць A, B, маємо tr(AT)=tr(A) та tr(AB)=tr(BA).

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Детермінант ― детермінант квадратної матриці A∈Rn×n визначеної |A| або det(A) виражений рекурсивно через A∖i,∖j, що є матрицею А без її і-го рядка і j-го стовпчика :

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Примітка : A може бути оберненою тільки якщо |A|≠0. Також, |AB|=|A||B| та |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Властивості матриць

<br>

**31. Definitions**

&#10230; Визначення

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Симетричний розклад матриці - дана матриця А може бути виражена в термінах своєї симетричної і антисиметричної частини наступним способом:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Симетрична, Антисиметрична]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Норма - це функція N:V⟶[0,+∞[ де V є векторним простором, таким що для для всіх x,y∈V, маємо :

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) для скаляру

<br>

**36. if N(x)=0, then x=0**

&#10230; якщо N(x)=0, тоді x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Для x∈V, найважливіші в уживанні норми вказано у наступній таблиці:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Норма, Нотація, Визначення, Спосіб вживання]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Лінійна залежність - набір векторів називається лінійно залежним якщо один з векторів в наборі може бути визначений через лінійну комбінацію інших.
<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Примітка: якщо жоден з векторів не може бути так визначений, тоді ветори називаються лінійно незалежними.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Ранг матриці ― ранг даної матриці A визначається rang(A) і є виміром векторного простору що заданий її рядками. Ранг є еквівалентом максимальної кількості лінійно незалежних стовпчиків в A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Додатноозначена матриця ― матриця A∈Rn×n є додатноозначеною і визначається A⪰0 якщо :

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Примітка: схожим чином, матриця є додатноозначеною і визначається A⪰0, якщо вона є додатноозначеною і для всіх ненульових векторів x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Власне значення, власний вектор ― маючи матрицю A∈Rn×n, λ називається власним значенням A якщо існує вектор z∈Rn∖{0}, названий власним вектором, таким що :

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Спектральна теорема ― нехай A∈Rn×n. Якщо A є симетричною, тоді A є діагоналізовною через ортогональну матрицю U∈Rn×n. Визначаючи Λ=diag(λ1,...,λn), маємо :

<br>

**46. diagonal**

&#10230; діагональ

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Сингулярний розклад матриці ― для даної матриці A з вимірами m×n, сингулярний розклад є технікою факторизації що гарантує існування U m×m, діагональної матриці Σ m×n та V n×n унітарної матриці, наступним чином :

<br>

**48. Matrix calculus**

&#10230; Матричне числення

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Градієнт ― нехай f:Rm×n→R буде функцією і A∈Rm×n буде матрицею. Градієнт f відносно A є матрицею m×n, визначеною ∇Af(A), такою що :

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Примітка: градієнт f є визначеним тільки коли f є функцією що повертає скаляр.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Матриця Гессе ― Нехай f:Rn→R буде функцією і x∈Rn буде вектором. Матриця Гессе f відносно x є симетричною матрицею n×n, визначеною ∇2xf(x), такою що :

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Примітка: матриця Гессе від f є визначеною тільки коли f є функцією що повертає скаляр.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Дії на градієнтах ― Для матриць A,B,C варто знати наступні властивості градієнтів :
