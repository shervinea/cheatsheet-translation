**Linear Algebra and Calculus translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)

<br>

**1. Linear Algebra and Calculus refresher**

&#10230; Памятка: Линейная Алгебра и Математический Анализ

<br>

**2. General notations**

&#10230; Общие обозначения

<br>

**3. Definitions**

&#10230; Определения

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Вектор ― Мы обозначаем x∈Rn вектор с n элементами, где xi∈R∈ i-й элемент:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Матрица ― Мы обозначаем A∈Rm×n матрица с m строками и n столбцами, где Ai,j∈R - запись, расположенная в i-й строке и j-м столбце:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Примечание: вектор x, определенный выше, можно рассматривать как матрицу 1×n и, в частности, называется вектор-столбец.

<br>

**7. Main matrices**

&#10230; Основные матрицы

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Матрица идентичности ― единичная матрица I∈Rn×n является квадратной матрицей с единицами на диагонали и нулем во всех остальных местах:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Примечание: для всех матриц A∈Rn×n имеем A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Диагональная матрица ― Диагональная матрица D∈Rn×n представляет собой квадратную матрицу с ненулевыми значениями на её диагонали и нулевыми везде:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Примечание: мы также отбозначаем D как diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Матричные операции

<br>

**13. Multiplication**

&#10230; Умножение

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Вектор-вектор ― существует два типа векторно-векторных произведений:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; внутреннее произведение: для x,y∈Rn, у нас есть:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; внешнее произведение: для x∈Rm,y∈Rn, у нас есть:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Матрица-вектор ― Произведение матрицы A∈Rm×n и вектора x∈Rn - это вектор размера Rn, такой что:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; где aTr,i - векторные строки, а ac,j - векторные столбцы A, а xi - элементы x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Матрица-матрица ― Произведение матриц A∈Rm×n и B∈Rn×p матрица размера Rn×p такая, что:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; где aTr,i,bTr,i - векторные строки и ac,j,bc,j - векторные столбцы A и B соответственно

<br>

**21. Other operations**

&#10230; Прочие операции

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Транспонирование ― Транспонирование матрицы A∈Rm×n, обозначается AT, таково, что её элементы переворачиваются:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Примечание: для матриц A,B имеем (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Обращение ― Обратная квадратная матрица A обозначается как A−1 и является единственной матрицей, такой что:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Примечание: не все квадратные матрицы обратимы. Также для матриц A,B, мы имеем (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; След (Trace) ― След квадратной матрицы A, обозначается tr(A), представляет собой сумму её диагональных элементов:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Примечание: для матриц A,B имеем tr(AT)=tr(A) и tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Определитель ― определитель квадратной матрицы A∈Rn×n, обозначается |A| или det(A) рекурсивно выражается через A∖i,∖j, которая является матрицей A без её i-й строки и j-го столбца, следующим образом:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Примечание: A обратима тогда и только тогда, когда |A|≠0. Также, |AB|=|A||B| и |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Свойства матрицы

<br>

**31. Definitions**

&#10230; Определения

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Симметричное разложение ― Данная матрица A может быть выражена в терминах её симметричной и антисимметричной частей следующим образом:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Симметричная, Антисимметричная]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Норма ― Норма - это функция N:V⟶[0,+∞[ где V - векторное пространство, и такая, что для всех x,y∈V, есть:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) для скаляра

<br>

**36. if N(x)=0, then x=0**

&#10230; если N(x)=0, тогда x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Для x∈V наиболее часто используемые нормы приведены в таблице ниже:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Норма, Обозначение, Определение, Вариант использования]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Линейная зависимость ― Набор векторов называется линейно зависимым, если один из векторов в наборе может быть определен как линейная комбинация других.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Примечание: если ни один вектор не может быть записан таким образом, то векторы называются линейно независимыми

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Ранг матрицы ― Ранг данной матрицы A обозначается rank(A) и является размерностью векторного пространства, порожденного его столбцами. Это эквивалентно максимальному количеству линейно независимых столбцов A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Положительная полуопределенная матрица ― Матрица A∈Rn×n является положительно полуопределенной (positive semi-definite, PSD) и обозначается как A⪰0, если у нас есть:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Примечание: аналогично, матрица A называется положительно определенной и обозначается как A≻0, если это матрица PSD, которая удовлетворяет всем ненулевым векторам x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Собственное значение, собственный вектор ― Для матрицы A∈Rn×n λ называется собственным значением A, если существует вектор z∈Rn∖{0}, называемый собственным вектором, такой, что у нас есть:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Спектральная теорема ― Пусть A∈Rn×n. Если A симметрична, то A диагонализуема действительной ортогональной матрицей U∈Rn×n. Обозначим Λ=diag(λ1,...,λn), у нас есть:

<br>

**46. diagonal**

&#10230; диагональ

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Сингулярное разложение ― Для данной матрицы A размеров m×n разложение по сингулярным числам (singular-value decomposition, SVD) представляет собой метод факторизации, который гарантирует существование унитарных матриц U m×m, диагональных Σ m×n и унитарных матриц V n×n, таких что:

<br>

**48. Matrix calculus**

&#10230; Матричное исчисление

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Градиент ― Пусть f:Rm×n→R - функция, а A∈Rm×n - матрица. Градиент f относительно A представляет собой матрицу размера m×n, отмеченную как ∇Af(A), такую, что:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Примечание: градиент f определяется только тогда, когда f - функция, возвращающая скаляр.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Гессиан ― Пусть f:Rn→R функция, а x∈Rn - вектор. Гессиан f относительно x является симметричной матрицей размера n×n, обозначенной как ∇2xf(x), такой что:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Примечание: гессиан функции f определяется только тогда, когда f является функцией, возвращающей скаляр

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Градиентные операции ― Для матриц A,B,C следует иметь в виду следующие свойства градиента:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Общие обозначения, Определения, Основные матрицы]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Матричные операции, Умножение, Прочие операции]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Свойства матрицы, Норма, Собственное значение/Собственный вектор, Сингулярное разложение]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Матричное исчисление, Градиент, Гессиан, Операции]

<br>

**58. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/shervinea

<br>

**59. Translated by X, Y and Z**

&#10230; Российская адаптация: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>

**60. Reviewed by X, Y and Z**

&#10230; Проверено X, Y и Z

<br>

**61. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>

**62. By X and Y**

&#10230; По X и Y

<br>
