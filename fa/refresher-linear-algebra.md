**1. Linear Algebra and Calculus refresher**

یادآوری جبر خطی و حسابان

<br>

**2. General notations**

نمادها

<br>

**3. Definitions**

تعاریف

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

بردار - $x \in \mathbb{R}^n$ یک بردار با $n$ درایه است، که $x_i \in \mathbb{R}$ درایه‌ی $i$ام می‌باشد:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with n rows and m, where Ai,j∈R is the entry located in the ith row and jth column:**

ماتریس - $A \in \mathbb{R} ^ {m \times n}$ یک بردار با $n$ سطر و $m$ ستون است، که در آن $A_{i, j} \in \mathbb{R}$ درایه‌ای است که در سطر $i$ام و ستون $j$ام قرار دارد:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

نکته: بردار $x$ که در بالا تعریف شد را می‌توان به صورت یک ماتریس $n \times 1$ در نظر گرفت که به طور خاص به آن بردار ستونی گویند.

<br>

**7. Main matrices**

ماتریس‌های اصلی:

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

ماتریس همانی - ماتریس همانی $I \in \mathbb{R}^{n \times n}$ یک ماتریس مربعی است که درایه‌های قطری آن همه مقدار ۱ و بقیه‌ی درایه‌ها مقدار ۰ دارند:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

نکته: برای همه‌ی ماتریس‌های $A \in \mathbb{R}^{n \times n}$ داریم $A \times I = I \times A = A$.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

ماتریس قطری - ماتریس $D \in \mathbb{R} ^ {n \times n}$ یک ماتریس مربعی است که درایه‌های قطری آن مقادیر غیرصفر دارند و بقیه‌ی درایه‌ها صفر هستند:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

نکته:‌$D$ همچنین به صورت $\text{diag}(d_1, \dots, d_n)$ هم نمایش داده می‌شود.

<br>

**12. Matrix operations**

عملیات ماتریسی

<br>

**13. Multiplication**

ضرب

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

بردار با بردار - دو نوع عملیات ضرب بردار با بردار وجود دارد:

<br>

**15. inner product: for x,y∈Rn, we have:**

ضرب داخلی: برای هر $x, y \in \mathbb{R}^n$ داریم:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

ضرب خارجی: برای هر $x \in \mathbb{R}^m$ و $y \in \mathbb{R}^n$ داریم:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

ماتریس با بردار - ضرب ماتریس $A \in \mathbb{R}^{m \times n}$ و بردار $x \in \mathbb{R}^n$ برداری با اندازه‌ی $m$ است به طوری که:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

که $a^T_{r, i} بردارهای سطری و $a_{c, j}$ بردارهای ستونی $A$، و $x_i$ درایه‌های $x$ هستند.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

ماتریس با ماتریس - ضرب ماتریس‌های $A \in \mathbb{R}^{n \times m}$ و $B \in \mathbb{R}^{n \times p}$ ماتریسی با اندازه‌ی $n \times p$ است که:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

که $a^T_{r, i}$ و $b^T_{r, i}$ بردارهای سطری و $a_{c, j}$ و b_{c, j}$ بردارهای ستونی $A$ و $B$ هستند.

<br>

**21. Other operations**

دیگر عملیات

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

ترانهاده - ترانهاده‌ی ماتریس $A \in \mathbb{R}^{m \times n}$ که با $A^T$ نمایش داده می‌شود، ماتریسی است که مکان درایه‌های آن نسبت به قطر ماتریس برعکس شده‌اند:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

نکته: برای ماتریس‌های $A$ و $B$، داریم $(AB)^T = B^T A^T$.

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

معکوس - معکوس یک ماتریس مربعی معکوس‌پذیر $A$ که با $A^{-1}$ نمایش داده می‌شود، تنها ماتریسی است که:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

نکته: همه‌ی ماتریس‌های مربعی معکوس‌پذیر نیستند. همچنین، برای ماتریس‌های مربعی معکوس‌پذیر $A$ و $B$ داریم $(AB)^{-1} = B^{-1} A^{-1}$.

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

اثر - اثر ماتریس مربعی $A$ که با $tr(A)$ نمایش داده می‌شود، مجموع همه‌ی درایه‌های قطری ماتریس است.

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

نکته: برای ماتریس‌های $A$ و $B$ داریم $tr(A^T) = tr(A)$ و $tr(AB) = tr(BA)$.

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

دترمینان - دترمینان یک ماتریس مربعی $A \in \mathbb{R}^{n \times n}$ که با $|A|$ یا $\det(A)$ نمایش داده می‌شود، به صورت یک عبارت بازگشتی بر روی $A_{\\i, \\j}$، که ماتریس $A$ بدون سطر $i$-ام و ستون $j$-ام است، به صورت زیر تعریف می‌شود: 

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

نکته: $A$ معکوس‌پذیر است اگر و فقط اگر $|A| \neq 0$. همچنین $|A B| = |A| |B|$ و $|A^T| = |A|$.

<br>

**30. Matrix properties**

ویژگی‌های ماتریس‌ها

<br>

**31. Definitions**

تعاریف

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

تجزیه‌ی متقارن - یک ماتریس دلخواه $A$ را می‌توان با استفاده از اجزای متقارن و غیرمتقارن آن به صورت زیر نشان داد:

<br>

**33. [Symmetric, Antisymmetric]**

[متقارن، غیرمتقارن]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

نرم - نرم تابع $N: \mathbb{V} \rightarrow [0, +\infty[$ است که $V$ یک فضای برداری است، و به گونه‌ای است که برای هر $x, y \in \mathbb{V}$ داریم:

<br>

**35. N(ax)=|a|N(x) for a scalar**

$N(a x) = |a| N(x)$ برای عدد اسکالر 

<br>

**36. if N(x)=0, then x=0**

اگر $N(x) = $ باشد در این صورت $x = 0$

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

برای $x \in \mathbb{V}$، نرم‌هایی که بیشتر استفاده می‌شوند در جدول زیر آمده‌اند:

<br>

**38. [Norm, Notation, Definition, Use case]**

[نُرم، نماد، تعریف، کاربرد]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

وابستگی خطی - مجموعه‌ای از بردارها وابستگی خطی دارند اگر یکی از بردارهای مجموعه را بتوان به صورت ترکیب خطی دیگر بردارها تعریف کرد.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

نکته: اگر نتوان هیچ برداری را به این شکل تعریف کرد، در این صورت بردارها استقلال خطی دارند.

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

رتبه ماتریس - رتبه‌ی یک ماتریس $A$ که با $\text{rank}(A)$ نمایش داده می‌شود، تعداد ابعاد فضایی است که توسط ستون‌های آن ایجاد می‌شود. این مقدار برابر است با حداکثر تعداد ستون‌های $A$ که استقلال خطی داشته باشند.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

ماتریس مثبت نیمه‌معین - ماتریس $A \in \mathbb{R}^{n \times n}$ یک ماتریس مثبت نیمه‌معین است که با $A \succeq 0$ نمایش داده می‌شود اگر داشته باشیم:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

نکته: به طور مشابه، یک ماتریس $A$ مثبت معین است ($A \succ 0$)، اگر یک ماتریس مثبت نیمه‌معین باشد که برای هر بردار غیرصفر $x$ داشته باشیم $x^T A > 0$.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

مقدار ویژه، بردار ویژه - برای یک ماتریس $A \in \mathbb{R}^{n \times n}$، گوییم $\lambda$ یک مقدار ویژه ماتریس $A$ است اگر وجود داشته باشد بردار $z \in \mathbb{R}^n \\ \{0\}$، که یک بردار ویژه نام دارد، به طوری که:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

قضیه‌ی طیفی - فرض کنید $A \in \mathbb{R}^{n \times n}$ باشد. اگر $A$ متقارن باشد، در این صورت $A$ توسط یک ماتریس حقیقی متعامد $U \in \mathbb{R} ^{n \times n}$ قطری‌پذیر است. با نمایش $\Lambda = \diag(\lambda_1, \dots, \lambda_n)$ داریم:

<br>

**46. diagonal**

قطری

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

تجزیه‌ی مقدار منفرد - برای یک ماتریس $A$ با ابعاد $m \times n$، تجزیه‌ی مقدار منفرد یک تکنیک تقسیم‌بندی است که تضمین می‌کند یک ماتریس یکانی $U \in \mathbb{R}^{n \times n}$، یک ماتریس قطری $\Sigma \in \mathbb{R}^{m \times n}$، و یک ماتریس یکانی $V \in \mathbb{R}^{n \times n}$ وجود دارند، به طوری که:

<br>

**48. Matrix calculus**

حسابان ماتریسی

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

گرادیان - فرض کنید $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ یک تابع و $A \in \mathbb{R}^{m \times n}$ یک ماتریس باشد. گرادیان $f$ نسبت به $A$ یک ماتریس با ابعاد $m \times n$ است و با $\nabla_A f(A)$ نمایش داده می‌شود، به طوری که:

<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

نکته: گرادیان $f$ تنها زمانی تعریف شده است که $f$ تابعی باشد که یک عدد اسکالر خروجی دهد.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

هسیان - فرض کنید $f: \mathbb{R}^n \rightarrow \mathbb{R}$ یک تابع و $x \in \mathbb{R}^n$ یک بردار باشد. هسیان $f$ نسبت به $x$ یک ماتریس متقارن با ابعاد $n \times n$ است و با $\nabla^2_x f(x)$ نمایش داده می‌شود، به طوری که:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

نکته: هسیان تابع $f$ تنها زمانی تعریف شده است که $f$ تابعی با خروجی اسکالر باشد.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

عملیات گرادیانی - برای ماتریس‌های $A$، $B$، و $C$، ویژگی‌های زیر را به خاطر داشته باشید:
