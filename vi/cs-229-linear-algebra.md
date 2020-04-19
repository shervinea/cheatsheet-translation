**Linear Algebra and Calculus translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)

<br>

**1. Linear Algebra and Calculus refresher**

&#10230; Đại số tuyến tính và Giải tích cơ bản

<br>

**2. General notations**

&#10230; Kí hiệu chung

<br>

**3. Definitions**

&#10230; Định nghĩa

<br>

**4. Vector ― We note x∈Rn a vector with n entries, where xi∈R is the ith entry:**

&#10230; Vector - Chúng ta kí hiệu x∈Rn là một vector với n phần tử, với xi∈R là phần tử thứ i:

<br>

**5. Matrix ― We note A∈Rm×n a matrix with m rows and n columns, where Ai,j∈R is the entry located in the ith row and jth column:**

&#10230; Ma trận - Kí hiệu A∈Rm×n là một ma trận với m hàng và n cột, Ai,j∈R là phần tử nằm ở hàng thứ i, cột j:

<br>

**6. Remark: the vector x defined above can be viewed as a n×1 matrix and is more particularly called a column-vector.**

&#10230; Ghi chú: Vector x được xác định ở trên có thể coi như một ma trận nx1 và được gọi là vector cột.

<br>

**7. Main matrices**

&#10230; Ma trận chính

<br>

**8. Identity matrix ― The identity matrix I∈Rn×n is a square matrix with ones in its diagonal and zero everywhere else:**

&#10230; Ma trận đơn vị - Ma trận đơn vị I∈Rn×n là một ma trận vuông với các phần tử trên đường chéo chính bằng 1 và các phần tử còn lại bằng 0:

<br>

**9. Remark: for all matrices A∈Rn×n, we have A×I=I×A=A.**

&#10230; Ghi chú: với mọi ma trận vuông A∈Rn×n, ta có A×I=I×A=A.

<br>

**10. Diagonal matrix ― A diagonal matrix D∈Rn×n is a square matrix with nonzero values in its diagonal and zero everywhere else:**

&#10230; Ma trận đường chéo - Ma trận đường chéo D∈Rn×n là một ma trận vuông với các phần tử trên đường chéo chính khác 0 và các phần tử còn lại bằng 0:

<br>

**11. Remark: we also note D as diag(d1,...,dn).**

&#10230; Ghi chú: Chúng ta kí hiệu D là diag(d1,...,dn).

<br>

**12. Matrix operations**

&#10230; Các phép toán ma trận

<br>

**13. Multiplication**

&#10230; Phép nhân

<br>

**14. Vector-vector ― There are two types of vector-vector products:**

&#10230; Vector-vector ― Có hai loại phép nhân vector-vector:

<br>

**15. inner product: for x,y∈Rn, we have:**

&#10230; Phép nhân inner: với x,y∈Rn, ta có:

<br>

**16. outer product: for x∈Rm,y∈Rn, we have:**

&#10230; Phép nhân outer: với x∈Rm,y∈Rn, ta có:

<br>

**17. Matrix-vector ― The product of matrix A∈Rm×n and vector x∈Rn is a vector of size Rn, such that:**

&#10230; Ma trận - Vector ― Phép nhân giữa ma trận A∈Rm×n và vector x∈Rn là một vector có kích thước Rn:

<br>

**18. where aTr,i are the vector rows and ac,j are the vector columns of A, and xi are the entries of x.**

&#10230; với aTr,i là các vector hàng và ac,j là các vector cột của A, và xi là các phần tử của x.

<br>

**19. Matrix-matrix ― The product of matrices A∈Rm×n and B∈Rn×p is a matrix of size Rn×p, such that:**

&#10230; Ma trận - ma trận ― Phép nhân giữa ma trận A∈Rm×n và B∈Rn×p là một ma trận kích thước Rn×p:

<br>

**20. where aTr,i,bTr,i are the vector rows and ac,j,bc,j are the vector columns of A and B respectively**

&#10230; với aTr,i,bTr,i là các vector hàng và ac,j,bc,j lần lượt là các vector cột của A and B.

<br>

**21. Other operations**

&#10230; Một số phép toán khác

<br>

**22. Transpose ― The transpose of a matrix A∈Rm×n, noted AT, is such that its entries are flipped:**

&#10230; Chuyển vị ― Chuyển vị của một ma trận A∈Rm×n, kí hiệu AT, khi các phần tử hàng cột hoán đổi vị trí cho nhau:

<br>

**23. Remark: for matrices A,B, we have (AB)T=BTAT**

&#10230; Ghi chú: với ma trận A,B, ta có (AB)T=BTAT

<br>

**24. Inverse ― The inverse of an invertible square matrix A is noted A−1 and is the only matrix such that:**

&#10230; Nghịch đảo ― Nghịch đảo của ma trận vuông khả đảo A được kí hiệu là A-1 và chỉ tồn tại duy nhất:

<br>

**25. Remark: not all square matrices are invertible. Also, for matrices A,B, we have (AB)−1=B−1A−1**

&#10230; Ghi chú: không phải tất cả các ma trận vuông đều khả đảo. Ngoài ra, với ma trận A,B, ta có (AB)−1=B−1A−1

<br>

**26. Trace ― The trace of a square matrix A, noted tr(A), is the sum of its diagonal entries:**

&#10230; Truy vết ― Truy vết của ma trận vuông A, kí hiệu tr(A), là tổng của các phần tử trên đường chéo chính của nó:

<br>

**27. Remark: for matrices A,B, we have tr(AT)=tr(A) and tr(AB)=tr(BA)**

&#10230; Ghi chú: với ma trận A,B, chúng ta có tr(AT)=tr(A) và tr(AB)=tr(BA)

<br>

**28. Determinant ― The determinant of a square matrix A∈Rn×n, noted |A| or det(A) is expressed recursively in terms of A∖i,∖j, which is the matrix A without its ith row and jth column, as follows:**

&#10230; Định thức ― Định thức của một ma trận vuông A∈Rn×n, kí hiệu |A| hay det(A) được tính đệ quy với A∖i,∖j, ma trận A xóa đi hàng thứ i và cột thứ j:

<br>

**29. Remark: A is invertible if and only if |A|≠0. Also, |AB|=|A||B| and |AT|=|A|.**

&#10230; Ghi chú: A khả đảo nếu và chỉ nếu |A|≠0. Ngoài ra, |AB|=|A||B| và |AT|=|A|.

<br>

**30. Matrix properties**

&#10230; Những tính chất của ma trận

<br>

**31. Definitions**

&#10230; Định nghĩa

<br>

**32. Symmetric decomposition ― A given matrix A can be expressed in terms of its symmetric and antisymmetric parts as follows:**

&#10230; Phân rã đối xứng - Một ma trận A đã cho có thể được biểu diễn dưới dạng các phần đối xứng và phản đối xứng của nó như sau:

<br>

**33. [Symmetric, Antisymmetric]**

&#10230; [Đối xứng, Phản đối xứng]

<br>

**34. Norm ― A norm is a function N:V⟶[0,+∞[ where V is a vector space, and such that for all x,y∈V, we have:**

&#10230; Chuẩn (norm) ― Một chuẩn (norm) là một hàm N:V⟶[0,+∞[ mà V là một không gian vector, và với mọi x,y∈V, ta có:

<br>

**35. N(ax)=|a|N(x) for a scalar**

&#10230; N(ax)=|a|N(x) với a là một số

<br>

**36. if N(x)=0, then x=0**

&#10230; nếu N(x)=0, thì x=0

<br>

**37. For x∈V, the most commonly used norms are summed up in the table below:**

&#10230; Với x∈V, các chuẩn thường dùng được tổng hợp ở bảng dưới đây:

<br>

**38. [Norm, Notation, Definition, Use case]**

&#10230; [Chuẩn, Kí hiệu, Định nghĩa, Trường hợp dùng]

<br>

**39. Linearly dependence ― A set of vectors is said to be linearly dependent if one of the vectors in the set can be defined as a linear combination of the others.**

&#10230; Sự phụ thuộc tuyến tính―- Một tập hợp các vectơ được cho là phụ thuộc tuyến tính nếu một trong các vectơ trong tập hợp có thể được biểu diễn bởi một tổ hợp tuyến tính của các vectơ khác.

<br>

**40. Remark: if no vector can be written this way, then the vectors are said to be linearly independent**

&#10230; Ghi chú: nếu không có vectơ nào có thể được viết theo cách này, thì các vectơ được cho là độc lập tuyến tính

<br>

**41. Matrix rank ― The rank of a given matrix A is noted rank(A) and is the dimension of the vector space generated by its columns. This is equivalent to the maximum number of linearly independent columns of A.**

&#10230; Hạng ma trận (rank) ― Hạng của một ma trận A kí hiệu rank(A) và là số chiều của không gian vectơ được tạo bởi các cột của nó. Điều này tương đương với số cột độc lập tuyến tính tối đa của A.

<br>

**42. Positive semi-definite matrix ― A matrix A∈Rn×n is positive semi-definite (PSD) and is noted A⪰0 if we have:**

&#10230; Ma trận bán xác định dương - Ma trận A∈Rn×n là bán xác định dương (PSD) kí hiệu A⪰0 nếu chúng ta có:

<br>

**43. Remark: similarly, a matrix A is said to be positive definite, and is noted A≻0, if it is a PSD matrix which satisfies for all non-zero vector x, xTAx>0.**

&#10230; Ghi chú: tương tự, một ma trận A được cho là xác định dương và được kí hiệu A≻0, nếu đó là ma trận PSD thỏa mãn cho tất cả các vectơ khác không x, xTAx>0.

<br>

**44. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Giá trị riêng, vector riêng - Cho ma trận A∈Rn×n, λ được gọi là giá trị riêng của A nếu tồn tại một vectơ z∈Rn∖{0}, được gọi là vector riêng, sao cho:

<br>

**45. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Định lý phổ - Cho A∈Rn×n. Nếu A đối xứng, thì A có thể chéo hóa bởi một ma trận trực giao thực U∈Rn×n. Bằng cách kí hiệu Λ=diag(1,...,n), chúng ta có:

<br>

**46. diagonal**

&#10230; đường chéo

<br>

**47. Singular-value decomposition ― For a given matrix A of dimensions m×n, the singular-value decomposition (SVD) is a factorization technique that guarantees the existence of U m×m unitary, Σ m×n diagonal and V n×n unitary matrices, such that:**

&#10230; Phân tích giá trị suy biến - Đối với một ma trận A có kích thước m×n, Phân tích giá trị suy biến (SVD) là một kỹ thuật phân tích nhân tố nhằm đảm bảo sự tồn tại của đơn vị U m×m, đường chéo Σm×n và đơn vị V n×n ma trận, sao cho:

<br>

**48. Matrix calculus**

&#10230; Giải tích ma trận

<br>

**49. Gradient ― Let f:Rm×n→R be a function and A∈Rm×n be a matrix. The gradient of f with respect to A is a m×n matrix, noted ∇Af(A), such that:**

&#10230; Gradient ― Cho f:Rm×n→R là một hàm và A∈Rm×n là một ma trận. Gradient của f đối với A là ma trận m×n, được kí hiệu là ∇Af(A), sao cho:



<br>

**50. Remark: the gradient of f is only defined when f is a function that returns a scalar.**

&#10230; Ghi chú: gradient của f chỉ được xác định khi f là hàm trả về một số.

<br>

**51. Hessian ― Let f:Rn→R be a function and x∈Rn be a vector. The hessian of f with respect to x is a n×n symmetric matrix, noted ∇2xf(x), such that:**

&#10230; Hessian - Cho f:Rn→R là một hàm và x∈Rn là một vector. Hessian của f đối với x là một ma trận đối xứng n×n, ghi chú ∇2xf(x), sao cho:

<br>

**52. Remark: the hessian of f is only defined when f is a function that returns a scalar**

&#10230; Ghi chú: hessian của f chỉ được xác định khi f là hàm trả về một số.

<br>

**53. Gradient operations ― For matrices A,B,C, the following gradient properties are worth having in mind:**

&#10230; Các phép toán của gradient ― Đối với ma trận A,B,C, các thuộc tính gradient sau cần để lưu ý:

<br>

**54. [General notations, Definitions, Main matrices]**

&#10230; [Kí hiệu chung, Định nghĩa, Ma trận chính]

<br>

**55. [Matrix operations, Multiplication, Other operations]**

&#10230; [Phép toán ma trận, Phép nhân, Các phép toán khác]

<br>

**56. [Matrix properties, Norm, Eigenvalue/Eigenvector, Singular-value decomposition]**

&#10230; [Các thuộc tính ma trận, Chuẩn, Giá trị riêng/Vector riêng, Phân tích giá trị suy biến]

<br>

**57. [Matrix calculus, Gradient, Hessian, Operations]**

&#10230; [Giải tích ma trận, Gradient, Hessian, Phép tính]
