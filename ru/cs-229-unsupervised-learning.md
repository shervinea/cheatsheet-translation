**Unsupervised Learning translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning)

<br>

**1. Unsupervised Learning cheatsheet**

&#10230; Шпаргалка по обучению без учителя

<br>

**2. Introduction to Unsupervised Learning**

&#10230; Введение в обучение без учителя

<br>

**3. Motivation ― The goal of unsupervised learning is to find hidden patterns in unlabeled data {x(1),...,x(m)}.**

&#10230; Мотивация ― цель обучения без учителя - найти скрытые закономерности в неразмеченных данных {x(1),...,x(m)}.

<br>

**4. Jensen's inequality ― Let f be a convex function and X a random variable. We have the following inequality:**

&#10230; Неравенство Йенсена ― Пусть f - выпуклая функция, а X - случайная величина. E - математическое ожидание. Имеем следующее неравенство:

<br>

**5. Clustering**

&#10230; Кластеризация

<br>

**6. Expectation-Maximization**

&#10230; Максимизация Ожидания

<br>

**7. Latent variables ― Latent variables are hidden/unobserved variables that make estimation problems difficult, and are often denoted z. Here are the most common settings where there are latent variables:**

&#10230; Скрытые величины ― это скрытые/ненаблюдаемые величины, которые затрудняют задачи оценки, и часто обозначаются буквой z. Вот наиболее распространенные настройки, в которых присутствуют скрытые величины:

<br>

**8. [Setting, Latent variable z, Comments]**

&#10230; [Настройка, Скрытая величина z, Комментарии]

<br>

**9. [Mixture of k Gaussians, Factor analysis]**

&#10230; [Смесь k гауссианов, Факторный анализ]

<br>

**10. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; Алгоритм ― Алгоритм ожидания-максимизации (Expectation-Maximization, EM) дает эффективный метод оценки параметра θ посредством оценки максимального правдоподобия путем многократного построения нижней границы правдоподобия (E-шаг) и оптимизации этой нижней границы (M-шаг) следующим образом:

<br>

**11. E-step: Evaluate the posterior probability Qi(z(i)) that each data point x(i) came from a particular cluster z(i) as follows:**

&#10230; E-шаг: Оценить апостериорную вероятность Qi(z(i)) того, что каждая точка данных x(i) пришла из определенного кластера z(i) следующим образом:

<br>

**12. M-step: Use the posterior probabilities Qi(z(i)) as cluster specific weights on data points x(i) to separately re-estimate each cluster model as follows:**

&#10230; M-шаг: Использовать апостериорные вероятности Qi(z(i)) в качестве весовых коэффициентов для конкретных кластеров точек данных x(i), чтобы отдельно переоценить каждую модель кластера следующим образом:

<br>

**13. [Gaussians initialization, Expectation step, Maximization step, Convergence]**

&#10230; [Гауссовская инициализация, Шаг ожидания, Шаг максимизации, Сходимость]

<br>

**14. k-means clustering**

&#10230; Метод k-средних

<br>

**15. We note c(i) the cluster of data point i and μj the center of cluster j.**

&#10230; Мы обозначаем c(i) кластер точки данных i и μj центр кластера j.

<br>

**16. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Алгоритм ― после случайной инициализации центроидов кластера μ1,μ2,...,μk∈Rn алгоритм k-средних повторяет следующий шаг до сходимости:

<br>

**17. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Инициализация средних, Назначение кластера, Обновление средних, Сходимость]

<br>

**18. Distortion function ― In order to see if the algorithm converges, we look at the distortion function defined as follows:**

&#10230; Функция искажения ― Чтобы увидеть, сходится ли алгоритм, мы смотрим на функцию искажения, определенную следующим образом:

<br>

**19. Hierarchical clustering**

&#10230; Иерархическая кластеризация

<br>

**20. Algorithm ― It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.**

&#10230; Алгоритм ― Это алгоритм кластеризации с агломеративным иерархическим подходом, который последовательно создает вложенные кластеры.

<br>

**21. Types ― There are different sorts of hierarchical clustering algorithms that aims at optimizing different objective functions, which is summed up in the table below:**

&#10230; Типы ― Существуют различные виды алгоритмов иерархической кластеризации, которые направлены на оптимизацию различных целевых функций, которые приведены в таблице ниже:

<br>

**22. [Ward linkage, Average linkage, Complete linkage]**

&#10230; [Связь Уорда, Средняя связь, Полная связь]

<br>

**23. [Minimize within cluster distance, Minimize average distance between cluster pairs, Minimize maximum distance of between cluster pairs]**

&#10230; [Минимизирует расстояние в пределах кластера, Минимизирует среднее расстояние между парами кластеров, Минимизирует максимальное расстояние между парами кластеров]

<br>

**24. Clustering assessment metrics**

&#10230; Кластеризация показателей оценки

<br>

**25. In an unsupervised learning setting, it is often hard to assess the performance of a model since we don't have the ground truth labels as was the case in the supervised learning setting.**

&#10230; В условиях обучения без учителя часто бывает трудно оценить производительность модели, поскольку у нас нет основных меток истинности, как это было в условиях обучения с учителем.

<br>

**26. Silhouette coefficient ― By noting a and b the mean distance between a sample and all other points in the same class, and between a sample and all other points in the next nearest cluster, the silhouette coefficient s for a single sample is defined as follows:**

&#10230; Коэффициент силуэта ― Обозначим a и b среднее расстояние между образцом и всеми другими точками в том же классе, а также между образцом и всеми другими точками в следующем ближайшем кластере, коэффициент силуэта s для одного образца определяется следующим образом:

<br>

**27. Calinski-Harabaz index ― By noting k the number of clusters, Bk and Wk the between and within-clustering dispersion matrices respectively defined as**

&#10230; Индекс Калински-Харабаза ― Обозначим k количество кластеров, Bk и Wk матрицы дисперсии между кластерами и внутри кластеров, соответственно определяемые как

<br>

**28. the Calinski-Harabaz index s(k) indicates how well a clustering model defines its clusters, such that the higher the score, the more dense and well separated the clusters are. It is defined as follows:**

&#10230; индекс Калински-Харабаза s(k) показывает, насколько хорошо модель кластеризации определяет свои кластеры, так что чем выше оценка, тем более плотными и хорошо разделенными являются кластеры. Это определяется следующим образом:

<br>

**29. Dimension reduction**

&#10230; Уменьшение размерности

<br>

**30. Principal component analysis**

&#10230; Метод главных компонент

<br>

**31. It is a dimension reduction technique that finds the variance maximizing directions onto which to project the data.**

&#10230; Это метод уменьшения размерности, который находит направления максимизации дисперсии для проецирования данных.

<br>

**32. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Собственное значение, собственный вектор ― Для матрицы A∈Rn×n λ называется собственным значением A, если существует вектор z∈Rn∖{0}, называемый собственным вектором, так что у нас есть:

<br>

**33. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Спектральная теорема ― Пусть A∈Rn×n. Если A симметрична, то A диагонализуема вещественной ортогональной матрицей U∈Rn×n. Обозначим Λ=diag(λ1,...,λn), у нас есть:

<br>

**34. diagonal**

&#10230; диагональ

<br>

**35. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Примечание: собственный вектор, связанный с наибольшим собственным значением, называется главным собственным вектором матрицы A.

<br>

**36. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Алгоритм ― Principal Component Analysis (PCA) - это метод уменьшения размерности, который проецирует данные на k измерений, максимизируя дисперсию данных следующим образом:

<br>

**37. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Шаг 1: Нормализовать данные, чтобы получить среднее значение 0 и стандартное отклонение 1.

<br>

**38. Step 2: Compute Σ=1mm∑i=1x(i)x(i)T∈Rn×n, which is symmetric with real eigenvalues.**

&#10230; Шаг 2: Вычислить Σ=1mm∑i=1x(i)x(i)T∈Rn×n, которое является симметричным с действительными собственными значениями.

<br>

**39. Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues.**

&#10230; Шаг 3: Вычислить u1,...,uk∈Rn k ортогональных главных собственных векторов матрицы Σ, то есть ортогональных собственных векторов k наибольших собственных значений.

<br>

**40. Step 4: Project the data on spanR(u1,...,uk).**

&#10230; Шаг 4: Спроецировать данные на spanR(u1,...,uk).

<br>

**41. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Эта процедура максимизирует дисперсию всех k-мерных пространств.

<br>

**42. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Данные в пространстве функций, Поиск главных компонент, Данные в пространстве главных компонент]

<br>

**43. Independent component analysis**

&#10230; Метод независимых компонент

<br>

**44. It is a technique meant to find the underlying generating sources.**

&#10230; Это метод, предназначенный для поиска основных источников генерации.

<br>

**45. Assumptions ― We assume that our data x has been generated by the n-dimensional source vector s=(s1,...,sn), where si are independent random variables, via a mixing and non-singular matrix A as follows:**

&#10230; Предположения ― Мы предполагаем, что наши данные x были сгенерированы n-мерным исходным вектором s=(s1,...,sn), где si - независимые случайные величины, посредством смешивающей и невырожденной матрицы A следующим образом:

<br>

**46. The goal is to find the unmixing matrix W=A−1.**

&#10230; Цель состоит в том, чтобы найти матрицу разложения W=A−1.

<br>

**47. Bell and Sejnowski ICA algorithm ― This algorithm finds the unmixing matrix W by following the steps below:**

&#10230; Алгоритм анализа независимых компонент Белла и Сейновского ― Bell and Sejnowski Independent Component Analysis, ICA - Этот алгоритм находит матрицу разложения W, выполнив следующие шаги:

<br>

**48. Write the probability of x=As=W−1s as:**

&#10230; Записать вероятность x=As=W−1s как:

<br>

**49. Write the log likelihood given our training data {x(i),i∈[[1,m]]} and by noting g the sigmoid function as:**

&#10230; Записать логарифмическое правдоподобие с учетом наших обучающих данных {x(i),i∈[[1,m]]} и обозначенной g сигмоидальной функции как:

<br>

**50. Therefore, the stochastic gradient ascent learning rule is such that for each training example x(i), we update W as follows:**

&#10230; Следовательно, правило обучения стохастическому градиентному восхождению таково, что для каждого обучающего примера x(i)мы обновляем W следующим образом:

<br>

**51. The Machine Learning cheatsheets are now available in [target language].**

&#10230; Шпаргалки по машинному обучению теперь доступны в формате [target language].

<br>

**52. Original authors**

&#10230; Авторы оригинала

<br>

**53. Translated by X, Y and Z**

&#10230; Переведено X, Y и Z

<br>

**54. Reviewed by X, Y and Z**

&#10230; Проверено X, Y и Z

<br>

**55. [Introduction, Motivation, Jensen's inequality]**

&#10230; [Введение, Мотивация, Неравенство Йенсена]

<br>

**56. [Clustering, Expectation-Maximization, k-means, Hierarchical clustering, Metrics]**

&#10230; [Кластеризация, Максимизация Ожидания, k-средние, Иерархическая кластеризация, Метрики]

<br>

**57. [Dimension reduction, PCA, ICA]**

&#10230; [Уменьшение размерности, PCA, ICA]
