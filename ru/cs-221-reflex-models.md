**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**

&#10230; Модели машинного обучения на основе рефлексов

<br>


**2. Linear predictors**

&#10230; Линейные предсказатели

<br>


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

&#10230; В этом разделе мы рассмотрим модели, основанные на рефлексах, которые улучшаются по мере накопления опыта, обучаясь на парах наблюдений вход-выход.

<br>


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

&#10230; Вектор признаков ― Вектор признаков входного сигнала x обозначается как ϕ(x) и таков, что:

<br>


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

&#10230; Оценка ― Оценка s(x,w) примера (ϕ(x),y) ∈Rd×R, связанного с линейной моделью весов w∈Rd, дается внутренним произведением:

<br>


**6. Classification**

&#10230; Классификация

<br>


**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

&#10230; Линейный классификатор ― Для вектора весов w∈Rd и вектора признаков ϕ(x)∈Rd бинарный линейный классификатор fw имеет вид:

<br>


**8. if**

&#10230; если

<br>


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

&#10230; Отступ ― Отступ m(x,y,w)∈R примера (ϕ(x),y)∈Rd×{−1,+1}, связанная с линейной моделью весов w∈Rd, количественно оценивает уверенность прогноз: большие значения лучше. Задается:

<br>


**10. Regression**

&#10230; Регрессия

<br>


**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**

&#10230; Линейная регрессия ― Для данного вектора весов w∈Rd и вектора признаков ϕ(x)∈Rd результат линейной регрессии весов w, обозначенный как fw, определяется выражением:

<br>


**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**

&#10230; Разность ― Разность res(x,y,w)∈R определяется как величина, на которую прогноз fw(x) превышает целевой y:

<br>


**13. Loss minimization**

&#10230; Минимизация потерь

<br>


**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**

&#10230; Функция потерь ― Функция потерь Loss(x,y,w) количественно определяет, насколько мы недовольны весами w модели в задаче прогнозирования выхода y на основе входа x. Это количество, которое мы хотим минимизировать во время процесса обучения.

<br>


**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**

&#10230; Случай классификации ― Классификация выборки x истинной метки y∈{−1,+1} с линейной моделью весов w может быть выполнена с помощью предиктора fw(x)≜sign(s(x,w)). В этой ситуации интересующий показатель, определяющий качество классификации, задается зазором m(x,y,w) и может использоваться со следующими функциями потерь:

<br>


**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**

&#10230; [Название, Иллюстрация, Zero-one loss, Hinge loss, Logistic loss]

<br>


**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**

&#10230; Случай регрессии ― Предсказание выборки x истинной метки y∈R с помощью линейной модели весов w может быть выполнено с помощью предиктора fw(x)≜s(x,w). В этой ситуации интересующий показатель, количественно оценивающий качество регрессии, задается отступом res(x,y,w) и может использоваться со следующими функциями потерь:

<br>


**18. [Name, Squared loss, Absolute deviation loss, Illustration]**

&#10230; [Название, Квадратичная потеря, Абсолютное отклонение, Иллюстрация]

<br>


**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**

&#10230; Фреймворк минимизации потерь ― чтобы обучить модель, мы хотим минимизировать потери при обучении, которые определяются следующим образом:

<br>


**20. Non-linear predictors**

&#10230; Нелинейные предсказатели

<br>


**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-ближайшие соседи ― Алгоритм k-ближайших соседей, широко известный как k-NN, представляет собой непараметрический подход, в котором ответ для точки данных определяется характером её k соседей из обучающего набора. Его можно использовать в настройках как классификации, так и регрессии.

<br>


**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Примечание: чем выше параметр k, тем выше смещение, а чем ниже параметр k, тем выше дисперсия.

<br>


**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Нейронные сети ― Нейронные сети - это класс моделей, построенных с использованием слоёв. Обычно используемые типы нейронных сетей включают сверточные и рекуррентные нейронные сети. Словарь архитектур нейронных сетей представлен на рисунке ниже:

<br>


**24. [Input layer, Hidden layer, Output layer]**

&#10230; [Входной слой, Скрытый слой, Выходной слой]

<br>


**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Обозначим i - это i-й уровень сети, а j - j-й скрытый блок слоя, у нас есть:

<br>


**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**

&#10230; где мы обозначаем w, b, x, z вес, смещение, вход и неактивированный выход нейрона соответственно.

<br>


**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**

&#10230; Для более подробного обзора приведенных выше концепций ознакомьтесь со шпаргалками по контролируемому обучению!

<br>


**28. Stochastic gradient descent**

&#10230; Стохастический градиентный спуск

<br>


**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**

&#10230; Градиентный спуск ― Gradient descent - Обозначим η∈R скорость обучения (также называемую размером шага), правило обновления для градиентного спуска выражается с помощью скорости обучения и функции потерь Loss(x,y,w) следующим образом:

<br>


**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**

&#10230; Стохастические обновления ― Stochastic gradient descent (SGD) обновляет параметры модели по одному обучающему примеру (ϕ(x),y)∈Dtrain за раз. Этот метод приводит к иногда шумным, но быстрым обновлениям.

<br>


**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**

&#10230; Пакетные обновления ― Batch gradient descent (BGD) обновляет параметры модели по одной партии примеров (например, половина обучающего набора) за раз. Этот метод вычисляет стабильные направления обновления с большими вычислительными затратами.

<br>


**32. Fine-tuning models**

&#10230; Доводка моделей

<br>


**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**

&#10230; Класс гипотез ― Класс гипотез F - это набор возможных предикторов с фиксированным ϕ(x) и изменяющимся w:

<br>


**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**

&#10230; Логистическая функция ― Логистическая функция σ, также называемая сигмовидной функцией, определяется как:

<br>


**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**

&#10230; Примечание: у нас есть σ′(z)=σ(z)(1−σ(z)).

<br>


**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**

&#10230; Обратное распространение ― Backpropagation - Прямой проход выполняется через fi, которое является значением подвыражения с индексом i, а обратный проход выполняется через gi=∂out∂fi и представляет, как fi влияет на вывод.

<br>


**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**

&#10230; Ошибка аппроксимации и оценки ― Ошибка аппроксимации ϵapprox представляет, как далеко весь класс гипотез F от целевого предиктора g∗, в то время как ошибка оценки ϵest количественно определяет, насколько хорош предиктор ^f по отношению к лучшему предиктору f∗ из класса гипотез F.

<br>


**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Регуляризация ― Процедура регуляризации направлена на то, чтобы модель не переобучалась на данных (запоминала их полностью), и, таким образом, решает проблемы с высокой дисперсией. В следующей таблице суммированы различные типы широко используемых методов регуляризации:

<br>


**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Уменьшает коэффициенты до 0, Подходит для выбора переменных, Делает коэффициенты меньше, Компромисс между выбором переменных и небольшими коэффициентами]

<br>


**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**

&#10230; Гиперпараметры ― это свойства алгоритма обучения и включают функции, параметр регуляризации λ, количество итераций T, размер шага η и так далее.

<br>


**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Наборы словарей ― при выборе модели мы выделяем 3 разные части данных, которые у нас есть, а именно:

<br>


**42. [Training set, Validation set, Testing set]**

&#10230; [Обучающий набор, Контрольный набор, Тестовый набор]

<br>


**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**

&#10230; [Модель обучена, Обычно 80% набора данных, Модель оценена, Обычно 20% набора данных, Также называется отложенным или набором для разработки, Модель дает прогнозы, Ранее невиданные данные]

<br>


**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Как только модель выбрана, она обучается на всем наборе данных и тестируется на невиданном тестовом наборе. Они представлены на рисунке ниже:

<br>


**45. [Dataset, Unseen data, train, validation, test]**

&#10230; [Набор данных, Ранее невиданные данные, обучение, контроль, тест]

<br>


**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**

&#10230; Для более подробного обзора приведенных выше концепций ознакомьтесь со шпаргалками с советами и приемами машинного обучения!

<br>


**47. Unsupervised Learning**

&#10230; Обучение без учителя

<br>


**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**

&#10230; Класс методов обучения без учителя направлен на обнаружение структуры данных, которые могут иметь богатые скрытые структуры.

<br>


**49. k-means**

&#10230; k-средние

<br>


**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**

&#10230; Кластеризация ― Дан обучающий набор входных точек Dtrain, цель алгоритма кластеризации состоит в том, чтобы назначить каждую точку ϕ(xi) кластеру zi∈{1,...,k}

<br>


**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**

&#10230; Целевая функция ― функция потерь для одного из основных алгоритмов кластеризации, k-средних, определяется выражением:

<br>


**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Алгоритм ― после случайной инициализации центроидов кластера μ1,μ2,...,μk∈Rn алгоритм k-средних повторяет следующий шаг до сходимости:

<br>


**53. and**

&#10230; и

<br>


**54. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Инициализация средних, Назначение кластера, Обновление средних, Сходимость]

<br>


**55. Principal Component Analysis**

&#10230; Метод главных компонент - Principal Component Analysis (PCA)

<br>


**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Собственное значение, собственный вектор ― Для данной матрицы A∈Rn×n, λ называется собственным значением A, если существует вектор z∈Rn∖{0}, называемый собственным вектором, такой, что у нас есть:

<br>


**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Спектральная теорема ― Пусть A∈Rn×n. Если A симметрична, то A диагонализуема действительной ортогональной матрицей U∈Rn×n. Обозначим Λ=diag(λ1,...,λn), у нас есть:

<br>


**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Примечание: собственный вектор, связанный с наибольшим собственным значением, называется главным собственным вектором матрицы A. (примечание переводчика: Смотри минимизацию нормы Фробениуса матрицы ошибок)

<br>


**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Алгоритм ― процедура метода главных компонент - это метод уменьшения размерности, который проецирует данные по k измерениям, максимизируя дисперсию данных следующим образом:

<br>


**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Шаг 1. Нормализовать данные, чтобы получить среднее значение 0 и стандартное отклонение 1.

<br>


**61. [where, and]**

&#10230; [где, и]

<br>


**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**

&#10230; [Шаг 2: Вычислить Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, которая симметрична действительным собственным значениям., Шаг 3: Вычислить u1,...,uk∈Rn k ортогональных главных собственных векторов Σ, т.е. ортогональные собственные векторы k наибольших собственных значений., Шаг 4: Спроецировать данные на spanR(u1,...,uk).]

<br>


**63. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Эта процедура максимизирует дисперсию всех k-мерных пространств.

<br>


**64. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Данные в пространстве функций, Поиск главных компонент, Данные в пространстве главных компонент]

<br>


**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**

&#10230; Более подробный обзор приведенных выше концепций можно найти в шпаргалках по Обучению без учителя!

<br>


**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin]**

&#10230; [Линейные предикторы, Вектор признаков, Линейный классификатор/регрессия, Отступn]

<br>


**67. [Loss minimization, Loss function, Framework]**

&#10230; [Минимизация потерь, Функция потерь, Фреймворк]

<br>


**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**

&#10230; [Нелинейные предикторы, k-ближайших соседей, Нейронные сети]

<br>


**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**

&#10230; [Стохастический градиентный спуск, Градиент, Стохастические обновления, Пакетные обновления]

<br>


**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**

&#10230; [Модели дообучения, Класс гипотез, Обратное распространение ошибки, Регуляризация, Наборы словарей]

<br>


**71. [Unsupervised Learning, k-means, Principal components analysis]**

&#10230; [Обучение без учителя, k-средние, Метод главных компонент]

<br>


**72. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**73. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>


**74. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>


**75. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>


**76. By X and Y**

&#10230; По X и Y

<br>


**77. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Шпаргалки по искусственному интеллекту теперь доступны в формате [target language].
