**Deep Learning Tips and Tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; Шпаргалка с советами и приемами глубокого обучения

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Глубокое обучение

<br>


**3. Tips and tricks**

&#10230; Советы и приемы

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [Обработка данных, Увеличение данных, Пакетная нормировка]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [Обучение нейронной сети, Эпоха, Мини-пакет, Функция потерь на основе перекрестной энтропии, Обратное распространение ошибки, Градиентный спуск, Обновление весов, Проверка градиента]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [Настройка параметров, Инициализация Ксавье, Трансферное обучение, Скорость обучения, Адаптивная скорость обучения]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [Регуляризация, Прореживание, Регуляризация весов, Ранняя остановка]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [Хорошие практики, Переобучение на небольших пакетах, Проверка градиента]

<br>


**9. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**10. Data processing**

&#10230; Обработка данных

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; Увеличение данных (augmentation) ― Для правильного обучения моделям глубокого обучения обычно требуется много данных. Часто бывает полезно получить больше данных из существующих, используя методы увеличения данных. Основные из них приведены в таблице ниже. Точнее, с учетом следующего входного изображения, вот методы, которые мы можем применить:

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [Оригинал, Отражение, Поворот, Случайное кадрирование]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [Изображение без изменений, Отражение изображения относительно оси, Вращение с небольшим углом, Имитирует неправильную калибровку горизонта, Случайный фокус на одной части изображения, Можно сделать несколько случайных обрезок подряд]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [Сдвиг цвета, Добавление шума, Потеря информации, Изменение контраста]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; [Немного изменены нюансы RGB, Получает возникающий при изменении освещения шум, Добавление шума, Толерантнее к качеству изображения, Части изображения игнорируются, Имитирует потенциальную потерю частей изображения, Изменяется яркость, Контролирует разницу в экспозиции в зависимости от времени суток]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; Примечание: данные обычно пополняются на лету во время обучения.

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Пакетная нормировка ― это шаг гиперпараметра γ,β , который нормирует пакет {xi}. Обозначим μB,σ2B среднее значение и дисперсию, которые мы хотим исправить для партии, это делается следующим образом:

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Обычно это делается после полносвязного/сверточного слоя и до уровня нелинейности и направлено на повышение скорости обучения и уменьшение сильной зависимости от инициализации.

<br>


**19. Training a neural network**

&#10230; Обучение нейронной сети

<br>


**20. Definitions**

&#10230; Определения

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; Эпоха ― в контексте обучения модели эпоха - это термин, используемый для обозначения одной итерации, когда модель видит весь обучающий набор для обновления своих весов.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; Мини-пакетный градиентный спуск ― на этапе обучения обновление весов обычно не основывается ни на всем обучающем наборе сразу (из-за сложности вычислений), ни на единственной точке данных (из-за проблем с шумом). Вместо этого этап обновления выполняется для мини-пакетов, где количество точек данных в пакете является гиперпараметром, который мы можем настроить.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; Функция потерь ― В целях количественного измерения работы конкретной модели зачастую используется функция потерь L для оценки того, в какой степени фактические выходные данные y правильно предсказываются выходными данными модели z.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Функция потерь на основе перекрестной энтропии ― В контексте бинарной классификации в нейронных сетях обычно используется потеря кросс-энтропии L(z,y) , которая определяется следующим образом:

<br>


**25. Finding optimal weights**

&#10230; Поиск оптимальных весов

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; Обратное распространение ошибки ― это метод обновления весов в нейронной сети с учетом фактических меток данных и желаемых выходов сети. Производная по каждому весу w вычисляется с использованием цепного правила.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; Используя этот метод, каждый вес обновляется с помощью правила:

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Обновление весов ― В нейронной сети веса обновляются следующим образом:

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [Шаг 1: Подать на вход сети пакет обучающих данных и выполнить прямой проход для расчета значения функции потерь, Шаг 2: Выполнить обратное распространение ошибки для получения градиентов по каждому весу, Шаг 3: Использовать градиенты для обновления весов сети.]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [Прямой проход, Обратное распространение ошибки, Обновление весов]

<br>


**31. Parameter tuning**

&#10230; Настройка параметров

<br>


**32. Weights initialization**

&#10230; Инициализация весов

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Метод инициализации Ксавье ― Вместо того, чтобы инициализировать веса чисто случайным образом, инициализация Ксавье позволяет иметь начальные веса, которые учитывают уникальные характеристики архитектуры.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; Трансферное обучение ― для обучения модели глубокого обучения требуется много данных и, что ещё более важно, много времени. Часто бывает полезно воспользоваться предварительно обученными на огромных наборах данных весами, обучение которых занимало дни/недели, и адаптировать их под наш случай использования. В зависимости от того, сколько данных у нас под рукой, существует несколько способов адаптации:

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [Размер тренировки, Иллюстрация, Пояснение]

<br>


**36. [Small, Medium, Large]**

&#10230; [Маленький, Средний, Большой]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [Замораживает все слои, обучает веса на softmax, Замораживает большинство слоев, дообучает веса на последних слоях и softmax, Дообучает веса на большинстве слоев и softmax]

<br>


**38. Optimizing convergence**

&#10230; Оптимизация сходимости

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Скорость обучения ― Скорость обучения, часто обозначаемая α или иногда η, указывает, с какой скоростью обновляются веса. Её можно исправить или адаптивно изменить. Самый популярный в настоящее время метод называется Adam, он адаптирует скорость обучения.

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; Адаптивная скорость обучения ― Изменение скорости обучения при обучении модели может сократить время обучения и улучшить численное оптимальное решение. Хотя оптимизатор Adam является наиболее часто используемым, другие также могут быть полезны. Они приведены в таблице ниже:

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [Метод, Объяснение, Обновление w, Обновление b]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [Momentum, Гасит колебания, Улучшение SGD, 2 параметра для настройки]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, Среднеквадратичное распространение ошибки (Root Mean Square propagation), Ускоряет алгоритм обучения за счет управления колебаниями]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [Adam, Оценка адаптивного момента, Самый популярный метод, 4 параметра для настройки]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; Примечание: другие методы включают Adadelta, Adagrad и SGD.

<br>


**46. Regularization**

&#10230; Регуляризация

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; Прореживание ― Dropout - это метод, используемый в нейронных сетях для предотвращения переобучения обучающих данных путем выпадения нейронов с вероятностью p>0. Это заставляет модель не слишком полагаться на определенные наборы функций.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; Примечание: большинство фреймворков глубокого обучения параметризуют исключение с помощью параметра 'keep' 1−p.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; Регуляризация весов ― Чтобы убедиться, что веса не слишком велики и что модель не переобучается на обучающей выборке, обычно выполняются методы регуляризации весов модели. Основные из них приведены в таблице ниже:

<br>


**50. [LASSO, Ridge, Elastic Net, Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [LASSO, Ridge, Elastic Net, Уменьшает коэффициенты до 0, Подходит для выбора переменных, Делает коэффициенты меньше, Компромисс между выбором переменных и небольшими коэффициентами]

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230; Ранняя остановка ― Этот метод регуляризации останавливает процесс обучения, как только функция потерь на валидационной выборке достигает плато или начинает увеличиваться.

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230; [Ошибка, Проверка, Обучение, ранняя остановка, Эпохи]

<br>


**53. Good practices**

&#10230; Хорошие практики

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230; Переобучение на небольших пакетах ― При отладке модели часто бывает полезно провести быстрые тесты, чтобы увидеть, есть ли какие-либо серьезные проблемы с архитектурой самой модели. В частности, чтобы убедиться, что модель должным образом обучена, на вход сети передается мини-пакет, чтобы увидеть, случится ли на нем переобучение. В случае если этого не происходит, то означает, что модель либо слишком сложна, либо недостаточно сложна, чтобы даже переобучиться на небольшой партии, не говоря уже об обучающем наборе нормального размера.

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230; Проверка градиента ― это метод, используемый во время реализации обратного прохода нейронной сети. Он сравнивает значение аналитического градиента с числовым градиентом в заданных точках и играет роль проверки реализации на корректность.

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230; [Тип, Числовой градиент, Аналитический градиент]

<br>


**57. [Formula, Comments]**

&#10230; [Формула, Комментарии]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230; [Дорого; потери должны вычисляться два раза для каждого измерения, Используется для проверки правильности аналитической реализации, Компромисс при выборе h не слишком малого (числовая нестабильность), но и не слишком большого (плохое приближение градиента)]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230; ['Точный' результат, Прямое вычисление, Используемое в окончательной реализации]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Шпаргалки по глубокому обучению теперь доступны в формате [target language].


**61. Original authors**

&#10230; Авторы оригинала: Afshine Amidi и Shervine Amidi ― https://github.com/afshinea и https://github.com/shervinea

<br>

**62. Translated by X, Y and Z**

&#10230; Переведено на русский язык: Пархоменко Александр ― https://github.com/AlexandrParkhomenko

<br>

**63. Reviewed by X, Y and Z**

&#10230; Проверено на русском языке: Труш Георгий (Georgy Trush) ― https://github.com/geotrush

<br>

**64. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>

**65. By X and Y**

&#10230; По X и Y

<br>
