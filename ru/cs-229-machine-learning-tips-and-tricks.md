**Machine Learning tips and tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

<br>

**1. Machine Learning tips and tricks cheatsheet**

&#10230; Шпаргалка с советами и приемами машинного обучения

<br>

**2. Classification metrics**

&#10230; Метрики классификации

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; В контексте бинарной классификации вот основные метрики, которые важно отслеживать, чтобы оценить производительность модели.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Матрица ошибок ― Матрица ошибок используется для получения более полной картины при оценке производительности модели. Она определяется следующим образом:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Прогнозируемый класс, Актуальный класс]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Основные метрики ― Для оценки эффективности моделей классификации обычно используются следующие показатели:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Метрика, Формула, Интерпретация]

<br>

**8. Overall performance of model**

&#10230; Эффективность работы модели

<br>

**9. How accurate the positive predictions are**

&#10230; Насколько точны положительные прогнозы

<br>

**10. Coverage of actual positive sample**

&#10230; Полнота показывает, какая часть положительных образцов была выделена классификатором

<br>

**11. Coverage of actual negative sample**

&#10230; Специфика выделенных отрицательных образцов

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Гибридная метрика полезна для несбалансированных классов

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; Receiver Operating (characteristic) Curve (ROC) ― Кривая рабочей характеристики приемника, также обозначенная как ROC, представляет собой график зависимости TPR от FPR при изменении порога. Эти показатели приведены в таблице ниже:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Метрика, Формула, Эквивалент]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; Area Under ROC Curve (AUC) ― Площадь под ROC кривой, также обозначаемая как AUC или AUROC, является областью под ROC, как показано на следующем рисунке:

<br>

**16. [Actual, Predicted]**

&#10230; [Актуальное, Прогнозируемое]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Базовые метрики ― Дана регрессионная модель f, для оценки производительности модели обычно используются следующие метрики:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Cумма квадратов общая (total), Сумма квадратов объясненная (regression), Сумма квадратов разностей (residual)]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Коэффициент детерминации ― Коэффициент детерминации, часто обозначаемый как R2 или r2, обеспечивает меру того, насколько хорошо наблюдаемые результаты воспроизводятся моделью, и определяется следующим образом:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Основные метрики ― Следующие метрики обычно используются для оценки эффективности регрессионных моделей с учетом количества переменных n, которые они принимают во внимание:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; где L - правдоподобие, а ˆσ2 - оценка дисперсии, связанной с каждым ответом.

<br>

**22. Model selection**

&#10230; Выбор модели

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Словарь ― При выборе модели мы выделяем 3 разные части имеющихся у нас данных:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Обучающий набор, Контрольный набор, Тестовый набор]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Модель обучена, Модель оценена, Модель дает прогнозы]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Обычно 80% набора данных, Обычно 20% набора данных]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Также называется набором для удержания или развития, Ранее невиданные данные]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Как только модель выбрана, она обучается на всем наборе данных и тестируется на невиданном тестовом наборе. Они представлены на рисунке ниже:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Кросс-валидация (CV) ― Перекрестная проверка - это метод, который используется для выбора модели, которая не слишком полагается на исходный обучающий набор. Различные типы суммированы в таблице ниже:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Тренировка на k−1 частях и оценка на оставшейся, Обучение на n−p наблюдениях и оценка на p оставшихся]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Обычно k=5 или 10, Случай p=1 называется исключение-разовое (leave-one-out)]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; Наиболее часто используемый метод называется k-кратной перекрестной проверкой и разбивает обучающие данные на k частей, чтобы проверить модель на одной выборке, одновременно обучая модель на k−1 других выборках, все это k раз. Затем ошибка усредняется по k результатам и называется ошибкой перекрестной проверки.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Регуляризация ― Процедура регуляризации направлена на то, чтобы модель не переобучалась на данных, и, таким образом, решает проблемы с высокой дисперсией. В следующей таблице суммированы различные типы широко используемых методов регуляризации:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Уменьшает коэффициенты до 0, Подходит для выбора переменных, Делает коэффициенты меньше, Компромисс между выбором переменных и небольшими коэффициентами]

<br>

**35. Diagnostics**

&#10230; Диагностика

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Смещение (Bias) ― Смещение модели - это разница между ожидаемым прогнозом и правильной моделью, которую мы пытаемся предсказать для заданных точек данных.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Дисперсия (Variance) ― Дисперсия модели - это изменчивость прогноза модели для заданных точек данных.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Компромисс смещения/дисперсии ― Чем проще модель, тем выше смещение, а чем сложнее модель, тем выше дисперсия.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Симптомы, Иллюстрация регрессии, Иллюстрация классификации, Иллюстрация глубокого обучения, Возможные исправления]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Высокая ошибка обучения, Ошибка обучения близка к ошибке теста, Высокое смещение, Ошибка обучения немного ниже ошибки теста, Очень низкая ошибка обучения, Ошибка обучения намного ниже, чем ошибка теста, Высокая дисперсия]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Усложнить модель, Добавьте больше параметров, Тренируйтесь дольше, Выполнить регуляризацию, Взять больше данных]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Анализ ошибок ― Анализ ошибок - это анализ основной причины разницы в производительности между текущей и идеальной моделями.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Абляционный анализ ― Абляционный анализ анализирует первопричину разницы в производительности между текущей и базовой моделями.

<br>

**44. Regression metrics**

&#10230; Метрики регрессии

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Метрики классификации, Матрица ошибок, доля правильных ответов (accuracy), точность (precision), полнота (recall), F1 мера, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Метрики регрессии, R в квадрате, Мэллоу CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Выбор модели, перекрестная проверка, регуляризация]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Диагностика, Компромисс смещения/дисперсии, ошибка/абляционный анализ]
