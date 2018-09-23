**1. Machine Learning tips and tricks cheatsheet**

&#10230; Uczenie maszynowe - ściąga z poradami

<br>

**2. Classification metrics**

&#10230; Miary efektywności klasyfikatorów

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; W przypadku klasyfikacji binarnej, następujące miary są użyteczne do ustalenia efektywności modelu.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Macierz pomyłek - Macierz pomyłek jest wykorzystywana w celu przedstawienia bardziej całościowego obrazu efektywności modelu. Definiuje się ją w następujący sposób:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Klasa predykowana, Klasa rzeczywista]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Główne miary - Następujące miary często wykorzystywane są do ustalenia efektywności modelu:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Miara, Wzór, Interpretacja]

<br>

**8. Overall performance of model**

&#10230; Dokładność - całościowa efektywność modelu

<br>

**9. How accurate the positive predictions are**

&#10230; Precyzja - jak dokładne są predykcje pozytywne

<br>

**10. Coverage of actual positive sample**

&#10230; Czułość - stosunek wyników prawdziwie dodatnich do sumy prawdziwie dodatnich i fałszywie ujemnych

<br>

**11. Coverage of actual negative sample**

&#10230; Swoistość - stosunek wyników prawdziwie ujemnych do sumy prawdziwie ujemnych i fałszywie dodatnich

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Hybrydowa miara, przydatna przy niezbalansowanych klasach

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC - jest to wykres TPR do FPR przy zmiennym progu. Podsumowanie tych miar znajduje się w tabeli poniżej:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Miara, Wzór, Odpowiednik]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - Powierzchna pola pod ROC, zwane także AUC lub AUROC, jest to powierzchnia pola pod wykresem ROC, jak to pokazano na wykresie obok:

<br>

**16. [Actual, Predicted]**

&#10230; [Rzeczywiste, Predykowane]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Miary podstawowe - Mając model regresyjny f, następujące miary są często używane do sprawdzenia efektywności modelu:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Całkowita suma kwadratów, Wyjaśniona suma kwadratów, Pozostała suma kwadratów]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Współczynnik determinacji - często zapisywany jako R2 lub r2, jest miarą tego, jak dobrze zaobserwowane wyniki są replikowane przez model. Definiuje się go następująco:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Główne miary - Następujące miary często wykorzystywane są do ustalenia efektywności modelu regresyjnego. Opierają się one na ilości zmiennych n, które model wykorzystuje:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; gdzie L jest prawdopodobieństwem i ˆσ2 jest estymatą wariancji związanej z każdą odpowiedzią.

<br>

**22. Model selection**

&#10230; Wybór modelu

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Słownictwo - Przy wybieraniu modelu rozróżniamy 3 różne porcje danych. Określamy je następująco:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Zbiór treningowy, Zbiór walidacyjny, Zbiór testowy]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Model jest trenowany, Model jest sprawdzany, Model generuje predykcje]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Zazwyczaj 80% zbioru danych, Zazwyczaj 20% zbioru danych]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Zwany także zbiorem zachowanym albo zbiorem deweloperskim, Niewidziane dane]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Po wyborze modelu, szkolimy go na całym zbiorze danych (treningowy + walidacyjny) i testujemy na niewidzianym zbiorze (testowy). Zbiory są przedstawione na obrazkach poniżej:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Walidacja krzyżowa - Cross-validation, zapisywana także jako CV, jest metodą która zakłada że przy wyborze modelu nie opieramy się tylko na jednych danych treningowych. Różne rodzaje tej metody opisane są poniżej w tabeli:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Trenowanie na k-1 podzbiorach i sprawdzanie na pozostałym podzbiorze, Trenowanie na n-p obserwacjach i sprawdzanie na p pozostałych]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Zazwyczaj k=5 lub 10, przypadek przy p=1 zwany jest leave-one-out]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; Najczęściej stosowanym rodzajem walidacji krzyżowej jest metoda zwana k-fold cross-validation (K-krotna walidacja krzyżowa). Dzieli ona dane treningowe na k równych podzbiorów. Model jest trenowany na k-1 podzbiorach i testowany na pozostałym jednym podzbiorze. Proces powtarzany jest k razy przy zmianie podzbioru walidacyjnej na następną. Błąd jest liczony jako średnia błędów ze wszytkich podzbiorów walidacyjnych.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularyzacja - jest to proces mający na celu uniknięcie nadmiernemu dopasowaniu (overfitting) modelu do danych treningowych i uniknięciu wysokiej wariancji modelu. Tabela obok przedstawia rodzaje często stosowanych motod regularyzacyjnych:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Zmniejsza współczynniki do 0, Dobra do doboru zmiennych, Zmniejsza współczynniki, Rozwiązanie pośrednie pomiędzy doborem zmiennych a małymi współczynnikami]

<br>

**35. Diagnostics**

&#10230; Diagnostyka

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Niewystarczające dopasowanie (bias, underfitting) - jest to różnica pomiędzy predykowanymi wynikami a wynikami rzeczywistymi. Predykcje modelu cechuje mała wariancja i słabe dopasowanie do danych treningowych.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Nadmierne dopasowanie (variance, overfitting) - predykcje modelu cechuje duża wariancja i dobre dopasowanie do danych treningowych.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Nadmierne/Niewystarczające dopasowanie modelu - im prostszy model tym będzie bardziej niewystarczająco dopasowany, im bradziej złożony tym będzie bardziej nadmiernie dopasowany.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Objawy, Regresja, Klasyfikacja, Deep learning, Co zrobić?]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Wysoki błąd treningowy, Błąd treningowy zbliżony do błędu testowego, Niewystarczające dopasowanie, Błąd treningowy odrobinę mniejszy niż błąd testowy, Bardzo mały błąd treningowy, Błąd treningowy o wiele mniejszy niż błąd testowy, Nadmierne dopasowanie]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Uczyń model bardziej złożonym, Dodaj zmiennych, Ucz model dłużej, Zastosuj regularyzacje, Zdobąć więcej danych]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Analiza błędu - Jest to analiza głównych powodów różnicy efektywności modelu testowanego i modelu doskonałego. W celu poprawy efektywności modelu.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Analiza ablacyjna - analiza głównych powodów różnicy efektywności modelu testowanego i modelu podstawowego. W celu uproszczenia modelu.

<br>

**44. Regression metrics**

&#10230; Miary regresji

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Miary klasyfikacji, macierz pomyłek, dokładność, precyzja, czułość, F1, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Miary regresji, R kwadrat, CP Mallow'a, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Wybór modelu, walidacja krzyżowa, regularyzacja]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Diagnostyka, Niedostateczne/nadmierne dopasowanie modelu, analiza ablacyjna/błędu]
