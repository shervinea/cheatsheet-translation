**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**

&#10230; Modelos reactivos con Aprendizaje Automático

<br>


**2. Linear predictors**

&#10230; Funciones de predicción lineales

<br>


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

&#10230; En esta sección, estudiaremos modelos reactivos que mejoran con experiencia, al procesar ejemplos con pares entrada-salida.

<br>


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

&#10230; Vector de características de aprendizaje ― El vector de características de aprendizaje de una entrada x se denota ϕ(x) y es tal que:

<br>


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

&#10230; Puntaje ― El puntaje s(x,w) de un ejemplo (ϕ(x),y)∈Rd×R asociado a un modelo lineal con pesos w∈Rd está dado por el producto escalar:

<br>


**6. Classification**

&#10230; Clasificación

<br>


**7. Linear classifier ― Given a weight vector w∈Rd wand a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

&#10230; Clasificación lineal ― Dado un vector de pesos w∈Rd y un vector de características de aprendizaje ϕ(x)∈Rd, el clasificador lineal binario fw está definido como:

<br>


**8. if**

&#10230; si

<br>


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

&#10230; Margen ― El margin m(x,y,w)∈R de un ejemplo (ϕ(x),y)∈Rd×{−1,+1} asociado a los pesos de un modelo lineal w∈Rd cuantifica la confianza en la predicción: se prefiere que tenga un valor grande. Está definido como:

<br>


**10. Regression**

&#10230; Regresión

<br>


**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**

&#10230; Regresión lineal ― Dados un vector de pesos w∈Rd y un vector de características de aprendizaje ϕ(x)∈Rd, la salida de un regresión lineal con pesos w y denotada por fw se define como:

<br>


**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**

&#10230; Residual ― El residual res(x,y,w)∈R está definido como la cantidad por la cual la predicción fw(x) sobreestima el objectivo y:

<br>


**13. Loss minimization**

&#10230; Minimización de pérdida

<br>


**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**

&#10230; Función de pérdida ― Una función de pérdida Loss(x,y,w) cuantifica qué tan insatisfechos estaremos con los pesos w de un modelo para la tarea de predecir la salida y a partir de la entrada x. Es una cantidad a minimizar en el proceso de entrenamiento.

<br>


**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**

&#10230; En clasificación - La clasificación de un ejemplo x con etiqueta real y∈{−1,+1} usando un modelo lineal con pesos w puede ser obtenida con la función de predicción fw(x)≜sign(s(x,w)). En este caso, una métrica de interés que cuantifica la calidad de la clasificación está dada por el margen m(x,y,w), y puede ser usada con las siguientes funciones de pérdida:

<br>


**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**

&#10230; [Nombre, Visualización, Pérdida cero-uno, Pérdida de articulación, Pérdida logística]

<br>


**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**

&#10230; En regresión - La predicción de un ejemplo x con valor objetivo real y∈R usando un modelo lineal con pesos w puede ser hecha con una función de predicción fw(x)≜s(x,w). En esta situación, una métrica de interés que cuantifica la calidad de la regresión está dada por el margen res(x,y,w) y puede ser usada con las siguientes funciones de pérdida:

<br>


**18. [Name, Squared loss, Absolute deviation loss, Illustration]**

&#10230; [Nombre, Pérdida al cuadrado, Pérdida de desviación absoluta, Visualización]

<br>


**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**

&#10230; Esquema de minimización de pérdida ― Para entrenar un modelo, queremos minimizar la pérdida durante el entrenamiento, definida como:

<br>


**20. Non-linear predictors**

&#10230; Funciones de predicción no-lineales

<br>


**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k vecinos más próximos ― El algoritmo de los k vecinos más próximos, conocido comunmente como k-NN por sus siglas en inglés, es una solución no-paramétrica donde la respuesta para cada valor de datos está determinada por los valores dados de k vecinos en el conjunto de entrenamiento. Puede ser usado tanto en clasificación como en regresión.

<br>


**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Observación: a mayor sea el parámetro k, mayor será el sesgo inductivo y a menor sea el parámetro k, mayor será la varianza.

<br>


**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Redes neuronales ― Las redes neuronales son un conjunto de modelos construidos a partir de capas. Tipos de redes neuronales comunes incluyen las redes convolucionales y las redes recurrentes. El vocabulario centrado en las arquitecturas de redes neuronales está descrito en la figura siguente:

<br>


**24. [Input layer, Hidden layer, Output layer]**

&#10230; [Capa de entrada, Capa oculta, Capa de salida]

<br>


**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Si llamamos i a la capa número i y j a la neurona número j de la capa, tenemos entonces que:

<br>


**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**

&#10230; donde definimos w, b, x, z como el peso, la ganancia, la entrada y la salida pre-activada de la neurona, respectivamente.

<br>


**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**

&#10230; Para una visión más detallada de estos conceptos, consulte las hojas de referencia en Aprendizaje Supervisado!

<br>


**28. Stochastic gradient descent**

&#10230; Descenso estocástico por el gradiente

<br>


**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**

&#10230; Descenso por el gradiente ― Denotando η∈R como la tasa de aprendizaje (también llamado factor de aprendizaje), la regla de actualización del descenso por el graidente se expresa usando la tasa de apredizaje y la función de pérdida Loss(x,y,w) como:

<br>


**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**

&#10230; Actualizaciones estocásticas ― El descenso estocástico por el gradiente (SGD por sus siglas en inglés) actualiza los parámetros del modelo un ejemplo de entrenamiento (ϕ(x),y)∈Dtrain a la vez. Este método genera actualizaciones muy veloces que contienen a veces mucho ruido.

<br>


**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**

&#10230; Actualizaciones por lotes ― Descenso por el gradiente por lotes (BGD por sus siglas en inglés) actualiza los parámetros del model cada vez que se han procesado un lote de ejemplos de entrenamiento (el conjunto completo de datos de entrenamiento). Este método computa direcciones de actualización estables, a un mucho mayor costo computacional.

<br>


**32. Fine-tuning models**

&#10230; Ajuste fino de modelos

<br>


**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**

&#10230; Clase de hipótesis ― Una clase de hipótesis F es el conjunto de todas las funciones de predicción con una ϕ(x) fija y variando w: 

<br>


**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**

&#10230; Función logística ― La función logística σ, también llamada función sigmoide, se define como:

<br>


**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**

&#10230; Observación: tenemos σ′(z)=σ(z)(1−σ(z)).

<br>


**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**

&#10230; Retropropagación ― La etapa de propagación hacia adelante se realiza usando fi, que es el valor de la subexpresión centrada en i, mientras que la retropropagación se hace a través de gi=∂out∂fi y representa como fi influencia la salida.

<br>


**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**

&#10230; Errores de aproximación y estimado ― El error de aproximación ϵapprox representa qué tan lejos la clase de hipótesis F está de la función de predicción buscada g∗, mientras que el error de estimado ϵest cuantifica qué tan buena es la función de predicción ^f con respecto a la mejor función de predicción f∗ en la clase de hipótesis F.

<br>


**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularización ― El procedimiento de regularización busca evitar que el modelo sobreajuste los datos y por lo tanto lidia con problemas de gran varianza. La tabla siguiente resume los diferentes tipos de técnicas de regularización más utilizadas en la práctica:

<br>


**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Reduce los coeficientes a 0, Ideal para selección de variables, Hace los coeficientes más pequeños, Balance entre selección de variables y coeficientes pequeños]

<br>


**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**

&#10230; Hiperparámetros ― Los hiperparámetros son propiedades del algoritmo de aprendizaje e incluyen las características de aprendizaje, el parámetro de regularización λ, el número de iteraciones T, la tasa de aprendizaje η, etc.

<br>


**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulario ― Al elegir un modelo, distinguimos las siguientes 3 partes distintas de los datos:

<br>


**42. [Training set, Validation set, Testing set]**

&#10230; [Conjunto de entrenamiento, Conjunto de validación, Conjunto de testeo]

<br>


**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**

&#10230; [Modelo se entrena, Usualmente el 80% de los datos, Modelo se evalúa, Usualmente el 20% de los datos, También llamado conjunto de desarrollo, Modelo hace predicción, Datos no vistos]

<br>


**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Una vez que el modelo ha sido elegio, se lo entrena en todo los datos y se lo evalúa en los datos no vistos. Ésto está representado en la siguiente figura:

<br>


**45. [Dataset, Unseen data, train, validation, test]**

&#10230; [Conjunto de datos, Datos no vistos, entrenamiento, validación, testeo]

<br>


**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**

&#10230; Para una visión más detallada de estos conceptos, consulte las hojas de referencia sobre Trucos y Consejos para el Aprendizaje Automático!

<br>


**47. Unsupervised Learning**

&#10230; Aprendizaje No Supervisado

<br>


**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**

&#10230; La clase de métodos de aprendizaje no supervisado busca descubrir la estructura de los datos, los cuales pueden tener un estructura latente muy rica.

<br>


**49. k-means**

&#10230; k-medias

<br>


**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**

&#10230; Agrupamiento ― Dada un conjunto de entrenamiento de entrada Dtrain consistente de puntos multidimensionales, el objetivo de un algoritmo de entrenamiento es asignar a cada punto ϕ(xi) un grupo zi∈{1,...,k}

<br>


**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**

&#10230; Función objetivo ― La función de pérdida de uno de los algoritmos de agrupamiento más conocidos, k-medias, está dada por:

<br>


**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algoritmo ― Después de inicializar al azar los centroides de los grupos μ1,μ2,...,μk∈Rn, el algoritmo k-medias repite los siguientes pasos hasta que converge:

<br>


**53. and**

&#10230; y

<br>


**54. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Inicializado de medias, Grupo asignado, Actualización de medias, Convergencia]

<br>


**55. Principal Component Analysis**

&#10230; Análisis de Componentes Principales

<br>


**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Autovalor, autovector ― Dada una matriz A∈Rn×n, decimos que λ es un autovalor de A si existe un vector z∈Rn∖{0}, llamado el autovector, tal que:

<br>


**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Teorema de descomposición espectral ― Dada A∈Rn×n, si A es simétrica, entonces A es diagonalizable por una matriz ortogonal en números reales U∈Rn×n. Llamando Λ=diag(λ1,...,λn) tenemos que:

<br>


**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Observación: al autovector asociado con el autovalor más grande lo llamaremos el autovector principal de la matriz A.

<br>


**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algoritmo ― El procedimiento de Análisis de Componentes Principales (PCA por sus siglas en inglés) es una técnica de reducción de dimensionalidad que proyecta los datos sobre k dimensiones maximizando la varianza de los datos del siguiente modo:

<br>


**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Paso 1: Normalizar los datos de forma tal que tengan media 0 y desviación estándar de 1.

<br>


**61. [where, and]**

&#10230; [donde, y]

<br>


**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**

&#10230; [Paso 2: Computar Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, que es simétrica y con autovalores en los números reales., Paso 3: Computar u1,...,uk∈Rn los k autovectores principales ortogonales de Σ, esto es, los autovectores ortogonales correspondientes a los k autovectores más grandes., Step 4: Proyectar los datos sobre spanR(u1,...,uk).]

<br>


**63. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; Este procedimiento maximiza la varianza entre los espacios k-dimensionales.

<br>


**64. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Datos en el espacio de características, Encontrar las componentes principales, Datos en el espacio de componentes principales]

<br>


**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**

&#10230; Para una visión más detallada de estos conceptos, consulte las hojas de referencia en Aprendizaje No Supervisado!

<br>


**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin, Residual]**

&#10230; [Funciones de predicción lineal, Vector de características de aprendizaje, Clasificación/regresión lineal, Margen, Residuo]

<br>


**67. [Loss minimization, Loss function, Framework]**

&#10230; [Minimización de pérdida, Función de pérdida, Esquema]

<br>


**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**

&#10230; [Función de predicción no lineal, k vecinos más próximos, Redes neuronales]

<br>


**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**

&#10230; [Descenso estocástico por el gradiente, Gradiente, Actualizaciones estocásticas, Actualiaciones por lotes]

<br>


**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**

&#10230; [Ajuste fino de modelos, Clase de hipótesis, Retropropagación, Regularización, Vocabulario]

<br>


**71. [Unsupervised Learning, k-means, Principal components analysis]**

&#10230; [Aprendizaje No Supervisado, k-medias, Análisis de componentes principales]

<br>


**72. View PDF version on GitHub**

&#10230; Ver la versión PDF en GitHub

<br>


**73. Original authors**

&#10230; Autoría Original

<br>


**74. Translated by X, Y and Z**

&#10230; Traducido por X, Y y Z

<br>


**75. Reviewed by X, Y and Z**

&#10230; Revisado por X, Y y Z

<br>


**76. By X and Y**

&#10230; Por X y Y

<br>


**77. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Las hojas de referencia en Inteligencia Artificial están ahora disponible en español.
