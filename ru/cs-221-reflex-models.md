**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

<br>

**1. Reflex-based models with Machine Learning**

&#10230; Reflex-based models with Machine Learning

<br>


**2. Linear predictors**

&#10230; Linear predictors

<br>


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

&#10230; In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.

<br>


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

&#10230; Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:

<br>


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

&#10230; Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:

<br>


**6. Classification**

&#10230; Classification

<br>


**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

&#10230; Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:

<br>


**8. if**

&#10230; if

<br>


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

&#10230; Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:

<br>


**10. Regression**

&#10230; Regression

<br>


**11. Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:**

&#10230; Linear regression ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the output of a linear regression of weights w denoted as fw is given by:

<br>


**12. Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:**

&#10230; Residual ― The residual res(x,y,w)∈R is defined as being the amount by which the prediction fw(x) overshoots the target y:

<br>


**13. Loss minimization**

&#10230; Loss minimization

<br>


**14. Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.**

&#10230; Loss function ― A loss function Loss(x,y,w) quantifies how unhappy we are with the weights w of the model in the prediction task of output y from input x. It is a quantity we want to minimize during the training process.

<br>


**15. Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:**

&#10230; Classification case - The classification of a sample x of true label y∈{−1,+1} with a linear model of weights w can be done with the predictor fw(x)≜sign(s(x,w)). In this situation, a metric of interest quantifying the quality of the classification is given by the margin m(x,y,w), and can be used with the following loss functions:

<br>


**16. [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]**

&#10230; [Name, Illustration, Zero-one loss, Hinge loss, Logistic loss]

<br>


**17. Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:**

&#10230; Regression case - The prediction of a sample x of true label y∈R with a linear model of weights w can be done with the predictor fw(x)≜s(x,w). In this situation, a metric of interest quantifying the quality of the regression is given by the margin res(x,y,w) and can be used with the following loss functions:

<br>


**18. [Name, Squared loss, Absolute deviation loss, Illustration]**

&#10230; [Name, Squared loss, Absolute deviation loss, Illustration]

<br>


**19. Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:**

&#10230; Loss minimization framework ― In order to train a model, we want to minimize the training loss is defined as follows:

<br>


**20. Non-linear predictors**

&#10230; Non-linear predictors

<br>


**21. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

&#10230; k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.

<br>


**22. Remark: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

&#10230; Примечание: the higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.

<br>


**23. Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Neural networks ― Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks. The vocabulary around neural networks architectures is described in the figure below:

<br>


**24. [Input layer, Hidden layer, Output layer]**

&#10230; [Input layer, Hidden layer, Output layer]

<br>


**25. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; By noting i the ith layer of the network and j the jth hidden unit of the layer, у нас есть:

<br>


**26. where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.**

&#10230; where we note w, b, x, z the weight, bias, input and non-activated output of the neuron respectively.

<br>


**27. For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!**

&#10230; For a more detailed overview of the concepts above, check out the Supervised Learning cheatsheets!

<br>


**28. Stochastic gradient descent**

&#10230; Stochastic gradient descent

<br>


**29. Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:**

&#10230; Gradient descent ― By noting η∈R the learning rate (also called step size), the update rule for gradient descent is expressed with the learning rate and the loss function Loss(x,y,w) as follows:

<br>


**30. Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.**

&#10230; Stochastic updates ― Stochastic gradient descent (SGD) updates the parameters of the model one training example (ϕ(x),y)∈Dtrain at a time. This method leads to sometimes noisy, but fast updates.

<br>


**31. Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.**

&#10230; Batch updates ― Batch gradient descent (BGD) updates the parameters of the model one batch of examples (e.g. the entire training set) at a time. This method computes stable update directions, at a greater computational cost.

<br>


**32. Fine-tuning models**

&#10230; Fine-tuning models

<br>


**33. Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:**

&#10230; Hypothesis class ― A hypothesis class F is the set of possible predictors with a fixed ϕ(x) and varying w:

<br>


**34. Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:**

&#10230; Logistic function ― The logistic function σ, also called the sigmoid function, is defined as:

<br>


**35. Remark: we have σ′(z)=σ(z)(1−σ(z)).**

&#10230; Примечание: we have σ′(z)=σ(z)(1−σ(z)).

<br>


**36. Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.**

&#10230; Backpropagation ― The forward pass is done through fi, which is the value for the subexpression rooted at i, while the backward pass is done through gi=∂out∂fi and represents how fi influences the output.

<br>


**37. Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.**

&#10230; Approximation and estimation error ― The approximation error ϵapprox represents how far the entire hypothesis class F is from the target predictor g∗, while the estimation error ϵest quantifies how good the predictor ^f is with respect to the best predictor f∗ of the hypothesis class F.

<br>


**38. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:

<br>


**39. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]

<br>


**40. Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.**

&#10230; Hyperparameters ― Hyperparameters are the properties of the learning algorithm, and include features, regularization parameter λ, number of iterations T, step size η, etc.

<br>


**41. Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Sets vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:

<br>


**42. [Training set, Validation set, Testing set]**

&#10230; [Training set, Validation set, Testing set]

<br>


**43. [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]**

&#10230; [Model is trained, Usually 80% of the dataset, Model is assessed, Usually 20% of the dataset, Also called hold-out or development set, Model gives predictions, Unseen data]

<br>


**44. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:

<br>


**45. [Dataset, Unseen data, train, validation, test]**

&#10230; [Dataset, Unseen data, train, validation, test]

<br>


**46. For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!**

&#10230; For a more detailed overview of the concepts above, check out the Machine Learning tips and tricks cheatsheets!

<br>


**47. Unsupervised Learning**

&#10230; Unsupervised Learning

<br>


**48. The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.**

&#10230; The class of unsupervised learning methods aims at discovering the structure of the data, which may have of rich latent structures.

<br>


**49. k-means**

&#10230; k-means

<br>


**50. Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}**

&#10230; Clustering ― Given a training set of input points Dtrain, the goal of a clustering algorithm is to assign each point ϕ(xi) to a cluster zi∈{1,...,k}

<br>


**51. Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:**

&#10230; Objective function ― The loss function for one of the main clustering algorithms, k-means, is given by:

<br>


**52. Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:**

&#10230; Algorithm ― After randomly initializing the cluster centroids μ1,μ2,...,μk∈Rn, the k-means algorithm repeats the following step until convergence:

<br>


**53. and**

&#10230; and

<br>


**54. [Means initialization, Cluster assignment, Means update, Convergence]**

&#10230; [Means initialization, Cluster assignment, Means update, Convergence]

<br>


**55. Principal Component Analysis**

&#10230; Principal Component Analysis

<br>


**56. Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that we have:**

&#10230; Eigenvalue, eigenvector ― Given a matrix A∈Rn×n, λ is said to be an eigenvalue of A if there exists a vector z∈Rn∖{0}, called eigenvector, such that у нас есть:

<br>


**57. Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), we have:**

&#10230; Spectral theorem ― Let A∈Rn×n. If A is symmetric, then A is diagonalizable by a real orthogonal matrix U∈Rn×n. By noting Λ=diag(λ1,...,λn), у нас есть:

<br>


**58. Remark: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.**

&#10230; Примечание: the eigenvector associated with the largest eigenvalue is called principal eigenvector of matrix A.

<br>


**59. Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:**

&#10230; Algorithm ― The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:

<br>


**60. Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.**

&#10230; Step 1: Normalize the data to have a mean of 0 and standard deviation of 1.

<br>


**61. [where, and]**

&#10230; [where, and]

<br>


**62. [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]**

&#10230; [Step 2: Compute Σ=1mm∑i=1ϕ(xi)ϕ(xi)T∈Rn×n, which is symmetric with real eigenvalues., Step 3: Compute u1,...,uk∈Rn the k orthogonal principal eigenvectors of Σ, i.e. the orthogonal eigenvectors of the k largest eigenvalues., Step 4: Project the data on spanR(u1,...,uk).]

<br>


**63. This procedure maximizes the variance among all k-dimensional spaces.**

&#10230; This procedure maximizes the variance among all k-dimensional spaces.

<br>


**64. [Data in feature space, Find principal components, Data in principal components space]**

&#10230; [Data in feature space, Find principal components, Data in principal components space]

<br>


**65. For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!**

&#10230; For a more detailed overview of the concepts above, check out the Unsupervised Learning cheatsheets!

<br>


**66. [Linear predictors, Feature vector, Linear classifier/regression, Margin]**

&#10230; [Linear predictors, Feature vector, Linear classifier/regression, Margin]

<br>


**67. [Loss minimization, Loss function, Framework]**

&#10230; [Loss minimization, Loss function, Framework]

<br>


**68. [Non-linear predictors, k-nearest neighbors, Neural networks]**

&#10230; [Non-linear predictors, k-nearest neighbors, Neural networks]

<br>


**69. [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]**

&#10230; [Stochastic gradient descent, Gradient, Stochastic updates, Batch updates]

<br>


**70. [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]**

&#10230; [Fine-tuning models, Hypothesis class, Backpropagation, Regularization, Sets vocabulary]

<br>


**71. [Unsupervised Learning, k-means, Principal components analysis]**

&#10230; [Unsupervised Learning, k-means, Principal components analysis]

<br>


**72. View PDF version on GitHub**

&#10230; Посмотреть PDF-версию на GitHub

<br>


**73. Original authors**

&#10230; Авторы оригинала

<br>


**74. Translated by X, Y and Z**

&#10230; Переведено X, Y и Z

<br>


**75. Reviewed by X, Y and Z**

&#10230; Проверено X, Y и Z

<br>


**76. By X and Y**

&#10230; По X и Y

<br>


**77. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Шпаргалки по искусственному интеллекту теперь доступны в формате [target language].
