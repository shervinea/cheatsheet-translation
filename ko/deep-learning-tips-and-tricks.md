**Deep Learning Tips and Tricks translation**

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; 딥 러닝 팁과 트릭 치트시트
 
<br>


**2. CS 230 - Deep Learning**

&#10230; CS230 - 딥 러닝

<br>


**3. Tips and tricks**

&#10230; 팁과 트릭 치트시트

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [데이터 처리, 데이터 증가, 배치 정규화]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [신경망 학습, 에포크, 미니-배치, 크로스-엔트로피 손실, 역전파, 경사하강법, 가중치 업데이트, 그레디언트 확인]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [parameter 조정, Xavier 초기화, 전이학습, 학습률, 데이터 맞춤 학습률]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [정규화, 드랍아웃, 가중치 정규화, 이른 정지]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [좋은 습관, 오버피팅 스몰 배치, 그레디언트 확인]

<br>


**9. View PDF version on GitHub**

&#10230; GitHub에서 PDF 버전을 확인할 수 있습니다.

<br>


**10. Data processing**

&#10230; 데이터 처리

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; 데이터 증가 - 딥러닝 모델들은 적절한 일반적으로 학습을 위해 많은 양의 데이터를 필요로 합니다. 데이터 증가 기술을 사용하여 기존의 데이터에서 더 많은 데이터를 얻는 것은 종종 유용합니다. 주요 내용은 아래 표에 요약되어 있습니다. 보다 정확하게, 주어진 이미지에 따라 우리가 적용할 수 있는 기술들이 있습니다. :

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [원본, 반전, 회전, 랜덤 이미지 패치]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [수정 없는 이미지, 원본 이미지 회손 없이 좌우 반전, 약간의 각도로 회전, 부정확 한 수평선 보정을 시뮬레이션합니다, 이미지의 한 부분을 임의의 초점으로 맞춥니다, 몇몇 무작위 이미지 패치는 연속으로 나타날 수 있습니다 ]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [색상변환, 잡음 추가, 정보 손실, 명암대비 변경]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; [RGB의 뉘앙스는 약간 변경됩니다, 빛 노출로 발생할 수 있는 잡음을 포착할 수 있습니다, 잡음 추가, 인풋의 품질변동에 대한 허용오차 증대, 이미지의 일부분 무시, 손실된 이미지 일부분을 모방할 가능성, 밝기 변화, 하루 중 시간에 따른 노출 변화 제어 ]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; 주석: 데이터는 일반적으로 학습중에 증가 됩니다. 

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 배치 정규화 - 배치{xi}를 정규화하는 하이퍼파라미터 γ,β 단계입니다. μB,σ2B를 우리가 배치에 정정하고자하는 평균과 분산으로 표기함으로써, 다음과 같이 진행됩니다.

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 일반적으로 완전연결/컨볼루셔널 계층 이후와 비선형 계층 이전에 사용되며 학습률을 높이고 초기화에 대한 의존성을 줄이는 데 그 목적이 있습니다.

<br>


**19. Training a neural network**

&#10230; 신경망 학습

<br>


**20. Definitions**

&#10230; 정의

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; 에포크 - 모델 학습의 맥락에서, 에포크는 모델이 전체 트레이닝 셋의 가중치를 업데이트 하는 한 번의 반복을 뜻하는 용어입니다.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; 미니-배치 경사하강법 - 학습 단계에서, 가중치 업데이트는 일반적으로 계산 복잡성이나 잡음 문제로 인한 하나의 데이터 포인트로 인해 전체 트레이닝 셋을 기반으로하지 않습니다 대신에, 업데이트 단계는 배치내에 있는 여러 데이터 포인트들을 튜닝할 수 있는 하이퍼파라미터인 미니 배치에서 진행됩니다.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; 손실함수 - 주어진 모델이 어떻게 수행되는지를 정량화하기 위해, 손실 함수 L은 보통 실제 출력값 y가 예측 모델 출력값 z에 의해 정확하게 예측되는 정도를 평가하는 데 사용됩니다.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 크로스-엔트로피 손실 - 신경망 학습에서 이진분류의 맥락으로 접근하면, 크로스-엔트로피 손실 L(z,y)는 일반적으로 사용되며 다음과 같이 정의됩니다.

<br>


**25. Finding optimal weights**

&#10230; 최적의 가중치 찾기

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; 역전파 - 역전파는 실제 출력값과 원하는 출력값을 계산하여 신경망의 가중치를 업데이트 하는 방법입니다. 각 가중치 w에 대한 미분은 체인규칙을 사용하여 계산됩니다.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; 이러한 방법을 사용하여, 각각의 가중치는 아래와 같은 규칙에 의해 업데이트 됩니다 :

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 가중치 업데이트 - 신경망에서, 다음과 같은 방법으로 가중치는 업데이트 됩니다 :

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [1단계 : 트레이닝 데이터의 배치를 가져와 순전파를 진행하여 손실을 계산합니다. 2단계 : 각 가중치와 관련하여 그레디언트 손실값을 얻기 위해 역전파를 하게됩니다. 3단계 : 네트워크의 가중치를 업데이트 하기 위해서 그레디언트를 사용합니다.]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [순전파, 역전파, 가중치 업데이트]

<br>


**31. Parameter tuning**

&#10230; 파라미터 조정

<br>


**32. Weights initialization**

&#10230; 가중치 초기화

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Xavier 초기화 - 완전 무작위 방식으로 가중치 초기화하는 대신, Xavier 초기화는 설계에 고유 한 특성을 고려한 초기 가중치를 가질 수 있습니다.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; 전이학습 - 딥 러닝 학습 모델을 학습하기 위해서는 많은 양의 데이터가 필요로하며 무엇 보다도 많은 시간을 필요로 합니다. 학습을 위해 수일,수주의 시간이 걸린 거대한 데이터 셋의 사전 훈련 된 가중치를 활용하며  케이스 활용에 유용합니다. 현재 보유하고있는 데이터의 양에 따라 여러가지 활용 방법이 있습니다.

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [학습 크기, 삽화, 설명]

<br>


**36. [Small, Medium, Large]**

&#10230; [작음, 중간, 큰]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [모든 층을 고정시키고, softmax에서 가중치 학습, 대부분의 층을 고정시키고, 마지막 층과 softmax에서 가중치 학습, 사전학습 된 가중치를 초기화 하여 layers 및 softamx에 가중치를 학습합니다.]

<br>


**38. Optimizing convergence**

&#10230; convergence 최적화 

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.
**

&#10230; 학습률 - α 또는 때때로 η로 표기되는 학습률은 가중치가 어느 속도로 업데이트 되는지를 나타내줍니다. 이것은 고정되거나 또는 적응적으로 변화될 수 있습니다. 현재 가장 널리 사용되는 방법은 Adam이며, 학습률을 조정하는 방법입니다.

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; 데이터 맞춤 학습률 - 모델 학습시 학습 속도가 달라지며 학습 시간이 단축되고 수치적인 최적화 솔루션이 향상 될 수 있습니다. Adam 최적화가 가장 일반적으로 사용되는 기술이지만 이외 방법들도 유용합니다. 아래 표에 요약되어 있습니다. :

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [방법, 설명, w 업데이트, b 업데이트]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [모멘텀, 감쇠진동, 확률적 경사하강법(SGD)을 개선, 튜닝할 2가지 파라미터]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, 루트평균제곱전파, 진동제어를 통해 학습 알고리즘 속도를 향상]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [아담, 적응 모멘트 추정, 가장 대중적인 방법, 4개의 파라미터를 조정]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; 주석 : 이외 방법으로 Adadelta, Adagrad 그리고 SGD가 포함됩니다. 

<br>


**46. Regularization**

&#10230; 정규화

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; 드랍아웃 - 드랍아웃은 확률 p>0인 뉴런을 제거하여 훈련 데이터의 과적합을 예방하기 위해 신경망에서 사용되는 기술입니다. 모델이 특정 변수 셋에 너무 의존하는것을 피하도록 해야합니다.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; 주석 : 대부분의 딥러닝 프레임워크에서 1-p '유지'파라미터 변수 통해 드랍아웃의 매개변수화 합니다.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; 가중치 정규화 - 가중치가 너무 크지 않고 모델이 트레이닝 셋에 과적합되지 않는것을 확인하기 위해서 정규화 기법이 일반적으로 가중치 모델에 사용됩니다. 주요 내용은 아래 표에 요약되어 있습니다.

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230;[라쏘, 리지, 엘라스틱 넷]

<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; 계수를 0으로 축소합니다, 변수 선택에 좋습니다, 계수를 더 작게 만듭니다, 변수 선택과 작은 계수간에 트레이드 오프

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230; 조기정지 - 해당 정규화 기술은 검증손실이 안정기에 도달하거나 증가하기 시작하는 즉시 트레이닝 과정을 중지합니다.

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230; [에러, 검증, 트레이닝, 조기정지, 에포크]

<br>


**53. Good practices**

&#10230; [좋은 습관]

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230; 과적합 작은 배치 - 모델 디버깅시, 종종 사용되는 방법으로 빠르게 테스트를 진행하여 모델 구조상 자체의 중대한 문제가 있는지 확인할 수 있습니다. 특히, 모델이 적절히 학습할 수 있는지 확인하기 위해 네트워크 내부에 미니배치가 전달되어 오버핏이 되는지 확입니다. 만약 그럿지 못하다면, 모델이 너무 복잡하거나 미니배치에 충분히 오버핏될 복잡도가 떨어지는것을 의미합니다, 일반 크기의 트레이닝 셋 또한 동일하다고 볼 수 있습니다. 

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230; 그레디언트 확인 - 그레디언트 확인은 신경망의 역방향 시행중에 사용되는 방법입니다. 주어진 점에서 분석 그레디언트의 값을 수치 그레디언트와 비교하여 정확성에 대한 민감정도 검사기 역할을 수행합니다.
<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230;[종류, 수치기울기, 분석 그레디언트]

<br>


**57. [Formula, Comments]**

&#10230; [공식, 언급]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230;[비용; 차원당 두번씩 손실값을 계산해야 합니다, 분석 구현의 정확성을 확인하는데 사용됩니다, h 선택에 있어 너무 작지 않으며(수치적 불안정성), 너무 크지않은(약한 그레디언트)상에서 트레이드 오프가 필요로합니다.]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230;['정확한' 결과, 직접계산, 최종 수행단계에서 사용]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].

&#10230; 딥 러닝 치트시트는 한국어로 이용가능 합니다.


**61. Original authors**

&#10230; 원 저자

<br>

**62.Translated by X, Y and Z**

&#10230; X,Y 그리고 Z로 번역됩니다.

<br>

**63.Reviewed by X, Y and Z**

&#10230; X,Y 그리고 Z에 의해 검토됩니다.

<br>

**64.View PDF version on GitHub**

&#10230; GitHub에서 PDF 버전으로 보실 수 있습니다.

<br>

**65.By X and Y**

&#10230; X와 Y로

<br>
