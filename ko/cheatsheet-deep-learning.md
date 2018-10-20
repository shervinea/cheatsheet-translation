**1. Deep Learning cheatsheet**

&#10230; 딥러닝 치트시트

<br>

**2. Neural Networks**

&#10230; 신경망

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; 신경망(neural network)은 층(layer)으로 구성되는 모델입니다. 합성곱 신경망(convolutional neural network)과 순환 신경망(recurrent neural network)이 널리 사용되는 신경망입니다.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; 구조 - 다음 그림에 신경망 구조에 관한 용어가 표현되어 있습니다:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [입력층(input layer), 은닉층(hidden layer), 출력층(output layer)]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; i는 네트워크의 i 번째 층을 나타내고 j는 각 층의 j 번째 은닉 유닛(hidden unit)을 지칭합니다:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; 여기에서 w, b, z는 각각 가중치(weight), 절편(bias), 출력입니다.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 활성화 함수(activation function) - 활성화 함수는 은닉 유닛 다음에 추가하여 모델에 비선형성을 추가합니다. 다음과 같은 함수들을 자주 사용합니다:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [시그모이드(Sigmoid), 하이퍼볼릭탄젠트(Tanh), 렐루(ReLU), Leaky 렐루(Leaky ReLU)]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 크로스 엔트로피(cross-entropy) 손실 - 신경망에서 널리 사용되는 크로스 엔트로피 손실 함수 L(z,y)는 다음과 같이 정의합니다: 

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 학습률 - 학습률은 종종 α 또는 η로 표시하며 가중치 업데이트 양을 조절합니다. 학습률을 일정하게 고정하거나 적응적으로 바꿀 수도 있습니다. 적응적 학습률 방법인 Adam이 현재 가장 인기가 많습니다.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; 역전파(backpropagation) - 역전파는 실제 출력과 기대 출력을 비교하여 신경망의 가중치를 업데이트하는 방법입니다. 연쇄 법칙(chain rule)으로 표현된 가중치 w에 대한 도함수는 다음과 같이 쓸 수 있습니다:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; 결국 가중치는 다음과 같이 업데이트됩니다:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 가중치 업데이트 - 신경망에서 가중치는 다음 단계를 따라 업데이트됩니다:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; 1 단계: 훈련 데이터의 배치(batch)를 만듭니다.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 2 단계: 정방향 계산을 수행하여 배치에 해당하는 손실(loss)을 얻습니다.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 3 단계: 손실을 역전파하여 그래디언트(gradient)를 구합니다.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; 4 단계: 그래디언트를 사용해 네트워크의 가중치를 업데이트합니다.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; 드롭아웃(dropout) - 드롭아웃은 신경망의 유닛을 꺼서 훈련 데이터에 과대적합(overfitting)되는 것을 막는 기법입니다. 실전에서는 확률 p로 유닛을 끄거나 확률 1-p로 유닛을 작동시킵니다.

<br>

**20. Convolutional Neural Networks**

&#10230; 합성곱 신경망

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 합성곱 층의 조건 - 입력 크기를 W, 합성곱 층의 커널(kernel) 크기를 F, 제로 패딩(padding)을 P, 스트라이드(stride)를 S라 했을 때 필요한 뉴런의 수 N은 다음과 같습니다:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 배치 정규화(batch normalization) - 하이퍼파라미터 γ,β로 배치 {xi}를 정규화하는 단계입니다. 조정하려는 배치의 평균과 분산을 각각 μB,σ2B라고 했을 때 배치 정규화는 다음과 같이 계산됩니다:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 보통 완전 연결(fully connected)/합성곱 층과 활성화 함수 사이에 위치합니다. 배치 정규화를 적용하면 학습률을 높일 수 있고 초기화에 대한 의존도를 줄일 수 있습니다.

<br>

**24. Recurrent Neural Networks**

&#10230; 순환 신경망

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; 게이트(gate) 종류 - 전형적인 순환 신경망에서 볼 수 있는 게이트 종류는 다음과 같습니다:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [입력 게이트, 삭제 게이트, 게이트, 출력 게이트]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [셀(cell) 정보의 기록 여부, 셀 정보의 삭제 여부, 셀의 입력 조절, 셀의 출력 조절]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - 장 단기 메모리(long short-term memory, LSTM) 네트워크는 삭제 게이트를 추가하여 그래디언트 소실 문제를 완화한 RNN 모델입니다.

<br>

**29. Reinforcement Learning and Control**

&#10230; 강화 학습(reinforcement learning)

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 강화 학습의 목표는 주어진 환경에서 진화할 수 있는 에이전트를 학습시키는 것입니다.

<br>

**31. Definitions**

&#10230; 정의

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; 마르코프 결정 과정(Markov decision process) - 마르코프 결정 과정(MDP)은 다섯 개의 요소 (S,A,{Psa},γ,R)로 구성됩니다:

<br>

**33. S is the set of states**

&#10230; S는 상태(state)의 집합입니다.

<br>

**34. A is the set of actions**

&#10230; A는 행동(action)의 집합입니다.

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa}는 상태 전이 확률(state transition probability)입니다. s∈S, a∈A 입니다.

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1]는 할인 계수(discount factor)입니다.

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R 또는 R:S⟶R 는 알고리즘이 최대화하려는 보상 함수(reward function)입니다.

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; 정책(policy) - 정책 π는 상태와 행동을 매핑하는 함수 π:S⟶A 입니다.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; 참고: 상태 s가 주어졌을 때 정책 π를 실행하여 행동 a=π(s)를 선택한다고 말합니다.

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; 가치 함수(value function) - 정책 π와 상태 s가 주어졌을 때 가치 함수 Vπ를 다음과 같이 정의합니다:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; 벨만(Bellman) 방정식 - 벨만 최적 방정식은 가치 함수 Vπ∗와 최적의 정책 π∗로 표현됩니다:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; 참고: 주어진 상태 s에 대한 최적 정책 π∗는 다음과 같이 나타냅니다:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; 가치 반복 알고리즘 - 가치 반복 알고리즘은 두 단계를 가집니다:

<br>

**44. 1) We initialize the value:**

&#10230; 1) 가치를 초기화합니다:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) 이전 가치를 기반으로 다음 가치를 반복합니다:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; 최대 가능도 추정 - 상태 전이 함수를 위한 최대 가능도(maximum likelihood) 추정은 다음과 같습니다:

<br>

**47. times took action a in state s and got to s′**

&#10230; 상태 s에서 행동 a를 선택하여 s′를 얻을 횟수

<br>

**48. times took action a in state s**

&#10230; 상태 s에서 행동 a를 선택한 횟수

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-러닝(learning) - Q-러닝은 다음과 같은 Q의 모델-프리(model-free) 추정입니다:

<br>

**50. View PDF version on GitHub**

&#10230; 깃허브(GitHub)에서 PDF 버전으로 보기

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;

<br> [신경망, 구조, 활성화 함수, 역전파, 드롭아웃]

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [합성곱 신경망, 합성곱 층, 배치 정규화]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [순환 신경망, 게이트, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [강화 학습, 마르코프 결정 과정, 가치/정책 반복, 근사 동적 계획법, 정책 탐색]
