**1. Deep Learning cheatsheet**

&#10230; Deep Learning cheatsheet

<br>

**2. Neural Networks**

&#10230; Mạng Neural

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; Mạng Neural là 1 lớp của các models được xây dựng với các tầng (layers). Các loại mạng Neural thường được sử dụng bao gồm: Mạng Neural tích chập (Convolutional Neural Networks) và Mạng Neural hồi quy (Recurrent Neural Networks).

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; Kiến trúc - Các thuật ngữ xoay quanh kiến trúc của mạng neural được mô tả như hình phía dưới

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [Tầng đầu vào, tầng ẩn, tầng đầu ra]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; Bằng việc kí hiệu i là tầng thứ i của mạng, j là đơn vị ẩn (hidden unit) thứ j của tầng, ta có:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; Chúng ta kí hiệu w, b, z tương ứng với trọng số (weights), bias và đầu ra.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; Hàm kích hoạt (Activation function) - Hàm kích hoạt được sử dụng ở phần cuối của đơn vị ẩn để đưa ra độ phức tạp phi tuyến tính (non-linear) cho mô hình (model). Đây là những trường hợp phổ biến nhất:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [Sigmoid, Tanh, ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Mất mát (loss) Cross-entropy - Trong bối cảnh của mạng neural, mất mát cross-entropy L(z, y) thường được sử dụng và định nghĩa như sau:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Tốc độ học (Learning rate) - Tốc độ học, thường được kí hiệu bởi α hoặc đôi khi là η, chỉ ra tốc độ mà trọng số được cập nhật. Thông số này có thể là cố định hoặc được thay đổi tuỳ biến. Phương thức (method) phổ biến nhất hiện tại là Adam, đó là phương thức thay đổi tốc độ học một cách phù hợp nhất có thể.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; Backpropagation (Lan truyền ngược) - Backpropagation là phương thức dùng để cập nhật trọng số trong mạng neural bằng cách tính toán đầu ra thực sự và đầu ra mong muốn. Đạo hàm liên quan tới trọng số w được tính bằng cách sử dụng quy tắc chuỗi (chain rule) theo như cách dưới đây:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; Như kết quả, trọng số được cập nhật như sau:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Cập nhật trọng số - Trong mạng neural, trọng số được cập nhật như sau:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; Bước 1: Lấy một mẻ (batch) dữ liệu huấn luyện (training data).

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; Bước 2: Thực thi lan truyền xuôi (forward propagation) để lấy được mất mát (loss) tương ứng.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; Bước 3: Lan truyền ngược mất mát để lấy được gradients (độ dốc).

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Bước 4: Sử dụng gradients để cập nhật trọng số của mạng (network).

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; Dropout - Dropout là thuật ngữ kĩ thuật dùng trong việc tránh overfitting tập dữ liệu huấn luyện

<br>

**20. Convolutional Neural Networks**

&#10230; Mạng neural tích chập (Convolutional Neural Networks)

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; Yêu cầu của tầng tích chập (Convolutional layer) - Bằng việc ghi chú W là kích cỡ của volume đầu vào, F là kích cỡ của neurals thuộc convolutional layer, P là số lượng zero padding, khi đó số lượng neurals N phù hợp với volume cho trước sẽ như sau:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Batch normalization (chuẩn hoá) - Đây là bước mà các hyperparameter γ,β chuẩn hoá batch (mẻ) {xi}. Bằng việc kí hiệu μB,σ2B là giá trị trung bình, phương sai mà ta muốn gán cho batch, nó được thực hiện như sau:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Nó thường được hoàn thành sau fully connected/convolutional layer và trước non-linearity layer và mục tiêu là cho phép tốc độ học cao hơn cũng như giảm đi sự phụ thuộc mạnh mẽ vào việc khởi tạo.

<br>

**24. Recurrent Neural Networks**

&#10230; Mạng neural hồi quy (Recurrent Neural Networks)

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; Các loại cổng - Đây là các loại cổng (gate) khác nhau mà chúng ta sẽ gặp ở một mạng neural hồi quy điển hình:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [Cổng đầu vào, cổng quên, cổng đầu ra]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [Ghi vào cell hay không?, Xoá cell hay không?, Ghi bao nhiêu vào cell?, Cần tiết lộ bao nhiêu về cell?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; LSTM - Mạng bộ nhớ ngắn dài (LSTM) là 1 loại RNN model tránh vấn đề vanishing gradient (độ dốc biến mất đột ngột) bằng cách thêm vào cổng 'quên' ('forget' gates).

<br>

**29. Reinforcement Learning and Control**

&#10230; Reinforcement Learning và Control

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; Mục tiêu của reinforcement learning đó là cho tác tử (agent) học cách làm sao để phát triển trong một môi trường

<br>

**31. Definitions**

&#10230; Định nghĩa

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; Tiến trình quyết định Markov (Markov decision processes) - Tiến trình quyết định Markov (MDP) là một dạng 5-tuple (S,A,{Psa},γ,R) mà ở đó:

<br>

**33. S is the set of states**

&#10230; S là tập hợp các trạng thái (states)

<br>

**34. A is the set of actions**

&#10230; A là tập hợp các hành động (actions)

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} là xác suất chuyển tiếp trạng thái cho s∈S và a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈[0,1[ là discount factor

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R hoặc R:S⟶R là reward function (hàm reward) mà giải thuật muốn tối đa hoá.

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; Policy - Policy π là 1 hàm π:S⟶A có nhiệm vụ ánh xạ states tới actions

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; Chú ý: Ta quy ước rằng ta thực thi policy π cho trước nếu cho trước state s ta có action a=π(s)

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; Hàm giá trị (Value function) - Với policy cho trước π và state s, ta định nghĩa value function Vπ như sau:

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; Phương trình Bellman - Phương trình tối ưu Bellman đặc trưng hoá value function Vπ∗ của policy tối ưu (optimal policy) π∗:

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; Chú ý: ta quy ước optimal policy π∗ đối với state s cho trước như sau:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; Giải thuật duyệt giá trị (Value iteration) - Giải thuật duyệt giá trị có 2 loại:

<br>

**44. 1) We initialize the value:**

&#10230; 1) Ta khởi tạo gái trị (value):

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; 2) Ta duyệt qua giá trị dựa theo giá trị phía trước:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; Ước lượng khả năng tối đa (Maximum likelihood estimate) - Ước lượng khả năng tối đa cho xác suất chuyển tiếp trạng thái (state) sẽ như sau:

<br>

**47. times took action a in state s and got to s′**

&#10230;

<br>

**48. times took action a in state s**

&#10230;

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; Q-learning ― Q-learning là 1 dạng phán đoán phi mô hình (model-free) của Q, được thực hiện như sau:

<br>

**50. View PDF version on GitHub**

&#10230; Xem bản PDF trên GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [Mạng neural, Kiến trúc, Hàm kích hoạt, Lan truyền ngược, Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Mạng neural tích chập, Tầng chập, Chuẩn hoá lô (batch)]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Mạng neural hồi quy, Gates, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
