**Deep Learning Tips and Tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

&#10230; Một số mẹo trong học sâu cheatsheet

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Học sâu

<br>


**3. Tips and tricks**

&#10230; Mẹo và thủ thuật

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

&#10230; [Xử lí dữ liệu, Thêm dữ liệu, Chuẩn hóa batch]

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

&#10230; [Huấn luyện một mô hình nhân tạo, Epoch, Mini-batch, Cross-entropy loss, Lan truyền ngược, Gradient descent, Cập nhật trọng số, Kiểm tra gradient]

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

&#10230; [Tinh chỉnh tham số, Khởi tạo Xavier, Học chuyển tiếp, Tốc độ học, Tốc độ học đáp ứng]

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

&#10230; [Sự phạt mô hình, Dropout, Khởi tạo trọng số, Dừng sớm]

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

&#10230; [Thói quen tốt, Quá khớp tập nhỏ, Kiểm tra đạo hàm]

<br>


**9. View PDF version on GitHub**

&#10230; [Xem bản PDF trên GitHub]

<br>


**10. Data processing**

&#10230; Xử lí dữ liệu

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

&#10230; Tăng cường dữ liệu - Các mô hình học sâu thường cần rất nhiều dữ liệu để có thể được huấn luyện đúng cách. Nó thường hữu ích để có được nhiều dữ liệu hơn từ những cái hiện có bằng cách sử dụng các kỹ thuật tăng dữ liệu. Những cái chính được tóm tắt trong bảng dưới đây. Chính xác hơn, với hình ảnh đầu vào sau đây, đây là những kỹ thuật mà chúng ta có thể áp dụng:

<br>


**12. [Original, Flip, Rotation, Random crop]**

&#10230; [Hình gốc, Lật, Xoay, Cắt ngẫu nhiên]

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

&#10230; [Hình ảnh không có bất kỳ sửa đổi nào, Lật đối với một trục mà ý nghĩa của hình ảnh được giữ nguyên, Xoay với một góc nhỏ, Mô phỏng hiệu chỉnh đường chân trời không chính xác, Lấy nét ngẫu nhiên trên một phần của hình ảnh, Một số cách cắt ngẫu nhiên có thể được thực hiện trên một hàng]

<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

&#10230; [Dịch chuyển màu, Thêm nhiễu, Mất mát thông tin, Thay đổi độ tương phản]

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

&#10230; [Các sắc thái của RGB bị thay đổi một chút, Nhiễu nhiễu có thể xảy ra khi tiếp xúc với ánh sáng nhẹ, Bổ sung nhiễu, Chịu được sự thay đổi chất lượng của các yếu tố đầu vào, Các phần của hình ảnh bị bỏ qua, Bắt chước mất khả năng của các phần của hình ảnh, Thay đổi độ sáng, Kiểm soát sự khác biệt do phơi sáng do thời gian trong ngày]

<br>


**16. Remark: data is usually augmented on the fly during training.**

&#10230; Ghi chú: dữ liệu thường được tăng cường khi huấn luyện

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; Chuẩn hóa batch ― Đây là một bước của siêu tham số γ,β chuẩn hóa tập dữ liệu {xi}. Kí hiệu μB,σ2B là trung bình và phương sai của tập dữ liệu ta muốn chuẩn hóa, tuân theo công thức sau:

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; Thường hoàn thành sau một lớp fully connected/nhân chập và trước lớp phi tuyến tính và mục đích cho phép tốc độc học cao hơn và giảm thiểu sự phụ thuộc vào khởi tạo

<br>


**19. Training a neural network**

&#10230; Huấn luyện một mô hình nhân tạo

<br>


**20. Definitions**

&#10230; Định nghĩa

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

&#10230; Vòng lặp ― Trong ngữ cảnh huấn luyện mô hình, vòng lặp là một từ chỉ một lần lặp qua toàn bộ dữ liệu huấn luyện để cập nhật tham số.

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

&#10230; Giảm độ dốc theo lô nhỏ - Trong giai đoạn đào tạo, việc cập nhật trọng số thường không dựa trên toàn bộ tập huấn cùng một lúc do độ phức tạp tính toán hoặc một điểm dữ liệu do vấn đề nhiễu. Thay vào đó, bước cập nhật được thực hiện trên các lô nhỏ, trong đó số lượng điểm dữ liệu trong một lô là một siêu tham số mà chúng ta có thể điều chỉnh.

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

&#10230; Hàm mất mát - Để định lượng cách thức một mô hình nhất định thực hiện, hàm mất L thường được sử dụng để đánh giá mức độ đầu ra thực tế y được dự đoán chính xác bởi mô hình đầu ra z.

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; Mất entropy chéo - Trong bối cảnh phân loại nhị phân trong các mạng thần kinh, tổn thất entropy chéo L(z,y) thường được sử dụng và được định nghĩa như sau:

<br>


**25. Finding optimal weights**

&#10230; Tìm trọng số tối ưu

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

&#10230; Lan truyền ngược - Lan truyền ngược là một phương pháp để cập nhật các trọng số trong mạng nhân tạo bằng cách tính đến đầu ra thực tế và đầu ra mong muốn. Đạo hàm tương ứng với từng trọng số w được tính bằng quy tắc chuỗi.

<br>


**27. Using this method, each weight is updated with the rule:**

&#10230; Sử dụng mô hình này, mỗi trọng số được cập nhật theo quy luật:

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; Cập nhật trọng số ― Trong một mô hình nhân tạo, trọng số được cập nhật như sau:

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

&#10230; [Bước 1: Lấy một loạt dữ liệu huấn luyện và thực hiện lan truyền thẳng để tính toán mất mát, Bước 2: Sao lưu lại mất mát để có được độ dốc của mất mát theo từng trọng số, Bước 3: Sử dụng độ dốc để cập nhật trọng số của mạng.]

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

&#10230; [Lan truyền thẳng, Lan truyền ngược, Cập nhật trọng số]

<br>


**31. Parameter tuning**

&#10230; Tinh chỉnh tham số

<br>


**32. Weights initialization**

&#10230; Khởi tạo trọng số

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

&#10230; Khởi tạo Xavier - Thay vì khởi tạo trọng số một cách ngẫu nhiên, khởi tạo Xavier cho chúng ta một cách khởi tạo tham số dựa trên một đặc tính độc nhất của mô hình.

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

&#10230; Học chuyển tiếp - Huấn luyện một mô hình học tập sâu đòi hỏi nhiều dữ liệu và quan trọng hơn là rất nhiều thời gian. Sẽ rất hữu ích để tận dụng các trọng số được đào tạo trước trên các bộ dữ liệu khổng lồ mất vài ngày / tuần để đào tạo và tận dụng nó cho trường hợp sử dụng của chúng ta. Tùy thuộc vào lượng dữ liệu chúng ta có trong tay, đây là các cách khác nhau để tận dụng điều này:

<br>


**35. [Training size, Illustration, Explanation]**

&#10230; [Kích thước tập huấn luyện, Mô phỏng, Giải thích]

<br>


**36. [Small, Medium, Large]**

&#10230; [Nhỏ, Trung bình, Lớn]

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

&#10230; [Đông đặc các lớp, huấn luyện hàm softmax, Đông đặc hầu hết các lớp, huấn luyện trên lớp cuối và hàm softmax, Huấn luyện trọng số trên tầng và softmax với khởi tạo trọng số trên mô hình đã huấn luyện sẵn]

<br>


**38. Optimizing convergence**

&#10230; Tối ưu hội tự

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; Tốc độ học - Tốc độ học, thường được kí hiệu là α hoặc đôi khi là η, cho biết mức độ thay đổi của các trọng số sau mỗi lần cập nhật. Nó có thể được cố định hoặc thay đổi thích ứng. Phương pháp phổ biến nhất hiện nay được gọi là Adam, đây là phương pháp thích nghi với tốc độ học.

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

&#10230; Tốc độ học thích nghi - Để tốc độ học thay đổi khi huấn luyện một mô hình có thể giảm thời gian huấn luyện và cải thiện giải pháp tối ưu số. Trong khi Adam tối ưu hóa là kỹ thuật được sử dụng phổ biến nhất, những phương pháp khác cũng có thể hữu ích. Chúng được tóm tắt trong bảng dưới đây:

<br>


**41. [Method, Explanation, Update of w, Update of b]**

&#10230; [Phương pháp, Giải thích, Cập nhật của w, Cập nhật của b]

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

&#10230; [Momentum, Làm giảm dao động, Cải thiện SGD, 2 tham số để tinh chỉnh]

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

&#10230; [RMSprop, lan truyền Root Mean Square, Thuật toán tăng tốc bằng kiểm soát dao động]

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

&#10230; [Adam, Ước lượng Adam Moment, Các phương pháp phổ biến, 4 tham số để tinh chỉnh]

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

&#10230; Chú ý: những phương pháp khác bao gồm Adadelta, Adagrad và SGD.

<br>


**46. Regularization**

&#10230; Phạt mô hình

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

&#10230; Dropout - Dropout là một kỹ thuật được sử dụng trong các mạng nhân tạo để ngăn chặn hiện tượng quá khớp bằng cách loại bỏ các nơ-ron với xác suất p>0. Nó buộc mô hình tránh phụ thuộc quá nhiều vào một tập thuộc tính nào đó.

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

&#10230; Ghi chú: hầu hết các framework học máy có cài đặt dropout thông qua biến 'keep' với tham số 1-p.

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

&#10230; Phạt trọng số - Để đảm bảo rằng các trọng số không quá lớn và mô hình không vượt quá tập huấn luyện, các kỹ thuật chính quy thường được thực hiện trên các trọng số mô hình. Những cái chính được tóm tắt trong bảng dưới đây:

<br>


**50. [LASSO, Ridge, Elastic Net]**

&#10230; [LASSO, Ridge, Elastic Net]

<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; bis. Thu hẹp hệ số về 0, Tốt cho lựa chọn biến, Làm cho hệ số nhỏ hơn, Trao đổi giữa lựa chọn biến và hệ số nhỏ]

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

&#10230; Dừng sớm - Kĩ thuật regularization này sẽ dừng quá trình huấn luyện một khi mất mát trên tập thẩm định đạt đến một độ nào đó hoặc bắt đầu tăng

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

&#10230; [Lỗi, Thẩm định, Huấn luyện, dừng sớm, Vòng]

<br>


**53. Good practices**

&#10230; Thói quen tốt

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

&#10230; Quá khớp batch nhỏ - Khi gỡ lỗi một mô hình, thường rất hữu ích khi thực hiện các thử nghiệm nhanh để xem liệu có bất kỳ vấn đề lớn nào với kiến ​​trúc của chính mô hình đó không. Đặc biệt, để đảm bảo rằng mô hình có thể được huấn luyện đúng cách, một batch nhỏ được truyền vào bên trong mạng để xem liệu nó có thể phù hợp với nó không. Nếu không thể, điều đó có nghĩa là mô hình quá phức tạp hoặc không đủ phức tạp để thậm chí vượt quá trên batch nhỏ, chứ đừng nói đến một tập huấn luyện có kích thước bình thường.

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

&#10230; Kiểm tra gradient - Kiểm tra gradient là một phương pháp được sử dụng trong quá trình thực hiện đường truyền ngược của mạng thần kinh. Nó so sánh giá trị của gradient phân tích với gradient số tại các điểm đã cho và đóng vai trò kiểm tra độ chính xác.

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

&#10230; [Loại, Gradient số, Gradient phân tích]

<br>


**57. [Formula, Comments]**

&#10230; [Công thức, Bình luận]

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

&#10230; [Đắt; Mất mát phải được tính hai lần cho mỗi chiều, Được sử dụng để xác minh tính chính xác của việc triển khai phân tích, Đánh đổi trong việc chọn h không quá nhỏ (mất ổn định số) cũng không quá lớn (xấp xỉ độ dốc kém)]

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

&#10230; [Kết quả 'Chính xác', Tính toán trực tiếp, Được sử dụng trong quá trình thực hiện cuối cùng]

<br>


**60. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Học sâu cheetsheets đã khả dụng trên [Tiếng Việt]


**61. Original authors**

&#10230; Những tác giả

<br>

**62.Translated by X, Y and Z**

&#10230; Dịch bởi X, Y và Z

<br>

**63.Reviewed by X, Y and Z**

&#10230; Đánh giá bởi X, Y và Z

<br>

**64.View PDF version on GitHub**

&#10230; Xem bản PDF trên GitHub

<br>

**65.By X and Y**

&#10230; Bởi X và Y

<br>
