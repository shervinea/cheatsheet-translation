**Convolutional Neural Networks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

<br>

**1. Convolutional Neural Networks cheatsheet**

&#10230;Convolutional Neural Networks cheatsheet

<br>


**2. CS 230 - Deep Learning**

&#10230; CS 230 - Deep Learning

<br>


**3. [Overview, Architecture structure]**

&#10230; [Tổng quan, Kết cấu kiến trúc]

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230; [Các kiểu tầng (layer), Tích chập, Pooling, Kết nối đầy đủ]

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230; [Các siêu tham số của bộ lọc, Các chiều, Stride, Padding]

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

&#10230; [Điều chỉnh các siêu tham số, Độ tương thích tham số, Độ phức tạp mô hình, Receptive field]

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230; [Các hàm kích hoạt, Rectified Linear Unit, Softmax]

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

&#10230; [Phát hiện vật thể, Các kiểu mô hình, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

&#10230; [Nhận diện/ xác nhận gương mặt, One shot learning, Siamese network, Triplet loss]

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230; [Neural style transfer, Activation, Style matrix, Style/content cost function]

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230; [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]

<br>


**12. Overview**

&#10230; Tổng quan

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230; Kiến trúc truyền thống của một mạng CNN ― Mạng neural tích chập (Convolutional neural networks), còn được biết đến với tên CNNs, là một dạng mạng neural được cấu thành bởi các tầng sau:

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230; Tầng tích chập và tầng pooling có thể được hiệu chỉnh theo các siêu tham số (hyperparameters) được mô tả ở những phần tiếp theo.

<br>


**15. Types of layer**

&#10230; Các kiểu tầng

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230; Tầng tích chập (CONV) ― Tầng tích chập (CONV) sử dụng các bộ lọc để thực hiện phép tích chập khi đưa chúng đi qua đầu vào I theo các chiều của nó. Các siêu tham số của các bộ lọc này bao gồm kích thước bộ lọc F và độ trượt (stride) S. Kết quả đầu ra O được gọi là feature map hay activation map.

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230; Lưu ý: Bước tích chập cũng có thể được khái quát hóa cả với trường hợp một chiều (1D) và ba chiều (3D).

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230; Pooling (POOL) ― Tầng pooling (POOL) là một phép downsampling, thường được sử dụng sau tầng tích chập, giúp tăng tính bất biến không gian. Cụ thể, max pooling và average pooling là những dạng pooling đặc biệt, mà tương ứng là trong đó giá trị lớn nhất và giá trị trung bình được lấy ra.

<br>


**19. [Type, Purpose, Illustration, Comments]**

&#10230; [Kiểu, Chức năng, Minh họa, Nhận xét]

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

&#10230; [Max pooling, Average pooling, Từng phép pooling chọn giá trị lớn nhất trong khu vực mà nó đang được áp dụng, Từng phép pooling tính trung bình các giá trị trong khu vực mà nó đang được áp dụng]

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

&#10230; [Bảo toàn các đặc trưng đã phát hiện, Được sử dụng thường xuyên, Giảm kích thước feature map, Được sử dụng trong mạng LeNet]

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230;  Fully Connected (FC) ― Tầng kết nối đầy đủ (FC) nhận đầu vào là các dữ liệu đã được làm phẳng, mà mỗi đầu vào đó được kết nối đến tất cả neuron. Trong mô hình mạng CNNs, các tầng kết nối đầy đủ thường được tìm thấy ở cuối mạng và được dùng để tối ưu hóa mục tiêu của mạng ví dụ như độ chính xác của lớp (class).

<br>


**23. Filter hyperparameters**

&#10230; Các siêu tham số của bộ lọc

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

&#10230; Tầng tích chập chứa các bộ lọc mà rất quan trọng cho ta khi biết ý nghĩa đằng sau các siêu tham số của chúng.

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

&#10230; Các chiều của một bộ lọc ― Một bộ lọc kích thước F×F áp dụng lên đầu vào chứa C kênh (channels) thì có kích thước tổng kể là F×F×C thực hiện phép tích chập trên đầu vào kích thước I×I×C và cho ra một  feature map (hay còn gọi là activation map) có kích thước O×O×1.

<br>


**26. Filter**

&#10230; Bộ lọc

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

&#10230; Lưu ý: Việc áp dụng K bộ lọc có kích thước F×F cho ra một feature map có kích thước O×O×K.

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

&#10230; Stride ― Đối với phép tích chập hoặc phép pooling, độ trượt S ký hiệu số pixel mà cửa sổ sẽ di chuyển sau mỗi lần thực hiện phép tính.

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230;  Zero-padding ― Zero-padding là tên gọi của quá trình thêm P số không vào các biên của đầu vào. Giá trị này có thể được lựa chọn thủ công hoặc một cách tự động bằng một trong ba những phương pháp mô tả bên dưới:

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

&#10230; [Phương pháp, Giá trị, Mục đích, Valid, Same, Full]

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

&#10230; [Không sử dụng padding, Bỏ phép tích chập cuối nếu số chiều không khớp, Sử dụng padding để làm cho feature map có kích  thước ⌈IS⌉, Kích thước đầu ra thuận lợi về mặt toán học, Còn được gọi là 'half' padding, Padding tối đa sao cho các phép tích chập có thể được sử dụng tại các rìa của đầu vào, Bộ lọc 'thấy' được đầu vào từ đầu đến cuối]

<br>


**32. Tuning hyperparameters**

&#10230; Điều chỉnh siêu tham số

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230; Tính tương thích của tham số trong tầng tích chập ― Bằng cách ký hiệu I là độ dài kích thước đầu vào, F là độ dài của bộ lọc, P là số lượng zero padding, S là độ trượt, ta có thể tính được độ dài O của feature map theo một chiều bằng công thức:

<br>


**34. [Input, Filter, Output]**

&#10230; [Đầu vào, Bộ lọc, Đầu ra]

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230; Lưu ý: Trong một số trường hợp, Pstart=Pend≜P, ta có thể thay thế Pstart+Pend bằng 2P trong công thức trên.

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230; Hiểu về độ phức tạp của mô hình ― Để đánh giá độ phức tạp của một mô hình, cách hữu hiệu là xác định số tham số mà mô hình đó sẽ có. Trong một tầng của mạng neural tích chập, nó sẽ được tính toán như sau:

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

&#10230; [Minh họa, Kích thước đầu vào, Kích thước đầu ra, Số lượng tham số, Lưu ý]

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230; [Một tham số bias với mỗi bộ lọc, Trong đa số trường hợp, S<F, Một lựa chọn phổ biến cho K là 2C]

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230; [Phép pooling được áp dụng lên từng kênh (channel-wise), Trong đa số trường hợp, S=F]

<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230; [Đầu vào được làm phẳng, Mỗi neuron có một tham số bias, Số neuron trong một tầng FC phụ thuộc vào ràng buộc kết cấu]

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230; Trường thụ cảm (Receptive field) ― Trường thụ cảm tại tầng k là vùng được ký hiệu Rk×Rk của đầu vào mà những pixel của activation map thứ k có thể "nhìn thấy". Bằng cách gọi Fj là kích thước bộ lọc của tầng j và Si là giá trị độ trượt của tầng i và để thuận tiện, ta mặc định S0=1, trường thụ cảm của tầng k được tính toán bằng công thức:

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230; Trong ví dụ bên dưới, ta có F1=F2=3 và S1=S2=1, nên cho ra được R2=1+2⋅1+2⋅1=5.

<br>


**43. Commonly used activation functions**

&#10230; Các hàm kích hoạt thường gặp

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230; Rectified Linear Unit ― Tầng rectified linear unit (ReLU) là một hàm kích hoạt g  được sử dụng trên tất cả các thành phần. Mục đích của nó là tăng tính phi tuyến tính cho mạng. Những biến thể khác của ReLU được tổng hợp ở bảng dưới:

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230; [ReLU, Leaky ReLU, ELU, with]

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230; [Độ phức tạp phi tuyến tính có thể thông dịch được về mặt sinh học, Gán vấn đề ReLU chết cho những giá trị âm, Khả vi tại mọi nơi]

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230; Softmax ― Bước softmax có thể được coi là một hàm logistic tổng quát lấy đầu vào là một vector chứa các giá trị x∈Rn và cho ra là một vector gồm các xác suất p∈Rn thông qua một hàm softmax ở cuối kiến trúc. Nó được định nghĩa như sau:

<br>


**48. where**

&#10230; với

<br>


**49. Object detection**

&#10230; Phát hiện vật thể (Object detection)

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230; Các kiểu mô hình ― Có 3 kiểu thuật toán nhận diện vật thể chính, vì thế mà bản chất của thứ được dự đoán sẽ khác nhau. Chúng được miêu tả ở bảng dưới:

<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230; [Phân loại hình ảnh, Phân loại cùng với khoanh vùng, Phát hiện]

<br>


**52. [Teddy bear, Book]**

&#10230; [Gấu bông, Sách]

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230; [Phân loại một tấm ảnh, Dự đoán xác suất của một vật thể, Phát hiện một vật thể trong ảnh, Dự đoán xác suất của vật thể và định vị nó, Phát hiện nhiều vật thể trong cùng một tấm ảnh, Dự đoán xác suất của các vật thể và định vị chúng]

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230; [CNN cổ điển, YOLO đơn giản hóa, R-CNN, YOLO, R-CNN]

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230; Detection ― Trong bối cảnh phát hiện vật thể, những phương pháp khác nhau được áp dụng tùy thuộc vào liệu chúng ta chỉ muốn định vị vật thể hay phát hiện được những hình dạng phức tạp hơn trong tấm ảnh. Hai phương pháp chính được tổng hợp ở bảng dưới: 

<br>


**56. [Bounding box detection, Landmark detection]**

&#10230; [Phát hiện hộp giới hạn (bounding box), Phát hiện landmark]

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230; [Phát hiện phần trong ảnh mà có sự xuất hiện của vật thể, Phát hiện hình dạng và đặc điểm của một đối tượng (vd: mắt), Nhiều hạt]

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230; [Hộp có tọa độ trung tâm (bx, by), chiều cao bh và chiều rộng bw, Các điểm tương quan (l1x,l1y), ..., (lnx,lny)]

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230; Intersection over Union ― Tỉ lệ vùng giao trên vùng hợp, còn được biết đến là IoU, là một hàm định lượng vị trí Bp của hộp giới hạn dự đoán được định vị đúng như thế nào so với hộp giới hạn thực tế Ba. Nó được định nghĩa:

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230; Lưu ý: ta luôn có IoU∈[0,1]. Để thuận tiện, một hộp giới hạn Bp được cho là khá tốt nếu IoU(Bp,Ba)⩾0.5.

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230; Anchor boxes ― Hộp mỏ neo là một kỹ thuật được dùng để dự đoán những hộp giới hạn nằm chồng lên nhau. Trong thực nghiệm, mạng được phép dự đoán nhiều hơn một hộp cùng một lúc, trong đó mỗi dự đoán được giới hạn theo một tập những tính chất hình học cho trước. Ví dụ, dự đoán đầu tiên có khả năng là một hộp hình chữ nhật có hình dạng cho trước, trong khi dự đoán thứ hai sẽ là một hộp hình chữ nhật nữa với hình dạng hình học khác.

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230; Non-max suppression ― Kỹ thuật non-max suppression hướng tới việc loại bỏ những hộp giới hạn bị trùng chồng lên nhau của cùng một đối tượng bằng cách chọn chiếc hộp có tính đặc trưng nhất. Sau khi loại bỏ tất cả các hộp có xác suất dự đoán nhỏ hơn 0.6, những bước tiếp theo được lặp lại khi vẫn còn tồn tại những hộp khác.

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230; [Với một lớp cho trước, Bước 1: Chọn chiếc hộp có xác suất dự đoán lớn nhất., Bước 2: Loại bỏ những hộp có IoU⩾0.5 với hộp đã chọn.]

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230; [Các dự đoán hộp, Chọn hộp với xác suất cao nhất, Loại bỏ trùng lặp trong cùng một lớp, Các hộp giới hạn cuối cùng]

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230; YOLO ― You Only Look Once (YOLO) là một thuật toán phát hiện vật thể thực hiện những bước sau:

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230; [Bước 1: Phân chia tấm ảnh đầu vào thành một lưới G×G., Bước 2: Với mỗi lưới, chạy một mạng CNN dự đoán y có dạng sau:, lặp lại k lần]

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230; với pc là xác suất dự đoán được một vật thể, bx,by,bh,bw là những thuộc tính của hộp giới hạn được dự đoán, c1,...,cp là biểu diễn one-hot của việc lớp nào trong p các lớp được dự đoán, và k là số lượng các hộp mỏ neo.

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230; Bước 3: Chạy thuật toán non-max suppression để loại bỏ bất kỳ hộp giới hạn có khả năng bị trùng lặp.

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230; [Ảnh gốc, Phân chia thành lưới GxG, Dự đoán hộp giới hạn, Non-max suppression]

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230; Lưu ý: khi pc=0, thì mạng không phát hiện bất kỳ vật thể nào. Trong trường hợp đó, Các dự đoán liên quan bx,...,cp sẽ bị lờ đi.

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230; R-CNN ― Region with Convolutional Neural Networks (R-CNN) là một thuật toán phát hiện vật thể mà đầu tiên phân chia ảnh thành các vùng để tìm các hộp giới hạn có khả năng liên quan cao rồi chạy một thuật toán phát hiện để tìm những thứ có khả năng cao là vật thể trong những hộp giới hạn đó. 

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230; [Ảnh gốc, Phân vùng, Dự đoán hộp giới hạn, Non-max suppression]

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230; Lưu ý: mặc dù thuật toán gốc có chi phí tính toán cao và chậm, những kiến trúc mới đã có thể cho phép thuật toán này chạy nhanh hơn, như là Fast R-CNN và Faster R-CNN.

<br>


**74. Face verification and recognition**

&#10230; Xác nhận khuôn mặt và nhận diện khuôn mặt

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230; Các kiểu mô hình ― Hai kiểu mô hình chính được tổng hợp trong bảng dưới:

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230; [Xác nhận khuôn mặt, Nhận diện khuôn mặt, Truy vấn, Tham vấn, Cơ sở dữ liệu]

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230; [Có đúng người không?, Tra cứu một-một, Đây có phải là 1 trong K người trong cơ sở dữ liệu không?, Tra cứu một với tất cả]

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230; One Shot Learning ― One Shot Learning là một thuật toán xác minh khuôn mặt sử dụng một tập huấn luyện hạn chế để học một hàm similarity nhằm ước lượng sự khác nhau giữa hai tấm hình. Hàm này được áp dụng cho hai tấm ảnh thường được ký hiệu d(image 1,image 2).

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230; Siamese Network ― Siamese Networks hướng tới việc học cách mã hóa tấm ảnh để rồi định lượng sự khác nhau giữa hai tấm ảnh. Với một tấm ảnh đầu vào x(i), đầu ra được mã hóa thường được ký hiệu là f(x(i)).

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230; Triplet loss ― Triplet loss ℓ là một hàm mất mát được tính toán dựa trên biểu diễn nhúng của bộ ba hình ảnh A (mỏ neo), P (dương tính) và N(âm tính). Ảnh mỏ neo và ảnh dương tính đều thuộc một lớp, trong khi đó ảnh âm tính thuộc về một lớp khác. Bằng các gọi α∈R+ là tham số margin, hàm mất mát này được định nghĩa như sau:

<br>


**81. Neural style transfer**

&#10230; Neural style transfer

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230; Ý tưởng ― Mục tiêu của neural style transfer là tạo ra một ảnh G dựa trên một nội dung C và một phong cách S. 

<br>


**83. [Content C, Style S, Generated image G]**

&#10230; [Nội dung C, Phong cách S, Ảnh tạo được G]

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230; Tầng kích hoạt ― Trong một tầng l cho trước, tầng kích hoạt được ký hiệu a[l] và có các chiều là nH×nw×nc

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230; Hàm mất mát nội dung ― Hàm mất mát nội dung Jcontent(C,G) được sử dụng để xác định nội dung của ảnh được tạo G khác biệt với nội dung gốc trong ảnh C. Nó được định nghĩa như dưới đây:

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230; Ma trận phong cách ― Ma trận phong cách G[l] của một tầng cho trước l  là một ma trận Gram mà mỗi thành phần G[l]kk′ của ma trận xác định sự tương quan giữa kênh k và kênh k'. Nó được định nghĩa theo tầng kích hoạt a[l] như sau:

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230; Lưu ý: ma trận phong cách cho ảnh phong cách và ảnh được tạo được ký hiệu tương ứng là G[l] (S) và G[l] (G).

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230; Hàm mất mát phong cách ― Hàm mất mát phong cách Jstyle(S,G) được sử dụng để xác định sự khác biệt về phong cách giữa ảnh được tạo G và ảnh phong cách S. Nó được định nghĩa như sau:

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230; Hàm mất mát tổng quát ― Hàm mất mát tổng quát được định nghĩa là sự kết hợp của hàm mất mát nội dung và hàm mất mát phong cách, độ quan trọng của chúng được xác định bởi hai tham số α,β, như dưới đây:

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230; Lưu ý: giá trị của α càng lớn dẫn tới việc mô hình sẽ quan tâm hơn cho nội dung, trong khi đó, giá trị của β càng lớn sẽ khiến nó quan tâm hơn đến phong cách.

<br>


**91. Architectures using computational tricks**

&#10230; Những kiến trúc sử dụng computational tricks

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**


&#10230; Generative Adversarial Network ― Generative adversarial networks, hay còn được gọi là GAN, là sự kết hợp giữa mô hình khởi tạo và mô hình phân biệt, khi mà mô hình khởi tạo cố gắng tạo ra hình ảnh đầu ra chân thực nhất, sau đó được đưa vô mô hình phân biệt, mà mục tiêu của nó là phân biệt giữa ảnh được tạo và ảnh thật.

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

&#10230; [Huấn luyện, Nhiễu, Ảnh thật, Mô hình khởi tạo, Mô hình phân biệt, Thật Giả]

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230; Lưu ý: có nhiều loại GAN khác nhau bao gồm từ văn bản thành ảnh, sinh nhạc và tổ hợp.

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230; ResNet ― Kiến trúc Residual Network (hay còn gọi là ResNet) sử dụng những khối residual (residual blocks) cùng với một lượng lớn các tầng để giảm lỗi huấn luyện. Những khối residual có những tính chất sau đây:

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230; Inception Network ― Kiến trúc này sử dụng những inception module và hướng tới việc thử các tầng tích chập khác nhau để tăng hiệu suất thông qua sự đa dạng của các feature. Cụ thể, kiến trúc này sử dụng thủ thuật tầng tích chập 1×1 để hạn chế gánh nặng tính toán. 

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

&#10230; Những cheatsheet về Deep Learning nay đã được dịch sang [target language].

<br>


**98. Original authors**

&#10230; Các tác giả

<br>


**99. Translated by X, Y and Z**

&#10230; Được dịch bởi X, Y và Z

<br>


**100. Reviewed by X, Y and Z**

&#10230; Xem qua bởi X, Y và Z

<br>


**101. View PDF version on GitHub**

&#10230; Xem bản PDF trên Github

<br>


**102. By X and Y**

&#10230; Bởi X và Y

<br>
