**1. Deep Learning cheatsheet**

برگه یاداشت یادگیری ژرف(عمیق)

<br>

**2. Neural Networks**

شبکه های عصبی

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

شبکه های عصبی یک کلاس از مدل هایی هستند که با لایه بندی ساخته میشوند(ساختاری لایه مانند دارند). شبکه های عصبی پیچشی ( کانولوشنی (CNN)) و شبکه های عصبی برگشتی (RNN) انواع رایج شبکه های عصبی هستند.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

معماری - واژه معماری در شبکه های عصبی به شکل زیر اطلاق میشود: 
<br>

**5. [Input layer, hidden layer, output layer]**

[لایه ورودی،لایه پنهان،لایه خروجی]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

با نمایش i به عنوان لایه iام و j به عنوان واحد jام مخفی از لایه ، داریم:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

که به ترتیب w،b،z وزن ، انحراف و خروجی هستند.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

تابع فعال سازی - توابع فعال سازی در انتهای لایه پنهان برای معرفی پیچیدگی غیر خطی به مدل استفاده میشود.در اینجا رایج ترین آنها نمایش داده شده است:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

[سیگموئید،تانژانت هذلولوی،یکسو ساز،یکسوساز رخنه گر]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

تابع هزینه آنتروپی متقابل - 
در متن شبکه های عصبی ، عموما از تابع هزینه آنتروپی متقابل L(z,y) استفاده میشود و به صورت زیر تعریف میشود:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

نرخ یادگیری - نرخ یادگیری اغلب با نماد α و گاهی اوقات با نماد η نمایش داده میشود و بیانگر سرعت (گام) بروزرسانی وزن ها است که میتواند مقداری ثابت یا به صورت تطبیقی تفییر کند .محبوبترین متد حال حاضر Adam نام دارد، متدی است که نرخ یادگیری را در حین فرآیند آموزش تنظیم می‌کند .

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

پس انتشار خطا - پس انتشار خطا روشی برای بروزرسانی وزن ها با توجه به خروجی واقعی و خروجی مورد انتظار در شبکه عصبی است . مشتق نسبت به وزن W توسط قاعده زنجیری
 محاسبه میشود و به شکل زیر است:
 
<br>

**13. As a result, the weight is updated as follows:**

در نتیجه، وزن به صورت زیر بروز رسانی میشود:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

بروز رسانی وزن ها - در یک شبکه عصبی، وزن ها به صورت زیر بروز رسانی میشوند:

<br>

**15. Step 1: Take a batch of training data.**

گام 1: یک دسته از نمونه های آموزشی را بگیر.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

گام دوم: الگوریتم انتشار رو به جلو را برای بدست آوردن هزینه مربوطه اجرا کن.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

گام 3: هزینه را به عقب انتشار بده تا گرادیان ها بدست آیند.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

گام 4: از گرادیان ها برای بروز رسانی وزن های شبکه استفاده کن.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

حذف تصادفی – حذف تصادفی یک تکنیک برای جلوگیری از بیش بر ارزش شدن داده های آموزشی با حذف تصادفی اتصال ها در یک شبکه عصبی است. در عمل، نرون ها با احتمال p حذف یا با احتمال 1-p حفظ میشوند. 

<br>

**20. Convolutional Neural Networks**

شبکه های عصبی پیچشی ( کانولوشنی)

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

الزامات لایه کانولوشنی - با نمایش W اندازه توده ورودی، F اندازه نرون های لایه کانولوشنی ، P اندازه گسترش مرز (صفر)، تعداد نرونهای N که در توده داده شده قرار میگیرند برابر است با :

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

نرمال سازی دسته ای - یک گام از ابر پارامتر های γ,β است که دسته {xi} را نرمال میکند.نماد μB و σ2B به میانگین و واریانس دسته ای که میخواهیم آن را اصلاح کنیم اشاره دارد که به صورت زیر انجام میشود:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

معمولا بعد از یک لایه تماما متصل/لایه کانولوشنی و قبل از لایه غیر خطی اعمال میشود و این جازه را به ما میدهد که نرخ یادگیری بالاتر داشته باشیم از طرفی وابستگی شدید مدل را به مقدار دهی اولیه کاهش میدهد. 

<br>

**24. Recurrent Neural Networks**

شبکه های عصبی برگشتی

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

انواع گیت ها - اینها انواع مختلف گیت هایی هستند که ما در یک شبکه عصبی برگشتی معمولی به آنها برمیخوریم :

<br>

**26. [Input gate, forget gate, gate, output gate]**

[گیت ورودی، گیت فراموشی،گیت،گیت خروجی]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

[در سلول بنویسد یا خیر؟، سلول را پاک کند یا خیر؟ چه مقدار در سلول بنویسد؟، چه مقدار به سلول بروز دهید؟]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

یک شبکه حافظه کوتاه‌-مدت طولانی (LSTM) یک نوع از مدل های RNN است که مشکل ناپدید شدن گرادیان را با اضافه کردن 'گیت فراموشی' حل میکند.

<br>

**29. Reinforcement Learning and Control**

یادگیری تقویتی و کنترل

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

هدف یادگیری تقویتی برای یک عامل این است که یادبگیرد در یک محیط چگونه رشد کند

<br>

**31. Definitions**

تعاریف 

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

فرایندهای تصمیم‌گیری مارکوف- یک فرآیند تصمیم گیری مارکوف (به اختصار MDP) شامل پنج عنصر (S,A,{Psa},γ,R) است بطوریکه:

<br>

**33. S is the set of states**

S مجموعه ای از حالات است

<br>

**34. A is the set of actions**

A مجموعه ای از اکشن هاست

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;

<br>

**36. γ∈[0,1[ is the discount factor**

γ∈[0,1]  عامل تنزیل است

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

R:S×A⟶R یا R:S⟶R تابع پاداش است که الگوریتم ما میخواهد آن را بیشینه کند

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

یک سیاست π تابعیست که π:S⟶A مجموعه حالات را به اکشن ها نگاشت میدهد. 

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230;

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

تابع ارزش - برای یک سیایت
<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

الگوریتم تکرار ارزش - الگوریتم تکرار ارزش دو گام دارد:

<br>

**44. 1) We initialize the value:**

1) ارزش را مقدار دهی اولیه میکنیم:

<br>

**45. 2) We iterate the value based on the values before:**

ارزش را با توجه به ارزشهای قبلی تکرار میکنیم:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;

<br>

**47. times took action a in state s and got to s′**

مدت زمانی که عمل a در حالت s است و به حالت 
s′ میرود

<br>

**48. times took action a in state s**

مدت زمانی که عمل a در حالت s است

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

یادگیری کیو - یادگیری کیو نوعی از یادگیری تقویتی بدون مدل برای تخمین Q است که به صورت زیر انجام میشود:

<br>

**50. View PDF version on GitHub**

نسخه PDF را در گیت هاب ببینید

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

[شبکه های عصبی،معماری،توابع فعالسازی،پس انتشار خطا،حذف تصادفی]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

[شبکه های عصبی پیچشی (کانولوشنی)،لایه پیچشی (کانولوشنی)،نرمال سازی دسته ای ]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

[شبکه های عصبی بازگشتی،گیت ها (Gates)،حافظه کوتاه‌-مدت طولانی (LSTM)]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

[ یادگیری تقویتی، فرایند های تصمیم گیری مارکوف، تکرار ارزش/سیاست ، برنامه نویسی پویا تقریبی، کنکاش سیاست]
