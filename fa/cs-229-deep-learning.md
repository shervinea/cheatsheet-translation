**1. Deep Learning cheatsheet**

<div dir="rtl">
راهنمای کوتاه یادگیری عمیق
</div>

<br>

**2. Neural Networks**

<div dir="rtl">
شبکه‌های عصبی
</div>

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

<div dir="rtl">
شبکه‌های عصبی دسته‌ای از مدل‌هایی هستند که با لایه‌بندی ساخته میشوند (ساختاری چند لایه دارند). شبکه‌های عصبی پیچشی ( کانولوشنی (CNN)) و شبکه‌های عصبی بازگشتی (RNN) انواع رایج شبکه‌های عصبی هستند.
</div>

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

<div dir="rtl">
معماری - واژه معماری در شبکه‌های عصبی در شکل زیر توصیف شده است:
</div>

<br>

**5. [Input layer, hidden layer, output layer]**

<div dir="rtl">
[لایه‌ی ورودی، لایه‌ی پنهان، لایه‌ی خروجی]
</div>

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

<div dir="rtl">
با نمایش $i$ به عنوان لایه $i$ام و $j$ به عنوان واحد $j$ام پنهان آن لایه، داریم:
</div>

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

<div dir="rtl">
که به ترتیب $w$، $b$، و $z$ وزن، پیش‌قدر، و خروجی لایه هستند.
</div>

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

<div dir="rtl">
تابع فعال‌سازی - توابع فعال‌سازی در انتهای واحد پنهان برای معرفی پیچیدگی غیر خطی به مدل استفاده می‌شوند. در اینجا رایج‌ترین آنها نمایش داده شده است:
</div>

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

<div dir="rtl">
[سیگموئید، تانژانت هذلولوی، یکسو ساز، یکسو ساز نشتی‌دار]
</div>

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

<div dir="rtl">
خطای آنتروپی متقاطع - در مضمون شبکه‌های عصبی، عموما از تابع خطای آنتروپی متقاطع $L(z, y)$ استفاده می‌شود که به صورت زیر تعریف می‌شود:
</div>

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

<div dir="rtl">
نرخ یادگیری - نرخ یادگیری اغلب با نماد $\alpha$ و گاهی اوقات با نماد $\eta$ نمایش داده می‌شود و بیانگر سرعت (گام) بروزرسانی وزن‌ها است که میتواند مقداری ثابت یا به سازگارشونده تغییر کند. محبوب‌ترین روش حال حاضر Adam نام دارد، متدی است که نرخ یادگیری را در حین فرآیند آموزش تنظیم می‌کند.
</div>

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

<div dir="rtl">
انتشار معکوس - انتشار معکوس روشی برای بروزرسانی وزن‌ها با توجه به خروجی واقعی و خروجی مورد انتظار در شبکه‌ی عصبی است. مشتق نسبت به وزن $W$ توسط قاعده‌ی زنجیری محاسبه می‌شود و به شکل زیر است:
</div>

<br>

**13. As a result, the weight is updated as follows:**

<div dir="rtl">
در نتیجه، وزن به صورت زیر بروز‌رسانی می‌شود:
</div>

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

<div dir="rtl">
بروزرسانی وزن‌ها - در یک شبکه‌ی عصبی، وزن‌ها به صورت زیر بروزرسانی می‌شوند:
</div>

<br>

**15. Step 1: Take a batch of training data.**

<div dir="rtl">
گام ۱: یک دسته از داده‌های آموزشی را تهیه می‌کنیم.
</div>

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

<div dir="rtl">
گام ۲: الگوریتم انتشار مستقیم را برای بدست آوردن خطای مربوطه اجرا می‌کنیم.
</div>

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

<div dir="rtl">
گام ۳: خطا را انتشار معکوس می‌دهیم تا گرادیان‌ها به دست بیایند.
</div>

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

<div dir="rtl">
گام ۴: از گرادیان‌ها برای بروزرسانی وزن‌های شبکه استفاده می‌کنیم.
</div>

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

<div dir="rtl">
برون‌اندازی – برون‌اندازی یک روش برای جلوگیری از بیش‌برازش بر روی داده‌های آموزشی با حذف تصادفی واحدها در یک شبکه‌ی عصبی است. در عمل، واحدها با احتمال $p$ حذف یا با احتمال $1-p$ حفظ می‌شوند. 
</div>

<br>

**20. Convolutional Neural Networks**

<div dir="rtl">
شبکه‌های عصبی پیچشی ( کانولوشنی)
</div>

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

<div dir="rtl">
الزامات لایه کانولوشنی - با نمایش $W$ اندازه توده‌ی ورودی، $F$ اندازه نورون‌های لایه‌ی کانولوشنی، $P$ اندازه‌ی حاشیه‌ی صفر، تعداد نورون‌های $N$ که در توده‌ی داده شده قرار می‌گیرند برابر است با:
</div>

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

<div dir="rtl">
نرمال‌سازی دسته‌ای - یک مرحله از فراعامل‌های $\gamma$ و $\beta$ که دسته‌ی $\{x_i\}$ را نرمال می‌کند در زیر آمده است.نماد $\mu_B$ و $\sigma^2_B$ به میانگین و واریانس دسته‌ای که میخواهیم آن را اصلاح کنیم اشاره دارد که به صورت زیر است:
</div>

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

<div dir="rtl">
معمولا بعد از یک لایه‌ی تمام‌متصل یا لایه‌ی کانولوشنی و قبل از یک لایه‌ی غیرخطی اعمال می‌شود و امکان استفاده از نرخ یادگیری بالاتر را می‌دهد و همچنین باعث می‌شود که وابستگی شدید مدل به مقداردهی اولیه کاهش یابد. 
</div>

<br>

**24. Recurrent Neural Networks**

<div dir="rtl">
شبکه‌های عصبی بازگشتی
</div>

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

<div dir="rtl">
انواع دروازه‌ها - انواع مختلف دروازه‌هایی که در یک شبکه‌ی عصبی بازگشتی معمولی به آنها برمی‌خوریم در زیر آمده‌اند:
</div>

<br>

**26. [Input gate, forget gate, gate, output gate]**

<div dir="rtl">
[دروازه‌ی ورودی، دروازه‌ی فراموشی، دروازه، دروازه‌ی خروجی]
</div>

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

<div dir="rtl">
[در سلول بنویسد یا خیر؟، سلول را پاک کند یا خیر؟، چه مقدار در سلول بنویسد؟، چه مقدار برای سلول آشکار کند؟]
</div>

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

<div dir="rtl">
LSTM - یک شبکه‌ی حافظه‌ی کوتاه‌-مدت طولانی (LSTM) یک نوع از مدل‌های RNN است که مشکل ناپدید شدن (صفر شدن) گرادیان را با اضافه کردن «دروازه‌ی فراموشی» حل می‌کند.
</div>

<br>

**29. Reinforcement Learning and Control**

<div dir="rtl">
یادگیری تقویتی و کنترل
</div>

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

<div dir="rtl">
هدف یادگیری تقویتی برای یک عامل این است که یاد بگیرد در یک محیط چگونه تکامل یابد.
</div>

<br>

**31. Definitions**

<div dir="rtl">
تعاریف 
</div>

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

<div dir="rtl">
فرایندهای تصمیم‌گیری مارکوف - یک فرآیند تصمیم‌گیری مارکوف (به اختصار MDP) شامل پنج‌تایی $(S, A, \{P_{s, a}\}, \gamma, R)$ است به طوری که:
</div>

<br>

**33. S is the set of states**

<div dir="rtl">
$S$ مجموعه‌ی  حالات است
</div>

<br>

**34. A is the set of actions**

<div dir="rtl">
$A$ مجموعه‌ای از کنش‌ها است
</div>

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

<div dir="rtl">
$\{P_{s, a}\}$ احتمالات انتقال وضعیت برای هر $a \in A$ و $s \in S$ هستند.
</div>

<br>

**36. γ∈[0,1[ is the discount factor**

<div dir="rtl">
$\gamma \in [0, 1[$ ضریب تخفیف است.
</div>

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

<div dir="rtl">
$R: S \times A \rightarrow R$ یا $R: S \rightarrow R$ تابع پاداشی است که الگوریتم سعی دارد آن را بیشینه بکند.
</div>

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

<div dir="rtl">
خط‌مشی - یک خط‌مشی $\pi$ تابعی است $\pi : S \rightarrow A$ که حالات را به کنش‌ها نگاشت می‌کند.
</div>

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

<div dir="rtl">
نکته: می‌گوییم ما در حال اجرای خط‌مشی $\pi$ هستیم اگر به ازای وضعیت $s$ کنش $a = \pi(s)$ را اجرا کنیم.
</div>

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

<div dir="rtl">
تابع ارزش - برای سیاست $\pi$ و وضعیت $s$، تابع ارزش $V_\pi$ را به صورت زیر تعریف می‌کنیم:
</div>

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

<div dir="rtl">
معادله‌ی بلمن - معادله‌ی بلمن بهینه‌ی تابع ارزش $V_{\pi^*}$ مربوط به خط‌مشی بهینه‌ی $\pi^*$ را مشخص می‌کند:
</div>

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

<div dir="rtl">
نکته: سیاست بهینه‌ی $\pi^*$ برای وضعیت $s$ به این صورت است که:
</div>

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

<div dir="rtl">
الگوریتم تکرار ارزش - الگوریتم تکرار ارزش دو گام دارد:
</div>

<br>

**44. 1) We initialize the value:**

<div dir="rtl">
۱) ارزش را مقداردهی اولیه می‌کنیم:
</div>

<br>

**45. 2) We iterate the value based on the values before:**

<div dir="rtl">
۲) ارزش را با توجه به ارزش‌های قبلی تکرار می‌کنیم:
</div>

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

<div dir="rtl">
تخمین درست‌نمایی بیشینه - تخمین‌های درست‌نمایی بیشینه برای احتمالات انتقال وضعیت به صورت زیر است:
</div>

<br>

**47. times took action a in state s and got to s′**

<div dir="rtl">
دفعاتی که کنش $a$ در وضعیت $s$ انتخاب شد و منجر به رفتن به وضعیت $s'$ شد.
</div>

<br>

**48. times took action a in state s**

<div dir="rtl">
دفعاتی که کنش $a$ در وضعیت $s$ اجرا شد.
</div>

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

<div dir="rtl">
یادگیری Q - یادگیری Q نوعی از یادگیری تقویتی بدون مدل برای تخمین Q است که به صورت زیر انجام می‌شود:
</div>

<br>

**50. View PDF version on GitHub**

<div dir="rtl">
نسخه‌ی PDF را در گیت‌هاب ببینید
</div>

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

<div dir="rtl">
[شبکه‌های عصبی، معماری، تابع فعال‌سازی، انتشار معکوس، برون‌اندازی]
</div>

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

<div dir="rtl">
[شبکه‌های عصبی پیچشی (کانولوشنی)، لایه‌ی پیچشی (کانولوشنی)، نرمال‌سازی دسته‌ای]
</div>

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

<div dir="rtl">
[شبکه های عصبی بازگشتی، دروازه‌ها، شبکه با حافظه‌ی کوتاه‌-مدت طولانی (LSTM)]
</div>

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

<div dir="rtl">
[یادگیری تقویتی، فرایندهای تصمیم‌گیری مارکوف، تکرار ارزش/خط‌مشی، برنامه‌نویسی پویای تقریبی، جست‌وجوی خط‌مشی]
</div>
