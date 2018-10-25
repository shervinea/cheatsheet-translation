
**1. Deep Learning cheatsheet**

&#10230;
ورقة غش التعلم العميق
<br> 

**2. Neural Networks**

&#10230;
الشبكة العصبونية الاصطناعية
<br> 
**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230;
الشبكة العصبونية الاصطناعيةهي عبارة عن نوع من النماذج يبنى من عدة طبقات , اكثر هذة الانواع استخداما هي الشبكات الالتفافية و الشبكات العصبونية المتكرره

<br> 

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230;
البنية - المصطلحات حول بنية الشبكة العصبونية موضح في الشكل ادناة
<br> 

**5. [Input layer, hidden layer, output layer]**

&#10230;
[طبقة ادخال, طبقة مخفية, طبقة اخراج ]
<br>  

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;
عبر تدوين i كالطبقة رقم i و j للدلالة على رقم الوحده الخفية في تلك الطبقة , نحصل على:
<br>  

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230;
حيث نعرف w, b, z كالوزن , و معامل التعديل , و الخرج حسب الترتيب.
<br>  

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230;
دالة التفعيل - دالة التفعيل تستخدم في نهاية الوحده الخفية لتضمن المكونات الغير خطية للنموذج. هنا بعض دوال التفعيل الشائعة
<br> 

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;
[Sigmoid, Tanh, ReLU, Leaky ReLU] 
<br> 

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;
دالة الانتروبيا التقاطعية للخسارة - في سياق الشبكات العصبونية, دالة الاتاروبيا L(z,y) تستخدم و تعرف كالاتي:
<br>  

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;
نسبة سرعة التعلم - نسبة التعلم, يرمز , و هو مؤشر في اي تجاة يتم تحديث الاوزان. يمكن تثبيت هذا المعامل او تحديثة بشكل تأقلمي . حاليا اكثر النسب شيوعا تدعى Adam , وهي طريقة تجعل هذه النسبة سرعة التعلم بشكل تأقلمي    α او η ب , 
<br>  

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;
التغذية الخلفية - التغذية الخلفية هي طريقة لتحديث الاوزان في الشبكة العصبونية عبر اعتبار القيم الحقيقة للخرج مع القيمة المطلوبة للخرج. المشتقة بالنسبة للوزن w يتم حسابها باستخدام قاعدة التسلسل و تكون عبر الشكل الاتي: 
<br>

**13. As a result, the weight is updated as follows:**

&#10230;
كنتيجة , الوزن سيتم تحديثة كالتالي:
<br> 

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;
تحديث الاوزان - في الشبكات العصبونية , يتم تحديث الاوزان كما يلي: 
<br>  

**15. Step 1: Take a batch of training data.**

&#10230;
الخطوة 1: خذ حزمة من بيانات التدريب
<br> 

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;
الخطوة 2: قم بعملية التغذيه الامامية لحساب الخسارة الناتجة
<br> 

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230;
الخطوة 3: قم بتغذية خلفية للخساره للحصول على دالة الانحدار
<br>  

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230;
الخطوة 4: استخدم قيم الانحدار لتحديث اوزان الشبكة
<br> 

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;
الاسقاط - الاسقاط هي طريقة الغرض منها منع التكيف الزائد للنموذج في بيانات التدريب عبر اسقاط بعض الواحدات في الشبكة العصبونية, العصبونات يتم اما اسقاطها باحتمالية p او الحفاظ عليها باحتمالية 1-p.
<br>  

**20. Convolutional Neural Networks**

&#10230;
الشبكات العصبونية الالتفافية
<br> 

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230;
احتياج الطبقة الالتفافية - عبر رمز w لحجم المدخل , F حجم العصبونات للطبقة الالتفافية , P عدد الحشوات الصفرية , فأن N عدد العصبونات لكل حجم معطى يحسب عبر الاتي: 
<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;
تنظيم الحزمة - هي خطوه من قيم التحسين الخاصة γ,β  والتي تعدل الحزمة {xi}. لنجعل μB,σ2B المتوسط و الانحراف للحزمة المعنية و نريد تصحيح هذه الحزمة, يتم ذلك كالتالي:    
<br> 

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;
في الغالب تتم بعد الطبقة الالتفافية المتصلة كليا و قبل طبقة التغيرات الغير خطية و تهدف للسماح للسرعات التعليم العالية للتقليل من الاعتمادية القوية للقيم الاولية.


<br>

**24. Recurrent Neural Networks**

&#10230;
الشبكات العصبونية التكرارية
<br> 

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;
انواع البوابات - هنا الانواع المختلفة التي ممكن مواجهتها في الشبكة العصبونية الاعتيادية:
<br>  

**26. [Input gate, forget gate, gate, output gate]**

&#10230;
[بوابة ادخال, بوابة نسيان, بوابة منفذ, بوابة اخراج ]
<br> 

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;
[كتابة ام عدم كتابة الى الخلية؟, مسح ام عدم مسح الخلية؟, كمية الكتابة الى الخلية ؟ , مدى الافصاح عن الخلية ؟ ]
<br> 

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;
LSTM - ذاكرة طويلة قصير الامد (long short-term memory) هي نوع من نموذج ال RNN تستخدم لتجنب مشكلة اختفاء الانحدار عبر اضافة بوابات النسيان.
<br>  

**29. Reinforcement Learning and Control**

&#10230;
التعلم و التحكم المعزز
<br> 

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;
الهدف من التعلم المعزز للعميل الذكي هو التعلم لكيفية التأقلم في اي بيئة.
<br>  

**31. Definitions**

&#10230;
تعريفات
<br> 

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;
عملية ماركوف لاتخاذ القرار - عملية ماركوف لاتخاذ القرار هي سلسلة خماسية (S,A,{Psa},γ,R) حيث

<br> 
**33. S is the set of states**

&#10230;
 S هي مجموعة من حالات البيئة
<br>

**34. A is the set of actions**

&#10230;
A هي مجموعة من حالات الاجراءات
<br> 
**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;
{Psa} هو حالة احتمال الانتقال من الحالة s∈S و a∈A
<br>  

**36. γ∈[0,1[ is the discount factor**

&#10230;
γ∈[0,1[ هي عامل الخصم
<br>   

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;
R:S×A⟶R or R:S⟶R  هي دالة المكافأة والتي تعمل الخوارزمية على جعلها اعلى قيمة
<br> 

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;
دالة القواعد - دالة القواعد π:S⟶A  هي التي تقوم بترجمة الاحالات الى اجراءات.
<br>  

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230;
تعليق: نقول ان النموذج ينفذ القاعدة المعينه π للحالة المعطاة s ان نتخذ الاجراءa=π(s).  
<br>  
 
**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;
دالة القاعدة - لاي قاعدة معطاة π و حالة s, نقوم بتعريف دالة القيمة Vπ  كما يلي:  
<br>    

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;
معادلة بيلمان - معادلات بيلمان المثلى تشخص دالة القيمة دالة القيمة Vπ∗  π∗:للقاعدة المثلى 
<br>  

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;
  π∗ للحالة المعطاه s تعطى كاالتالي: تعليق: نلاحظ ان القاعدة المثلى
<br>  

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;
خوارزمية تكرار القيمة - خوارزمية تكرار القيمة تكون في خطوتين:
<br>  

**44. 1) We initialize the value:**

&#10230;
 1) نقوم بوضع قيمة اولية:
<br> 

**45. 2) We iterate the value based on the values before:**

&#10230;
2) نقوم بتكرير القيمة حسب القيم السابقة: 

<br> 
**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;
تقدير الامكانية القصوى - تقديرات الامكانية القصوى (تقدير الاحتمال الأرجح) لحتماليات انتقال الحالة تكون كما يلي : 
<br>   

**47. times took action a in state s and got to s′**

&#10230;
اوقات تنفيذ الاجراء a في الحالة s و انتقلت الى s' 

<br> 
**48. times took action a in state s**

&#10230;
اوقات تنفيذ الاجراء a في الحالة s
<br>  

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
التعلم-Q (Q-learning) -هي طريقة لاتحتاج لنموذج للبيئة لتقدير Q , و تتم كالاتي:
<br>  
**50. View PDF version on GitHub**

&#10230;
قم باستعراض نسخة ال PDF على GitHub
<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;
 [شبكات عصبونية, البنية , دالة التفعيل , التغذية الخلفية , الاسقاط ]
<br> 

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;
[ الشبكة العصبونية الالتفافية , طبقة التفافية , تنظيم الحزمة ] 
<br>  

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230;
[الشبكة العصبونية التكرارية , البوابات , LSTM]
<br>  

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
[التعلم المعزز , عملية ماركوف لاتخاذ القرار , تكرير القيمة / القاعدة , بحث القاعدة]
