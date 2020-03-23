
**1. Deep Learning cheatsheet**

&#10230;
<div dir="rtl">
ملخص مختصر التعلم العميق
</div>
<br> 

**2. Neural Networks**

&#10230;
<div dir="rtl">
الشبكة العصبونية الاصطناعية(Neural Networks)
</div>
<br> 
**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230;
<div dir="rtl">
الشبكة العصبونية الاصطناعيةهي عبارة عن نوع من النماذج يبنى من عدة طبقات , اكثر هذة الانواع استخداما هي الشبكات الالتفافية و الشبكات العصبونية المتكرره

</div>
<br> 

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230;
<div dir="rtl">
البنية - المصطلحات حول بنية الشبكة العصبونية موضح في الشكل ادناة
</div>
<br> 

**5. [Input layer, hidden layer, output layer]**

&#10230;
<div dir="rtl">
[طبقة ادخال, طبقة مخفية, طبقة اخراج ]
</div>
<br>  

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;
<div dir="rtl">
عبر تدوين i كالطبقة رقم i و j للدلالة على رقم الوحده الخفية في تلك الطبقة , نحصل على:
</div>
<br>  

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230;
<div dir="rtl">
حيث نعرف w, b, z كالوزن , و معامل التعديل , و الناتج حسب الترتيب.
</div>
<br>  

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230;
<div dir="rtl">
دالة التفعيل(Activation function) - دالة التفعيل تستخدم في نهاية الوحده الخفية لتضمن المكونات الغير خطية للنموذج. هنا بعض دوال التفعيل الشائعة
</div>
<br> 

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;
<div dir="rtl">
[Sigmoid, Tanh, ReLU, Leaky ReLU] 
</div>
<br> 

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;
<div dir="rtl">
دالة الانتروبيا التقاطعية للخسارة(Cross-entropy loss) - في سياق الشبكات العصبونية, دالة الأنتروبيا L(z,y) تستخدم و تعرف كالاتي:
</div>
<br>  

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;
<div dir="rtl">
معدل التعلم(Learning rate) - معدل التعلم, يرمز , و هو مؤشر في اي تجاة يتم تحديث الاوزان. يمكن تثبيت هذا المعامل او تحديثة بشكل تأقلمي . حاليا اكثر النسب شيوعا تدعى Adam , وهي طريقة تجعل هذه النسبة سرعة التعلم بشكل تأقلمي    α او η ب , 
</div>
<br>  

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;
<div dir="rtl">
التغذية الخلفية(Backpropagation) - التغذية الخلفية هي طريقة لتحديث الاوزان في الشبكة العصبونية عبر اعتبار القيم الحقيقة للناتج مع القيمة المطلوبة للخرج. المشتقة بالنسبة للوزن w يتم حسابها باستخدام قاعدة التسلسل و تكون عبر الشكل الاتي: 
</div>
<br>

**13. As a result, the weight is updated as follows:**

&#10230;
<div dir="rtl">
كنتيجة , الوزن سيتم تحديثة كالتالي:
</div>
<br> 

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;
<div dir="rtl">
تحديث الاوزان - في الشبكات العصبونية , يتم تحديث الاوزان كما يلي: 
</div>
<br>  

**15. Step 1: Take a batch of training data.**

&#10230;
<div dir="rtl">
الخطوة 1: خذ حزمة من بيانات التدريب
</div>
<br> 

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;
<div dir="rtl">
الخطوة 2: قم بعملية التغذيه الامامية لحساب الخسارة الناتجة
</div>
<br> 

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230;
<div dir="rtl">
الخطوة 3: قم بتغذية خلفية للخساره للحصول على دالة الانحدار
</div>
<br>  

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230;
<div dir="rtl">
الخطوة 4: استخدم قيم الانحدار لتحديث اوزان الشبكة
</div>
<br> 

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;
<div dir="rtl">
الاسقاط(Dropout) - الاسقاط هي طريقة الغرض منها منع التكيف الزائد للنموذج في بيانات التدريب عبر اسقاط بعض الواحدات في الشبكة العصبونية, العصبونات يتم اما اسقاطها باحتمالية p او الحفاظ عليها باحتمالية 1-p.
</div>
<br>  

**20. Convolutional Neural Networks**

&#10230;
<div dir="rtl">
الشبكات العصبونية الالتفافية(CNN) 
</div>
<br> 

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230;
<div dir="rtl">
احتياج الطبقة الالتفافية - عبر رمز w لحجم المدخل , F حجم العصبونات للطبقة الالتفافية , P عدد الحشوات الصفرية , فأن N عدد العصبونات لكل حجم معطى يحسب عبر الاتي: 
</div>
<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;
<div dir="rtl">
تنظيم الحزمة(Batch normalization) - هي خطوه من قيم التحسين الخاصة γ,β  والتي تعدل الحزمة {xi}. لنجعل μB,σ2B المتوسط و الانحراف للحزمة المعنية و نريد تصحيح هذه الحزمة, يتم ذلك كالتالي:    
</div>
<br> 

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;
<div dir="rtl">
في الغالب تتم بعد الطبقة الالتفافية أو المتصلة كليا و قبل طبقة التغيرات الغير خطية و تهدف للسماح للسرعات التعليم العالية للتقليل من الاعتمادية القوية للقيم الاولية.

</div>
<br>

**24. Recurrent Neural Networks**

&#10230;
<div dir="rtl">
(RNN)الشبكات العصبونية التكرارية
</div>
<br> 

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;
<div dir="rtl">
انواع البوابات - هنا الانواع المختلفة التي ممكن مواجهتها في الشبكة العصبونية الاعتيادية:
</div>
<br>  

**26. [Input gate, forget gate, gate, output gate]**

&#10230;
<div dir="rtl">
[بوابة ادخال, بوابة نسيان, بوابة منفذ, بوابة اخراج ]
</div>
<br> 

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;
<div dir="rtl">
[كتابة ام عدم كتابة الى الخلية؟, مسح ام عدم مسح الخلية؟, كمية الكتابة الى الخلية ؟ , مدى الافصاح عن الخلية ؟ ]
</div>
<br> 

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;
<div dir="rtl">
LSTM - ذاكرة طويلة قصير الامد (long short-term memory) هي نوع من نموذج ال RNN تستخدم لتجنب مشكلة اختفاء الانحدار عبر اضافة بوابات النسيان.
</div>
<br>  

**29. Reinforcement Learning and Control**

&#10230;
<div dir="rtl">
التعلم و التحكم المعزز(Reinforcement Learning)
</div>
<br> 

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;
<div dir="rtl">
الهدف من التعلم المعزز للعميل الذكي هو التعلم لكيفية التأقلم في اي بيئة.
</div>
<br>  

**31. Definitions**

&#10230;
<div dir="rtl">
تعريفات
</div>
<br> 

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;
<div dir="rtl">
عملية ماركوف لاتخاذ القرار - عملية ماركوف لاتخاذ القرار هي سلسلة خماسية (S,A,{Psa},γ,R) حيث
</div>
<br> 
**33. S is the set of states**

&#10230;
<div dir="rtl">
 S هي مجموعة من حالات البيئة
</div>
<br>

**34. A is the set of actions**

&#10230;
<div dir="rtl">
A هي مجموعة من حالات الاجراءات
</div>
<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;
<div dir="rtl">
{Psa} هو حالة احتمال الانتقال من الحالة s∈S و a∈A
</div>
<br>  

**36. γ∈[0,1[ is the discount factor**

&#10230;
<div dir="rtl">
γ∈[0,1[ هي عامل الخصم
</div>
<br>   

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;
<div dir="rtl">
R:S×A⟶R or R:S⟶R  هي دالة المكافأة والتي تعمل الخوارزمية على جعلها اعلى قيمة
</div>
<br> 

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;
<div dir="rtl">
دالة القواعد - دالة القواعد π:S⟶A  هي التي تقوم بترجمة الحالات الى اجراءات.
</div>
<br>  

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230;
<div dir="rtl">
ملاحظة: نقول ان النموذج ينفذ القاعدة المعينه π للحالة المعطاة s ان نتخذ الاجراءa=π(s).  
</div>
<br>  
 
**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;
<div dir="rtl">
دالة القاعدة - لاي قاعدة معطاة π و حالة s, نقوم بتعريف دالة القيمة Vπ  كما يلي:  
</div>
<br>    

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;
<div dir="rtl">
معادلة بيلمان - معادلات بيلمان المثلى تشخص دالة القيمة دالة القيمة Vπ∗  π∗:للقاعدة المثلى 
</div>
<br>  

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;
<div dir="rtl">
  π∗ للحالة المعطاه s تعطى كاالتالي: ملاحظة: نلاحظ ان القاعدة المثلى
</div>
<br>  

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;
<div dir="rtl">
خوارزمية تكرار القيمة(Value iteration algorithm) - خوارزمية تكرار القيمة تكون في خطوتين:
</div>
<br>  

**44. 1) We initialize the value:**

&#10230;
<div dir="rtl">
 1) نقوم بوضع قيمة اولية:
</div>
<br> 

**45. 2) We iterate the value based on the values before:**

&#10230;
<div dir="rtl">
2) نقوم بتكرير القيمة حسب القيم السابقة: 
</div>
<br> 

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;
<div dir="rtl">
تقدير الامكانية القصوى - تقديرات الامكانية القصوى (تقدير الاحتمال الأرجح) لحتماليات انتقال الحالة تكون كما يلي : 
</div>
<br>   

**47. times took action a in state s and got to s′**

&#10230;
<div dir="rtl">
اوقات تنفيذ الاجراء a في الحالة s و انتقلت الى s' 
</div>
<br> 

**48. times took action a in state s**

&#10230;
<div dir="rtl">
اوقات تنفيذ الاجراء a في الحالة s
</div>
<br>  

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
<div dir="rtl">
التعلم-Q (Q-learning) -هي طريقة غير منمذجة لتقدير Q , و تتم كالاتي:
</div>
<br> 

**50. View PDF version on GitHub**

&#10230;
<div dir="rtl">
قم باستعراض نسخة ال PDF على GitHub
</div>
<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;
<div dir="rtl">
 [شبكات عصبونية, البنية , دالة التفعيل , التغذية الخلفية , الاسقاط ]
</div>
<br> 

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230;
<div dir="rtl">
[ الشبكة العصبونية الالتفافية , طبقة التفافية , تنظيم الحزمة ] 
</div>
<br>  

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230;
<div dir="rtl">
[الشبكة العصبونية التكرارية , البوابات , LSTM]
</div>
<br>  

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230;
<div dir="rtl">
[التعلم المعزز , عملية ماركوف لاتخاذ القرار , تكرير القيمة / القاعدة , بحث القاعدة]
</div>
