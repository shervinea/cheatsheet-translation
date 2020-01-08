**1. Supervised Learning cheatsheet**

<div dir="rtl">
مرجع سريع للتعلّم المُوَجَّه
</div> 
<br>

**2. Introduction to Supervised Learning**

<div dir="rtl">
مقدمة للتعلّم المُوَجَّه
</div> 
<br>

**3. Given a set of data points {x(1),...,x(m)} associated to a set of outcomes {y(1),...,y(m)}, we want to build a classifier that learns how to predict y from x.**

<div dir="rtl">
إذا كان لدينا مجموعة من نقاط البيانات {x(1),...,x(m)} مرتبطة بمجموعة مخرجات {y(1),...,y(m)}، نريد أن نبني مُصَنِّف يتعلم كيف يتوقع y من x.
</div> 
<br>

**4. Type of prediction ― The different types of predictive models are summed up in the table below:**

<div dir="rtl">
نوع التوقّع - أنواع نماذج التوقّع المختلفة موضحة في الجدول التالي:
</div> 
<br>

**5. [Regression, Classifier, Outcome, Examples]**

<div dir="rtl">
[الانحدار (Regression)، التصنيف (Classification)، المُخرَج، أمثلة]
</div> 
<br>

**6. [Continuous, Class, Linear regression, Logistic regression, SVM, Naive Bayes]**

<div dir="rtl">
[مستمر، صنف، انحدار خطّي (Linear regression)، انحدار لوجستي (Logistic regression)، آلة المتجهات الداعمة (SVM)، بايز البسيط (Naive Bayes)]
</div> 
<br>

**7. Type of model ― The different models are summed up in the table below:**

<div dir="rtl">
نوع النموذج - أنواع النماذج المختلفة موضحة في الجدول التالي:
</div> 
<br>

**8. [Discriminative model, Generative model, Goal, What's learned, Illustration, Examples]**

<div dir="rtl">
[نموذج تمييزي (discriminative)، نموذج توليدي (Generative)، الهدف، ماذا يتعلم، توضيح، أمثلة]
</div> 
<br>

**9. [Directly estimate P(y|x), Estimate P(x|y) to then deduce P(y|x), Decision boundary,  	Probability distributions of the data, Regressions, آلة المتجهات الداعمة (SVM), GDA, Naive Bayes]**

<div dir="rtl">
[التقدير المباشر لـ P(y|x)، تقدير P(x|y) ثم استنتاج P(y|x)، حدود القرار، التوزيع الاحتمالي للبيانات، الانحدار (Regression)، آلة المتجهات الداعمة (SVM)، GDA، بايز البسيط (Naive Bayes)]
</div> 
<br>

**10. Notations and general concepts**

<div dir="rtl">
الرموز ومفاهيم أساسية
</div> 
<br>

**11. Hypothesis ― The hypothesis is noted hθ and is the model that we choose. For a given input data x(i) the model prediction output is hθ(x(i)).**

<div dir="rtl">
الفرضية (Hypothesis) - الفرضية، ويرمز لها بـ hθ، هي النموذج الذي نختاره. إذا كان لدينا المدخل x(i)، فإن المخرج الذي سيتوقعه النموذج هو hθ(x(i)).
</div> </div> 
<br>

**12. Loss function ― A loss function is a function L:(z,y)∈R×Y⟼L(z,y)∈R that takes as inputs the predicted value z corresponding to the real data value y and outputs how different they are. The common loss functions are summed up in the table below:**

<div dir="rtl">
دالة الخسارة (Loss function) - دالة الخسارة هي الدالة L:(z,y)∈R×Y⟼L(z,y)∈R التي تأخذ كمدخلات القيمة المتوقعة z والقيمة الحقيقية y وتعطينا الاختلاف بينهما. الجدول التالي يحتوي على بعض دوال الخسارة الشائعة:
</div> 
<br>

**13. [Least squared error, Logistic loss, Hinge loss, Cross-entropy]**

<div dir="rtl">
[خطأ أصغر تربيع (Least squared error)، خسارة لوجستية (Logistic loss)، خسارة مفصلية (Hinge loss)، الانتروبيا التقاطعية (Cross-entropy)]
</div> 
<br>

**14. [Linear regression, Logistic regression, SVM, Neural Network]**

<div dir="rtl">
[الانحدار الخطّي (Linear regression)، الانحدار اللوجستي (Logistic regression)، آلة المتجهات الداعمة (SVM)، الشبكات العصبية (Neural Network)]
</div> 
<br>

**15. Cost function ― The cost function J is commonly used to assess the performance of a model, and is defined with the loss function L as follows:**

<div dir="rtl">
دالة التكلفة (Cost function) - دالة التكلفة J تستخدم عادة لتقييم أداء نموذج ما، ويتم تعريفها مع دالة الخسارة L كالتالي:
</div> 
<br>

**16. Gradient descent ― By noting α∈R the learning rate, the update rule for gradient descent is expressed with the learning rate and the cost function J as follows:**

<div dir="rtl">
النزول الاشتقاقي (Gradient descent) - لنعرّف معدل التعلّم α∈R، يمكن تعريف القانون الذي يتم تحديث خوارزمية النزول الاشتقاقي من خلاله باستخدام معدل التعلّم ودالة التكلفة J كالتالي:
</div> 
<br>

**17. Remark: Stochastic gradient descent (SGD) is updating the parameter based on each training example, and batch gradient descent is on a batch of training examples.**

<div dir="rtl">
ملاحظة: في النزول الاشتقاقي العشوائي (Stochastic gradient descent (SGD)) يتم تحديث المُعاملات (parameters) بناءاً على كل عينة تدريب على حدة، بينما في النزول الاشتقاقي الحُزَمي (batch gradient descent) يتم تحديثها باستخدام حُزَم من عينات التدريب.
</div> 
<br>

**18. Likelihood ― The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through maximizing the likelihood. In practice, we use the log-likelihood ℓ(θ)=log(L(θ)) which is easier to optimize. We have:**

<div dir="rtl">
الأرجحية (Likelihood) - تستخدم أرجحية النموذج L(θ)، حيث أن θ هي المُدخلات، للبحث عن المُدخلات θ الأحسن عن طريق تعظيم (maximizing) الأرجحية. عملياً يتم استخدام الأرجحية اللوغاريثمية (log-likelihood) ℓ(θ)=log(L(θ)) حيث أنها أسهل في التحسين (optimize). فيكون لدينا:
</div> 
<br>

**19. Newton's algorithm ― The Newton's algorithm is a numerical method that finds θ such that ℓ′(θ)=0. Its update rule is as follows:**

<div dir="rtl">
خوارزمية نيوتن (Newton's algorithm) - خوارزمية نيوتن هي طريقة حسابية للعثور على θ بحيث يكون ℓ′(θ)=0. قاعدة التحديث للخوارزمية كالتالي:
</div> 
<br>

**20. Remark: the multidimensional generalization, also known as the Newton-Raphson method, has the following update rule:**

<div dir="rtl">
ملاحظة: هناك خوارزمية أعم وهي متعددة الأبعاد (multidimensional)، يطلق عليها خوارزمية نيوتن-رافسون (Newton-Raphson)، ويتم تحديثها عبر القانون التالي:
</div> 
<br>

**21. Linear models**

<div dir="rtl">
النماذج الخطيّة (Linear models)
</div> 
<br>

**22. Linear regression**

<div dir="rtl">
الانحدار الخطّي (Linear regression)
</div> 
<br>

**23. We assume here that y|x;θ∼N(μ,σ2)**

<div dir="rtl">
هنا نفترض أن y|x;θ∼N(μ,σ2)
</div> 
<br>

**24. Normal equations ― By noting X the matrix design, the value of θ that minimizes the cost function is a closed-form solution such that:**

<div dir="rtl">
المعادلة الطبيعية/الناظمية (Normal) - إذا كان لدينا المصفوفة X، القيمة θ التي تقلل من دالة التكلفة يمكن حلها رياضياً بشكل مغلق (closed-form) عن طريق:
</div> 
<br>

**25. LMS algorithm ― By noting α the learning rate, the update rule of the Least Mean Squares (LMS) algorithm for a training set of m data points, which is also known as the Widrow-Hoff learning rule, is as follows:**

<div dir="rtl">
خوارزمية أصغر معدل تربيع LMS - إذا كان لدينا معدل التعلّم α، فإن قانون التحديث لخوارزمية أصغر معدل تربيع (Least Mean Squares (LMS)) لمجموعة بيانات من m عينة، ويطلق عليه قانون تعلم ويدرو-هوف (Widrow-Hoff)، كالتالي:
</div> 
<br>

**26. Remark: the update rule is a particular case of the gradient ascent.**

<div dir="rtl">
ملاحظة: قانون التحديث هذا يعتبر حالة خاصة من النزول الاشتقاقي (Gradient descent).
</div> 
<br>

**27. LWR ― Locally Weighted Regression, also known as LWR, is a variant of linear regression that weights each training example in its cost function by w(i)(x), which is defined with parameter τ∈R as:**

<div dir="rtl">
الانحدار الموزون محليّاً (LWR) - الانحدار الموزون محليّاً (Locally Weighted Regression)، ويعرف بـ LWR، هو نوع من الانحدار الخطي يَزِن كل عينة تدريب أثناء حساب دالة التكلفة باستخدام w(i)(x)، التي يمكن تعريفها باستخدام المُدخل (parameter) τ∈R كالتالي:
</div> 
<br>

**28. Classification and logistic regression**

<div dir="rtl">
التصنيف والانحدار اللوجستي
</div> 
<br>

**29. Sigmoid function ― The sigmoid function g, also known as the logistic function, is defined as follows:**

<div dir="rtl">
دالة سيجمويد (Sigmoid) - دالة سيجمويد g، وتعرف كذلك بالدالة اللوجستية، تعرّف كالتالي:
</div> 
<br>

**30. Logistic regression ― We assume here that y|x;θ∼Bernoulli(ϕ). We have the following form:**

<div dir="rtl">
الانحدار اللوجستي (Logistic regression) - نفترض هنا أن  y|x;θ∼Bernoulli(ϕ). فيكون لدينا:
</div> 
<br>

**31. Remark: there is no closed form solution for the case of logistic regressions.**

<div dir="rtl">
ملاحظة: ليس هناك حل رياضي مغلق للانحدار اللوجستي.
</div> 
<br>

**32. Softmax regression ― A softmax regression, also called a multiclass logistic regression, is used to generalize logistic regression when there are more than 2 outcome classes. By convention, we set θK=0, which makes the Bernoulli parameter ϕi of each class i equal to:**

<div dir="rtl">
انحدار سوفت ماكس (Softmax) - ويطلق عليه الانحدار اللوجستي متعدد الأصناف (multiclass logistic regression)، يستخدم لتعميم الانحدار اللوجستي إذا كان لدينا أكثر من صنفين. في العرف يتم تعيين θK=0، بحيث تجعل مُدخل بيرنوللي (Bernoulli) ϕi لكل فئة i يساوي:
</div> 
<br>

**33. Generalized Linear Models**

<div dir="rtl">
النماذج الخطية العامة (Generalized Linear Models - GLM)
</div> 
<br>

**34. Exponential family ― A class of distributions is said to be in the exponential family if it can be written in terms of a natural parameter, also called the canonical parameter or link function, η, a sufficient statistic T(y) and a log-partition function a(η) as follows:**

<div dir="rtl">
العائلة الأُسيّة (Exponential family) - يطلق على صنف من التوزيعات (distributions) بأنها تنتمي إلى العائلة الأسيّة إذا كان يمكن كتابتها بواسطة مُدخل قانوني (canonical parameter) η، إحصاء كافٍ (sufficient statistic) T(y)، ودالة تجزئة لوغاريثمية a(η)، كالتالي:
</div> 
<br>

**35. Remark: we will often have T(y)=y. Also, exp(−a(η)) can be seen as a normalization parameter that will make sure that the probabilities sum to one.**

<div dir="rtl">
ملاحظة: كثيراً ما سيكون T(y)=y. كذلك فإن exp(−a(η)) يمكن أن تفسر كمُدخل تسوية (normalization) للتأكد من أن الاحتمالات يكون حاصل جمعها يساوي واحد.
</div> 
<br>

**36. Here are the most common exponential distributions summed up in the following table:**

<div dir="rtl">
تم تلخيص أكثر التوزيعات الأسيّة استخداماً في الجدول التالي:
</div> 
<br>

**37. [Distribution, Bernoulli, Gaussian, Poisson, Geometric]**

<div dir="rtl">
[التوزيع، بِرنوللي (Bernoulli)، جاوسي (Gaussian)، بواسون (Poisson)، هندسي (Geometric)]
</div> 
<br>

**38. Assumptions of GLMs ― Generalized Linear Models (GLM) aim at predicting a random variable y as a function fo x∈Rn+1 and rely on the following 3 assumptions:**

<div dir="rtl">
افتراضات GLMs - تهدف النماذج الخطيّة العامة (GLM) إلى توقع المتغير العشوائي y كدالة لـ x∈Rn+1، وتستند إلى ثلاثة افتراضات:
</div> 
<br>

**39. Remark: ordinary least squares and logistic regression are special cases of generalized linear models.**

<div dir="rtl">
ملاحظة: أصغر تربيع (least squares) الاعتيادي و الانحدار اللوجستي يعتبران من الحالات الخاصة للنماذج الخطيّة العامة.
</div> 
<br>

**40. Support Vector Machines**

<div dir="rtl">
آلة المتجهات الداعمة (Support Vector Machines)
</div> 
<br>

**41: The goal of support vector machines is to find the line that maximizes the minimum distance to the line.**

<div dir="rtl">
تهدف آلة المتجهات الداعمة (SVM) إلى العثور على الخط الذي يعظم أصغر مسافة إليه:
</div> 
<br>

**42: Optimal margin classifier ― The optimal margin classifier h is such that:**

<div dir="rtl">
مُصنِّف الهامش الأحسن (Optimal margin classifier) - يعرَّف مُصنِّف الهامش الأحسن h كالتالي:
</div> 
<br>

**43: where (w,b)∈Rn×R is the solution of the following optimization problem:**

<div dir="rtl">
حيث (w,b)∈Rn×R هو الحل لمشكلة التحسين (optimization) التالية:
</div> 
<br>

**44. such that**

<div dir="rtl">
بحيث أن
</div> 
<br>

**45. support vectors**

<div dir="rtl">
المتجهات الداعمة (support vectors)
</div> 
<br>

**46. Remark: the line is defined as wTx−b=0.**

<div dir="rtl">
ملاحظة: يتم تعريف الخط بهذه المعادلة wTx−b=0.
</div> 
<br>

**47. Hinge loss ― The hinge loss is used in the setting of SVMs and is defined as follows:**

<div dir="rtl">
الخسارة المفصلية (Hinge loss) - تستخدم الخسارة المفصلية في حل SVM ويعرف على النحو التالي:
</div> 
<br>

**48. Kernel ― Given a feature mapping ϕ, we define the kernel K to be defined as:**

<div dir="rtl">
النواة (Kernel) - إذا كان لدينا دالة ربط الخصائص (features) ϕ، يمكننا تعريف النواة K كالتالي:
</div> 
<br>

**49. In practice, the kernel K defined by K(x,z)=exp(−||x−z||22σ2) is called the Gaussian kernel and is commonly used.**

<div dir="rtl">
عملياً، يمكن أن تُعَرَّف الدالة K عن طريق المعادلة K(x,z)=exp(−||x−z||22σ2)، ويطلق عليها النواة الجاوسية (Gaussian kernel)، وهي تستخدم بكثرة.
</div> 
<br>

**50. [Non-linear separability, Use of a kernel mapping, Decision boundary in the original space]**

<div dir="rtl">
[قابلية الفصل غير الخطي، استخدام ربط النواة، حد القرار في الفضاء الأصلي]
</div> 
<br>

**51. Remark: we say that we use the "kernel trick" to compute the cost function using the kernel because we actually don't need to know the explicit mapping ϕ, which is often very complicated. Instead, only the values K(x,z) are needed.**

<div dir="rtl">
ملاحظة: نقول أننا نستخدم "حيلة النواة" (kernel trick) لحساب دالة التكلفة عند استخدام النواة لأننا في الحقيقة لا نحتاج أن نعرف التحويل الصريح ϕ، الذي يكون في الغالب شديد التعقيد. ولكن، نحتاج أن فقط أن نحسب القيم K(x,z).
</div> 
<br>

**52. Lagrangian ― We define the Lagrangian L(w,b) as follows:**

<div dir="rtl">
اللّاغرانجي (Lagrangian) - يتم تعريف اللّاغرانجي L(w,b) على النحو التالي: 
</div> 
<br>

**53. Remark: the coefficients βi are called the Lagrange multipliers.**

<div dir="rtl">
ملاحظة: المعامِلات (coefficients) βi يطلق عليها مضروبات لاغرانج (Lagrange multipliers).
</div> 
<br>

**54. Generative Learning**

<div dir="rtl">
التعلم التوليدي (Generative Learning)
</div> 
<br>

**55. A generative model first tries to learn how the data is generated by estimating P(x|y), which we can then use to estimate P(y|x) by using Bayes' rule.**

<div dir="rtl">
النموذج التوليدي في البداية يحاول أن يتعلم كيف تم توليد البيانات عن طريق تقدير P(x|y)، التي يمكن حينها استخدامها لتقدير P(y|x) باستخدام قانون بايز (Bayes' rule).
</div> 
<br>

**56. Gaussian Discriminant Analysis**

<div dir="rtl">
تحليل التمايز الجاوسي (Gaussian Discriminant Analysis)
</div> 
<br>

**57. Setting ― The Gaussian Discriminant Analysis assumes that y and x|y=0 and x|y=1 are such that:**

<div dir="rtl">
الإطار - تحليل التمايز الجاوسي يفترض أن y و x|y=0 و x|y=1 بحيث يكونوا كالتالي:
</div> 
<br>

**58. Estimation ― The following table sums up the estimates that we find when maximizing the likelihood:**

<div dir="rtl">
التقدير - الجدول التالي يلخص التقديرات التي يمكننا التوصل لها عند تعظيم الأرجحية (likelihood):
</div> 
<br>

**59. Naive Bayes**

<div dir="rtl">
بايز البسيط (Naive Bayes)
</div> 
<br>

**60. Assumption ― The Naive Bayes model supposes that the features of each data point are all independent:**

<div dir="rtl">
الافتراض - يفترض نموذج بايز البسيط أن جميع الخصائص لكل عينة بيانات مستقلة (independent):
</div> 
<br>

**61. Solutions ― Maximizing the log-likelihood gives the following solutions, with k∈{0,1},l∈[[1,L]]**

<div dir="rtl">
الحل - تعظيم الأرجحية اللوغاريثمية (log-likelihood) يعطينا الحلول التالية إذا كان k∈{0,1}، l∈[[1,L]]:
</div> 
<br>

**62. Remark: Naive Bayes is widely used for text classification and spam detection.**

<div dir="rtl">
ملاحظة: بايز البسيط يستخدم بشكل واسع لتصنيف النصوص واكتشاف البريد الإلكتروني المزعج.
</div> 
<br>

**63. Tree-based and ensemble methods**

<div dir="rtl">
الطرق الشجرية (tree-based) والتجميعية (ensemble)
</div> 
<br>

**64. These methods can be used for both regression and classification problems.**

<div dir="rtl">
هذه الطرق يمكن استخدامها لكلٍ من مشاكل الانحدار (regression) والتصنيف (classification).
</div> 
<br>

**65. CART ― Classification and Regression Trees (CART), commonly known as decision trees, can be represented as binary trees. They have the advantage to be very interpretable.**

<div dir="rtl">
التصنيف والانحدار الشجري (CART) - والاسم الشائع له أشجار القرار (decision trees)، يمكن أن يمثل كأشجار ثنائية (binary trees). من المزايا لهذه الطريقة إمكانية تفسيرها بسهولة.
</div> 
<br>

**66. Random forest ― It is a tree-based technique that uses a high number of decision trees built out of randomly selected sets of features. Contrary to the simple decision tree, it is highly uninterpretable but its generally good performance makes it a popular algorithm.**

<div dir="rtl">
الغابة العشوائية (Random forest) - هي أحد الطرق الشجرية التي تستخدم عدداً كبيراً من أشجار القرار مبنية باستخدام مجموعة عشوائية من الخصائص. بخلاف شجرة القرار البسيطة لا يمكن تفسير النموذج بسهولة، ولكن أدائها العالي جعلها أحد الخوارزمية المشهورة.
</div> 
<br>

**67. Remark: random forests are a type of ensemble methods.**

<div dir="rtl">
ملاحظة: أشجار القرار نوع من الخوارزميات التجميعية (ensemble).
</div> 
<br>

**68. Boosting ― The idea of boosting methods is to combine several weak learners to form a stronger one. The main ones are summed up in the table below:**

<div dir="rtl">
التعزيز (Boosting) - فكرة خوارزميات التعزيز هي دمج عدة خوارزميات تعلم ضعيفة لتكوين نموذج قوي. الطرق الأساسية ملخصة في الجدول التالي:
</div> 
<br>

**69. [Adaptive boosting, Gradient boosting]**

<div dir="rtl">
[التعزيز التَكَيُّفي (Adaptive boosting)، التعزيز الاشتقاقي (Gradient boosting)]
</div> 
<br>

**70. High weights are put on errors to improve at the next boosting step**

<div dir="rtl">
يتم التركيز على مواطن الخطأ لتحسين النتيجة في الخطوة التالية.
</div> 
<br>

**71. Weak learners trained on remaining errors**

<div dir="rtl">
يتم تدريب خوارزميات التعلم الضعيفة على الأخطاء المتبقية.
</div> 
<br>

**72. Other non-parametric approaches**

<div dir="rtl">
طرق أخرى غير بارامترية (non-parametric)
</div> 
<br>

**73. k-nearest neighbors ― The k-nearest neighbors algorithm, commonly known as k-NN, is a non-parametric approach where the response of a data point is determined by the nature of its k neighbors from the training set. It can be used in both classification and regression settings.**

<div dir="rtl">
خوارزمية أقرب الجيران (k-nearest neighbors) - تعتبر خوارزمية أقرب الجيران، وتعرف بـ k-NN، طريقة غير بارامترية، حيث يتم تحديد نتيجة عينة من البيانات من خلال عدد k من البيانات المجاورة في مجموعة التدريب. ويمكن استخدامها للتصنيف والانحدار.
</div> 
<br>

**74. Remark: The higher the parameter k, the higher the bias, and the lower the parameter k, the higher the variance.**

<div dir="rtl">
ملاحظة: كلما زاد المُدخل k، كلما زاد الانحياز (bias)، وكلما نقص k، زاد التباين (variance).
</div> 
<br>

**75. Learning Theory**

<div dir="rtl">
نظرية التعلُّم
</div> 
<br>

**76. Union bound ― Let A1,...,Ak be k events. We have:**

<div dir="rtl">
حد الاتحاد (Union bound) - لنجعل A1,...,Ak تمثل k حدث. فيكون لدينا:
</div> 
<br>

**77. Hoeffding inequality ― Let Z1,..,Zm be m iid variables drawn from a Bernoulli distribution of parameter ϕ. Let ˆϕ be their sample mean and γ>0 fixed. We have:**

<div dir="rtl">
متراجحة هوفدينج (Hoeffding) - لنجعل Z1,..,Zm تمثل m متغير مستقلة وموزعة بشكل مماثل (iid) مأخوذة من توزيع بِرنوللي (Bernoulli distribution) ذا مُدخل ϕ. لنجعل ˆϕ متوسط العينة (sample mean) و γ>0 ثابت. فيكون لدينا:
</div> 
<br>

**78. Remark: this inequality is also known as the Chernoff bound.**

<div dir="rtl">
ملاحظة: هذه المتراجحة تعرف كذلك بحد تشرنوف (Chernoff bound).
</div> 
<br>

**79. Training error ― For a given classifier h, we define the training error ˆϵ(h), also known as the empirical risk or empirical error, to be as follows:**

<div dir="rtl">
خطأ التدريب - ليكن لدينا المُصنِّف h، يمكن تعريف خطأ التدريب ˆϵ(h)، ويعرف كذلك بالخطر التجريبي أو الخطأ التجريبي، كالتالي:
</div> 
<br>

**80. Probably Approximately Correct (PAC) ― PAC is a framework under which numerous results on learning theory were proved, and has the following set of assumptions: **

<div dir="rtl">
تقريباً صحيح احتمالياً (Probably Approximately Correct (PAC)) - هو إطار يتم من خلاله إثبات العديد من نظريات التعلم، ويحتوي على الافتراضات التالية:
</div> 
<br>

**81: the training and testing sets follow the same distribution **

<div dir="rtl">
مجموعتي التدريب والاختبار يتبعان نفس التوزيع.
</div> 
<br>

**82. the training examples are drawn independently**

<div dir="rtl">
عينات التدريب تؤخذ بشكل مستقل.
</div> 
<br>

**83. Shattering ― Given a set S={x(1),...,x(d)}, and a set of classifiers H, we say that H shatters S if for any set of labels {y(1),...,y(d)}, we have:**

<div dir="rtl">
مجموعة تكسيرية (Shattering Set) - إذا كان لدينا المجموعة S={x(1),...,x(d)}، ومجموعة مُصنٍّفات H، نقول أن H تكسر S (H shatters S) إذا كان لكل مجموعة علامات (labels) {y(1),...,y(d)} لدينا:
</div> 
<br>

**84. Upper bound theorem ― Let H be a finite hypothesis class such that |H|=k and let δ and the sample size m be fixed. Then, with probability of at least 1−δ, we have:**

<div dir="rtl">
مبرهنة الحد الأعلى (Upper bound theorem) - لنجعل H فئة فرضية محدودة (finite hypothesis class) بحيث |H|=k، و δ وحجم العينة m ثابتين. حينها سيكون لدينا، مع احتمال على الأقل 1−δ، التالي:
</div> 
<br>

**85. VC dimension ― The Vapnik-Chervonenkis (VC) dimension of a given infinite hypothesis class H, noted VC(H) is the size of the largest set that is shattered by H.**

<div dir="rtl">
بُعْد فابنيك-تشرفونيكس (Vapnik-Chervonenkis - VC) لفئة فرضية غير محدودة (infinite hypothesis class) H، ويرمز له بـ VC(H)، هو حجم أكبر مجموعة (set) التي تم تكسيرها بواسطة H (shattered by H).
</div> 
<br>

**86. Remark: the VC dimension of H={set of linear classifiers in 2 dimensions} is 3.**

<div dir="rtl">
ملاحظة: بُعْد فابنيك-تشرفونيكس VC لـ H = {مجموعة التصنيفات الخطية في بُعدين} يساوي 3.
</div> 
<br>

**87. Theorem (Vapnik) ― Let H be given, with VC(H)=d and m the number of training examples. With probability at least 1−δ, we have:**

<div dir="rtl">
مبرهنة فابنيك (Vapnik theorem) - ليكن لدينا H، مع VC(H)=d وعدد عيّنات التدريب m. سيكون لدينا، مع احتمال على الأقل 1−δ، التالي:
</div> 
<br>

**88. [Introduction, Type of prediction, Type of model]**

<div dir="rtl">
[مقدمة، نوع التوقع، نوع النموذج]
</div> 
<br>

**89. [Notations and general concepts, loss function, gradient descent, likelihood]**

<div dir="rtl">
[الرموز ومفاهيم أساسية، دالة الخسارة، النزول الاشتقاقي، الأرجحية]
</div> 
<br>

**90. [Linear models, linear regression, logistic regression, generalized linear models]**

<div dir="rtl">
[النماذج الخطيّة، الانحدار الخطّي، الانحدار اللوجستي، النماذج الخطية العامة]
</div> 
<br>

**91. [Support vector machines, Optimal margin classifier, Hinge loss, Kernel]**

<div dir="rtl">
[آلة المتجهات الداعمة (SVM)، مُصنِّف الهامش الأحسن، الفرق المفصلي، النواة]
</div> 
<br>

**92. [Generative learning, Gaussian Discriminant Analysis, Naive Bayes]**

<div dir="rtl">
[التعلم التوليدي، تحليل التمايز الجاوسي، بايز البسيط]
</div> 
<br>

**93. [Trees and ensemble methods, CART, Random forest, Boosting]**

<div dir="rtl">
[الطرق الشجرية والتجميعية، التصنيف والانحدار الشجري (CART)، الغابة العشوائية (Random forest)، التعزيز (Boosting)]
</div> 
<br>

**94. [Other methods, k-NN]**

<div dir="rtl">
[طرق أخرى، خوارزمية أقرب الجيران (k-NN)]
</div> 
<br>

**95. [Learning theory, Hoeffding inequality, PAC, VC dimension]**

<div dir="rtl">
[نظرية التعلُّم، متراجحة هوفدنك، تقريباً صحيح احتمالياً (PAC)، بُعْد فابنيك-تشرفونيكس (VC dimension)]
</div> 
