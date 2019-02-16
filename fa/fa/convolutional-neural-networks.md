**1. Convolutional Neural Networks cheatsheet**

&#10230;
<div dir="rtl">
راهنمای کوتاه شبکه عصبی پیچشی (کانولوشنی)
</div>  

<br>


**2. CS 230 - Deep Learning**

&#10230;

<div dir="rtl">
یادگیری عمیق - CS 230 کلاس
</div>
<br>


**3. [Overview, Architecture structure]**

&#10230;

<div dir="rtl">
[نمای کلی، ساختار معماری]
</div>
<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

&#10230;

<div dir="rtl">
[انواع لایه، کانولوشنی، ادغام، تمام متصل]
</div>
<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

&#10230;

<div dir="rtl">
[ابرفراسنج‌های فیلتر، ابعاد، گام، لایه گذاری] 
</div>
<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

 
&#10230;

<div dir="rtl">
[ تنظیم ابرفراسنج‌ها، سازش پذیری فراسنج، پیچیدگی مدل، ناحیه‌ی تاثیر]
</div>
<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

&#10230;

<div dir="rtl">
[توابع فعال سازی، واحد یکسو ساز خطی، تابع بیشینه‌ی هموار] 
</div>
<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

 
&#10230;

<div dir="rtl">
[تشخیص شئی، انواع مدل ها، تشخیص، نسبت هم‌پوشانی اشتراک به اجتماع /نسبت همبری به اجتماع، حذف مقادیر غیربیشینه، YOLO ,  شبکه R-CNN]
</div> 
<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

 
&#10230;

<div dir="rtl">
[تایید\بازشناسایی چهره، یادگیری یکباره‌ای، شبکه Siamese، خطای سه‌گانه]  
</div> 
<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

&#10230;

<div dir="rtl">
[انتقالِ سبکِ عصبی، فعال سازی، ماتریسِ سبک، تابع هزینه‌ی محتوا/سبک]
</div>
<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

&#10230;

<div dir="rtl">
[معماری ترفند محاسباتی، شبکه مولد تخاصمی ، ResNet، شبکه Inception]
</div>
<br>


**12. Overview**

&#10230;

<div dir="rtl">
نمای کلی
</div>
<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

&#10230;

<div dir="rtl">
معماری یک CNN سنتی - شبکه های عصبی مصنوعی پیچشی، که همچنین با عنوان CNN شناخته می شوند، یک نوع خاص از شبکه های عصبی هستند که عموما از لایه های زیر تشکیل شده اند:
</div>  
<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

&#10230;

<div dir="rtl">
لایه کانولوشنی و لایه ادغام می‌توانند به نسبت ابرفراسنج‌هایی که در بخش‌های بعدی بیان شده‌اند تنظیم و تعدیل شوند.
</div>  
<br>


**15. Types of layer**

&#10230;

<div dir="rtl">
انواع لایه‌ها
</div> 
<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

&#10230;

<div dir="rtl">
لایه کانولوشنی (CONV) ― لایه کانولوشنی (CONV) از فیلترهایی استفاده می‌کند که عملیات کانولوشنی را در هنگام پویش ورودی I به نسبت ابعادش، اجرا می‌کند. ابرفراسنج‌های آن شامل اندازه فیلتر F و گام S هستند. خروجی حاصل شده O نگاشت ویژگی یا نگاشت فعالسازی نامیده می‌شود.
</div>  
<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

&#10230;

<div dir="rtl">
نکته: گام کانولوشنی همچنین می‌تواند به موارد یک بُعدی و سه بُعدی تعمیم داده شود.
</div>  
<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

&#10230;
<div dir="rtl">
(POOL) ― لایه ادغام (POOL) یک عمل نمونه‌کاهی است، که معمولا بعد از یک لایه کانولوشنی اعمال میشود، و یکسری ناوردایی مکانی را اعمال می‌کند. به طور خاص، ادغام بیشینه و میانگین انواع خاص ادغام هستند جایی‌که به ترتیب مقدار بیشینه و میانگین گرفته می‌شود.
</div>  

<br>


**19. [Type, Purpose, Illustration, Comments]**

  
&#10230;

<div dir="rtl">
[نوع، هدف، نگاره، توضیحات،]
</div>
<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

 
&#10230;

<div dir="rtl">
[ادغام بیشینه، ادغام میانگین، هر عمل ادغام مقدار بیشینه‌ی نمای فعلی را انتخاب می‌کند، هر عمل ادغام مقدار میانگینِ نمای فعلی را انتخاب می‌کند]
</div> 
<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

  
&#10230;

<div dir="rtl">
[ویژگی‌های شناسایی شده را حفظ می‌کند، اغلب مورد استفاده قرار می‌گیرد، کاستن نگاشت ویژگی، در (معماری) LeNet استفاده شد]
</div>
<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

&#10230;

<div dir="rtl">
تمام‌متصل (FC) – لایه تمام‌متصل (FC) بر روی یک ورودی مسطح جایی‌که هر ورودی به تمامی نورون‌ها متصل است، عمل می‌کند. در صورت ارائه، لایه‌های FC معمولا در انتهای معماری‌های CNN یافت می‌شوند و میتوان آنها برای بهینه‌سازی اهدافی مثل امتیازات دسته به‌کار برد.
</div>  
<br>


**23. Filter hyperparameters**

 
&#10230;

<div dir="rtl">
ابرفراسنج‌های فیلتر
</div> 
<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

 
&#10230;

<div dir="rtl">
لایه کانولوشنی شامل فیلترهایی است که دانستن مفهوم نهفته در فراسنج‌های آن اهمیت دارد.
</div>  
<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

 
&#10230;

<div dir="rtl">
ابعاد یک فیلتر -  یک فیلتر به سایز F×F که شامل ورودی کانال C است، که مقدار ماتریس F×F×C را با انجام عملیات پیچش بر روی ورودی به سایز I×I×C و تولید خروجی نگاشت ویژگی(فعالساز نگاشت) به سایز 0×0×1.
</div> 
<br>


**26. Filter**

  
&#10230;

<div dir="rtl">
فیلتر
</div>
<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

 
&#10230;

<div dir="rtl">
توجه:  کابرد فیلترهای K در سایز0×0×K نتیجه نهایی که می‌دهند، به صورت نگاشت ویژگی به سایز F×F است.
</div> 
<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

 
&#10230;

<div dir="rtl">
گام - در عملیات کانولوشنی یا ادغام، گام S نشان دهنده تعداد پیکسل در هر پنجره است که بعد از هر عملیات می‌آید.
</div> 
<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

&#10230;

<div dir="rtl">
گسترش مرزِ صفر – گسترش مرزِ صفر به فرآیند افزودن P صفر به هر طرف از کرانه‌های ورودی اشاره دارد. این مقدار می‌تواند به طور دستی مشخص شود یا به طور خودکار از طریق یکی از سه نوع مشروح زیر تعیین گردد:
</div>
<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

  
&#10230;

<div dir="rtl">
[نوع، مقدار، نگاره، هدف، Valid، Same، Full]
</div>
<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

  
&#10230;

<div dir="rtl">
[فاقد گسترش مرز، اگر ابعاد مطابقت ندارد آخرین کانولوشنی را رها کن، (اعمال) گسترش مرز به طوری که اندازه نگاشت ویژگی ⌈IS⌉ باشد، (محاسبه) اندازه خروجی به لحاظ ریاضیاتی آسان است، همچنین گسترش مرزِ 'نیمه' نامیده میشود، بالاترین گسترش مرز (اعمال میشود) به طوری که (عملیات) کانولوشنی انتهایی بر روی مرزهای ورودی اعمال میشود، فیلتر ورودی را به صورت پکپارچه 'می‌پیماید']
</div>
<br>


**32. Tuning hyperparameters**

&#10230;

<div dir="rtl">
تنظیم ابرفراسنج‌ها
</div>
<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

&#10230;

<div dir="rtl">
سازگاری پارامتر در لایه پیچشی - باتوجه به اینکه I حجم داده ورودی، F طول فیلتر، P مجموع تعداد لایه گذاری صفر، S تعداد گام و در آخر سایز خروجی O که به صورت نگاشت ویژگی بر اساس ابعاد می‌دهد به صورت :
</div>
<br>


**34. [Input, Filter, Output]**

 
&#10230;

<div dir="rtl">
[ورودی ، فیلتر ، خروجی]
</div> 
<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

&#10230;

<div dir="rtl">
توجه : اغلب اوقات، Pstart=Pend≜P را با فرمول دیگری میتوان جایگزین کرد به صورت Pstart+Pend by 2P نسبت به فرمول بالا  
<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

&#10230;

<div dir="rtl">
درک پیچیدگی مدل -  برای ارزیابی پیچیدگی یک مدل، اغلب مفید است که تعداد پارامترهای معماری آن را مشخص کرد. در یک لایه داده شده از شبکه عصبی پیچشی، به شرح زیر است:
</div>
<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

 
&#10230;

<div dir="rtl">
[نگاره، سایز ورودی، سایز خروجی، تعداد پارامترها، ملاحظات]
</div> 
<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

&#10230;

<div dir="rtl">
[یک پارامتر سوگیری به ازای هر فیلتر، در بیشتر موارد، S<F ، یک انتخاب مشترک برای K همان 2C است]  
</div>
<br>



**39. [Pooling operation done channel-wise, In most cases, S=F]**

&#10230;

<div dir="rtl">
[حوضچه‌ای که عملیات چنل-وایز را انجام داده، در بیشتر موارد به صورت S=F است] 
</div>  
<br>


**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

&#10230;

<div dir="rtl">
[دارای ورودی مسطح، یک پارامتر سوگیری به ازای هر نورون، تعداد نورون‌ها در لایه تماما متصل دارای محدودیت ساختاری نیستند ] 
</div>  
<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

&#10230;

<div dir="rtl">
پهنه پذیرا - پهنه پذیرا در لایه K در فضایی که نشان دهنده Rk×Rk در داده ورودی به ازای هر پیکسل Kام از نگاشت فعالساز  را میتواند  مشاهده کرد. که با فراخوانی Fj  سایز فیلتر در لایه j و  میزان گام Si در لایهi است که هم‌آیی S0=1 در پهنه پذیرا در لایه K میتواند به صورت زیر محاسبه شود:
</div>
<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

&#10230;

<div dir="rtl">
در مثال پایین، داریم دو مقدار F1=F2=3 و S1=S2=1 که حاصل میشود R2=1+2⋅1+2⋅1=5
</div>  
<br>


**43. Commonly used activation functions**

  
&#10230;

<div dir="rtl">
معمولا از تابع فعالساز استفاده می‌شود.
</div>
<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

&#10230;

<div dir="rtl">
واحد یک‌سو ساز خطی - لایه یک‌سو ساز خطی (ReLU) که تابع فعالساز آن g است برای تمامی مقادیر این گذاره استفاده می‌شود. که در نهایت هدف آن معرفی یک شبکه غیر خطی است. انواع آن در جدول زیر به مختصر آمده است: 
</div>
<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

&#10230;

<div dir="rtl">
[یک‌سو ساز، یک‌سو ساز غیر خطی، ELU] 
</div>
<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

&#10230;

<div dir="rtl">
[پیچیدگی غیر خطی که به صورت بیولوژیکی قابل تفسیر است، مقادیری که کمترین ارزش یا حتی ارزش منفی در یک‌سو ساز دارند،مشتق‌پذیر بودن در همه جا]
</div>      
<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

&#10230;

<div dir="rtl">
بیشینه‌ی هموار -  در گام بیشینه‌ی فعال شاهد تعمیم تابع لجیستیک هستیم که به ورودی به صورت بردار اعداد به مجموعه x∈Rn و همچنین به بردار خروجی با احتمال خروجی مجموعه p∈Rn اعمال میشود، که تابع بیشینه‌ی هموار در لایه نهایی معماری به آنها اعمال می‌شود. که توضیح آن در ادامه :
</div>
<br>


**48. where**

&#10230;

<div dir="rtl">
جایی که
</div>
<br>


**49. Object detection**

<div dir="rtl">
تشخیص شی
</div>  
&#10230;

<br>
@@ -348,19 +450,25 @@
**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

&#10230;

<div dir="rtl">
انواع مدل‌ها - سه نوع اصلی الگوریتم شناسایی شیئ وجود دارد، و طبعیت هر الگوریتم در پیش‌بینی متفاوت است. در جدول زیر توضیحاتی داده شده:
</div>
<br>


**51. [Image classification, Classification w. localization, Detection]**

&#10230;

<div dir="rtl">
[طبقه‌بندی تصاویر، طبقه‌بندی بر اساس محل قرارگیری، شناسایی]  
</div>  
<br>


**52. [Teddy bear, Book]**

<div dir="rtl">
خرس عروسکی تدی، کتاب
</div>
&#10230;

<br>
@@ -369,26 +477,34 @@
**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

&#10230;

<div dir="rtl">
[طبقه‌بندی عکس، پیش‌بینی احتمال  شی بودن، شناسایی شی در عکس، پیش‌بینی احتمال شی بودن و محل قرار گیری آن، شناسایی دیگر عناصر در عکس، پیش‌بینی احتمالات شی بودن و محل قرارگیری شی]
</div>
<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

&#10230;

<div dir="rtl">
[شبکه عصبی پیچشی سنتی، ساده‌سازی YOLO، R-CNN، YOLO  ]
</div>
<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

&#10230;

<div dir="rtl">
شناسایی - در حیطه شناسایی شی، شیوه‌های متفاوتی استفاده می‌شود و بستگی دارد به اینکه در یک شی میخواهیم محل شی را شناسایی کنیم یا یک الگوی پیچیده در تصویر. دو مدلی که بیشترین استفاده را دارد در جدول زیر آمده:
</div>
<br>


**56. [Bounding box detection, Landmark detection]**

<div dir="rtl">
[شناسایی با محدود سازی، شناسایی با برجسته سازی]
</div>
&#10230;

<br>
@@ -397,173 +513,225 @@
**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

&#10230;

<div dir="rtl">
[شناسایی قسمتی از تصویر که شی در آن قرار دارد،  شناسایی شکل یا کارکترها یک شی (به مانند چشم)، و یا دانه‌ها ]
</div>
<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

&#10230;

<div dir="rtl">
[محدوده‌ای در مرکز به صورت (bx,by)، طول bh و عرضbw، نقاط منبع (l1x,l1y), ..., (lnx,lny ]
</div>
<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

&#10230;

<div dir="rtl">
تقاطع بیش از اتحاد - تقاطع بیش از اتحاد، که به صورت IoU نیز شناخته می‌شود، یک تابع است که نتیجه گیری می‌کنیم که محل قرار گرفتن درست را پیش‌بینی کنیم در جعبه محدود ساز Bp نسبت به مقدار واقعی  جعبه محدود سازBa. بدین صورت تعریف می‌شود: 
</div>
<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

&#10230;

<div dir="rtl">
توجه:  ما همیشه داریمIoU∈[0,1]. باتوافق، پیش‌بینی جعبه محدودساز Bp با در نظر گرفتن خوب بودن نتایج به شرط وجود IoU(Bp,Ba)⩾0.5.
</div>
<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

&#10230;

<div dir="rtl">
جعبه لنگر - جعبه لنگر یک تکنیک برای پیش‌بینی تداخل در جعبه محدود ساز است. در عمل یک شبکه اجازه پیش‌بینی بیش از یک جعبه را به طور همزمان می‌دهد، در جایی که هر کدام از جعبه‌های پیش‌بینی مقید به دادن مجموعه‌ای از خواص هندسی است. برای مثال،  در پیش‌بینی اولیه  این پتانسیل وجود دارد که جعبه به صورت مستطیلی را به فرم متفاوت هندسی و پیش‌بینی جعبه دو م را به شیوه متفاوتی انجام دهد.
</div>
<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

&#10230;

<div dir="rtl">
محو غیر حداکثری - محو غیر حداکثری تکنیکی است به هدف حذف کردن همپوشانی تکراری در جعبه محدود ساز برای شی مشابه و انتخاب بیشترین تشابه شی‌ای. بعد از حذف همه جعبه‌هایی با احتمال پیش‌بینی کمتر از ۰/۶،   و این گام‌ها تکرار می‌شود برای جعبه‌های باقی‌مانده. 
</div>
<br>



**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

&#10230;

<div dir="rtl">
[برای یک کلاس داده شده، گام اول : انتخاب جعبه‌ای با داشتن بیشترین احتمال پیش‌بینی است.، گام دوم: حذف تمامی جعبه‌هایی که دارای IoU⩾0.5 با جعبه پیشین]
</div>
<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

&#10230;

<div dir="rtl">
[جعبه پیش‌بینی، انتخاب جعبه‌ای با حداکثر احتمال، حذف کلاس‌هایی که همپوشانی دارند، جعبه محدودساز نهایی]
</div>
<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

&#10230;

<div dir="rtl">
YOLO -  فقط یک بار نگاه کن () یک الگوریتم شناسایی شی است که بر اساس مراحل زیر است:
</div>
<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

&#10230;

<div dir="rtl">
[گام اول: تقسیم تصویر ورودی به یک شبکه توری ().، گام دوم: برای هر  سلول شبکه توری، شبکه  عصبی پیچشی را برای پیش‌بینی مقدار () به شکل زیر اجرا می‌شود: ]
</div>
<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

&#10230;

<div dir="rtl">
جایی که pc احتمال شناسایی شی، bx,by,bh,bw به صورت شناسایی خواص جعبه محدودساز، c1,...,cp یک نماینده بردار مهم است که p کلاس شناسایی، و k تعداد جعبه‌های لنگر است. 
</div>
<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

&#10230;

<div dir="rtl">
گام سوم: اجرای الگوریتم محو غیر حداکثری و حذف پتانسیل همپوشانی تکراری در جعبه‌های محدودساز است.
</div>  
<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

&#10230;

<div dir="rtl">
[عکس اصلی، تقسیم به شبکه توری GxG، پیش‌بینی جعبه محدودساز، حذف مقادیر غیر بیشینه ]
</div>  
<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

&#10230;

<div dir="rtl">
توجه: زمانی pc=0، بعد از آن شبکه شی‌ای شناسایی نمی‌کند. در این صورت پیش‌بینی متناظر bx,...,cp مورد توجه قرار نمی‌گیرد.
</div>  
<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

&#10230;

<div dir="rtl">
 R-CNN - شبکه عصبی پیچشی با توجه به موقعیت مکانی () یک الگوریتم شناسایی است که در قسمت اول تصویر پیدا کردن پتانسیل مربوطه جعبه محدودساز  است و سپس الگوریتم شناسایی برای پیدا کردن بیشترین احتمال شی بودن در یک جعبه محدودساز اجرا می‌شود.
</div>  
<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

&#10230;

<div dir="rtl">
[تصویر اصلی، قطعه بندی، پیش‌بینی جعبه محدودساز، محو غیر حداکثری]
</div>  
<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

&#10230;

<div dir="rtl">
توجه : اگرچه در الگوریتم اصلی هزینه محاسبات بالا و کند عمل می‌کند، ولی معماری جدید سریع‌تر اجرا می‌شود، و الگوریتم جدید با سرعتی به سرعت  الگوریتم شبکه‌های عصبی پیچشی که وابسته به ناحیه هستند عمل می‌کند.
</div>  
<br>


**74. Face verification and recognition**

  
&#10230;

<div dir="rtl">
تایید چهره و تشخیص چهره
</div>
<br>


**75. Types of models ― Two main types of model are summed up in table below:**

&#10230;

<div dir="rtl">
انواع مدل‌ها  -  دو مدل اصلی وجود دارد که به طور خلاصه در جدول زیر آمده است:  
</div>
<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

&#10230;

<div dir="rtl">
[تایید چهره، تسخیص چهره،  جست‌و‌جو، منبع ، مجموعه داده]
</div>
<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

&#10230;

<div dir="rtl">
[آیا این شخص مورد نظر است؟، جست‌و جوی نفر به نفر، آیا این شخص Kامین شخص در مجموعه داده است؟، جست‌و‌جوی یک به چند ]
</div>
<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

&#10230;

<div dir="rtl">
یادگیری در یک گذار - یادگیری در یک گذار یک الگوریتم تایید چهره است که برای مجموعه آموزش محدود کاربرد دارد، که  یاد می‌گیرد تابع تشابه چگونه، تفاوت دو تصویر داده شده را پیدا کند. تابع تشابه گاهی روی دو تصویر اعمال میشود تا نشان دهد()(تصویر ۱، تصویر ۲). 
</div>
<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

&#10230;

<div dir="rtl">
شبکه Siamese  -  شبکه Siamese به هدف چگونگی آموزش رمز کردن تصاویر با توجه به متغییر‌های متفاوت در دو تصویر ایجاد شد. برای نشان دادن تصویر ورودی x(i)، و کدگذاری خروجی معمولا به صورت f(x(i) است.  
</div>
<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

&#10230;

<div dir="rtl">
خطای سه‌گانه - خطای سه‌گانه ℓ یک تابع خطای محاسبه شده است که با توجه به نمای سه بعدی سازی تصویر A (لنگر)،P مثبت) و N (منفی). 
   میزان خطای پارامتر حاشیه به صورت α∈R+لنگر و مثال مثبت مربوط به یک کلاس هستند، و مثال منفی مربوط به کلاس دیگر و تا زمانی که 
</div>
<br>


**81. Neural style transfer**

<div dir="rtl">
انتقال به سبک شبکه عصبی
</div>
&#10230;

<br>
@@ -572,117 +740,152 @@
**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

&#10230;

<div dir="rtl">
انگیختگی - هدف انتقال به سبک شبکه عصبی تولید تصویر G بر پایه دادن محتوای C و دادن شیوه S است.   
</div>
<br>


**83. [Content C, Style S, Generated image G]**

&#10230;

<div dir="rtl">
[محتوای C، شیوه S،  G تصویر تولید شده ]
</div>
<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

&#10230;

<div dir="rtl">
فعالساز - در یک لایه داده شده L، که فعالساز نشان دهنده a[l] و ابعاد nH×nw×nc است. 
</div>
<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

&#10230;

<div dir="rtl">
محتوای تابع هزینه -  محتوای یک تابع هزینه J که شامل محتوای (C,G) است ، نشان دهنده چگونگی ایجاد تصویر G متفاوت‌تر نسبت به تصویر و محتوای اصلی C است. که در ادامه مختصر توضیح داده شده:
</div>
<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

&#10230;

<div dir="rtl">
سبک ماتریس - سبک ماتریس G[l] که یک لایه داده شده l از ماتریس گرام در جایی که عناصر G[l]kk′ برای تعیین میزان همبستگی کانال‌های k و k′ است. با توجه به فعالساز a[l] بدین شکل تعریف شده است: 
</div>
<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

&#10230;

<div dir="rtl">
توجه : سبک ماتریس نیز برای سبک تصویر و تولید تصویر است که به صورت G[l] (S) و به ترتیب G[l] (G) است. 
</div>
<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

&#10230;

<div dir="rtl">
سبک تابع هزینه  -  سبک تابع هزینه J که به سبک (S,G) است، برای استفاده و نمایش چگونگی تولید تصویر G و متفاوت از سبک S است که به شکل زیر تعریف می‌شود.
</div>
<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

&#10230;

<div dir="rtl">
تابع هزینه کلی - تابع هزینه کلی به این صورت تعریف می‌شود که از ترکیب محتوا و سبک میزان تابع هزینه بدست می‌آید، که وزن‌ها بر اساس پارامترهای α,β به شکل زیر است:
</div>
<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

&#10230;

<div dir="rtl">
توجه : میزان بالای α باعث می‌شود که مدل به محتوا حساس‌تر باشد و همچنین میزان بالای β باعث می‌شود که نسبت به سبک و روش حساس بشود.
</div>
<br>


**91. Architectures using computational tricks**

&#10230;

<div dir="rtl">
معماری که از ترفندهای محاسباتی استفده می‌کند.
</div>
<br>



**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

&#10230;

<div dir="rtl">
شبکه مولد تخاصمی - شبکه مولد تخاصمی، که به ()ها شناخته می‌شوند، که شامل یک مولد و یک مدل تشخیص دهنده است، در جایی که مدل مولد به هدف تولید حقیقی‌ترین خروجی می‌پردازد، و از این طریق تشخیص دهنده را تغذیه می‌کند به هدف تشخیص تصاویر حقیقی از جعلی کاربرد دارد. 
</div>  
<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

 
&#10230;

<div dir="rtl">
[آموزش، خطا، تصاویر واقعی، تولید کننده، تشخیص دهنده، جعلی نزدیک به اصل]
</div> 
<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

&#10230;

<div dir="rtl">
توجه: موارد استفاده از شبکه GAN شامل تبدیل متن به تصویر، تولید موزیک و ترکیب‌های دیگر است. 
</div> 
<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

&#10230;

<div dir="rtl">
 ResNet - معماری شبکه باقی‌مانده که به (ResNet) نیر معروف است، در بلوک‌های باقی‌مانده با تعداد بالای لایه‌ها مورد استفاده قرار می‌گیرد به منظور کاهش میزان خطا در مدت دوره یادگیری است. بلوک‌های باقی‌مانده به صورت معادله‌ای در زیر نمایش داده می‌شود: 
</div> 
<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

&#10230;

<div dir="rtl">
شبکه Inception -  این معماری از ماژول Inception استفاده می‌کند، و  هدف استفاده از نوع متفاوتی از شبکه پیچشی در جهت  افزایش کارایی، از طریق افزایش ویژگی‌ها است. بخصوص که استفاده از مدل 1×1 پیچشی در جهت ترفند‌های محاسباتی محدود است.
</div> 
<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

<div dir="rtl">
راهنمای یادگیری عمیق هم اکنون در دسترس است به زبان فارسی
</div>  
&#10230;

<br>


**98. Original authors**

<div dir="rtl">
مولف اصلی
</div>
&#10230;

<br>
