**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

<div dir="rtl">
راهنمای کوتاه شبکه‌های عصبی پیچشی (کانولوشنی)
</div>  

<br>


**2. CS 230 - Deep Learning**

<div dir="rtl">
کلاس CS 230 - یادگیری عمیق
</div>
<br>

<br>


**3. [Overview, Architecture structure]**

<div dir="rtl">
[نمای کلی، ساختار معماری]
</div>

<br>


**4. [Types of layer, Convolution, Pooling, Fully connected]**

<div dir="rtl">
[انواع لایه، کانولوشنی، ادغام، تمام‌متصل]
</div>

<br>


**5. [Filter hyperparameters, Dimensions, Stride, Padding]**

<div dir="rtl">
[ابرفراسنج‌های فیلتر، ابعاد، گام، حاشیه] 
</div>
<br>

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

<div dir="rtl">
[تنظیم ابرفراسنج‌ها، سازش‌پذیری فراسنج، پیچیدگی مدل،  ناحیه‌ی تاثیر]
</div>

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

<div dir="rtl">
[توابع فعال‌سازی، تابع یکسوساز خطی، تابع بیشینه‌ی هموار] 
</div>

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

<div dir="rtl">
[شناسایی شیء، انواع مدل‌ها، شناسایی، نسبت هم‌پوشانی اشتراک به اجتماع، فروداشت غیربیشینه، YOLO، R-CNN]
</div>

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

<div dir="rtl">
[تایید/بازشناسایی چهره، یادگیری یک‌باره‌ای (One shot)، شبکه‌ی Siamese، خطای سه‌گانه]
</div> 

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

<div dir="rtl">
[انتقالِ سبکِ عصبی، فعال سازی، ماتریسِ سبک، تابع هزینه‌ی محتوا/سبک]
</div>

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

<div dir="rtl">
[معماری‌های با ترفندهای محاسباتی، شبکه‌ی هم‌آوردِ مولد، ResNet، شبکه‌ی Inception]
</div>

<br>


**12. Overview**

<div dir="rtl">
نمای کلی
</div>

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

<div dir="rtl">
معماری یک CNN سنتی – شبکه‌های عصبی مصنوعی پیچشی، که همچنین با عنوان CNN شناخته می شوند، یک نوع خاص از شبکه های عصبی هستند که عموما از لایه‌های زیر تشکیل شده‌اند:
</div>

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

<div dir="rtl">
لایه‌ی کانولوشنی و لایه‌ی ادغام می‌توانند به نسبت ابرفراسنج‌هایی که در بخش‌های بعدی بیان شده‌اند تنظیم و تعدیل شوند.
</div>

<br>


**15. Types of layer**

<div dir="rtl">
انواع لایه‌ها
</div> 

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

<div dir="rtl">
لایه کانولوشنی (CONV) - لایه کانولوشنی (CONV) از فیلترهایی استفاده می‌کند که عملیات کانولوشنی را در هنگام پویش ورودی I به نسبت ابعادش، اجرا می‌کند. ابرفراسنج‌های آن شامل اندازه فیلتر F و گام S هستند. خروجی حاصل شده O نگاشت ویژگی یا نگاشت فعال‌سازی نامیده می‌شود.
</div>

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

<div dir="rtl">
نکته: مرحله کانولوشنی همچنین می‌تواند به موارد یک بُعدی و سه بُعدی تعمیم داده شود.
</div>

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

<div dir="rtl">
لایه ادغام (POOL) - لایه ادغام (POOL) یک عمل نمونه‌کاهی است، که معمولا بعد از یک لایه کانولوشنی اعمال می‌شود، که تا حدی منجر به ناوردایی مکانی می‌شود. به طور خاص، ادغام بیشینه و میانگین انواع خاص ادغام هستند که به ترتیب مقدار بیشینه و میانگین گرفته می‌شود.
</div>

<br>


**19. [Type, Purpose, Illustration, Comments]**

<div dir="rtl">
[نوع، هدف، نگاره، توضیحات]
</div>

<br>


**20. [Max pooling, Average pooling, Each pooling operation selects the maximum value of the current view, Each pooling operation averages the values of the current view]**

<div dir="rtl">
[ادغام بیشینه، ادغام میانگین، هر عمل ادغام مقدار بیشینه‌ی نمای فعلی را انتخاب می‌کند، هر عمل ادغام مقدار میانگینِ نمای فعلی را انتخاب می‌کند]
</div>

<br>


**21. [Preserves detected features, Most commonly used, Downsamples feature map, Used in LeNet]**

<div dir="rtl">
[ویژگی‌های شناسایی شده را حفظ می‌کند، اغلب مورد استفاده قرار می‌گیرد، کاستن نگاشت ویژگی، در (معماری) LeNet استفاده شده است]
</div>

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

<div dir="rtl">
تمام‌متصل (FC) - لایه‌ی تمام‌متصل (FC) بر روی یک ورودی مسطح به طوری ‌که هر ورودی به تمامی نورون‌ها متصل است، عمل می‌کند. در صورت وجود، لایه‌های FC معمولا در انتهای معماری‌های CNN یافت می‌شوند و می‌توان آن‌ها را برای بهینه‌سازی اهدافی مثل امتیازات کلاس به‌ کار برد.
</div>
<br>


**23. Filter hyperparameters**

<div dir="rtl">
ابرفراسنج‌های فیلتر
</div>

<br>


**24. The convolution layer contains filters for which it is important to know the meaning behind its hyperparameters.**

<div dir="rtl">
لایه کانولوشنی شامل فیلترهایی است که دانستن مفهوم نهفته در فراسنج‌های آن اهمیت دارد.
</div>

<br>


**25. Dimensions of a filter ― A filter of size F×F applied to an input containing C channels is a F×F×C volume that performs convolutions on an input of size I×I×C and produces an output feature map (also called activation map) of size O×O×1.**

<div dir="rtl">
ابعاد یک فیلتر - یک فیلتر به اندازه F×F اعمال شده بر روی یک ورودیِ حاوی C کانال، یک توده F×F×C است که (عملیات) پیچشی بر روی یک ورودی به اندازه I×I×C اعمال می‌کند و یک نگاشت ویژگی خروجی (که همچنین نگاشت فعال‌سازی نامیده می‌شود) به اندازه O×O×1 تولید می‌کند.
</div>

<br>


**26. Filter**

<div dir="rtl">
فیلتر
</div>

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

<div dir="rtl">
نکته: اعمال K فیلتر به اندازه‌ی F×F، منتج به یک نگاشت ویژگی خروجی به اندازه O×O×K می‌شود.
</div>

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

<div dir="rtl">
گام – در یک عملیات ادغام یا پیچشی، اندازه گام S به تعداد پیکسل‌هایی که پنجره بعد از هر عملیات جابه‌جا می‌شود، اشاره دارد.
</div>

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

<div dir="rtl">
حاشیه‌ی صفر – حاشیه‌ی صفر به فرآیند افزودن P صفر به هر طرف از کرانه‌های ورودی اشاره دارد. این مقدار می‌تواند به طور دستی مشخص شود یا به طور خودکار به سه روش زیر تعیین گردد:
</div>

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

<div dir="rtl">
[نوع، مقدار، نگاره، هدف، Valid، Same، Full]
</div>

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

<div dir="rtl">
[فاقد حاشیه، اگر ابعاد مطابقت ندارد آخرین کانولوشنی را رها کن، (اعمال) حاشیه به طوری که اندازه نگاشت ویژگی ⌈IS⌉ باشد، (محاسبه) اندازه خروجی به لحاظ ریاضیاتی آسان است، همچنین حاشیه‌ی 'نیمه' نامیده می‌شود، بالاترین حاشیه (اعمال می‌شود) به طوری که (عملیات) کانولوشنی انتهایی بر روی مرزهای ورودی اعمال می‌شود، فیلتر ورودی را به صورت پکپارچه 'می‌پیماید']
</div>

<br>


**32. Tuning hyperparameters**

<div dir="rtl">
تنظیم ابرفراسنج‌ها
</div>

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

<div dir="rtl">
سازش‌پذیری فراسنج در لایه کانولوشنی – با ذکر I به عنوان طول اندازه توده ورودی، F طول فیلتر، P میزان حاشیه‌ی صفر، S گام، اندازه خروجی نگاشت ویژگی O در امتداد ابعاد خواهد بود:
</div>

<br>


**34. [Input, Filter, Output]**

<div dir="rtl">
[ورودی، فیلتر، خروجی]
</div>

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

<div dir="rtl">
نکته: اغلب Pstart=Pend≜P است، در این صورت Pstart+Pend را می‌توان با  2 Pدر فرمول بالا جایگزین کرد.
</div>

<br>


**36. Understanding the complexity of the model ― In order to assess the complexity of a model, it is often useful to determine the number of parameters that its architecture will have. In a given layer of a convolutional neural network, it is done as follows:**

<div dir="rtl">
درک پیچیدگی مدل – برای برآورد پیچیدگی مدل، اغلب تعیین تعداد فراسنج‌هایی که معماری آن می‌تواند داشته باشد، مفید است. در یک لایه مفروض شبکه پیچشی عصبی این امر به صورت زیر انجام می‌شود:
</div>

<br>


**37. [Illustration, Input size, Output size, Number of parameters, Remarks]**

<div dir="rtl">
[نگاره، اندازه ورودی، اندازه خروجی، تعداد فراسنج‌ها، ملاحظات]
</div>

<br>


**38. [One bias parameter per filter, In most cases, S<F, A common choice for K is 2C]**

<div dir="rtl">
[یک پیش‌قدر به ازای هر فیلتر، در بیشتر موارد S&lt;F است، یک انتخاب رایج برای K، 2C است]
</div>


<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

<div dir="rtl">
[عملیات ادغام به صورت کانال‌به‌کانال انجام میشود، در بیشتر موارد S=F است]
</div>

<br>

**40. [Input is flattened, One bias parameter per neuron, The number of FC neurons is free of structural constraints]**

<div dir="rtl">
[ورودی مسطح شده است، یک پیش‌قدر به ازای هر نورون، تعداد نورون‌های FC فاقد محدودیت‌های ساختاری‌ست]
</div>

<br>


**41. Receptive field ― The receptive field at layer k is the area denoted Rk×Rk of the input that each pixel of the k-th activation map can 'see'. By calling Fj the filter size of layer j and Si the stride value of layer i and with the convention S0=1, the receptive field at layer k can be computed with the formula:**

<div dir="rtl">
ناحیه تاثیر – ناحیه تاثیر در لایه k محدوده‌ای از ورودی Rk×Rk است که هر پیکسلِ kاٌم نگاشت ویژگی می‌تواند 'ببیند'. با ذکر Fj به عنوان اندازه فیلتر لایه j و Si مقدار گام لایه i و با این توافق که S0=1 است، ناحیه تاثیر در لایه k با فرمول زیر محاسبه می‌شود:
</div>

<br>


**42. In the example below, we have F1=F2=3 and S1=S2=1, which gives R2=1+2⋅1+2⋅1=5.**

<div dir="rtl">
در مثال زیر داریم، F1=F2=3 و S1=S2=1 که منتج به R2=1+2⋅1+2⋅1=5 می‌شود.
</div>

<br>


**43. Commonly used activation functions**

<div dir="rtl">
توابع فعال‌سازی پرکاربرد
</div>

<br>


**44. Rectified Linear Unit ― The rectified linear unit layer (ReLU) is an activation function g that is used on all elements of the volume. It aims at introducing non-linearities to the network. Its variants are summarized in the table below:**

<div dir="rtl">
تابع یکسوساز خطی – تابع یکسوساز خطی (ReLU) یک تابع فعال‌سازی g است که بر روی تمامی عناصر توده اعمال می‌شود. هدف آن ارائه (رفتار) غیرخطی به شبکه است. انواع آن در جدول زیر به‌صورت خلاصه آمده‌اند:
</div>

<br>


**45. [ReLU, Leaky ReLU, ELU, with]**

<div dir="rtl">
[ReLU ، ReLUنشت‌دار، ELU، با]
</div>

<br>


**46. [Non-linearity complexities biologically interpretable, Addresses dying ReLU issue for negative values, Differentiable everywhere]**

<div dir="rtl">
[پیچیدگی‌های غیر خطی که از دیدگاه زیستی قابل تفسیر هستند، مسئله افول ReLU برای مقادیر منفی را مهار می‌کند، در تمامی نقاط مشتق‌پذیر است]
</div>

<br>


**47. Softmax ― The softmax step can be seen as a generalized logistic function that takes as input a vector of scores x∈Rn and outputs a vector of output probability p∈Rn through a softmax function at the end of the architecture. It is defined as follows:**

<div dir="rtl">
بیشینه‌ی هموار – مرحله بیشینه‌ی هموار را می‌توان به عنوان یک تابع لجستیکی تعمیم داده شده که یک بردار x∈Rn را از ورودی می‌گیرد و یک بردار خروجی احتمال p∈Rn، به‌واسطه‌ی تابع بیشینه‌ی هموار در انتهای معماری، تولید می‌کند. این تابع به‌صورت زیر تعریف می‌شود:
</div>

<br>


**48. where**

<div dir="rtl">
که
</div>

<br>


**49. Object detection**

<div dir="rtl">
شناسایی شیء
</div>

<br>


**50. Types of models ― There are 3 main types of object recognition algorithms, for which the nature of what is predicted is different. They are described in the table below:**

<div dir="rtl">
انواع مدل‌ – سه نوع اصلی از الگوریتم‌های بازشناسایی وجود دارد، که ماهیت آنچه‌که شناسایی شده متفاوت است. این الگوریتم‌ها در جدول زیر توضیح داده شده‌اند:
</div>

<br>


**51. [Image classification, Classification w. localization, Detection]**

<div dir="rtl">
[دسته‌بندی تصویر، دسته‌بندی با موقعیت‌یابی، شناسایی]
</div>

<br>


**52. [Teddy bear, Book]**

<div dir="rtl">
[خرس تدی، کتاب]
</div>

<br>


**53. [Classifies a picture, Predicts probability of object, Detects an object in a picture, Predicts probability of object and where it is located, Detects up to several objects in a picture, Predicts probabilities of objects and where they are located]**

<div dir="rtl">
[یک عکس را دسته‌بندی می‌کند، احتمال شیء را پیش‌بینی می‌کند، یک شیء را در یک عکس شناسایی می‌کند، احتمال یک شیء و موقعیت آن را پیش‌بینی میکند، چندین شیء در یک عکس را شناسایی می‌کند، احتمال اشیاء و موقعیت آنها را پیش‌بینی می‌کند]
</div>

<br>


**54. [Traditional CNN, Simplified YOLO, R-CNN, YOLO, R-CNN]**

<div dir="rtl">
[CNN سنتی، YOLO ساده شده، R-CNN، YOLO، R-CNN]
</div>

<br>


**55. Detection ― In the context of object detection, different methods are used depending on whether we just want to locate the object or detect a more complex shape in the image. The two main ones are summed up in the table below:**

<div dir="rtl">
شناسایی – در مضمون شناسایی شیء، روشهای مختلفی بسته به اینکه آیا فقط می‌خواهیم موقعیت قرارگیری شیء را پیدا کنیم یا شکل پیچیده‌تری در تصویر را شناسایی کنیم، استفاده می‌شوند. دو مورد از اصلی ترین آنها در جدول زیر به‌صورت خلاصه آورده‌ شده‌اند:
</div>

<br>


**56. [Bounding box detection, Landmark detection]**

<div dir="rtl">
[پیش‌بینی کادر محصورکننده، ]شناسایی نقاط(برجسته)
</div>

<br>


**57. [Detects the part of the image where the object is located, Detects a shape or characteristics of an object (e.g. eyes), More granular]**

<div dir="rtl">
[بخشی از تصویر که شیء در آن قرار گرفته را شناسایی می‌کند، یک شکل یا مشخصات یک شیء (مثل چشم‌ها) را شناسایی می‌کند، موشکافانه‌تر]
</div>

<br>


**58. [Box of center (bx,by), height bh and width bw, Reference points (l1x,l1y), ..., (lnx,lny)]**

<div dir="rtl">
[مرکزِ کادر (bx,by)، ارتفاع bh و عرض bw، نقاط مرجع (l1x,l1y), ..., (lnx,lny)]
</div>

<br>


**59. Intersection over Union ― Intersection over Union, also known as IoU, is a function that quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding box Ba. It is defined as:**

<div dir="rtl">
نسبت هم‌پوشانی اشتراک به اجتماع - نسبت هم‌پوشانی اشتراک به اجتماع، همچنین به عنوان IoU شناخته می‌شود، تابعی‌ است که میزان موقعیت دقیق کادر محصورکننده Bp نسبت به کادر محصورکننده حقیقی Ba را می‌سنجد. این تابع به‌صورت زیر تعریف می‌شود:
</div>

<br>


**60. Remark: we always have IoU∈[0,1]. By convention, a predicted bounding box Bp is considered as being reasonably good if IoU(Bp,Ba)⩾0.5.**

<div dir="rtl">
نکته: همواره داریم IoU∈[0,1]. به صورت قرارداد، یک کادر محصورکننده Bp را می‌توان نسبتا خوب در نظر گرفت اگر IoU(Bp,Ba)⩾0.5 باشد.
</div>

<br>


**61. Anchor boxes ― Anchor boxing is a technique used to predict overlapping bounding boxes. In practice, the network is allowed to predict more than one box simultaneously, where each box prediction is constrained to have a given set of geometrical properties. For instance, the first prediction can potentially be a rectangular box of a given form, while the second will be another rectangular box of a different geometrical form.**

<div dir="rtl">
کادرهای محوری – کادر بندی محوری روشی است که برای پیش‌بینی کادرهای محصورکننده هم‌پوشان استفاده می‌شود. در عمل، شبکه این اجازه را دارد که بیش از یک کادر به‌صورت هم‌زمان پیش‌بینی کند جایی‌که هر پیش‌بینی کادر مقید به داشتن یک مجموعه خصوصیات هندسی مفروض است. به عنوان مثال، اولین پیش‌بینی می‌تواند یک کادر مستطیلی با قالب خاص باشد حال آنکه کادر دوم، یک کادر مستطیلی محوری با قالب هندسی متفاوتی خواهد بود.
</div>

<br>


**62. Non-max suppression ― The non-max suppression technique aims at removing duplicate overlapping bounding boxes of a same object by selecting the most representative ones. After having removed all boxes having a probability prediction lower than 0.6, the following steps are repeated while there are boxes remaining:**

<div dir="rtl">
فروداشت غیربیشینه – هدف روش فروداشت غیربیشینه، حذف کادرهای محصورکننده هم‌پوشان تکراریِ دسته یکسان با انتخاب معرف‌ترین‌ها است. بعد از حذف همه کادرهایی که احتمال پیش‌بینی پایین‌تر از 0.6 دارند، مراحل زیر  با وجود آنکه کادرهایی باقی می‌مانند، تکرار می‌شوند:
</div>

<br>


**63. [For a given class, Step 1: Pick the box with the largest prediction probability., Step 2: Discard any box having an IoU⩾0.5 with the previous box.]**

<div dir="rtl">
[برای یک دسته مفروض، گام اول: کادر با بالاترین احتمال پیش‌بینی را انتخاب کن، گام دوم: هر کادری که IoU≥0.5 نسبت به کادر پیشین دارد را رها کن.]
</div>

<br>


**64. [Box predictions, Box selection of maximum probability, Overlap removal of same class, Final bounding boxes]**

<div dir="rtl">
[پیش‌بینی کادرها، انتخاب کادرِ با احتمال بیشینه، حذف (کادر) همپوشان دسته یکسان، کادرهای محصورکننده نهایی]
</div>

<br>


**65. YOLO ― You Only Look Once (YOLO) is an object detection algorithm that performs the following steps:**

<div dir="rtl">
YOLO -  «شما فقط یک‌بار نگاه می‌کنید» (YOLO) یک الگوریتم شناسایی شیء است که مراحل زیر را اجرا می‌کند:
</div>

<br>


**66. [Step 1: Divide the input image into a G×G grid., Step 2: For each grid cell, run a CNN that predicts y of the following form:, repeated k times]**

<div dir="rtl">
[گام اول: تصویر ورودی را به یک مشبک G×G تقسیم کن، گام دوم: برای هر سلول مشبک، یک CNN که y را به شکل زیر پیش‌بینی می‌کند، اجرا کن:، k مرتبه تکرارشده]
</div>

<br>


**67. where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were detected, and k is the number of anchor boxes.**

<div dir="rtl">
که pc احتمال شناسایی یک شیء است، bx,by,bh,bw اندازه‌های نسبی کادر محیطی شناسایی شده است، c1,...,cp نمایش «تک‌فعال» یک دسته از p دسته که تشخیص داده شده است، و k تعداد کادرهای محوری است.

</div>

<br>


**68. Step 3: Run the non-max suppression algorithm to remove any potential duplicate overlapping bounding boxes.**

<div dir="rtl">
گام سوم: الگوریتم فروداشت غیربیشینه را برای حذف هر کادر محصورکننده هم‌پوشان تکراری بالقوه، اجرا کن.
</div>

<br>


**69. [Original image, Division in GxG grid, Bounding box prediction, Non-max suppression]**

<div dir="rtl">
[تصویر اصلی، تقسیم به GxG مشبک، پیش‌بینی کادر محصورکننده، فروداشت غیربیشینه]
</div>

<br>


**70. Remark: when pc=0, then the network does not detect any object. In that case, the corresponding predictions bx,...,cp have to be ignored.**

<div dir="rtl">
نکته: زمانی‌که pc=0 است، شبکه هیچ شیئی را شناسایی نمی‌کند. در چنین حالتی، پیش‌بینی‌های متناظر bx,…,cp بایستی نادیده گرفته شوند.
</div>

<br>


**71. R-CNN ― Region with Convolutional Neural Networks (R-CNN) is an object detection algorithm that first segments the image to find potential relevant bounding boxes and then run the detection algorithm to find most probable objects in those bounding boxes.**

<div dir="rtl">
R-CNN - ناحیه با شبکه‌های عصبی پیچشی (R-CNN) یک الگوریتم شناسایی شیء است که ابتدا تصویر را برای یافتن کادرهای محصورکننده مربوط بالقوه قطعه‌بندی می‌کند و سپس الگوریتم شناسایی را برای یافتن محتمل‌ترین اشیاء در این کادرهای محصور کننده اجرا می‌کند.
</div>

<br>


**72. [Original image, Segmentation, Bounding box prediction, Non-max suppression]**

<div dir="rtl">
[تصویر اصلی، قطعه بندی، پیش‌بینی کادر محصور کننده، فروداشت غیربیشینه]
</div>

<br>


**73. Remark: although the original algorithm is computationally expensive and slow, newer architectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.**

<div dir="rtl">
نکته: هرچند الگوریتم اصلی به لحاظ محاسباتی پرهزینه و کند است، معماری‌های جدید از قبیل Fast R-CNN و Faster R-CNN باعث شدند که الگوریتم سریعتر اجرا شود.
</div>

<br>


**74. Face verification and recognition**

<div dir="rtl">
تایید چهره و بازشناسایی
</div>

<br>


**75. Types of models ― Two main types of model are summed up in table below:**

<div dir="rtl">
انواع مدل – دو نوع اصلی از مدل در جدول زیر به‌صورت خلاصه آورده‌ شده‌اند:
</div>

<br>


**76. [Face verification, Face recognition, Query, Reference, Database]**

<div dir="rtl">
[تایید چهره، بازشناسایی چهره، جستار، مرجع، پایگاه داده]
</div>

<br>


**77. [Is this the correct person?, One-to-one lookup, Is this one of the K persons in the database?, One-to-many lookup]**

<div dir="rtl">
[فرد مورد نظر است؟، جستجوی یک‌به‌یک، این فرد یکی از K فرد پایگاه داده است؟، جستجوی یک‌به‌چند]
</div>

<br>


**78. One Shot Learning ― One Shot Learning is a face verification algorithm that uses a limited training set to learn a similarity function that quantifies how different two given images are. The similarity function applied to two images is often noted d(image 1,image 2).**

<div dir="rtl">
یادگیری یک‌باره‌ای – یادگیری یک‌باره‌ای یک الگوریتم تایید چهره است که از یک مجموعه آموزشی محدود برای یادگیری یک تابع مشابهت که میزان اختلاف دو تصویر مفروض را تعیین می‌کند، بهره می‌برد. تابع مشابهت اعمال‌شده بر روی دو تصویر اغلب با نماد  d(image 1, image 2) نمایش داده می‌شود.
</div>

<br>


**79. Siamese Network ― Siamese Networks aim at learning how to encode images to then quantify how different two images are. For a given input image x(i), the encoded output is often noted as f(x(i)).**

<div dir="rtl">
شبکه‌ی Siamese - هدف شبکه‌ی Siamese یادگیری طریقه رمزنگاری تصاویر و سپس تعیین اختلاف دو تصویر است. برای یک تصویر مفروض ورودی x(i)، خروجی رمزنگاری شده اغلب با نماد f(x(i)) نمایش داده می‌شود.
</div>

<br>


**80. Triplet loss ― The triplet loss ℓ is a loss function computed on the embedding representation of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive example belong to a same class, while the negative example to another one. By calling α∈R+ the margin parameter, this loss is defined as follows:**

<div dir="rtl">
خطای سه‌گانه – خطای سه‌گانه ℓ یک تابع خطا است که بر روی بازنمایی تعبیه‌ی سه‌گانه‌ی تصاویر A (محور)، P (مثبت) و N (منفی)  محاسبه می‌شود. نمونه‌های محور (anchor) و مثبت به دسته یکسانی تعلق دارند، حال آنکه نمونه منفی به دسته دیگری تعلق دارد. با نامیدن α∈R+ (به عنوان) فراسنج حاشیه، این خطا به‌صورت زیر تعریف می‌شود:
</div>

<br>


**81. Neural style transfer**

<div dir="rtl">
انتقالِ سبک عصبی
</div>

<br>


**82. Motivation ― The goal of neural style transfer is to generate an image G based on a given content C and a given style S.**

<div dir="rtl">
انگیزه – هدف انتقالِ سبک عصبی تولید یک تصویر G بر مبنای یک محتوای مفروض C و سبک مفروض S است.
</div>

<br>


**83. [Content C, Style S, Generated image G]**

<div dir="rtl">
[محتوای  C، سبک S، تصویر تولیدشده‌ی  G]
</div>

<br>


**84. Activation ― In a given layer l, the activation is noted a[l] and is of dimensions nH×nw×nc**

<div dir="rtl">
فعال‌سازی – در یک لایه مفروض l، فعال‌سازی با a[l] نمایش داده می‌شود و به ابعاد nH×nw×nc است
</div>

<br>


**85. Content cost function ― The content cost function Jcontent(C,G) is used to determine how the generated image G differs from the original content image C. It is defined as follows:**

<div dir="rtl">
تابع هزینه‌ی محتوا – تابع هزینه‌ی محتوا Jcontent(C,G) برای تعیین میزان اختلاف تصویر تولیدشده G از تصویر اصلی C استفاده می‌شود. این تابع به‌صورت زیر تعریف می‌شود:
</div>

<br>


**86. Style matrix ― The style matrix G[l] of a given layer l is a Gram matrix where each of its elements G[l]kk′ quantifies how correlated the channels k and k′ are. It is defined with respect to activations a[l] as follows:**

<div dir="rtl">
ماتریسِ سبک - ماتریسِ سبک G[l] یک لایه مفروض l، یک ماتریس گرَم (Gram) است که هر کدام از عناصر G[l]kk′ میزان همبستگی کانال‌های k و k′ را می‌سنجند. این ماتریس نسبت به فعال‌سازی‌های a[l] به‌صورت زیر محاسبه می‌شود:
</div>

<br>


**87. Remark: the style matrix for the style image and the generated image are noted G[l] (S) and G[l] (G) respectively.**

<div dir="rtl">
نکته: ماتریس سبک برای تصویر سبک و تصویر تولید شده، به ترتیب با G[l] (S) و G[l] (G) نمایش داده می‌شوند.
</div>

<br>


**88. Style cost function ― The style cost function Jstyle(S,G) is used to determine how the generated image G differs from the style S. It is defined as follows:**

<div dir="rtl">
تابع هزینه‌ی سبک – تابع هزینه‌ی سبک Jstyle(S,G) برای تعیین میزان اختلاف تصویر تولیدشده G و سبک S استفاده می‌شود. این تابع به صورت زیر تعریف می‌شود:
</div>

<br>


**89. Overall cost function ― The overall cost function is defined as being a combination of the content and style cost functions, weighted by parameters α,β, as follows:**

<div dir="rtl">
تابع هزینه‌ی کل – تابع هزینه‌ی کل به صورت ترکیبی از توابع هزینه‌ی سبک و محتوا تعریف شده است که با فراسنج‌های α,β, به شکل زیر وزن‌دار شده است:
</div>

<br>


**90. Remark: a higher value of α will make the model care more about the content while a higher value of β will make it care more about the style.**

<div dir="rtl">
نکته: مقدار بیشتر α مدل را به توجه بیشتر به محتوا وا می‌دارد حال آنکه مقدار بیشتر β مدل را به توجه بیشتر به سبک وا می‌دارد.
</div>

<br>


**91. Architectures using computational tricks**

<div dir="rtl">
معماری‌هایی که از ترفندهای محاسباتی استفاده می‌کنند.
</div>

<br>


**92. Generative Adversarial Network ― Generative adversarial networks, also known as GANs, are composed of a generative and a discriminative model, where the generative model aims at generating the most truthful output that will be fed into the discriminative which aims at differentiating the generated and true image.**

<div dir="rtl">
شبکه‌ی هم‌آوردِ مولد – شبکه‌ی هم‌آوردِ مولد، همچنین با نام GANs شناخته می‌شوند، ترکیبی از یک مدل مولد و تمیزدهنده هستند، جایی‌که مدل مولد هدفش تولید واقعی‌ترین خروجی است که به (مدل) تمیزدهنده تغذیه می‌شود و این (مدل) هدفش تفکیک بین تصویر تولیدشده و واقعی است.
</div>

<br>


**93. [Training, Noise, Real-world image, Generator, Discriminator, Real Fake]**

<div dir="rtl">
[آموزش، نویز، تصویر دنیای واقعی، مولد، تمیز دهنده، واقعی بدلی]
</div>

<br>


**94. Remark: use cases using variants of GANs include text to image, music generation and synthesis.**

<div dir="rtl">
نکته: موارد استفاده متنوع GAN ها شامل تبدیل متن به تصویر، تولید موسیقی و تلفیقی از آنهاست.
</div>

<br>


**95. ResNet ― The Residual Network architecture (also called ResNet) uses residual blocks with a high number of layers meant to decrease the training error. The residual block has the following characterizing equation:**

<div dir="rtl">
ResNet – معماری شبکه‌ی پسماند (همچنین با عنوان ResNet شناخته می‌شود) از بلاک‌های پسماند با تعداد لایه‌های زیاد به منظور کاهش خطای آموزش استفاده می‌کند. بلاک پسماند معادله‌ای با خصوصیات زیر دارد:
</div>

<br>


**96. Inception Network ― This architecture uses inception modules and aims at giving a try at different convolutions in order to increase its performance through features diversification. In particular, it uses the 1×1 convolution trick to limit the computational burden.**

<div dir="rtl">
شبکه‌ی Inception – این معماری از ماژول‌های inception استفاده می‌کند و هدفش فرصت دادن به (عملیات) کانولوشنی مختلف برای افزایش کارایی از طریق تنوع‌بخشی ویژگی‌ها است. به طور خاص، این معماری از ترفند کانولوشنی 1×1 برای محدود سازی بار محاسباتی استفاده می‌کند.
</div>

<br>


**97. The Deep Learning cheatsheets are now available in [target language].**

<div dir="rtl">
راهنمای یادگیری عمیق هم اکنون به زبان ]فارسی[ در دسترس است.
</div>

<br>


**98. Original authors**

<div dir="rtl">
نویسندگان اصلی
</div>

<br>


**99. Translated by X, Y and Z**

<div dir="rtl">
ترجمه شده توسط X،Y و Z
</div>

<br>


**100. Reviewed by X, Y and Z**

<div dir="rtl">
بازبینی شده توسط توسط X،Y و Z
</div>

<br>


**101. View PDF version on GitHub**

<div dir="rtl">
نسخه پی‌دی‌اف را در گیت‌هاب ببینید
</div>

<br>


**102. By X and Y**

<div dir="rtl">
توسط X و Y
</div>

<br>

