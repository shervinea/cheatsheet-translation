**Convolutional Neural Networks translation**

<br>

**1. Convolutional Neural Networks cheatsheet**

<div dir="rtl">
راهنمای کوتاه شبکه عصبی پیچشی (کانولوشنی)
</div>  

<br>


**2. CS 230 - Deep Learning**

<div dir="rtl">
یادگیری عمیق - CS 230کلاس 
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
[ابرفراسنج‌های فیلتر، ابعاد، گام، گسترش مرز] 
</div>
<br>

<br>


**6. [Tuning hyperparameters, Parameter compatibility, Model complexity, Receptive field]**

<div dir="rtl">
[تنظیم ابرفراسنج‌ها، سازش پذیری فراسنج، پیچیدگی مدل،  ناحیه‌ی تاثیر]
</div>

<br>


**7. [Activation functions, Rectified Linear Unit, Softmax]**

<div dir="rtl">
[توابع فعال سازی، واحد یکسو ساز خطی، تابع بیشینه‌ی هموار] 
</div>

<br>


**8. [Object detection, Types of models, Detection, Intersection over Union, Non-max suppression, YOLO, R-CNN]**

<div dir="rtl">
[شناسایی شئی، انواع مدل ها، شناسایی، نسبت هم‌پوشانی اشتراک به اجتماع، فروداشت غیربیشینه، YOLO، ]R-CNN
</div>

<br>


**9. [Face verification/recognition, One shot learning, Siamese network, Triplet loss]**

<div dir="rtl">
[تایید/بازشناسایی چهره، یادگیری یک‌باره‌ای (One shot)، شبکه Siamese، خطای سه‌گانه]
</div> 

<br>


**10. [Neural style transfer, Activation, Style matrix, Style/content cost function]**

<div dir="rtl">
[انتقالِ سبکِ عصبی، فعال سازی، ماتریسِ سبک، تابع هزینه‌ی محتوا/سبک]
</div>

<br>


**11. [Computational trick architectures, Generative Adversarial Net, ResNet, Inception Network]**

<div dir="rtl">
[معماری‌هایی با ترفند محاسباتی، شبکه مولد هماورد، ResNet، شبکه Inception]
</div>

<br>


**12. Overview**

<div dir="rtl">
نمای کلی
</div>

<br>


**13. Architecture of a traditional CNN ― Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:**

<div dir="rtl">
معماری یک CNN سنتی - شبکه های عصبی مصنوعی پیچشی، که همچنین با عنوان CNN شناخته می شوند، یک نوع خاص از شبکه های عصبی هستند که عموما از لایه های زیر تشکیل شده اند:
</div>

<br>


**14. The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters that are described in the next sections.**

<div dir="rtl">
لایه کانولوشنی و لایه ادغام می‌توانند به نسبت ابرفراسنج‌هایی که در بخش‌های بعدی بیان شده‌اند تنظیم و تعدیل شوند.
</div>

<br>


**15. Types of layer**

<div dir="rtl">
انواع لایه‌ها
</div> 

<br>


**16. Convolution layer (CONV) ― The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S. The resulting output O is called feature map or activation map.**

<div dir="rtl">
لایه کانولوشنی (CONV)  ― لایه کانولوشنی (CONV) از فیلترهایی استفاده می‌کند که عملیات کانولوشنی را در هنگام پویش ورودی I به نسبت ابعادش، اجرا می‌کند. ابرفراسنج‌های آن شامل اندازه فیلتر F و گام S هستند. خروجی حاصل شده O نگاشت ویژگی یا نگاشت فعالسازی نامیده می‌شود.
</div>

<br>


**17. Remark: the convolution step can be generalized to the 1D and 3D cases as well.**

<div dir="rtl">
نکته: گام کانولوشنی همچنین می‌تواند به موارد یک بُعدی و سه بُعدی تعمیم داده شود.
</div>

<br>


**18. Pooling (POOL) ― The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance. In particular, max and average pooling are special kinds of pooling where the maximum and average value is taken, respectively.**

<div dir="rtl">
(POOL)  ― لایه ادغام (POOL) یک عمل نمونه‌کاهی است، که معمولا بعد از یک لایه کانولوشنی اعمال میشود، که یکسری ناوردایی مکانی را اعمال می‌کند. به طور خاص، ادغام بیشینه و میانگین انواع خاص ادغام هستند جایی‌که به ترتیب مقدار بیشینه و میانگین گرفته می‌شود.
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
[ویژگی‌های شناسایی شده را حفظ می‌کند، اغلب مورد استفاده قرار می‌گیرد، کاستن نگاشت ویژگی، در (معماری) LeNet استفاده شد]
</div>

<br>


**22. Fully Connected (FC) ― The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons. If present, FC layers are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.**

<div dir="rtl">
تمام‌متصل (FC) – لایه تمام‌متصل (FC) بر روی یک ورودی مسطح جایی‌که هر ورودی به تمامی نورون‌ها متصل است، عمل می‌کند. در صورت ارائه، لایه‌های FC معمولا در انتهای معماری‌های CNN یافت می‌شوند و میتوان آنها برای بهینه‌سازی اهدافی مثل امتیازات کلاس به‌کار برد.
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
ابعاد یک فیلتر – یک فیلتر به اندازه F×F اعمال شده برروی یک ورودیِ حاوی C کانال، یک توده F×F×C است که (عملیات) پیچشی برروی یک ورودی به اندازه I×I×C اعمال میکند و یک نگاشت ویژگی خروجی (که همچنین نگاشت فعالسازی نامیده میشود) به اندازه O×O×1 تولید میکند.
</div>

<br>


**26. Filter**

<div dir="rtl">
فیلتر
</div>

<br>


**27. Remark: the application of K filters of size F×F results in an output feature map of size O×O×K.**

<div dir="rtl">
نکته: اعمال K فیلتر به اندازه F×F، منتج به یک نگاشت ویژگی خروجی به اندازه O×O×K میشود.
</div>

<br>


**28. Stride ― For a convolutional or a pooling operation, the stride S denotes the number of pixels by which the window moves after each operation.**

<div dir="rtl">
گام – در یک عملیات ادغام یا پیچشی، اندازه گام S به تعداد پیکس‌هایی که پنجره بعد از هر عملیات جابه‌جا می‌شود، اشاره دارد.
</div>

<br>


**29. Zero-padding ― Zero-padding denotes the process of adding P zeroes to each side of the boundaries of the input. This value can either be manually specified or automatically set through one of the three modes detailed below:**

<div dir="rtl">
گسترش مرزِ صفر – گسترش مرزِ صفر به فرآیند افزودن P صفر به هر طرف از کرانه‌های ورودی اشاره دارد. این مقدار می‌تواند به طور دستی مشخص شود یا به طور خودکار از طریق یکی از سه نوع مشروح زیر تعیین گردد:
</div>

<br>


**30. [Mode, Value, Illustration, Purpose, Valid, Same, Full]**

<div dir="rtl">
[نوع، مقدار، نگاره، هدف، Valid، Same، Full]
</div>

<br>


**31. [No padding, Drops last convolution if dimensions do not match, Padding such that feature map size has size ⌈IS⌉, Output size is mathematically convenient, Also called 'half' padding, Maximum padding such that end convolutions are applied on the limits of the input, Filter 'sees' the input end-to-end]**

<div dir="rtl">
[فاقد گسترش مرز، اگر ابعاد مطابقت ندارد آخرین کانولوشنی را رها کن، (اعمال) گسترش مرز به طوری که اندازه نگاشت ویژگی ⌈IS⌉ باشد، (محاسبه) اندازه خروجی به لحاظ ریاضیاتی آسان است، همچنین گسترش مرزِ 'نیمه' نامیده می‌شود، بالاترین گسترش مرز  (اعمال می‌شود) به طوری که (عملیات) کانولوشنی انتهایی بر روی مرزهای ورودی اعمال می‌شود، فیلتر ورودی را به صورت پکپارچه 'می‌پیماید']
</div>

<br>


**32. Tuning hyperparameters**

<div dir="rtl">
تنظیم ابرفراسنج‌های
</div>

<br>


**33. Parameter compatibility in convolution layer ― By noting I the length of the input volume size, F the length of the filter, P the amount of zero padding, S the stride, then the output size O of the feature map along that dimension is given by:**

<div dir="rtl">
سازش‌پذیری فراسنج در لایه کانولوشنی – با ذکر I به عنوان طول اندازه توده ورودی،F طول فیلتر، P میزان گسترش مرزِ صفر، S گام، اندازه خروجی نگاشت ویژگی O در امتداد ابعاد مفروض است:
</div>

<br>


**34. [Input, Filter, Output]**

<div dir="rtl">
[ورودی، فیلتر، خروجی]
</div>

<br>


**35. Remark: often times, Pstart=Pend≜P, in which case we can replace Pstart+Pend by 2P in the formula above.**

<div dir="rtl">
نکته: اغلب Pstart=Pend≜P است، در این صورت Pstart+Pend را می‌توان با 2P در فرمول بالا جایگزین کرد.
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
[یک پیش‌قدر به ازای هر فیلتر، در بیشتر موارد S<F است، یک انتخاب رایج برای K، 2C است]
</div>

<br>


**39. [Pooling operation done channel-wise, In most cases, S=F]**

<div dir="rtl">
[عملیات ادغام به کانال‌به‌کانال انجام میشود، در بیشتر موارد S=F است]
</div>

<br>


