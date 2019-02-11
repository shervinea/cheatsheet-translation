
**Deep Learning Tips and Tricks translation**

<br>

**1. Deep Learning Tips and Tricks cheatsheet**

<div dir="rtl">
راهنمای کوتاه نکات و ترفندهای یادگیری عمیق
</div>

<br>


**2. CS 230 - Deep Learning**

<div dir="rtl">
کلاس CS 230 - یادگیری عمیق
</div>

<br>


**3. Tips and tricks**

<div dir="rtl">
نکات و ترفندها
</div>

<br>


**4. [Data processing, Data augmentation, Batch normalization]**

<div dir="rtl">
[پردازش داده، داده‌افزایی، نرمال‌سازی دسته‌ای]
</div>

<br>


**5. [Training a neural network, Epoch, Mini-batch, Cross-entropy loss, Backpropagation, Gradient descent, Updating weights, Gradient checking]**

<div dir="rtl">
[آموزش یک شبکه‌ی عصبی، تکرار(Epoch)، دسته‌ی کوچک، خطای آنتروپی متقاطع، انتشار معکوس، گرادیان نزولی، به‌روزرسانی وزن‌ها، وارسی گرادیان]
</div>

<br>


**6. [Parameter tuning, Xavier initialization, Transfer learning, Learning rate, Adaptive learning rates]**

<div dir="rtl">
[تنظیم فراسنج، مقداردهی اولیه Xavier،یادگیری انتقالی، نرخ یادگیری، نرخ یادگیری سازگارشونده]
</div>

<br>


**7. [Regularization, Dropout, Weight regularization, Early stopping]**

<div dir="rtl">
[نظام‌بخشی، برون‌اندازی، نظام‌بخشی وزن، توقف زودهنگام]
</div>

<br>


**8. [Good practices, Overfitting small batch, Gradient checking]**

<div dir="rtl">
[عادت‌های خوب، بیش‌برارزش دسته‌ی کوچک، وارسی گرادیان]
</div>

<br>


**9. View PDF version on GitHub**

<div dir="rtl">
نسخه پی‌دی‌اف را در گیت‌هاب ببینید 
</div>

<br>


**10. Data processing**

<div dir="rtl">
پردازش داده
</div>

<br>


**11. Data augmentation ― Deep learning models usually need a lot of data to be properly trained. It is often useful to get more data from the existing ones using data augmentation techniques. The main ones are summed up in the table below. More precisely, given the following input image, here are the techniques that we can apply:**

<div dir="rtl">
داده‌افزایی ― مدل‌های یادگیری عمیق معمولا به داده‌های زیادی نیاز دارند تا بتوانند به خوبی آموزش ببینند. اغلب، استفاده از روش‌های داده‌افزایی برای گرفتن داده‌ی بیشتر از داده‌های موجود، مفید است. اصلی‌ترین آنها در جدول زیر به اختصار آمده‌اند. به عبارت دقیق‌تر، با در نظر گرفتن تصویر ورودی زیر، روش‌هایی که می‌توان اعمال کرد بدین شرح هستند:
</div>

<br>

**12. [Original, Flip, Rotation, Random crop]**

<div dir="rtl">
[تصویر اصلی، قرینه، چرخش، برش تصادفی]
</div>

<br>


**13. [Image without any modification, Flipped with respect to an axis for which the meaning of the image is preserved, Rotation with a slight angle, Simulates incorrect horizon calibration, Random focus on one part of the image, Several random crops can be done in a row]**

<div dir="rtl">
[تصویر (آغازین) بدون هیچ‌گونه تغییری، قرینه‌شده نسبت به محوری که معنای (محتوای) تصویر را حفظ می‌کند، چرخش با زاویه‌ی اندک، خط افق نادرست را شبیه‌سازی می‌کند، روی ناحیه‌ای تصادفی از تصویر متمرکز می‌شود، چندین برش تصادفی را میتوان پشت‌سرهم انجام داد]
</div>
<br>


**14. [Color shift, Noise addition, Information loss, Contrast change]**

<div dir="rtl">
[تغییر رنگ، اضافه‌کردن نویز، هدررفت اطلاعات، تغییر تباین(کُنتراست)]
</div>

<br>


**15. [Nuances of RGB is slightly changed, Captures noise that can occur with light exposure, Addition of noise, More tolerance to quality variation of inputs, Parts of image ignored, Mimics potential loss of parts of image, Luminosity changes, Controls difference in exposition due to time of day]**

<div dir="rtl">
[عناصر RGB کمی تغییر کرده است، نویزی که در هنگام مواجه شدن با نور رخ می‌دهد را شبیه‌سازی می‌کند، افزودگی نویز، مقاومت بیشتر نسبت به تغییر کیفیت تصاویر ورودی، بخش‌هایی از تصویر نادیده گرفته می‌شوند، تقلید (شبیه سازی) هدررفت بالقوه بخش‌هایی از تصویر، تغییر درخشندگی، با توجه به زمان روز تفاوت نمایش (تصویر) را کنترل می‌کند]
</div>

<br>


**16. Remark: data is usually augmented on the fly during training.**

<div dir="rtl">
نکته: داده‌ها معمولا در فرآیند آموزش (به صورت درجا) افزایش پیدا می‌کنند.
</div>

<br>


**17. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

<div dir="rtl">
نرمال‌سازی دسته‌ای ― یک مرحله از فراسنج‌های γ و β که دسته‌ی {xi} را نرمال می‌کند. نماد μB و σ2B به میانگین و وردایی دسته‌ای که می‌خواهیم آن را اصلاح کنیم اشاره دارد که به صورت زیر است:
</div>

<br>


**18. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

<div dir="rtl">
معمولا بعد از یک لایه‌ی تمام‌متصل یا لایه‌ی کانولوشنی و قبل از یک لایه‌ی غیرخطی اعمال می‌شود و امکان استفاده از نرخ یادگیری بالاتر را می‌دهد و همچنین باعث می‌شود که وابستگی شدید مدل به مقداردهی اولیه کاهش یابد.
</div>

<br>


**19. Training a neural network**

<div dir="rtl">
آموزش یک شبکه‌ی عصبی
</div>

<br>


**20. Definitions**

<div dir="rtl">
تعاریف
</div>

<br>


**21. Epoch ― In the context of training a model, epoch is a term used to refer to one iteration where the model sees the whole training set to update its weights.**

<div dir="rtl">
تکرار (epoch) ― در مضمون آموزش یک مدل، تکرار اصطلاحی است که مدل در یک دوره تکرار تمامی نمونه‌های آموزشی را برای به‌روزرسانی وزن‌ها می‌بیند.
</div>

<br>


**22. Mini-batch gradient descent ― During the training phase, updating weights is usually not based on the whole training set at once due to computation complexities or one data point due to noise issues. Instead, the update step is done on mini-batches, where the number of data points in a batch is a hyperparameter that we can tune.**

<div dir="rtl">
گرادیان نزولی دسته‌ی‌کوچک ―  در فاز آموزش، به‌روزرسانی وزن‌ها معمولا بر مبنای تمامی مجموعه آموزش به علت پیچیدگی‌های محاسباتی، یا یک نمونه داده به علت مشکل نویز، نیست. در عوض، گام به‌روزرسانی بر روی دسته‌های کوچک انجام می شود، که تعداد نمونه‌های داده در یک دسته یک ابرفراسنج است که میتوان آن را تنظیم کرد.
</div>

<br>


**23. Loss function ― In order to quantify how a given model performs, the loss function L is usually used to evaluate to what extent the actual outputs y are correctly predicted by the model outputs z.**

<div dir="rtl">
تابع خطا ―  به منظور سنجش کارایی یک مدل مفروض، معمولا از تابع خطای L برای ارزیابی اینکه تا چه حد خروجی حقیقی y به شکل صحیح توسط خروجی z مدل پیش‌بینی شده‌اند، استفاده می‌شود. 
</div>

<br>


**24. Cross-entropy loss ― In the context of binary classification in neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

<div dir="rtl">
خطای آنتروپی متقاطع – در مضمون دسته‌بندی دودویی در شبکه‌های عصبی، عموما از تابع خطای آنتروپی متقاطع L(z,y) استفاده و به صورت زیر تعریف میشود:
</div>

<br>


**25. Finding optimal weights**

<div dir="rtl">
یافتن وزن‌های بهینه
</div>

<br>


**26. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to each weight w is computed using the chain rule.**

<div dir="rtl">
انتشار معکوس ―  انتشار معکوس روشی برای به‌روزرسانی وزن‌ها با توجه به خروجی واقعی و خروجی مورد انتظار در شبکه‌ی عصبی است. مشتق نسبت به هر وزن w توسط قاعده‌ی زنجیری محاسبه می‌شود.
</div>

<br>


**27. Using this method, each weight is updated with the rule:**

<div dir="rtl">
با استفاده از این روش، هر وزن با قانون زیر به‌روزرسانی می‌شود:
</div>

<br>


**28. Updating weights ― In a neural network, weights are updated as follows:**

<div dir="rtl">
به‌روزرسانی وزن‌ها – در یک شبکه‌ی عصبی، وزن‌ها به شکل زیر به‌روزرسانی می‌شوند:
</div>

<br>


**29. [Step 1: Take a batch of training data and perform forward propagation to compute the loss, Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight, Step 3: Use the gradients to update the weights of the network.]**

<div dir="rtl">
[گام 1: یک دسته از داده‌های آموزشی گرفته شده و با استفاده از انتشار مستقیم خطا محاسبه می‌شود، گام 2: با استفاده از انتشار معکوس مشتق خطا نسبت به هر وزن محاسبه می‌شود، گام 3: با استفاده از مشتقات، وزن‌های شبکه به‌روزرسانی می‌شوند.]
</div>

<br>


**30. [Forward propagation, Backpropagation, Weights update]**

<div dir="rtl">
[انتشار مستقیم، انتشار معکوس، به‌روزرسانی وزنها]
</div>

<br>


**31. Parameter tuning**

<div dir="rtl">
تنظیم فراسنج
</div>

<br>


**32. Weights initialization**

<div dir="rtl">
مقداردهی اولیه‌ی وزن‌ها
</div>

<br>


**33. Xavier initialization ― Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture.**

<div dir="rtl">
مقداردهی‌ اولیه Xavier ―  به‌جای مقداردهی اولیه‌ی وزن‌ها به شیوه‌ی کاملا تصادفی، مقداردهی اولیه Xavier  این امکان را فراهم می‌سازد تا وزن‌های اولیه‌ای داشته باشیم که ویژگی‌های منحصر به فرد معماری را به حساب می‌آورند.
</div>

<br>


**34. Transfer learning ― Training a deep learning model requires a lot of data and more importantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets that took days/weeks to train, and leverage it towards our use case. Depending on how much data we have at hand, here are the different ways to leverage this:**

<div dir="rtl">
یادگیری انتقالی ― آموزش یک مدل یادگیری عمیق به داده‌های زیاد و مهم‌تر از آن به زمان زیادی احتیاج دارد. اغلب بهتر است که از وزن‌های پیش‌آموخته روی پایگاه داده‌های عظیم که آموزش بر روی آن‌ها روزها یا هفته‌ها طول می‌کشند استفاده کرد، و آن‌ها را برای موارد استفاده‌ی خود به کار برد. بسته به میزان داده‌هایی که در اختیار داریم، در زیر روش‌های مختلفی که می‌توان از آنها بهره جست آورده شده‌اند:
</div>

<br>


**35. [Training size, Illustration, Explanation]**

<div dir="rtl">
[تعداد داده‌های آموزش، نگاره، توضیح]
</div>

<br>


**36. [Small, Medium, Large]**

<div dir="rtl">
[کوچک، متوسط، بزرگ]
</div>

<br>


**37. [Freezes all layers, trains weights on softmax, Freezes most layers, trains weights on last layers and softmax, Trains weights on layers and softmax by initializing weights on pre-trained ones]**

<div dir="rtl">
[منجمد کردن تمامی لایه‌ها، آموزش وزن‌ها در بیشینه‌ی هموار، منجمد کردن اکثر لایه‌ها، آموزش وزن‌ها در لایه‌های آخر و بیشینه‌ی هموار، آموزش وزن‌ها در (تمامی) لایه‌ها و بیشینه‌ی هموار با مقداردهی‌اولیه‌ی وزن‌ها بر طبق مقادیر پیش‌آموخته]
</div>

<br>


**38. Optimizing convergence**

<div dir="rtl">
بهینه‌سازی همگرایی
</div>

<br>


**39. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. It can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.
**

<div dir="rtl">
نرخ یادگیری – نرخ یادگیری اغلب با نماد α و گاهی اوقات با نماد η نمایش داده می‌شود و بیانگر سرعت (گام) به‌روزرسانی وزن‌ها است که می‌تواند مقداری ثابت داشته باشد یا به صورت سازگارشونده تغییر کند. محبوب‌ترین روش حال حاضر Adam نام دارد، روشی است که نرخ یادگیری را در حین فرآیند آموزش تنظیم می‌کند.
</div>

<br>


**40. Adaptive learning rates ― Letting the learning rate vary when training a model can reduce the training time and improve the numerical optimal solution. While Adam optimizer is the most commonly used technique, others can also be useful. They are summed up in the table below:**

<div dir="rtl">
نرخ‌های یادگیری سازگارشونده ― داشتن نرخ یادگیری متغیر در فرآیند آموزش یک مدل،  می‌تواند زمان آموزش را کاهش دهد و راه‌حل بهینه عددی را بهبود ببخشد. با آنکه بهینه‌ساز Adam محبوب‌ترین روش مورد استفاده است، دیگر روش‌ها نیز می‌توانند مفید باشند. این روش‌ها در جدول زیر به اختصار آمده‌اند:
</div>

<br>


**41. [Method, Explanation, Update of w, Update of b]**

<div dir="rtl">
[روش، توضیح، به‌روزرسانی w، به‌روزرسانی  b]
</div>

<br>


**42. [Momentum, Dampens oscillations, Improvement to SGD, 2 parameters to tune]**

<div dir="rtl">
[تکانه، نوسانات را تعدیل می‌دهد، بهبود SGD، دو  فراسنج که نیاز به تنظیم دارند]
</div>

<br>


**43. [RMSprop, Root Mean Square propagation, Speeds up learning algorithm by controlling oscillations]**

<div dir="rtl">
[RMSprop، انتشار جذر میانگین مربعات، سرعت بخشیدن به الگوریتم یادگیری با کنترل نوسانات]
</div>

<br>


**44. [Adam, Adaptive Moment estimation, Most popular method, 4 parameters to tune]**

<div dir="rtl">
[Adam، تخمین سازگارشونده ممان، محبوب‌ترین روش، چهار فراسنج که نیاز به تنظیم دارند]
</div>

<br>


**45. Remark: other methods include Adadelta, Adagrad and SGD.**

<div dir="rtl">
نکته: سایر متدها  شامل Adadelta، Adagrad و SGD هستند.
</div>

<br>


**46. Regularization**

<div dir="rtl">
نظام‌بخشی
</div>

<br>


**47. Dropout ― Dropout is a technique used in neural networks to prevent overfitting the training data by dropping out neurons with probability p>0. It forces the model to avoid relying too much on particular sets of features.**

<div dir="rtl">
برون‌اندازی – برون‌اندازی روشی است که در شبکه‌های عصبی برای جلوگیری از بیش‌برارزش بر روی داده‌های آموزشی با حذف تصادفی نورون‌ها با احتمال p>0 استفاده می‌شود. این روش مدل را مجبور می‌کند تا از تکیه کردن بیش‌از‌حد بر روی مجموعه خاصی از ویژگی‌ها خودداری کند.
</div>

<br>


**48. Remark: most deep learning frameworks parametrize dropout through the 'keep' parameter 1−p.**

<div dir="rtl">
نکته: بیشتر کتابخانه‌های یادگیری عمیق برون‌اندازی را با استفاده از فراسنج 'نگه‌داشتن' 1-p کنترل می‌کنند.
</div>

<br>


**49. Weight regularization ― In order to make sure that the weights are not too large and that the model is not overfitting the training set, regularization techniques are usually performed on the model weights. The main ones are summed up in the table below:**

<div dir="rtl">
نظام‌بخشی وزن – برای اطمینان از اینکه (مقادیر) وزن‌ها بیش‌ازحد بزرگ نیستند و مدل به مجموعه‌ی آموزش بیش‌برارزش نمی‌کند، روشهای نظام‌بخشی معمولا بر روی وزن‌های مدل اجرا می‌شوند. اصلی‌ترین آنها در جدول زیر به اختصار آمده‌اند:
</div>

<br>


**50. [LASSO, Ridge, Elastic Net]**

<div dir="rtl">
[LASSO, Ridge, Elastic Net]
</div>
<br>

**50 bis. Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

<div dir="rtl">
ضرایب را تا صفر کاهش می‌دهد، برای انتخاب متغیر مناسب است، ضرایب را کوچکتر می‌کند، بین انتخاب متغیر و ضرایب کوچک مصالحه می‌کند
</div>

<br>

**51. Early stopping ― This regularization technique stops the training process as soon as the validation loss reaches a plateau or starts to increase.**

<div dir="rtl">
توقف زودهنگام ― این روش نظام‌بخشی، فرآیند آموزش را به محض اینکه خطای اعتبارسنجی ثابت می‌شود یا شروع به افزایش پیدا کند، متوقف می‌کند.
</div>

<br>


**52. [Error, Validation, Training, early stopping, Epochs]**

<div dir="rtl">
[خطا، اعتبارسنجی، آموزش، توقف زودهنگام، تکرارها]
</div>

<br>


**53. Good practices**

<div dir="rtl">
عادت‌های خوب
</div>

<br>


**54. Overfitting small batch ― When debugging a model, it is often useful to make quick tests to see if there is any major issue with the architecture of the model itself. In particular, in order to make sure that the model can be properly trained, a mini-batch is passed inside the network to see if it can overfit on it. If it cannot, it means that the model is either too complex or not complex enough to even overfit on a small batch, let alone a normal-sized training set.**

<div dir="rtl">
بیش‌برارزش روی دسته‌ی ‌کوچک ―  هنگام اشکال‌زدایی یک مدل، اغلب مفید است که یک سری آزمایش‌های سریع برای اطمینان از اینکه هیچ مشکل عمده‌ای در معماری مدل وجود ندارد، انجام شود. به طورخاص، برای اطمینان از اینکه مدل می‌تواند به شکل صحیح آموزش ببیند، یک دسته‌ی‌ کوچک (از داده‌ها) به شبکه داده می‌شود تا دریابیم که مدل می‌تواند به آنها بیش‌برارزش کند. اگر نتواند، بدین معناست که مدل از پیچیدگی بالایی برخوردار است یا پیچیدگی کافی برای بیش‌برارزش شدن روی دسته‌ی‌ کوچک ندارد، چه برسد به یک مجموعه آموزشی با اندازه عادی.
</div>

<br>


**55. Gradient checking ― Gradient checking is a method used during the implementation of the backward pass of a neural network. It compares the value of the analytical gradient to the numerical gradient at given points and plays the role of a sanity-check for correctness.**

<div dir="rtl">
وارسی گرادیان – وارسی گرادیان روشی است که در طول پیاده‌سازی گذر روبه‌عقبِ یک شبکه‌ی عصبی استفاده می‌شود. این روش مقدار گرادیان تحلیلی را با گرادیان عددی در نقطه‌های مفروض مقایسه می‌کند و نقش بررسی درستی را ایفا میکند. 
</div>

<br>


**56. [Type, Numerical gradient, Analytical gradient]**

<div dir="rtl">
[نوع، گرادیان عددی، گرادیان تحلیلی]
</div>

<br>


**57. [Formula, Comments]**

<div dir="rtl">
[فرمول، توضیحات]
</div>

<br>


**58. [Expensive; loss has to be computed two times per dimension, Used to verify correctness of analytical implementation, Trade-off in choosing h not too small (numerical instability) nor too large (poor gradient approximation)]**

<div dir="rtl">
[پرهزینه (از نظر محاسباتی)،  خطا باید دو بار برای هر بُعد محاسبه شود، برای تایید صحت پیاده‌سازی تحلیلی استفاده می‌شود، مصالحه در انتخاب h: نه بسیار کوچک (ناپایداری عددی) و نه خیلی بزرگ (تخمین گرادیان ضعیف) باشد]
</div>

<br>


**59. ['Exact' result, Direct computation, Used in the final implementation]**

<div dir="rtl">
[نتیجه 'عینی'، محاسبه مستقیم، در پیاده‌سازی نهایی استفاده می‌شود]
</div>

<br>


**60. The Deep Learning cheatsheets are now available in [target language].**

<div dir="rtl">
راهنمای یادگیری عمیق هم اکنون به زبان [فارسی] در دسترس است.
</div>

**61. Original authors**

<div dir="rtl">
نویسندگان اصلی
</div>

<br>

**62.Translated by X, Y and Z**

<div dir="rtl">
ترجمه شده توسط X،Y و Z
</div>

<br>

**63.Reviewed by X, Y and Z**

<div dir="rtl">
بازبینی شده توسط توسط X،Y و Z
</div>

<br>

**64.View PDF version on GitHub**

<div dir="rtl">
نسخه پی‌دی‌اف را در گیت‌هاب ببینید
</div>

<br>

**65.By X and Y**

<div dir="rtl">
توسط X و Y
</div>

<br>
