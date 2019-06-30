**Recurrent Neural Networks translation**

<br>

**1. Recurrent Neural Networks cheatsheet**

<div dir="rtl">
راهنمای کوتاه شبکه‌های عصبی برگشتی 
</div>
 
<br>


**2. CS 230 - Deep Learning**

<div dir="rtl">
کلاس CS 230 - یادگیری عمیق
</div>

<br>


**3. [Overview, Architecture structure, Applications of RNNs, Loss function, Backpropagation]**

<div dir="rtl">
[نمای کلی، ساختار معماری، کاربردهایRNN  ها، تابع خطا، انتشار معکوس]
</div>

<br>


**4. [Handling long term dependencies, Common activation functions, Vanishing/exploding gradient, Gradient clipping, GRU/LSTM, Types of gates, Bidirectional RNN, Deep RNN]**

<div dir="rtl">
[کنترل وابستگی‌های بلندمدت، توابع فعال‌سازی رایج، مشتق صفرشونده/منفجرشونده، برش گرادیان، GRU/LSTM، انواع دروازه، RNN دوسویه، RNN عمیق]
</div>

<br>


**5. [Learning word representation, Notations, Embedding matrix, Word2vec, Skip-gram, Negative sampling, GloVe]**

<div dir="rtl">
[یادگیری بازنمائی کلمه، نمادها، ماتریس تعبیه، Word2vec،skip-gram، نمونه‌برداری منفی، GloVe]
</div>

<br>


**6. [Comparing words, Cosine similarity, t-SNE]**

<div dir="rtl">
[مقایسه‌ی کلمات، شباهت کسینوسی، t-SNE]
</div>

<br>


**7. [Language model, n-gram, Perplexity]**

<div dir="rtl">
[مدل زبانی،ان‌گرام، سرگشتگی]
</div>

<br>


**8. [Machine translation, Beam search, Length normalization, Error analysis, Bleu score]**

<div dir="rtl">
[ترجمه‌ی ماشینی، جستجوی پرتو، نرمال‌سازی طول، تحلیل خطا، امتیاز Bleu]
</div>

<br>


**9. [Attention, Attention model, Attention weights]**

<div dir="rtl">
[ژرف‌نگری، مدل ژرف‌نگری، وزن‌های ژرف‌نگری]
</div>

<br>


**10. Overview**

<div dir="rtl">
نمای کلی
</div>

<br>


**11. Architecture of a traditional RNN ― Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:**

<div dir="rtl">
معماری RNN سنتی ــ شبکه‌های عصبی برگشتی که همچنین با عنوان RNN شناخته می‌شوند، دسته‌ای از شبکه‌های عصبی‌اند که این امکان را می‌دهند خروجی‌های قبلی به‌عنوان ورودی استفاده شوند و در عین حال حالت‌های نهان داشته باشند. این شبکه‌ها به‌طور معمول عبارت‌اند از:</div>

<br>


**12. For each timestep t, the activation a<t> and the output y<t> are expressed as follows:**

<div dir="rtl">
به‌ازای هر گام زمانی t، فعال‌سازی a<t> و خروجی y<t> به‌صورت زیر بیان می‌شود:
 </div>

<br>


**13. and**

<div dir="rtl">
و
</div>

<br>


**14. where Wax,Waa,Wya,ba,by are coefficients that are shared temporally and g1,g2 activation functions.**

<div dir="rtl">
که در آن Wax,Waa,Wya,ba,by ضرایبی‌اند که در راستای زمان به ‌اشتراک گذاشته می‌شوند و g1، g2 توابع فعال‌سازی‌ هستند.
</div>

<br>


**15. The pros and cons of a typical RNN architecture are summed up in the table below:**

<div dir="rtl">
مزایا و معایب معماری RNN به‌صورت خلاصه در جدول زیر آورده شده‌اند:
</div>

<br>


**16. [Advantages, Possibility of processing input of any length, Model size not increasing with size of input, Computation takes into account historical information, Weights are shared across time]**

<div dir="rtl">
مزایا، امکان پردازش ورودی با هر طولی، اندازه‌ی مدل مطابق با اندازه‌ی ورودی افزایش نمی‌یابد، اطلاعات (زمان‌های) گذشته در محاسبه در نظر گرفته می‌شود، وزن‌ها در طول زمان به‌ اشتراک گذاشته می‌شوند]
</div>

<br>


**17. [Drawbacks, Computation being slow, Difficulty of accessing information from a long time ago, Cannot consider any future input for the current state]**

<div dir="rtl">
[معایب، محاسبه کند می‌شود، دشوار بودن دسترسی به اطلاعات مدت‌ها پیش، در نظر نگرفتن ورودی‌های بعدی در وضعیت جاری]
</div>

<br>


**18. Applications of RNNs ― RNN models are mostly used in the fields of natural language processing and speech recognition. The different applications are summed up in the table below:**

<div dir="rtl">
کاربردهایRNN  ها ــ مدل‌های RNN غالباً در حوزه‌ی پردازش زبان طبیعی و حوزه‌ی بازشناسایی گفتار به کار می‌روند. کاربردهای مختلف آنها به صورت خلاصه در جدول زیر آورده شده‌اند:
</div>

<br>


**19. [Type of RNN, Illustration, Example]**

<div dir="rtl">
[نوع RNN، نگاره، مثال]
</div>

<br>


**20. [One-to-one, One-to-many, Many-to-one, Many-to-many]**

<div dir="rtl">
[یک به یک، یک به چند، چند به یک، چند به چند]
</div>

<br>


**21. [Traditional neural network, Music generation, Sentiment classification, Name entity recognition, Machine translation]**

<div dir="rtl">
[شبکه‌ی عصبی سنتی، تولید موسیقی، دسته‌بندی حالت احساسی، بازشناسایی موجودیت اسمی، ترجمه ماشینی]
</div>

<br>


**22. Loss function ― In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:**

<div dir="rtl">
تابع خطا ــ در شبکه عصبی برگشتی، تابع خطا L برای همه‌ی گام‌های زمانی براساس خطا در هر گام به صورت زیر محاسبه می‌شود:
</div>

<br>


**23. Backpropagation through time ― Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is expressed as follows:**

<div dir="rtl">
انتشار معکوس در طول زمان ـــ انتشار معکوس در هر نقطه از زمان انجام می‌شود. در گام زمانی T، مشتق خطا L با توجه به ماتریس وزن W به‌صورت زیر بیان می‌شود:
</div>

<br>


**24. Handling long term dependencies**

<div dir="rtl">
کنترل وابستگی‌های بلندمدت
</div>

<br>


**25. Commonly used activation functions ― The most common activation functions used in RNN modules are described below:**

<div dir="rtl">
توابع فعال‌سازی پرکاربرد ـــ رایج‌ترین توابع فعال‌سازی به‌کاررفته در ماژول‌های RNN به شرح زیر است:
</div>

<br>


**26. [Sigmoid, Tanh, RELU]**

<div dir="rtl">
[سیگموید، تانژانت هذلولوی، یکسو ساز]
</div>

<br>


**27. Vanishing/exploding gradient ― The vanishing and exploding gradient phenomena are often encountered in the context of RNNs. The reason why they happen is that it is difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.**

<div dir="rtl">
مشتق صفرشونده/منفجرشونده ــ  پدیده مشتق صفرشونده و منفجرشونده غالبا در بستر RNNها رخ می‌دهند. علت چنین رخدادی این است که به دلیل گرادیان ضربی، که می‌تواند با توجه به تعداد لایه‌ها به صورت نمایی کاهش/افزایش می‌یابد، به‌دست آوردن وابستگی‌های بلندمدت سخت است.
</div>


<br>


**28. Gradient clipping ― It is a technique used to cope with the exploding gradient problem sometimes encountered when performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in practice.**

<div dir="rtl">
برش گرادیان ــ یک روش برای مقابله با انفجار گرادیان است که گاهی اوقات هنگام انتشار معکوس رخ می‌دهد. با تعیین حداکثر مقدار برای گرادیان، این پدیده در عمل کنترل می‌شود.
</div>

<br>


**29. clipped**

<div dir="rtl">
برش ‌داده‌شده
</div>

<br>


**30. Types of gates ― In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and are equal to:**

<div dir="rtl">
انواع دروازه ـــ برای حل مشکل مشتق صفرشونده/منفجرشونده، در برخی از انواع RNN ها، دروازه‌های خاصی استفاده می‌شود و این دروازه‌ها عموما هدف معینی دارند. این  دروازه‌ها عموما با نمادΓ  نمایش داده می‌شوند و برابرند با:
</div>

<br>


**31. where W,U,b are coefficients specific to the gate and σ is the sigmoid function. The main ones are summed up in the table below:**

<div dir="rtl">
که W,U,b ضرایب خاص دروازه و σ تابع سیگموید است. دروازه‌های اصلی به صورت خلاصه در جدول زیر آورده شده‌اند:
</div>

<br>


**32. [Type of gate, Role, Used in]**

<div dir="rtl">
[نوع دروازه، نقش، به‌کار رفته در]
</div>

<br>


**33. [Update gate, Relevance gate, Forget gate, Output gate]**

<div dir="rtl">
33. [دروازه‌ی به‌روزرسانی، دروازه‌ی ربط(میزان اهمیت)، دروازه‌ی فراموشی، دروازه‌ی خروجی]
</div>

<br>


**34. [How much past should matter now?, Drop previous information?, Erase a cell or not?, How much to reveal of a cell?]**

<div dir="rtl">
34. [چه میزان از گذشته اکنون اهمیت دارد؟ اطلاعات گذشته رها شوند؟ سلول حذف شود یا خیر؟ چه میزان از (محتوای) سلول آشکار شود؟]
</div>

<br>


**35. [LSTM, GRU]**

<div dir="rtl">
[LSTM، GRU]
</div>

<br>


**36. GRU/LSTM ― Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being a generalization of GRU. Below is a table summing up the characterizing equations of each architecture:**

<div dir="rtl">
GRU/LSTM ـــ واحد برگشتی دروازه‌دار (GRU) و واحدهای حافظه‌ی کوتاه‌-مدت طولانی (LSTM) مشکل مشتق صفرشونده که در RNNهای سنتی رخ می‌دهد، را بر طرف می‌کنند، درحالی‌که LSTM شکل عمومی‌تر  GRU است. در جدول زیر، معادله‌های توصیف‌کنندهٔ هر معماری به صورت خلاصه آورده شده‌اند:
</div>

<br>


**37. [Characterization, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Dependencies]**

<div dir="rtl">
37. [توصیف، واحد برگشتی دروازه‌دار (GRU)، حافظه‌ی کوتاه-مدت طولانی (LSTM)، وابستگی‌ها]
</div>
<br>


**38. Remark: the sign ⋆ denotes the element-wise multiplication between two vectors.**

<div dir="rtl">
نکته: نشانه‌ی * نمایان‌گر ضرب عنصربه‌عنصر دو بردار است.
</div>

<br>


**39. Variants of RNNs ― The table below sums up the other commonly used RNN architectures:**

<div dir="rtl">
انواع RNN ها ــ جدول زیر سایر معماری‌های پرکاربرد RNN را به صورت خلاصه نشان می‌دهد.
</div>

<br>


**40. [Bidirectional (BRNN), Deep (DRNN)]**

<div dir="rtl">
[دوسویه  (BRNN)، عمیق (DRNN)]
</div>

<br>


**41. Learning word representation**

<div dir="rtl">
یادگیری بازنمائی کلمه
</div>

<br>


**42. In this section, we note V the vocabulary and |V| its size.**

<div dir="rtl">
در این بخش، برای اشاره به واژگان از V و برای اشاره به اندازه‌ی آن از |V| استفاده می‌کنیم.
</div>

<br>


**43. Motivation and notations**

<div dir="rtl">
انگیزه و نمادها
</div>

<br>


**44. Representation techniques ― The two main ways of representing words are summed up in the table below:**

<div dir="rtl">
روش‌های بازنمائی ― دو روش اصلی برای بازنمائی کلمات به صورت خلاصه در جدول زیر آورده شده‌اند:
</div>

<br>


**45. [1-hot representation, Word embedding]**

<div dir="rtl">
[بازنمائی تک‌فعال، تعبیه‌ی کلمه]
</div>

<br>


**46. [teddy bear, book, soft]**

<div dir="rtl">
[خرس تدی، کتاب، نرم]
</div>

<br>


**47. [Noted ow, Naive approach, no similarity information, Noted ew, Takes into account words similarity]**

<div dir="rtl">
[نشان داده شده با نماد ow، رویکرد ساده، فاقد اطلاعات تشابه، نشان داده شده با نماد ew، به‌حساب‌آوردن تشابه کلمات]
</div>

<br>


**48. Embedding matrix ― For a given word w, the embedding matrix E is a matrix that maps its 1-hot representation ow to its embedding ew as follows:**

<div dir="rtl">
ماتریس تعبیه ـــ به‌ ازای کلمه‌ی مفروض w ، ماتریس تعبیه E ماتریسی است که بازنمائی تک‌فعال  ow را به نمایش تعبیه‌ی ew نگاشت می‌دهد:
</div>

<br>


**49. Remark: learning the embedding matrix can be done using target/context likelihood models.**

<div dir="rtl">
نکته: یادگیری ماتریس تعبیه را می‌توان با استفاده از مدل‌های درست‌نمایی هدف/متن(زمینه) انجام داد.
</div>

<br>


**50. Word embeddings**

<div dir="rtl">
(نمایش) تعبیه‌ی کلمه
</div>

<br>


**51. Word2vec ― Word2vec is a framework aimed at learning word embeddings by estimating the likelihood that a given word is surrounded by other words. Popular models include skip-gram, negative sampling and CBOW.**

<div dir="rtl">
Word2vec ― Word2vec چهارچوبی است که با محاسبه‌ی احتمال قرار گرفتن یک کلمه‌ی خاص در میان سایر کلمات، تعبیه‌های کلمه را یاد می‌گیرد. مدل‌های متداول شامل Skip-gram، نمونه‌برداری منفی و CBOW هستند.
</div>

<br>


**52. [A cute teddy bear is reading, teddy bear, soft, Persian poetry, art]**

<div dir="rtl">
[یک خرس تدی بامزه در حال مطالعه است، خرس تدی، نرم، شعر فارسی، هنر]
</div>

<br>


**53. [Train network on proxy task, Extract high-level representation, Compute word embeddings]**

<div dir="rtl">
[آموزش شبکه بر روی مسئله‌ی جایگزین، استخراج بازنمائی سطح بالا، محاسبه‌ی نمایش تعبیه‌ی کلمات]
</div>

<br>


**54. Skip-gram ― The skip-gram word2vec model is a supervised learning task that learns word embeddings by assessing the likelihood of any given target word t happening with a context word c. By noting θt a parameter associated with t, the probability P(t|c) is given by:**

<div dir="rtl">
Skip-gram ــ مدل اسکیپ‌گرام word2vec یک وظیفه‌ی یادگیری بانظارت است که تعبیه‌های کلمه را با ارزیابی احتمال وقوع کلمه‌ی t هدف با کلمه‌ی زمینه c یاد می‌گیرد. با توجه به اینکه نماد θt پارامتری مرتبط با t است، احتمال P(t|c) به‌صورت زیر به‌دست می‌آید:
</div>

<br>


**55. Remark: summing over the whole vocabulary in the denominator of the softmax part makes this model computationally expensive. CBOW is another word2vec model using the surrounding words to predict a given word.**

<div dir="rtl">
نکته: جمع کل واژگان در بخش مقسوم‌الیه بیشینه‌ی‌هموار باعث می‌شود که این مدل از لحاظ محاسباتی گران شود. مدل CBOW مدل word2vec دیگری ست که از کلمات اطراف برای پیش‌بینی یک کلمهٔ مفروض استفاده می‌کند.
</div>

<br>


**56. Negative sampling ― It is a set of binary classifiers using logistic regressions that aim at assessing how a given context and a given target words are likely to appear simultaneously, with the models being trained on sets of k negative examples and 1 positive example. Given a context word c and a target word t, the prediction is expressed by:**

<div dir="rtl">
نمونه‌گیری منفی ― مجموعه‌ای از دسته‌بندی‌های دودویی با استفاده از رگرسیون لجستیک است که مقصودش ارزیابی احتمال ظهور همزمان کلمه‌ی مفروض هدف و کلمه‌ی مفروض زمینه است، که در اینجا مدل‌ها براساس مجموعه k مثال منفی و 1 مثال مثبت آموزش می‌بینند. با توجه به کلمه‌ی مفروض زمینه c و کلمه‌ی مفروض هدف t، پیش‌بینی به صورت زیر بیان می‌شود:
</div>

<br>


**57. Remark: this method is less computationally expensive than the skip-gram model.**

<div dir="rtl">
نکته: این روش از لحاظ محاسباتی ارزان‌تر از مدل skip-gram است.
</div>

<br>


**57bis. GloVe ― The GloVe model, short for global vectors for word representation, is a word embedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of times that a target i occurred with a context j. Its cost function J is as follows:**

<div dir="rtl">
GloVe ― مدل GloVe، مخفف بردارهای سراسری بازنمائی کلمه، یکی از روش‌های تعبیه کلمه است که از ماتریس هم‌رویدادی X استفاده می‌کند که در آن هر Xi,j به تعداد دفعاتی اشاره دارد که هدف i با زمینهٔ j رخ می‌دهد. تابع هزینه‌ی J به‌صورت زیر است:
</div>

<br>


**58. where f is a weighting function such that Xi,j=0⟹f(Xi,j)=0.
Given the symmetry that e and θ play in this model, the final word embedding e(final)w is given by:**

<div dir="rtl">
که در آن f تابع وزن‌دهی است، به‌طوری که Xi,j=0⟹f(Xi,j)=0. با توجه به تقارنی که e و θ در این مدل دارند، نمایش تعبیه‌ی نهایی کلمه‌ e(final)w به صورت زیر محاسبه می‌شود:
</div>

<br>


**59. Remark: the individual components of the learned word embeddings are not necessarily interpretable.**

<div dir="rtl">
تذکر: مولفه‌های مجزا در نمایش تعبیه‌ی یادگرفته‌شده‌ی کلمه الزاما قابل تفسیر نیستند.
</div>

<br>


**60. Comparing words**

<div dir="rtl">
مقایسه‌ی کلمات
</div>

<br>


**61. Cosine similarity ― The cosine similarity between words w1 and w2 is expressed as follows:**

<div dir="rtl">
شباهت کسینوسی - شباهت کسینوسی بین کلمات w1 و w2 به ‌صورت زیر بیان می‌شود:
</div>

<br>


**62. Remark: θ is the angle between words w1 and w2.**

<div dir="rtl">
نکته: θ زاویهٔ بین کلمات w1 و w2 است.
</div>

<br>


**63. t-SNE ― t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at reducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly used to visualize word vectors in the 2D space.**

<div dir="rtl">
t-SNE ― t-SNE (نمایش تعبیه‌ی همسایه‌ی تصادفی توزیع‌شده توسط توزیع t) روشی است که هدف آن کاهش تعبیه‌های ابعاد بالا به فضایی با ابعاد پایین‌تر است. این روش در تصویرسازی بردارهای کلمه در فضای 2 بعدی کاربرد فراوانی دارد.
</div>

<br>


**64. [literature, art, book, culture, poem, reading, knowledge, entertaining, loveable, childhood, kind, teddy bear, soft, hug, cute, adorable]**

<div dir="rtl">
[ادبیات، هنر، کتاب، فرهنگ، شعر، دانش، مفرح، دوست‌داشتنی، دوران کودکی، مهربان، خرس تدی، نرم، آغوش، بامزه، ناز]
</div>

<br>


**65. Language model**

<div dir="rtl">
مدل زبانی
</div>

<br>


**66. Overview ― A language model aims at estimating the probability of a sentence P(y).**

<div dir="rtl">
نمای کلی ـــ هدف مدل زبان تخمین احتمال جمله‌ی P(y) است.
</div>

<br>


**67. n-gram model ― This model is a naive approach aiming at quantifying the probability that an expression appears in a corpus by counting its number of appearance in the training data.**

<div dir="rtl">
مدل  ان‌گرام ــ این مدل یک رویکرد ساده با هدف اندازه‌گیری احتمال نمایش یک عبارت در یک نوشته است که با دفعات تکرار آن در داده‌های آموزشی محاسبه می‌شود.
</div>

<br>


**68. Perplexity ― Language models are commonly assessed using the perplexity metric, also known as PP, which can be interpreted as the inverse probability of the dataset normalized by the number of words T. The perplexity is such that the lower, the better and is defined as follows:**

<div dir="rtl">
سرگشتگی ـــ مدل‌های زبانی معمولاً با معیار سرگشتی، که با PP هم نمایش داده می‌شود، سنجیده می‌شوند، که مقدار آن معکوس احتمال یک مجموعه‌ داده است که تقسیم بر تعداد کلمات T می‌شود. هر چه سرگشتگی کمتر باشد بهتر است و به صورت زیر تعریف می‌شود:
</div>

<br>


**69. Remark: PP is commonly used in t-SNE.**

<div dir="rtl">
نکته: PP عموما در t-SNE کاربرد دارد.
</div>

<br>


**70. Machine translation**

<div dir="rtl">
ترجمه ماشینی
</div>

<br>


**71. Overview ― A machine translation model is similar to a language model except it has an encoder network placed before. For this reason, it is sometimes referred as a conditional language model. The goal is to find a sentence y such that:**

<div dir="rtl">
نمای کلی ― مدل ترجمه‌ی ماشینی مشابه مدل زبانی است با این تفاوت که یک شبکه‌ی رمزنگار قبل از آن قرار گرفته است. به همین دلیل، گاهی اوقات به آن مدل زبان شرطی می‌گویند. هدف آن یافتن جمله y است بطوری که:
</div>

<br>


**72. Beam search ― It is a heuristic search algorithm used in machine translation and speech recognition to find the likeliest sentence y given an input x.**

<div dir="rtl">
جستجوی پرتو ― یک الگوریتم جستجوی اکتشافی است که در ترجمه‌ی ماشینی و بازتشخیص گفتار برای یافتن محتمل‌ترین جمله‌ی y باتوجه به ورودی مفروض x بکار برده می‌شود.
</div>

<br>


**73. [Step 1: Find top B likely words y<1>, Step 2: Compute conditional probabilities y<k>|x,y<1>,...,y<k−1>, Step 3: Keep top B combinations x,y<1>,...,y<k>, End process at a stop word]**

<div dir="rtl">
[گام 1: یافتن B کلمه‌ی محتمل برتر y<1>، گام 2: محاسبه احتمالات شرطی y|x,y<1>,...,y<k−1>، گام 3: نگه‌داشتن B ترکیب برتر x,y<1>,…,y، خاتمه فرآیند با کلمه‌ی توقف]
 </div>

<br>


**74. Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.**

<div dir="rtl">
نکته: اگر پهنای پرتو 1 باشد، آنگاه با جست‌وجوی حریصانهٔ ساده برابر خواهد بود.
</div>

<br>


**75. Beam width ― The beam width B is a parameter for beam search. Large values of B yield to better result but with slower performance and increased memory. Small values of B lead to worse results but is less computationally intensive. A standard value for B is around 10.**

<div dir="rtl">
پهنای پرتو ـــ پهنای پرتوی B پارامتری برای جستجوی پرتو است. مقادیر بزرگ B به نتیجه بهتر منتهی می‌شوند اما عملکرد آهسته‌تری دارند و حافظه را افزایش می‌دهند. مقادیر کوچک B به نتایج بدتر منتهی می‌شوند اما بار محاسباتی پایین‌تری دارند. مقدار استاندارد B حدود 10 است.
</div>

<br>


**76. Length normalization ― In order to improve numerical stability, beam search is usually applied on the following normalized objective, often called the normalized log-likelihood objective, defined as:**

<div dir="rtl">
نرمال‌سازی طول ―‌ برای بهبود ثبات عددی، جستجوی پرتو معمولا با تابع هدف نرمال‌شده‌ی زیر اعمال می‌شود، که اغلب اوقات هدف درست‌نمایی لگاریتمی نرمال‌شده نامیده می‌شود و به‌صورت زیر تعریف می‌شود:
</div>

<br>


**77. Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.**

<div dir="rtl">
تذکر: پارامتر α را می‌توان تعدیل‌کننده نامید و مقدارش معمولا بین 0.5 و 1 است.
</div>

<br>


**78. Error analysis ― When obtaining a predicted translation ˆy that is bad, one can wonder why we did not get a good translation y∗ by performing the following error analysis:**

<div dir="rtl">
تحلیل خطا ―زمانی‌که ترجمه‌ی پیش‌بینی‌شده‌ی ^y ی به‌دست می‌آید که مطلوب نیست، می‌توان با انجام تحلیل خطای زیر از خود پرسید که چرا ترجمه y* خوب نیست:
</div>

<br>


**79. [Case, Root cause, Remedies]**

<div dir="rtl">
[قضیه، ریشه‌ی مشکل، راه‌حل]
</div>

<br>


**80. [Beam search faulty, RNN faulty, Increase beam width, Try different architecture, Regularize, Get more data]**

<div dir="rtl">
[جستجوی پرتوی معیوب، RNN معیوب، افزایش پهنای پرتو، امتحان معماری‌های مختلف، استفاده از تنظیم‌کننده، جمع‌آوری داده‌های بیشتر]</div>

<br>


**81. Bleu score ― The bilingual evaluation understudy (bleu) score quantifies how good a machine translation is by computing a similarity score based on n-gram precision. It is defined as follows:**

<div dir="rtl">
امتیاز Bleu ― جایگزین ارزشیابی دوزبانه  (bleu) میزان خوب بودن ترجمه ماشینی را با محاسبه‌ی امتیاز تشابه برمبنای دقت ان‌گرام اندازه‌گیری می‌کند. (این امتیاز) به صورت زیر تعریف می‌شود:
</div>

<br>


**82. where pn is the bleu score on n-gram only defined as follows:**

<div dir="rtl">
که pn امتیاز bleu تنها براساس ان‌گرام است و به صورت زیر تعریف می‌شود:
</div>

<br>


**83. Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially inflated bleu score.**

<div dir="rtl">
تذکر: ممکن است برای پیشگیری از امتیاز اغراق آمیز تصنعیbleu ، برای ترجمه‌های پیش‌بینی‌شده‌ی کوتاه از جریمه اختصار استفاده شود.</div>

<br>


**84. Attention**

<div dir="rtl">
ژرف‌نگری
</div>

<br>


**85. Attention model ― This model allows an RNN to pay attention to specific parts of the input that is considered as being important, which improves the performance of the resulting model in practice. By noting α<t,t′> the amount of attention that the output y<t> should pay to the activation a<t′> and c<t> the context at time t, we have:**

<div dir="rtl">
مدل ژرف‌نگری ― این مدل به RNN این امکان را می‌دهد که به بخش‌های خاصی از ورودی که حائز اهمیت هستند توجه نشان دهد که در عمل باعث بهبود عملکرد مدل حاصل‌شده خواهد شد. اگر α<t,t′> به معنای مقدار توجهی باشد که خروجی y باید به فعال‌سازی a<t′>  داشته باشد و c نشان‌دهنده‌ی زمینه (متن) در زمان t باشد، داریم:
 </div>

<br>


**86. with**

<div dir="rtl">
با
</div>

<br>


**87. Remark: the attention scores are commonly used in image captioning and machine translation.**

<div dir="rtl">
نکته: امتیازات ژرف‌نگری عموما در عنوان‌سازی متنی برای تصویر (image captioning) و ترجمه ماشینی کاربرد دارد.
</div>

<br>


**88. A cute teddy bear is reading Persian literature.**

<div dir="rtl">
یک خرس تدی بامزه در حال خواندن ادبیات فارسی است.
</div>

<br>


**89. Attention weight ― The amount of attention that the output y<t> should pay to the activation a<t′> is given by α<t,t′> computed as follows:**

<div dir="rtl">
وزن ژرف‌نگری ― مقدار توجهی که خروجی y باید به فعال‌سازی a<t′> داشته باشد به‌وسیله‌ی α<t,t′> به‌دست می‌آید که به‌صورت زیر محاسبه می‌شود:
</div>

<br>


**90. Remark: computation complexity is quadratic with respect to Tx.**

<div dir="rtl">
نکته: پیچیدگی محاسباتی به نسبت Tx از نوع درجه‌ی دوم است.
</div>

<br>


**91. The Deep Learning cheatsheets are now available in [target language].**

<div dir="rtl">
راهنمای یادگیری عمیق هم اکنون به زبان [فارسی] در دسترس است.
</div>

<br>

**92. Original authors**

<div dir="rtl">
نویسندگان اصلی
</div>

<br>

**93. Translated by X, Y and Z**

<div dir="rtl">
ترجمه شده توسط X،Y و Z
</div>

<br>

**94. Reviewed by X, Y and Z**

<div dir="rtl">
بازبینی شده توسط توسط X،Y و Z
</div>

<br>

**95. View PDF version on GitHub**

<div dir="rtl">
نسخه پی‌دی‌اف را در گیت‌هاب ببینید
</div>

<br>

**96. By X and Y**

<div dir="rtl">
توسط X و Y
</div>

<br>
