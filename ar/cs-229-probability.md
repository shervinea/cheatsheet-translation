**Probabilities and Statistics translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)

<br>

**1. Probabilities and Statistics refresher**
<div dir="rtl">
مراجعة للاحتمالات والإحصاء
</div>
<br>

**2. Introduction to Probability and Combinatorics**
<div dir="rtl">
مقدمة في الاحتمالات والتوافيق
</div>
<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**
<div dir="rtl">
فضاء العينة ― يعرَّف فضاء العينة لتجربة ما بمجموعة كل النتائج الممكنة لهذه التجربة ويرمز لها بـ S.
</div>
<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**
<div dir="rtl">
الحدث ― أي مجموعة جزئية E من فضاء العينة تعتبر حدثاً. أي، الحدث هو مجموعة من النتائج الممكنة للتجربة. إذا كانت نتيجة التجربة محتواة في E، عندها نقول أن الحدث E وقع.
</div>
<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**
<div dir="rtl">
مسلَّمات الاحتمالات. من أجل كل حدث E، نرمز لإحتمال وقوعه بـ P(E).
</div>
<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**
<div dir="rtl">
المسلَّمة 1 ― كل احتمال يأخد قيماً بين الـ 0 والـ 1 مضمَّنة، على سبيل المثال:
</div>
<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**
<div dir="rtl">
المسلَّمة 2 ― احتمال وقوع حدث ابتدائي واحد على الأقل من الأحداث الابتدائية في فضاء العينة يساوي الـ 1، على سبيل المثال:
</div>
<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**
<div dir="rtl">
المسلَّمة 3 ― من أجل أي سلسلة من الأحداث الغير متداخلة E1,...,En، لدينا:
</div>
<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**
<div dir="rtl">
التباديل ― التبديل هو عبارة عن ترتيب معين لـ r غرض مختارة من مجموعة من n غرض. عدد هكذا تراتيب يرمز له بـ P(n, r)، المعرف كالتالي:</div>
<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**
<div dir="rtl">
التوافيق ― التوفيق هو اختيار لـ r غرض من مجموعة مكونة من n غرض بدون إعطاء الترتيب أية أهمية. عدد هكذا توافيق يرمز له بـ C(n, r)، المعرف كالتالي:
</div>
<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**
<div dir="rtl">
ملاحظة: من أجل <span dir="ltr">0⩽r⩽n</span>، يكون لدينا P(n,r)⩾C(n,r)
</div>
<br>

**12. Conditional Probability**
<div dir="rtl">
الاحتمال الشرطي
</div>
<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**
<div dir="rtl">
قاعدة بايز ― من أجل الأحداث A و B بحيث P(B)>0، يكون لدينا:
</div>
<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**
<div dir="rtl">
ملاحظة: لدينا P(A∩B)=P(A)P(B|A)=P(A|B)P(B)
</div>
<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**
<div dir="rtl">
القسم ― ليكن {Ai,i∈[[1,n]]} بحيث من أجل كل i، لدينا<span dir="ltr">Ai≠∅ </span>. نقول أن {Ai} قسم إذا كان لدينا: 
</div>
<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**
<div dir="rtl">
ملاحظة: من أجل أي حدث B من فضاء العينة، لدينا P(B)=n∑i=1P(B|Ai)P(Ai).
</div>
<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**
<div dir="rtl">
النسخة الموسعة من قاعدة بايز ― ليكن {Ai,i∈[[1,n]]} قسم من فضاء العينة. لدينا:
</div>
<br>

**18. Independence ― Two events A and B are independent if and only if we have:**
<div dir="rtl">
الاستقلال ― يكون حدثين A و B مستقلين إذا وفقط إذا كان لدينا:
</div>
<br>

**19. Random Variables**
<div dir="rtl">
المتحولات العشوائية
</div>
<br>

**20. Definitions**
<div dir="rtl">
تعاريف
</div>
<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**
<div dir="rtl">
المتحول العشوائي ― المتحول العشوائي، المرمز له عادة ب X، هو دالة تربط كل عنصر من فضاء العينة إلى خط الأعداد الحقيقية.
</div>
<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**
<div dir="rtl">
دالة التوزيع التراكمي (CDF) ― تعرف دالة التوزيع التراكمي F، والتي تكون غير متناقصة بشكل دائم وتحقق limx→−∞F(x)=0 و limx→+∞F(x)=1، كالتالي:
</div>
<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**
<div dir="rtl">
ملاحظة: لدينا P(a&lt;X⩽B)=F(b)−F(a).
</div>
<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**
<div dir="rtl">
دالة الكثافة الإحتمالية (PDF) ― دالة الكثافة الاحتمالية f هي احتمال أن يأخذ X قيماً بين قيمتين متجاورتين من قيم المتحول العشوائي.
</div>
<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**
<div dir="rtl">
علاقات تتضمن دالة الكثافة الاحتمالية ودالة التوزع التراكمي ― هذه بعض الخصائص التي من المهم معرفتها في الحالتين المتقطعة (D) والمستمرة (C).
</div>
<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**
<div dir="rtl">
[الحالة، دالة التوزع التراكمي F، دالة الكثافة الاحتمالية f، خصائص دالة الكثافة الاحتمالية]
</div>
<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**
<div dir="rtl">
التوقع وعزوم التوزيع ― فيما يلي المصطلحات المستخدمة للتعبير عن القيمة المتوقعة E[X]، الصيغة العامة للقيمة المتوقعة E[g(X)]، العزم رقم K  <span dir="ltr">E[XK]</span>  ودالة السمة ψ(ω) من أجل الحالات المتقطعة والمستمرة:
</div>
<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**
<div dir="rtl">
التباين ― تباين متحول عشوائي، والذي يرمز له عادةً ب Var(X) أو σ2، هو مقياس لانتشار دالة توزيع هذا المتحول. يحسب بالشكل التالي:
</div>
<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**
<div dir="rtl">
الانحراف المعياري ― الانحراف المعياري لمتحول عشوائي، والذي يرمز له عادةً ب σ، هو مقياس لانتشار دالة توزيع هذا المتحول بما يتوافق مع وحدات قياس المتحول العشوائي. يحسب بالشكل التالي:
</div>
<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**
<div dir="rtl">
تحويل المتحولات العشوائية ― لتكن المتحولات العشوائية X وY مرتبطة من خلال دالة ما. باعتبار fX وfY دالتا التوزيع لX وY على التوالي، يكون لدينا:</div>
<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**
<div dir="rtl">
قاعدة لايبنتز للتكامل ― لتكن g دالة لـ x وربما لـ c، ولتكن a وb حدود قد تعتمد على c. يكون لدينا:
</div>
<br>

**32. Probability Distributions**
<div dir="rtl">
التوزعات الاحتمالية
</div>
<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**
<div dir="rtl">
متراجحة تشيبشيف ― ليكن X متحولاً عشوائياً قيمته المتوقعة تساوي μ. من أجل k ،σ>0، لدينا المتراجحة التالية:
</div>
<br>

**34. Main distributions ― Here are the main distributions to have in mind:**
<div dir="rtl">
التوزعات الأساسية ― فيما يلي التوزعات الأساسية لأخذها بالاعتبار:
</div>
<br>

**35. [Type, Distribution]**
<div dir="rtl">
[الحالة، التوزع]
</div>
<br>

**36. Jointly Distributed Random Variables**
<div dir="rtl">
المتحولات العشوائية الموزعة بشكل مشترك
</div>
<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**
<div dir="rtl">
الكثافة الهامشية والتوزع التراكمي ― من دالة الكثافة الاحتمالية المشتركة fXY، لدينا:
</div>
<br>

**38. [Case, Marginal density, Cumulative function]**
<div dir="rtl">
[الحالة، الكثافة الهامشية، الدالة التراكمية]
</div>
<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**
<div dir="rtl">
الكثافة الشرطية ― الكثافة الشرطية لـ X بالنسبة لـ Y، والتي يرمز لها عادةً بـ fX|Y، تعرف بالشكل التالي: 
</div>
<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**
<div dir="rtl">
الاستقلال ― يقال عن متحولين عشوائيين X و Y أنهما مستقلين إذا كان لدينا:
</div>
<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**
<div dir="rtl">
التغاير ― نعرف تغاير متحولين عشوائيين X و Y، والذي نرمز له بـ σ2XY أو بالرمز الأكثر شيوعاً Cov(X,Y)، كالتالي:
</div>
<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**
<div dir="rtl">
الارتباط ― بأخذ σX، σY كانحراف معياري لـ X و Y، نعرف الارتباط بين المتحولات العشوائية X و Y، و المرمز بـ ρXY، كالتالي:
</div>
<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**
<div dir="rtl">
ملاحظة 1: من أجل أية متحولات عشوائية X، Y، لدينا ρXY∈[−1,1].
</div>
<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**
<div dir="rtl">
ملاحظة 2: إذا كان X و Y مستقلين، فإن ρXY=0.
</div>
<br>

**45. Parameter estimation**
<div dir="rtl">
تقدير المُدخَل
</div>
<br>

**46. Definitions**
<div dir="rtl">
تعاريف
</div>
<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**
<div dir="rtl">
العينة العشوائية ― العينة العشوائية هي مجموعة من n متحول عشوائي X1,...,Xn والتي تكون مستقلة وموزعة بشكل متطابق مع X.
</div>
<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**
<div dir="rtl">
المُقَدِّر ― المُقَدِّر هو تابع للبيانات المستخدمة لاستنباط قيمة متحول غير معلوم ضمن نموذج إحصائي.
</div>
<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**
<div dir="rtl">
الانحياز ― انحياز مُقَدِّر ^θ هو الفرق بين القيمة المتوقعة لتوزع ^θ والقيمة الحقيقية، كمثال:
</div>
<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**
<div dir="rtl">
ملاحظة: يقال عن مُقَدِّر أنه غير منحاز عندما يكون لدينا E[^θ]=θ.
</div>
<br>

**51. Estimating the mean**
<div dir="rtl">
تقدير المتوسط
</div>
<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**
<div dir="rtl">
متوسط العينة ― يستخدم متوسط عينة عشوائية لتقدير المتوسط الحقيقي μ لتوزع ما، عادةً ما يرمز له بـ ¯¯¯¯¯X ويعرف كالتالي:
</div>
<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**
<div dir="rtl">
ملاحظة: متوسط العينة غير منحاز، أي E[¯¯¯¯¯X]=μ.
</div>
<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**
<div dir="rtl">
مبرهنة النهاية المركزية ― ليكن لدينا عينة عشوائية X1,...,Xn والتي تتبع لتوزع معطى له متوسط μ وتباين σ2، فيكون:
</div>
<br>

**55. Estimating the variance**
<div dir="rtl">
تقدير التباين
</div>
<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**
<div dir="rtl">
تباين العينة ― يستخدم تباين عينة عشوائية لتقدير التباين الحقيقي σ2 لتوزع ما، والذي يرمز له عادةً بـ s2 أو ^σ2 ويعرّف بالشكل التالي:
</div>
<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**
<div dir="rtl">
ملاحظة: تباين العينة غير منحاز، أي E[s2]=σ2.
</div>
<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**
<div dir="rtl">
علاقة مربع كاي مع تباين العينة ― ليكن s2 تباين العينة لعينة عشوائية. لدينا:
</div>
<br>

**59. [Introduction, Sample space, Event, Permutation]**
<div dir="rtl">
[مقدمة، فضاء العينة، الحدث، التبديل]
</div>
<br>

**60. [Conditional probability, Bayes' rule, Independence]**
<div dir="rtl">
[الاحتمال الشرطي، قاعدة بايز، الاستقلال]
</div>
<br>

**61. [Random variables, Definitions, Expectation, Variance]**
<div dir="rtl">
[المتحولات العشوائية، تعاريف، القيمة المتوقعة، التباين]
</div>
<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**
<div dir="rtl">
[التوزعات الاحتمالية، متراجحة تشيبشيف، توزعات رئيسية]
</div>
<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**
<div dir="rtl">
[المتحولات العشوائية الموزعة بشكل مشترك، الكثافة، التغاير، الارتباط]
</div>
<br>

**64. [Parameter estimation, Mean, Variance]**
<div dir="rtl">
[تقدير المُدخَل، المتوسط، التباين]
</div>
