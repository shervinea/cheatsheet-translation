**Reflex-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models)

مدل‌های عکس‌العمل محور

**2. Linear predictors**
تخمین‌گرهای خطی


**3. In this section, we will go through reflex-based models that can improve with experience, by going through samples that have input-output pairs.**

 در این بخش ما مدل‌های عکس‌العمل محور را که می‌توانند با تجربه،‌ با بررسی نمونه‌هایی که  جفت ورودی خروجی دارند، بهبود یابند را بررسی کنیم.


**4. Feature vector ― The feature vector of an input x is noted ϕ(x) and is such that:**

بردار ویژگی: بردار ویژگی ورودی x که با ... نمایش داده می‌شود و به صورتی است که:


**5. Score ― The score s(x,w) of an example (ϕ(x),y)∈Rd×R associated to a linear model of weights w∈Rd is given by the inner product:**

امتیاز: امتیاز .... برای مثال ...اختصاص داده شده به یک مدل خطی با وزن‌های ...که با ضرب داخلی داده شده است


**6. Classification**

**7. Linear classifier ― Given a weight vector w∈Rd and a feature vector ϕ(x)∈Rd, the binary linear classifier fw is given by:**

دسته‌بند خطی:‌ با فرض داده شدن بردار وزنی مانند ... و بردار ویژگی ...، دسته بند دودویی خطی ... داده شده است توسط:


**8. if**
اگر


**9. Margin ― The margin m(x,y,w)∈R of an example (ϕ(x),y)∈Rd×{−1,+1} associated to a linear model of weights w∈Rd quantifies the confidence of the prediction: larger values are better. It is given by:**

حاشیه










