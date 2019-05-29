**1. Probabilities and Statistics refresher**

&#10230;Review Probabilitas dan Statistik

<br>

**2. Introduction to Probability and Combinatorics**

&#10230;Pengenalan Probabilitas dan Kombinatorik

<br>

**3. Sample space ― The set of all possible outcomes of an experiment is known as the sample space of the experiment and is denoted by S.**

&#10230;Sample space - Set dari semua kemungkinan keluaran dari sebuah eksperimen didefinisikan sebagai ruang sampel dari eksperimen dan dituliskan sebagai S.

<br>

**4. Event ― Any subset E of the sample space is known as an event. That is, an event is a set consisting of possible outcomes of the experiment. If the outcome of the experiment is contained in E, then we say that E has occurred.**

&#10230;Event - Subset apapun dari dari ruang sampel dinamakan sebuah event. Sebuah event adalah sebuah set yang berisi kemungkinan keluaran dari sebuah eksperimen (persitiwa). Jika keluaran dari eksperiment berisi E, maka kita katakan bahwa event E terjadi.

<br>

**5. Axioms of probability For each event E, we denote P(E) as the probability of event E occuring.**

&#10230;Aksioma probabilitas untuk setiap event E, kita definisikan p(E) sebagai probabilitas terjadinya event E.

<br>

**6. Axiom 1 ― Every probability is between 0 and 1 included, i.e:**

&#10230;Aksioma 1 - Setiap probabilitas bernilai diantara 0 hingga 1, sebagai contoh:

<br>

**7. Axiom 2 ― The probability that at least one of the elementary events in the entire sample space will occur is 1, i.e:**

&#10230;Aksioma 2 - Probabilitas bahwa setidaknya satu dari event-event dasar pada keseluruhan ruang sampel akan terjadi adalah 1, sebagai contoh:

<br>

**8. Axiom 3 ― For any sequence of mutually exclusive events E1,...,En, we have:**

&#10230;Axiom 3- Untuk runtutan event-event yang mutual exclusive (exclusive satu sama lain) E1,...,En, kita memiliki:

<br>

**9. Permutation ― A permutation is an arrangement of r objects from a pool of n objects, in a given order. The number of such arrangements is given by P(n,r), defined as:**

&#10230;Permutasi - Sebuah permutasi adalah sebuah penyusunan dari objek-objek r dari sebuah kelompok objek-objek n. Jumlah dari penyusunan tersebut dituliskan sebagai P(n,r), dan diformulasikan sebagai:

<br>

**10. Combination ― A combination is an arrangement of r objects from a pool of n objects, where the order does not matter. The number of such arrangements is given by C(n,r), defined as:**

&#10230;Kombinasi - Sebuah kombinasi adalah sebuah penyusunan objek-objek r dari sebuah kumpulan objek-objek n, dimana urutan tidak menjadi permasalahan. Jumlah dari penyusunan tersebut dituliskan sebagai C(n,r), dan diformulasikan sebagai:

<br>

**11. Remark: we note that for 0⩽r⩽n, we have P(n,r)⩾C(n,r)**

&#10230;Perlu diperhatikan: kita mendefinisikan bahwa untuk 0⩽r⩽n, kita memiliki P(n,r)⩾C(n,r)

<br>

**12. Conditional Probability**

&#10230;Probabilitas Kondisional

<br>

**13. Bayes' rule ― For events A and B such that P(B)>0, we have:**

&#10230;Bayes' rule - Untuk event A dan B dimana P(B)>0, kita memiliki:

<br>

**14. Remark: we have P(A∩B)=P(A)P(B|A)=P(A|B)P(B)**

&#10230;Perlu diperhatikan: kita memiliki P(A∩B)=P(A)P(B|A)=P(A|B)P(B)

<br>

**15. Partition ― Let {Ai,i∈[[1,n]]} be such that for all i, Ai≠∅. We say that {Ai} is a partition if we have:**

&#10230;Partisi - Diketahui {Ai,i∈[[1,n]]} untuk semua i, Ai≠∅. Kita menyatakan {Ai} sebagai sebuah partisi jika kita memiliki:

<br>

**16. Remark: for any event B in the sample space, we have P(B)=n∑i=1P(B|Ai)P(Ai).**

&#10230;Perlu diperhatikan: untuk event apapun B pada ruang sampel, kita memiliki P(B)=n∑i=1P(B|Ai)P(Ai).

<br>

**17. Extended form of Bayes' rule ― Let {Ai,i∈[[1,n]]} be a partition of the sample space. We have:**

&#10230;Bentuk extended dari Bayes' rule - Diketahui {Ai,i∈[[1,n]]} sebagai sebuah partisi dari ruang sampel. Kita memiliki:

<br>

**18. Independence ― Two events A and B are independent if and only if we have:**

&#10230;Independence - Dua event A dan B adalah independen jika dan hanyak jika kita memiliki:

<br>

**19. Random Variables**

&#10230;Variabel Acak:

<br>

**20. Definitions**

&#10230;Definisi-definisi

<br>

**21. Random variable ― A random variable, often noted X, is a function that maps every element in a sample space to a real line.**

&#10230;Variabel acak - Sebuah variabel, sering dituliskan sebagai X, adalah sebuah fungsi yang memetakan setiap elemen pada ruang sampel ke sebuah garis numerik.

<br>

**22. Cumulative distribution function (CDF) ― The cumulative distribution function F, which is monotonically non-decreasing and is such that limx→−∞F(x)=0 and limx→+∞F(x)=1, is defined as:**

&#10230;Fungsi distribusi kumulatif (CDF) - Fungsi distribusi kumulatif F, yang secara monoton non-decreasing (tidak menurun) dan sehingga limx→−∞F(x)=0 dan limx→+∞F(x)=1, didefinisikan sebagai

<br>

**23. Remark: we have P(a<X⩽B)=F(b)−F(a).**

&#10230;Perlu diperhatikan: kita memiliki P(a<X⩽B)=F(b)−F(a).

<br>

**24. Probability density function (PDF) ― The probability density function f is the probability that X takes on values between two adjacent realizations of the random variable.**

&#10230;Fungsi probabilitas densitas (PDF) - Fungsi probabilitas densitas f adalah probabilitas bahwa X memiliki nilai diantara dua variabel acak yang bernilai berdekatan.

<br>

**25. Relationships involving the PDF and CDF ― Here are the important properties to know in the discrete (D) and the continuous (C) cases.**

&#10230;Hubungan antara PDF dan CDF - Dibawah ini adalah properti penting yang harus diketahui pada kasus diskrit (D) dan kontinyu (C).

<br>

**26. [Case, CDF F, PDF f, Properties of PDF]**

&#10230;[Kasus, CDF F, PDF F, properti dari PDF]

<br>

**27. Expectation and Moments of the Distribution ― Here are the expressions of the expected value E[X], generalized expected value E[g(X)], kth moment E[Xk] and characteristic function ψ(ω) for the discrete and continuous cases:**

&#10230;Ekspektasi dan Momen dari Distribusi - Inilah ekspresi dari nilai yang diharapkan E[X], generalisasi dari nilai yang diharapkan E[g(x)], momen ke-k E[Xk] dan fungsi karakteristik ψ(ω) untuk kasus diskrit dan kontinyu.

<br>

**28. Variance ― The variance of a random variable, often noted Var(X) or σ2, is a measure of the spread of its distribution function. It is determined as follows:**

&#10230;Variansi - Variansi dari sebuah variabel acak, sering dituliskan sebagai Var(x) atau σ2, adalah sebuah ukuran penyebaran fungsi distrubusi. Variansi diformulasikan sebagai berikut:

<br>

**29. Standard deviation ― The standard deviation of a random variable, often noted σ, is a measure of the spread of its distribution function which is compatible with the units of the actual random variable. It is determined as follows:**

&#10230;Standar deviasi - Standar deviasi dari sebuah variabel acak, sering dinyatakan sebagai σ, adalah sebuah ukuran penyebaran dari fungsi distribusi yang sesuai dengan unit-unit dari variabel acak sesungguhnya. Standar deviasi diformulasikan sebagai berikut:

<br>

**30. Transformation of random variables ― Let the variables X and Y be linked by some function. By noting fX and fY the distribution function of X and Y respectively, we have:**

&#10230;Transformasi dari variabel acak - Diketahui bahwa variabel X dan Y dihubungkan oleh beberapa fungsi. Dengan mendefinisikan fX dan fY sebagai masing-masing fungsi distribusi dari X dan Y, kita memiliki

<br>

**31. Leibniz integral rule ― Let g be a function of x and potentially c, and a,b boundaries that may depend on c. We have:**

&#10230;Aturan Leibniz integral - Diketahui g sebagai sebuah fungsi x dan kemungkinan c, dan batasan a,b yang mungkin tergantung pada c. Kita memiliki:

<br>

**32. Probability Distributions**

&#10230;Distribusi probabilitas

<br>

**33. Chebyshev's inequality ― Let X be a random variable with expected value μ. For k,σ>0, we have the following inequality:**

&#10230;Ketidaksamaan Chebyshev - Misal X adalah sebuah variabel acak dengan nilai yang diharapkan μ. Untuk k,σ>0, kita memiliki ketidaksamaan sebagai berikut:

<br>

**34. Main distributions ― Here are the main distributions to have in mind:**

&#10230;Distribusi-distribusi yang utama - Berikut adalah distribusi-distribusi yang utama dan perlu diingat:

<br>

**35. [Type, Distribution]**

&#10230;[Type, Distribution]

<br>

**36. Jointly Distributed Random Variables**

&#10230;Variabel Acak yang Terdistribusi Bersamaan

<br>

**37. Marginal density and cumulative distribution ― From the joint density probability function fXY , we have**

&#10230;Densitas marginal dan distribusi kumulativ - Dari fungsi probabilitas join densitas fXY, kita mendapatkan

<br>

**38. [Case, Marginal density, Cumulative function]**

&#10230;[Kasus, Densitas marginal, Fungsi kumulativ]

<br>

**39. Conditional density ― The conditional density of X with respect to Y, often noted fX|Y, is defined as follows:**

&#10230;Densitas Kondisional - Densitas kondisional dari X terhadap Y, sering dituliskan sebagai fX|Y, didefinisikan sebagai berikut:

<br>

**40. Independence ― Two random variables X and Y are said to be independent if we have:**

&#10230;Keindependenan - Dua variabel X dan Y dikatakan independen jika kita memiliki

<br>

**41. Covariance ― We define the covariance of two random variables X and Y, that we note σ2XY or more commonly Cov(X,Y), as follows:**

&#10230;Kovarians adalah - Kita definsikan kovarians dari dua variabel acak X dan Y, yang kita tuliskan sebagai σ2XY atau lebih umumnya Cov(X,Y), sebagai berikut:

<br>

**42. Correlation ― By noting σX,σY the standard deviations of X and Y, we define the correlation between the random variables X and Y, noted ρXY, as follows:**

&#10230;Korelasi - Dengan menyatakan σX,σY sebagai standar deviasi dari X dan Y, kita mendefinisikan korelasi diantara variabel X dan Y, dituliskan ρXY, sebagai berikut:

<br>

**43. Remark 1: we note that for any random variables X,Y, we have ρXY∈[−1,1].**

&#10230;Poin penting 1: kita menyatakan bahwa baik untuk variabel acak X,Y, kita memiliki ρXY∈[−1,1].

<br>

**44. Remark 2: If X and Y are independent, then ρXY=0.**

&#10230;Poin penting 2: Jika X dan Y independen, maka ρXY=0.

<br>

**45. Parameter estimation**

&#10230;Estimasi parameter

<br>

**46. Definitions**

&#10230;Definisi-definisi

<br>

**47. Random sample ― A random sample is a collection of n random variables X1,...,Xn that are independent and identically distributed with X.**

&#10230;Sampel acak - sebuah sampel acak adalah koleksi dari n variabel acak X1,...,Xn yang independen dan terdistribusi secara identik dengan X.

<br>

**48. Estimator ― An estimator is a function of the data that is used to infer the value of an unknown parameter in a statistical model.**

&#10230;Estimator - Sebuah estimator adalah sebuah fungsi dari data yang digunakan untuk menduga nilai dari sebuah parameter yang tidak diketahui pada sebuah model statistik.

<br>

**49. Bias ― The bias of an estimator ^θ is defined as being the difference between the expected value of the distribution of ^θ and the true value, i.e.:**

&#10230;Bias - Bias dari sebuah estimator ^θ didefinisikan sebagai perbedaan antara distribusi dari nilai yang diharapkan  ^θ dan nilai yang sesungguhnya, sebagai contoh:

<br>

**50. Remark: an estimator is said to be unbiased when we have E[^θ]=θ.**

&#10230;Perlu diperhatikan: sebuah estimator dikatakan tidak bias ketika kita memiliki E[^θ]=θ.

<br>

**51. Estimating the mean**

&#10230;Mengestimasi nilai rata-rata

<br>

**52. Sample mean ― The sample mean of a random sample is used to estimate the true mean μ of a distribution, is often noted ¯¯¯¯¯X and is defined as follows:**

&#10230;Rata-rata dari sampel - Nilai rata-rata sample dari sebuah sampel acak digunakan untuk mengestimasi nilai rata-rata sesungguhnya μ dari sebuah distribusi, sering dinotasikan sebagai ¯¯¯¯¯X dan didefinsikan sebagai berikut:

<br>

**53. Remark: the sample mean is unbiased, i.e E[¯¯¯¯¯X]=μ.**

&#10230;Perlu diperhatikan: nilai rata-rata sampel adalah tidak bias, sebagai contoh  i.e E[¯¯¯¯¯X]=μ.

<br>

**54. Central Limit Theorem ― Let us have a random sample X1,...,Xn following a given distribution with mean μ and variance σ2, then we have:**

&#10230;Kaidah Central Limit - Diketahui sebuah sampel acak  X1,...,Xn yang berasal dari distribusi dengan nilai rata-rata μ dan nilai variansi σ2, maka kita memiliki:

<br>

**55. Estimating the variance**

&#10230;Estimasi nilai variansi

<br>

**56. Sample variance ― The sample variance of a random sample is used to estimate the true variance σ2 of a distribution, is often noted s2 or ^σ2 and is defined as follows:**

&#10230;Variansi sampel - Variansi sampel dari sebuah sampel acak digunakan untuk mengestimasi nilai variansi sesungguhnya σ2 dari sebuah distribusi, sering dituliskan sebagai s2 atau ^σ2 dan didefinisikan sebagai berikut:

<br>

**57. Remark: the sample variance is unbiased, i.e E[s2]=σ2.**

&#10230;Perlu diperhatikan: variansi sampel adalah tidak bias, sabagai contoh E[s2]=σ2.

<br>

**58. Chi-Squared relation with sample variance ― Let s2 be the sample variance of a random sample. We have:**

&#10230;Relasi Chi-Squared dengan variansi sampel - Diketahui s2 adalah variansi sampel dari sebuah sampel acak. Kita memiliki:

<br>

**59. [Introduction, Sample space, Event, Permutation]**

&#10230;[Pengenalan, Ruang sampel, Even, Permutasi]

<br>

**60. [Conditional probability, Bayes' rule, Independence]**

&#10230;[Probabilitas Kondisional, Bayes' rule, Independece]

<br>

**61. [Random variables, Definitions, Expectation, Variance]**

&#10230;[Variabel acak, definisi-definisi, Ekspektasi, Variansi]

<br>

**62. [Probability distributions, Chebyshev's inequality, Main distributions]**

&#10230;[Distribusi-distribusi probabilitas, Pertidaksamaan inequality, Distribusi-distribusi utama]

<br>

**63. [Jointly distributed random variables, Density, Covariance, Correlation]**

&#10230;[Variabel acak yang terdistribusi bersamaan, Densitas, Kovariansi, Korelasi]

<br>

**64. [Parameter estimation, Mean, Variance]**

&#10230;[Estimasi parameter, Rata-rata, Variansi]**
