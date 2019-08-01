**Variables-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models)

<br>

**1. Variables-based models with CSP and Bayesian networks**

&#10230; 1. CSP  ile değişken-temelli modeller ve Bayesçi ağlar

<br>


**2. Constraint satisfaction problems**

&#10230; 2. Kısıt memnuniyet problemleri

<br>


**3. In this section, our objective is to find maximum weight assignments of variable-based models. One advantage compared to states-based models is that these algorithms are more convenient to encode problem-specific constraints.**

&#10230; 3. Bu bölümde hedefimiz değişken-temelli modellerin maksimum ağırlık seçimlerini bulmaktır. Durum temelli modellerle kıyaslandığında, bu algoritmaların probleme özgü kısıtları kodlamak için daha uygun olmaları bir avantajdır.  

<br>


**4. Factor graphs**

&#10230; 4. Faktör grafikleri

<br>


**5. Definition ― A factor graph, also referred to as a Markov random field, is a set of variables X=(X1,...,Xn) where Xi∈Domaini and m factors f1,...,fm with each fj(X)⩾0.**

&#10230;5. Tanımlama - Markov rasgele alanı olarak da adlandırılan faktör grafiği, Xi∈Domaini ve herbir fj(X)⩾0 olan f1,...,fm m faktör olmak üzere X=(X1,...,Xn) değişkenler kümesidir.

<br>


**6. Domain**

&#10230; 6. Etki Alanı (Domain)

<br>


**7. Scope and arity ― The scope of a factor fj is the set of variables it depends on. The size of this set is called the arity.**

&#10230; 7. Kapsam ve ilişki derecesi - Fj faktörünün kapsamı, dayandığı değişken kümesidir. Bu kümenin boyutuna ilişki derecesi (arity) denir.

<br>


**8. Remark: factors of arity 1 and 2 are called unary and binary respectively.**

&#10230; 8. Not: Faktörlerin ilişki derecesi 1 ve 2 olanlarına sırasıyla tek ve ikili denir.

<br>


**9. Assignment weight ― Each assignment x=(x1,...,xn) yields a weight Weight(x) defined as being the product of all factors fj applied to that assignment. Its expression is given by:**

&#10230;9. Atama ağırlığı - Her atama x = (x1, ..., xn), o atamaya uygulanan tüm faktörlerin çarpımı olarak tanımlanan bir Ağırlık (x) ağırlığı verir.Şöyle ifade edilir:

<br> 


**10. Constraint satisfaction problem ― A constraint satisfaction problem (CSP) is a factor graph where all factors are binary; we call them to be constraints:**

&#10230; 10. Kısıt memnuniyet problemi - Kısıtlama memnuniyet problemi (constraint satisfaction problem-CSP), tüm faktörlerin ikili olduğu bir faktör grafiğidir; bunları kısıt olarak adlandırıyoruz:

<br>


**11. Here, the constraint j with assignment x is said to be satisfied if and only if fj(x)=1.**

&#10230;11.Burada, j kısıtlı x ataması ancak ve ancak fj(x)=1 olduğunda memnundur denir.

<br>


**12. Consistent assignment ― An assignment x of a CSP is said to be consistent if and only if Weight(x)=1, i.e. all constraints are satisfied.**

&#10230; 12.Tutarlı atama - Bir CSP'nin bir x atamasının, yalnızca Ağırlık (x) = 1 olduğunda, yani tüm kısıtların yerine getirilmesi durumunda tutarlı olduğu söylenir.

<br>


**13. Dynamic ordering**

&#10230; 13. Dinamik düzenleşim

<br>


**14. Dependent factors ― The set of dependent factors of variable Xi with partial assignment x is called D(x,Xi), and denotes the set of factors that link Xi to already assigned variables.**

&#10230;14.Bağımlı faktörler - X değişkeninin kısmi atamaya sahip bağımlı X değişken faktörlerinin kümesi D (x, Xi) ile gösterilir ve Xi'yi önceden atanmış değişkenlere bağlayan faktörler kümesini belirtir.

<br>


**15. Backtracking search ― Backtracking search is an algorithm used to find maximum weight assignments of a factor graph. At each step, it chooses an unassigned variable and explores its values by recursion. Dynamic ordering (i.e. choice of variables and values) and lookahead (i.e. early elimination of inconsistent options) can be used to explore the graph more efficiently, although the worst-case runtime stays exponential: O(|Domain|n).**

&#10230; 15. Geri izleme araması - Geri izleme araması, bir faktör grafiğinin maksimum ağırlık atamalarını bulmak için kullanılan bir algoritmadır. Her adımda, atanmamış bir değişken seçer ve değerlerini özyineleme ile arar. Dinamik düzenleşim (yani değişkenlerin ve değerlerin seçimi) ve bakış açısı (yani tutarsız seçeneklerin erken elenmesi), en kötü durum çalışma süresi üssel olarak olsa da grafiği daha verimli aramak için kullanılabilir. O (| Domain | n).

<br>


**16. [Forward checking ― It is a one-step lookahead heuristic that preemptively removes inconsistent values from the domains of neighboring variables. It has the following characteristics:, After assigning a variable Xi, it eliminates inconsistent values from the domains of all its neighbors., If any of these domains becomes empty, we stop the local backtracking search., If we un-assign a variable Xi, we have to restore the domain of its neighbors.]**

&#10230; 16. [İleri kontrol - Tutarsız değerleri komşu değişkenlerin etki alanlarından öncelikli bir şekilde ortadan kaldıran sezgisel bakış açısıdır. Aşağıdaki özelliklere sahiptir :, Bir Xi değişkenini atadıktan sonra, tüm komşularının etki alanlarından tutarsız değerleri eler., Bu etki alanlardan herhangi biri boş olursa, yerel geri arama araması durdurulur. , komşularının etki alanını eski haline getirilmek zorundadır.]

<br>


**17. Most constrained variable ― It is a variable-level ordering heuristic that selects the next unassigned variable that has the fewest consistent values. This has the effect of making inconsistent assignments to fail earlier in the search, which enables more efficient pruning.**

&#10230; 17. En kısıtlı değişken - En az tutarlı değere sahip bir sonraki atanmamış değişkeni seçen, değişken seviyeli sezgisel düzenleşimdir. Bu, daha verimli budama olanağı sağlayan aramada daha önce başarısız olmak için tutarsız atamalar yapma etkisine sahiptir.

<br>


**18. Least constrained value ― It is a value-level ordering heuristic that assigns the next value that yields the highest number of consistent values of neighboring variables. Intuitively, this procedure chooses first the values that are most likely to work.**

&#10230; 18. En düşük kısıtlı değer - Komşu değişkenlerin en yüksek tutarlı değerlerini elde ederek bir sonraki değeri veren değer seviyesi düzenleyici sezgisel bir değerdir. Sezgisel olarak, bu prosedür önce çalışması en muhtemel olan değerleri seçer.

<br>


**19. Remark: in practice, this heuristic is useful when all factors are constraints.**

&#10230; 19. Not: Uygulamada, bu sezgisel yaklaşım tüm faktörler kısıtlı olduğunda kullanışlıdır.

<br>


**20. The example above is an illustration of the 3-color problem with backtracking search coupled with most constrained variable exploration and least constrained value heuristic, as well as forward checking at each step.**

&#10230; 20. Yukarıdaki örnek, en kısıtlı değişken keşfi ve sezgisel en düşük kısıtlı değerin yanı sıra, her adımda ileri kontrol ile birleştirilmiş geri izleme arama ile 3 renkli problemin bir gösterimidir.

<br>


**21. [Arc consistency ― We say that arc consistency of variable Xl with respect to Xk is enforced when for each xl∈Domainl:, unary factors of Xl are non-zero, there exists at least one xk∈Domaink such that any factor between Xl and Xk is non-zero.]**

&#10230; 21. [Ark tutarlılığı - Xl değişkeninin ark tutarlılığının Xk'ye göre her bir xl∈Domainl için geçerli olduğu söylenir :, Xl'in birleşik faktörleri sıfır olmadığında, en az bir xk∈Domaink vardır, öyle ki Xl ve Xk arasında sıfır olmayan herhangi bir faktör vardır.

<br>


**22. AC-3 ― The AC-3 algorithm is a multi-step lookahead heuristic that applies forward checking to all relevant variables. After a given assignment, it performs forward checking and then successively enforces arc consistency with respect to the neighbors of variables for which the domain change during the process.**

&#10230; 22. AC-3 - AC-3 algoritması, tüm ilgili değişkenlere ileri kontrol uygulayan çok adımlı sezgisel bir bakış açısıdır. Belirli bir görevden sonra ileriye doğru kontrol yapar ve ardından işlem sırasında etki alanının değiştiği değişkenlerin komşularına göre ark tutarlılığını ardı ardına uygular.

<br>


**23. Remark: AC-3 can be implemented both iteratively and recursively.**

&#10230; 23. Not: AC-3, tekrarlı ve özyinelemeli olarak uygulanabilir.

<br>


**24. Approximate methods**

&#10230;24. Yaklaşık yöntemler

<br>


**25. Beam search ― Beam search is an approximate algorithm that extends partial assignments of n variables of branching factor b=|Domain| by exploring the K top paths at each step. The beam size K∈{1,...,bn} controls the tradeoff between efficiency and accuracy. This algorithm has a time complexity of O(n⋅Kblog(Kb)).**

&#10230; 25. Işın araması - Işın araması, her adımda K en üst yollarını keşfederek, b=|Domain| dallanma faktörünün n değişkeninin kısmi atamalarını genişleten yaklaşık bir algoritmadır.

<br>


**26. The example below illustrates a possible beam search of parameters K=2, b=3 and n=5.**

&#10230; 26. Aşağıdaki örnek, K = 2, b = 3 ve n = 5 parametreleri ile muhtemel kiriş aramasını (beam search) göstermektedir.

<br>


**27. Remark: K=1 corresponds to greedy search whereas K→+∞ is equivalent to BFS tree search.**

&#10230; 27. Not: K = 1 açgözlü aramaya (greedy search) karşılık gelirken K → + ∞, BFS ağaç aramasına eşdeğerdir.

<br>


**28. Iterated conditional modes ― Iterated conditional modes (ICM) is an iterative approximate algorithm that modifies the assignment of a factor graph one variable at a time until convergence. At step i, we assign to Xi the value v that maximizes the product of all factors connected to that variable.**

&#10230;28. Tekrarlanmış koşullu modlar - Tekrarlanmış koşullu modlar (Iterated conditional modes-ICM), yakınsamaya kadar bir seferde bir değişkenli bir faktör grafiğinin atanmasını değiştiren yinelemeli bir yaklaşık algoritmadır. İ adımında, Xi'ye, bu değişkene bağlı tüm faktörlerin çarpımını maksimize eden v değeri atanır.

<br>


**29. Remark: ICM may get stuck in local minima.**

&#10230; 29. Not: ICM yerel minimumda takılıp kalabilir.

<br>


**30. [Gibbs sampling ― Gibbs sampling is an iterative approximate method that modifies the assignment of a factor graph one variable at a time until convergence. At step i:, we assign to each element u∈Domaini a weight w(u) that is the product of all factors connected to that variable, we sample v from the probability distribution induced by w and assign it to Xi.]**

&#10230; 30. [Gibbs örneklemesi - Gibbs örneklemesi, yakınsamaya kadar bir seferde bir değişken grafik faktörünün atanmasını değiştiren yinelemeli bir yaklaşık yöntemdir. İ adımında, her bir u∈Domain olan öğeye , bu değişkene bağlı tüm faktörlerin çarpımı olan bir ağırlık w (u) atanır, v'yi w tarafından indüklenen olasılık dağılımından örnek alır ve Xi'ye atanır.]

<br>


**31. Remark: Gibbs sampling can be seen as the probabilistic counterpart of ICM. It has the advantage to be able to escape local minima in most cases.**

&#10230; 31. Not: Gibbs örneklemesi, ICM'nin olasılıksal karşılığı olarak görülebilir. Çoğu durumda yerel minimumlardan kaçabilme avantajına sahiptir.

<br>


**32. Factor graph transformations**

&#10230; 32. Faktör grafiği dönüşümleri

<br>


**33. Independence ― Let A,B be a partitioning of the variables X. We say that A and B are independent if there are no edges between A and B and we write:**

&#10230; 33. Bağımsızlık - A, B, X değişkenlerinin bir bölümü olsun. A ve B arasında kenar yoksa A ve B'nin bağımsız olduğu söylenir ve şöyle ifade edilir:

<br>


**34. Remark: independence is the key property that allows us to solve subproblems in parallel.**

&#10230; 34. Not: bağımsızlık, alt sorunları paralel olarak çözmemize olanak sağlayan bir kilit özelliktir.

<br>


**35. Conditional independence ― We say that A and B are conditionally independent given C if conditioning on C produces a graph in which A and B are independent. In this case, it is written:**

&#10230; 35. Koşullu bağımsızlık - Eğer C'nin şartlandırılması, A ve B'nin bağımsız olduğu bir grafik üretiyorsa A ve B verilen C koşulundan bağımsızdır. Bu durumda şöyle yazılır:

<br>


**36. [Conditioning ― Conditioning is a transformation aiming at making variables independent that breaks up a factor graph into smaller pieces that can be solved in parallel and can use backtracking. In order to condition on a variable Xi=v, we do as follows:, Consider all factors f1,...,fk that depend on Xi, Remove Xi and f1,...,fk, Add gj(x) for j∈{1,...,k} defined as:]**

&#10230; 36. [Koşullandırma - Koşullandırma, bir faktör grafiğini paralel olarak çözülebilen ve geriye doğru izlemeyi kullanabilen daha küçük parçalara bölen değişkenleri bağımsız kılmayı amaçlayan bir dönüşümdür. Xi = v değişkeninde koşullandırmak için aşağıdakileri yaparız: Xi'ye bağlı tüm f1, ..., fk faktörlerini göz önünde bulundurun, Xi ve f1, ..., fk öğelerini kaldırın, j∈ {1, ..., k} için gj (x) ekleyin:]

<br>


**37. Markov blanket ― Let A⊆X be a subset of variables. We define MarkovBlanket(A) to be the neighbors of A that are not in A.**

&#10230; 37. Markov blanket - A⊆X değişkenlerin bir alt kümesi olsun. MarkovBlanket'i (A), A'da olmayan A'nın komşuları olarak tanımlıyoruz.

<br>


**38. Proposition ― Let C=MarkovBlanket(A) and B=X∖(A∪C). Then we have:**

&#10230; Önerme - C = MarkovBlanket (A) ve B = X ∖ (A∪C) olsun.Bu durumda:

<br>


**39. [Elimination ― Elimination is a factor graph transformation that removes Xi from the graph and solves a small subproblem conditioned on its Markov blanket as follows:, Consider all factors fi,1,...,fi,k that depend on Xi, Remove Xi
and fi,1,...,fi,k, Add fnew,i(x) defined as:]**

&#10230; 39. [Eliminasyon - Eliminasyon, Xi'yi grafikten ayıran ve Markov blanket de şartlandırılmış küçük bir alt sorunu çözen bir faktör grafiği dönüşümüdür: Xi'ye bağlı tüm fi, 1, ..., fi, k faktörlerini göz önünde bulundurun, Xi ve fi, 1, ..., fi, k, kaldır, fnew ekleyin, i (x) şöyle tanımlanır:]

<br>


**40. Treewidth ― The treewidth of a factor graph is the maximum arity of any factor created by variable elimination with the best variable ordering. In other words,**

&#10230; 40. Ağaç genişliği - Bir faktör grafiğinin ağaç genişliği, değişken elemeli en iyi değişken sıralamasıyla oluşturulan herhangi bir faktörün maksimum ilişki derecesidir. Diğer bir deyişle,

<br>


**41. The example below illustrates the case of a factor graph of treewidth 3.**

&#10230; 41. Aşağıdaki örnek, ağaç genişliği 3 olan faktör grafiğini gösterir.

<br>


**42. Remark: finding the best variable ordering is a NP-hard problem.**

&#10230; 42. Not: en iyi değişken sıralamasını bulmak NP-zor (NP-hard) bir problemdir.

<br>


**43. Bayesian networks**

&#10230; 43. Bayesçi ağlar

<br>


**44. In this section, our goal will be to compute conditional probabilities. What is the probability of a query given evidence?**

&#10230;44. Bu bölümün amacı koşullu olasılıkları hesaplamak olacaktır. Bir sorgunun kanıt verilmiş olma olasılığı nedir?

<br>


**45. Introduction**

&#10230; 45. Giriş

<br>


**46. Explaining away ― Suppose causes C1 and C2 influence an effect E. Conditioning on the effect E and on one of the causes (say C1) changes the probability of the other cause (say C2). In this case, we say that C1 has explained away C2.**

&#10230; 47. Açıklamalar - C1 ve C2 sebeplerinin E etkisini yarattığını varsayalım. E etkisinin durumu ve sebeplerden biri (C1 olduğunu varsayalım) üzerindeki etkisi, diğer sebep olan C2'nin olasılığını değiştirir. Bu durumda, C1'in C2'yi açıkladığı söylenir.

<br>


**47. Directed acyclic graph ― A directed acyclic graph (DAG) is a finite directed graph with no directed cycles.**

&#10230;47. Yönlü çevrimsiz çizge - Yönlü çevrimsiz bir çizge (Directed acyclic graph-DAG), yönlendirilmiş çevrimleri olmayan sonlu bir yönlü çizgedir.

<br>


**48. Bayesian network ― A Bayesian network is a directed acyclic graph (DAG) that specifies a joint distribution over random variables X=(X1,...,Xn) as a product of local conditional distributions, one for each node:**

&#10230;48. Bayesçi ağ - Her düğüm için bir tane olmak üzere, yerel koşullu dağılımların bir çarpımı olarak, X = (X1, ..., Xn) rasgele değişkenleri üzerindeki bir ortak dağılımı belirten yönlü bir çevrimsiz çizgedir:

<br>


**49. Remark: Bayesian networks are factor graphs imbued with the language of probability.**

&#10230; 49. Not: Bayesçi ağlar olasılık diliyle bütünleşik faktör grafikleridir.

<br>


**50. Locally normalized ― For each xParents(i), all factors are local conditional distributions. Hence they have to satisfy:**

&#10230; 50. Yerel olarak normalleştirilmiş - Her xParents (i) için tüm faktörler yerel koşullu dağılımlardır. Bu nedenle yerine getirmek zorundalar:

<br>


**51. As a result, sub-Bayesian networks and conditional distributions are consistent.**

&#10230;51. Sonuç olarak, alt-Bayesçi ağlar ve koşullu dağılımlar tutarlıdır.

<br>


**52. Remark: local conditional distributions are the true conditional distributions.**

&#10230; 52. Not: Yerel koşullu dağılımlar gerçek koşullu dağılımlardır.

<br>


**53. Marginalization ― The marginalization of a leaf node yields a Bayesian network without that node.**

&#10230; 53. Marjinalleşme - Bir yaprak düğümünün marjinalleşmesi, o düğüm olmaksızın bir Bayesçi ağı sağlar.

<br>


**54. Probabilistic programs**

&#10230; 54. Olasılık programları

<br>


**55. Concept ― A probabilistic program randomizes variables assignment. That way, we can write down complex Bayesian networks that generate assignments without us having to explicitly specify associated probabilities.**

&#10230; 55. Konsept - Olasılıklı bir program değişkenlerin atanmasını randomize eder. Bu şekilde, ilişkili olasılıkları açıkça belirtmek zorunda kalmadan atamalar üreten karmaşık Bayesçi ağlar yazılabilir.

<br>


**56. Remark: examples of probabilistic programs include Hidden Markov model (HMM), factorial HMM, naive Bayes, latent Dirichlet allocation, diseases and symptoms and stochastic block models.**

&#10230; 56. Not: Olasılık programlarına örnekler arasında Gizli Markov modeli (Hidden Markov model-HMM), faktöriyel HMM, naif Bayes (naive Bayes), gizli Dirichlet tahsisi (latent Dirichlet allocation), hastalıklar ve semptomlar ve stokastik blok modelleri bulunmaktadır.

<br>


**57. Summary ― The table below summarizes the common probabilistic programs as well as their applications:**

&#10230; 57. Özet - Aşağıdaki tablo, ortak olasılıklı programları ve bunların uygulamalarını özetlemektedir:

<br>


**58. [Program, Algorithm, Illustration, Example]**

&#10230; 58. [Program, Algoritma, İllüstrasyon, Örnek]

<br>


**59. [Markov Model, Hidden Markov Model (HMM), Factorial HMM, Naive Bayes, Latent Dirichlet Allocation (LDA)]**

&#10230; 59. [Markov Modeli, Gizli Markov Modeli (HMM), Faktöriyel HMM, Naif Bayes, Gizli Dirichlet Tahsisi (Latent Dirichlet Allocation-LDA)]

<br>


**60. [Generate, distribution]**

&#10230; 60. [Üretim, Dağılım]

<br>


**61. [Language modeling, Object tracking, Multiple object tracking, Document classification, Topic modeling]**

&#10230; 61. [Dil modelleme, Nesne izleme, Çoklu nesne izleme, Belge sınıflandırma, Konu modelleme]

<br>


**62. Inference**

&#10230; 62. Çıkarım

<br>


**63. [General probabilistic inference strategy ― The strategy to compute the probability P(Q|E=e) of query Q given evidence E=e is as follows:, Step 1: Remove variables that are not ancestors of the query Q or the evidence E by marginalization, Step 2: Convert Bayesian network to factor graph, Step 3: Condition on the evidence E=e, Step 4: Remove nodes disconnected from the query Q by marginalization, Step 5: Run a probabilistic inference algorithm (manual, variable elimination, Gibbs sampling, particle filtering)]**

&#10230; 63. [Genel olasılıksal çıkarım stratejisi - E = e kanıtı verilen Q sorgusunun P (Q | E = e) olasılığını hesaplama stratejisi aşağıdaki gibidir :, Adım 1: Q sorgusunun ataları olmayan değişkenlerini ya da marjinalleştirme yoluyla E kanıtını silin, Adım 2: Bayesçi ağı faktör grafiğine dönüştürün, Adım 3: Kanıtın koşulu E = e, Adım 4: Q sorgusu ile bağlantısı kesilen düğümleri marjinalleştirme yoluyla silin, Adım 5: Olasılıklı bir çıkarım algoritması çalıştırın (kılavuz, değişken eleme, Gibbs örneklemesi, parçacık filtreleme)]

<br>


**64. Forward-backward algorithm ― This algorithm computes the exact value of P(H=hk|E=e) (smoothing query) for any k∈{1,...,L} in the case of an HMM of size L. To do so, we proceed in 3 steps:**

&#10230; 64. İleri-geri algoritma - Bu algoritma, L boyutunda bir HMM durumunda herhangi bir k∈ {1, ..., L} için P (H = hk | E = e) (düzeltme sorgusu) değerini hesaplar. Bunu yapmak için 3 adımda ilerlenir:

<br>


**65. Step 1: for ..., compute ...**

&#10230; 65. Adım 1: ... için (for), hesapla ...

<br>


**66. with the convention F0=BL+1=1. From this procedure and these notations, we get that**

&#10230; 66. F0 = BL + 1 = 1 kuralı ile. Bu prosedürden ve bu notasyonlardan anlıyoruz ki

<br>


**67. Remark: this algorithm interprets each assignment to be a path where each edge hi−1→hi is of weight p(hi|hi−1)p(ei|hi).**

&#10230; 67. Not: bu algoritma, her bir atamada her bir kenarın hi − 1 → hi'nin p (hi | hi − 1) p (ei | hi) olduğu bir yol olduğunu yorumlar.

<br>


**68. [Gibbs sampling ― This algorithm is an iterative approximate method that uses a small set of assignments (particles) to represent a large probability distribution. From a random assignment x, Gibbs sampling performs the following steps for i∈{1,...,n} until convergence:, For all u∈Domaini, compute the weight w(u) of assignment x where Xi=u, Sample v from the probability distribution induced by w: v∼P(Xi=v|X−i=x−i), Set Xi=v]**

&#10230; 68. [Gibbs örneklemesi - Bu algoritma, büyük olasılık dağılımını temsil etmek için küçük bir dizi atama (parçacık) kullanan tekrarlı bir yaklaşık yöntemdir. Rasgele bir x atamasından Gibbs örneklemesi, i∈ {1, ..., n} için yakınsamaya kadar aşağıdaki adımları uygular :, Tüm u∈Domaini için, x atamasının x (u) ağırlığını hesaplayın, burada Xi = u, Sample w: v∼P (Xi = v | X − i = x − i), Set Xi = v] ile indüklenen olasılık dağılımından

<br>


**69. Remark: X−i denotes X∖{Xi} and x−i represents the corresponding assignment.**

&#10230; 69. Not: X − i, X ∖ {Xi} ve x − i, karşılık gelen atamayı temsil eder.

<br>


**70. [Particle filtering ― This algorithm approximates the posterior density of state variables given the evidence of observation variables by keeping track of K particles at a time. Starting from a set of particles C of size K, we run the following 3 steps iteratively:, Step 1: proposal - For each old particle xt−1∈C, sample x from the transition probability distribution p(x|xt−1) and add x to a set C′., Step 2: weighting - Weigh each x of the set C′ by w(x)=p(et|x), where et is the evidence observed at time t., Step 3: resampling - Sample K elements from the set C′ using the probability distribution induced by w and store them in C: these are the current particles xt.]**

&#10230;70. [Parçacık filtreleme - Bu algoritma, bir seferde K parçacıklarını takip ederek gözlem değişkenlerinin kanıtı olarak verilen durum değişkenlerinin önceki yoğunluğuna yaklaşır.K boyutunda bir C parçacığı kümesinden başlayarak, aşağıdaki 3 adım tekrarlı olarak çalıştırılır: Adım 1: teklif - Her eski parçacık xt − 1∈C için, geçiş olasılığı dağılımından p (x | xt − 1) örnek x'i alın ve C ′ye ekleyin. Adım 2: ağırlıklandırma - C ′nin her x değerini w (x) = p (et | x) ile ağırlıklandırın, burada et t zamanında gözlemlenen kanıttır, Adım 3: yeniden örnekleme - w ile indüklenen olasılık dağılımını kullanarak C kümesinden örnek K elemanlarını C cinsinden saklayın: bunlar şuanki xt parçacıklarıdır.]

<br>


**71. Remark: a more expensive version of this algorithm also keeps track of past particles in the proposal step.**

&#10230; 71. Not: Bu algoritmanın daha pahalı bir versiyonu da teklif adımındaki geçmiş katılımcıların kaydını tutar.

<br>


**72. Maximum likelihood ― If we don't know the local conditional distributions, we can learn them using maximum likelihood.**

&#10230; 72. Maksimum olabilirlik - Yerel koşullu dağılımları bilmiyorsak, maksimum olasılık kullanarak bunları öğrenebiliriz.

<br>


**73. Laplace smoothing ― For each distribution d and partial assignment (xParents(i),xi), add λ to countd(xParents(i),xi), then normalize to get probability estimates.**

&#10230; 73. Laplace yumuşatma - Her d dağılımı ve (xParents (i), xi) kısmi ataması için, countd(xParents (i), xi)'a λ ekleyin, ardından olasılık tahminlerini almak için normalleştirin.

<br>


**74. Algorithm ― The Expectation-Maximization (EM) algorithm gives an efficient method at estimating the parameter θ through maximum likelihood estimation by repeatedly constructing a lower-bound on the likelihood (E-step) and optimizing that lower bound (M-step) as follows:**

&#10230; 74. Algoritma - Beklenti-Maksimizasyon (EM) algoritması, olasılığa art arda bir alt sınır oluşturarak (E-adım) tekrarlayarak ve bu alt sınırın (M-adımını) optimize ederek θ parametresini maksimum olasılık tahmini ile tahmin etmede aşağıdaki gibi etkin bir yöntem sunar :

<br>


**75. [E-step: Evaluate the posterior probability q(h) that each data point e came from a particular cluster h as follows:, M-step: Use the posterior probabilities q(h) as cluster specific weights on data points e to determine θ through maximum likelihood.]**

&#10230; 75. [E-adım: Her bir (e) veri noktasının belirli bir (h) kümesinden geldiği gerideki q (h) durumunu şu şekilde değerlendirin: M-adım: (maksimum olasılığını belirlemek için e veri noktalarındaki küme özgül ağırlıkları olarak gerideki olasılıklar q (h) kullanın.]

<br>


**76. [Factor graphs, Arity, Assignment weight, Constraint satisfaction problem, Consistent assignment]**

&#10230; 76. [Faktör grafikleri, İlişki Derecesi, Atama ağırlığı, Kısıt memnuniyet sorunu, Tutarlı atama]

<br>


**77. [Dynamic ordering, Dependent factors, Backtracking search, Forward checking, Most constrained variable, Least constrained value]**

&#10230; 77. [Dinamik düzenleşim, Bağımlı faktörler, Geri izleme araması, İleriye dönük kontrol, En kısıtlı değişken, En düşük kısıtlanmış değer]

<br>


**78. [Approximate methods, Beam search, Iterated conditional modes, Gibbs sampling]**

&#10230; 78. [Yaklaşık yöntemler, Işın arama , Tekrarlı koşullu modlar, Gibbs örneklemesi]

<br>


**79. [Factor graph transformations, Conditioning, Elimination]**

&#10230; 79. [Faktör grafiği dönüşümleri, Koşullandırma, Eleme]

<br>


**80. [Bayesian networks, Definition, Locally normalized, Marginalization]**

&#10230; 80. [Bayesçi ağlar, Tanım, Yerel normalleştirme, Marjinalleşme]

<br>


**81. [Probabilistic program, Concept, Summary]**

&#10230; 81. [Olasılık programı, Kavram, Özet]

<br>


**82. [Inference, Forward-backward algorithm, Gibbs sampling, Laplace smoothing]**

&#10230; 82. [Çıkarım, İleri-geri algoritması, Gibbs örneklemesi, Laplace yumuşatması]

<br>


**83. View PDF version on GitHub**

&#10230; 83. GitHub'da PDF versiyonun görüntüleyin

<br>


**84. Original authors**

&#10230; 84. Orijinal yazarlar

<br>


**85. Translated by X, Y and Z**

&#10230; 85. X, Y ve Z tarafından çevrilmiştir.

<br>


**86. Reviewed by X, Y and Z**

&#10230; 86. X,Y,Z tarafından kontrol edilmiştir.

<br>


**87. By X and Y**

&#10230; 87. X ve Y ile

<br>


**88. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230;88. Yapay Zeka el kitapları artık [hedef dilde] mevcuttur.
