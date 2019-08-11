**States-based models translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-states-models)

<br>

**1. States-based models with search optimization and MDP**

&#10230; Arama optimizasyonu ve Markov karar sürecine (MDP) sahip durum-temelli modeller

<br>


**2. Search optimization**

&#10230;  Arama optimizasyonu

<br>


**3. In this section, we assume that by accomplishing action a from state s, we deterministically arrive in state Succ(s,a). The goal here is to determine a sequence of actions (a1,a2,a3,a4,...) that starts from an initial state and leads to an end state. In order to solve this kind of problem, our objective will be to find the minimum cost path by using states-based models.**

&#10230; Bu bölümde, s durumunda a eylemini gerçekleştirdiğimizde, Succ(s,a) durumuna varacağımızı varsayıyoruz. Burada amaç, başlangıç durumundan başlayıp bitiş durumuna götüren bir eylem dizisi (a1,a2,a3,a4,...) belirlenmesidir. Bu tür bir problemi çözmek için, amacımız durum-temelli modelleri kullanarak asgari (minimum) maliyet yolunu bulmak olacaktır.

<br>


**4. Tree search**

&#10230; Ağaç arama

<br>


**5. This category of states-based algorithms explores all possible states and actions. It is quite memory efficient, and is suitable for huge state spaces but the runtime can become exponential in the worst cases.**

&#10230; Bu durum-temelli algoritmalar, olası bütün durum ve eylemleri araştırırlar. Oldukça bellek verimli ve büyük durum uzayları için uygundurlar ancak çalışma zamanı en kötü durumlarda üstel olabilir.

<br>


**6. [Self-loop, More than a parent, Cycle, More than a root, Valid tree]**

&#10230; [Kendinden-Döngü(Self-loop), Bir ebeveynden (parent) daha fazlası, Çevrim, Bir kökten daha fazlası, Geçerli ağaç]

<br>


**7. [Search problem ― A search problem is defined with:, a starting state sstart, possible actions Actions(s) from state s, action cost Cost(s,a) from state s with action a, successor Succ(s,a) of state s after action a, whether an end state was reached IsEnd(s)]**

&#10230; [Arama problemi ― Bir arama problemi aşağıdaki şekilde tanımlanmaktadır:, bir başlangıç durumu sstart, s durumunda gerçekleşebilecek olası eylemler Actions(s), s durumunda gerçekleşen a eyleminin eylem maliyeti Cost(s,a), a eyleminden sonraki varılacak durum Succ(s,a), son duruma ulaşılıp ulaşılamadığı IsEnd(s)]

<br>


**8. The objective is to find a path that minimizes the cost.**

&#10230; Amaç, maliyeti en aza indiren bir yol bulmaktır.

<br>


**9. Backtracking search ― Backtracking search is a naive recursive algorithm that tries all possibilities to find the minimum cost path. Here, action costs can be either positive or negative.**

&#10230; Geri izleme araması ― Geri izleme araması, asgari (minimum) maliyet yolunu bulmak için tüm olasılıkları deneyen saf (naive) bir özyinelemeli algoritmadır. Burada, eylem maliyetleri pozitif ya da negatif olabilir.

<br>


**10. Breadth-first search (BFS) ― Breadth-first search is a graph search algorithm that does a level-by-level traversal. We can implement it iteratively with the help of a queue that stores at each step future nodes to be visited. For this algorithm, we can assume action costs to be equal to a constant c⩾0.**

&#10230; Genişlik öncelikli arama (Breadth-first search-BFS) ― Genişlik öncelikli arama, seviye seviye arama yapan bir çizge arama algoritmasıdır. Gelecekte her adımda ziyaret edilecek düğümleri tutan bir kuyruk yardımıyla yinelemeli olarak gerçekleyebiliriz. Bu algoritma için, eylem maliyetlerinin belirli bir sabite c⩾0 eşit olduğunu kabul edebiliriz.

<br>


**11. Depth-first search (DFS) ― Depth-first search is a search algorithm that traverses a graph by following each path as deep as it can. We can implement it recursively, or iteratively with the help of a stack that stores at each step future nodes to be visited. For this algorithm, action costs are assumed to be equal to 0.**

&#10230; Derinlik öncelikli arama (Depth-first search-DFS) ― Derinlik öncelikli arama, her bir yolu olabildiğince derin bir şekilde takip ederek çizgeyi dolaşan bir arama algoritmasıdır. Bu algoritmayı, ziyaret edilecek gelecek düğümleri her adımda bir yığın yardımıyla saklayarak, yinelemeli (recursively) ya da tekrarlı (iteratively) olarak uygulayabiliriz. Bu algoritma için eylem maliyetlerinin 0 olduğu varsayılmaktadır.

<br>


**12. Iterative deepening ― The iterative deepening trick is a modification of the depth-first search algorithm so that it stops after reaching a certain depth, which guarantees optimality when all action costs are equal. Here, we assume that action costs are equal to a constant c⩾0.**

&#10230; Tekrarlı derinleşme ― Tekrarlı derinleşme hilesi, derinlik-ilk arama algoritmasının değiştirilmiş bir halidir, böylece belirli bir derinliğe ulaştıktan sonra durur, bu da tüm işlem maliyetleri eşit olduğunda en iyiliği (optimal) garanti eder. Burada, işlem maliyetlerinin c⩾0 gibi sabit bir değere eşit olduğunu varsayıyoruz.

<br>


**13. Tree search algorithms summary ― By noting b the number of actions per state, d the solution depth, and D the maximum depth, we have:**

&#10230; Ağaç arama algoritmaları özeti ― B durum başına eylem sayısını, d çözüm derinliğini ve D en yüksek (maksimum) derinliği ifade ederse, o zaman:

<br>


**14. [Algorithm, Action costs, Space, Time]**

&#10230; [Algoritma, Eylem maliyetleri, Arama uzayı, Zaman]

<br>


**15. [Backtracking search, any, Breadth-first search, Depth-first search, DFS-Iterative deepening]**

&#10230; [Geri izleme araması, herhangi bir şey, Genişlik öncelikli arama, Derinlik öncelikli arama, DFS - Tekrarlı derinleşme]

<br>


**16. Graph search**

&#10230; Çizge arama

<br>


**17. This category of states-based algorithms aims at constructing optimal paths, enabling exponential savings. In this section, we will focus on dynamic programming and uniform cost search.**

&#10230; Bu durum-temelli algoritmalar kategorisi, üssel tasarruf sağlayan en iyi (optimal) yolları oluşturmayı amaçlar. Bu bölümde, dinamik programlama ve tek tip maliyet araştırması üzerinde duracağız.

<br>


**18. Graph ― A graph is comprised of a set of vertices V (also called nodes) as well as a set of edges E (also called links).**

&#10230; Çizge ― Bir çizge, V köşeler (düğüm olarak da adlandırılır) kümesi ile E kenarlar (bağlantı olarak da adlandırılır) kümesinden oluşur.

<br>


**19. Remark: a graph is said to be acylic when there is no cycle.**

&#10230; Not: çevrim olmadığında, bir çizgenin asiklik (çevrimsiz) olduğu söylenir.

<br>


**20. State ― A state is a summary of all past actions sufficient to choose future actions optimally.**

&#10230; Durum ― Bir durum gelecekteki eylemleri en iyi (optimal) şekilde seçmek için, yeterli tüm geçmiş eylemlerin özetidir.

<br>


**21. Dynamic programming ― Dynamic programming (DP) is a backtracking search algorithm with memoization (i.e. partial results are saved) whose goal is to find a minimum cost path from state s to an end state send. It can potentially have exponential savings compared to traditional graph search algorithms, and has the property to only work for acyclic graphs. For any given state s, the future cost is computed as follows:**

&#10230; Dinamik programlama ― Dinamik programlama (DP), amacı s durumundan bitiş durumu olan send'e kadar asgari(minimum) maliyet yolunu bulmak olan hatırlamalı (memoization) (başka bir deyişle kısmi sonuçlar kaydedilir) bir geri izleme (backtracking) arama algoritmasıdır. Geleneksel çizge arama algoritmalarına kıyasla üstel olarak tasarruf sağlayabilir ve yalnızca asiklik (çevrimsiz) çizgeler ile çalışma özelliğine sahiptir. Herhangi bir durum için gelecekteki maliyet aşağıdaki gibi hesaplanır:

<br>


**22. [if, otherwise]**

&#10230; [eğer, aksi taktirde]

<br>


**23. Remark: the figure above illustrates a bottom-to-top approach whereas the formula provides the intuition of a top-to-bottom problem resolution.**

&#10230; Not: Yukarıdaki şekil, aşağıdan yukarıya bir yaklaşımı sergilerken, formül ise yukarıdan aşağıya bir önsezi ile problem çözümü sağlar.

<br>


**24. Types of states ― The table below presents the terminology when it comes to states in the context of uniform cost search:**

&#10230; Durum türleri ― Tek tip maliyet araştırması bağlamındaki durumlara ilişkin terminoloji aşağıdaki tabloda sunulmaktadır:

<br>


**25. [State, Explanation]**

&#10230; [Durum, Açıklama]

<br>


**26. [Explored, Frontier, Unexplored]**

&#10230; [Keşfedilmiş, Sırada (Frontier), Keşfedilmemiş]

<br>


**27. [States for which the optimal path has already been found, States seen for which we are still figuring out how to get there with the cheapest cost, States not seen yet]**

&#10230; [En iyi (optimal) yolun daha önce bulunduğu durumlar, Görülen ancak hala en ucuza nasıl gidileceği hesaplanmaya çalışılan durumlar, Daha önce görülmeyen durumlar]

<br>


**28. Uniform cost search ― Uniform cost search (UCS) is a search algorithm that aims at finding the shortest path from a state sstart to an end state send. It explores states s in increasing order of PastCost(s) and relies on the fact that all action costs are non-negative.**

&#10230; Tek tip maliyet araması ― Tek tip maliyet araması (Uniform cost search - UCS) bir başlangıç durumu olan Sstart, ile bir bitiş durumu olan Send arasındaki en kısa yolu bulmayı amaçlayan bir arama algoritmasıdır. Bu algoritma s durumlarını artan geçmiş maliyetleri olan PastCost(s)'a göre araştırır ve eylem maliyetlerinin negatif olmayacağı kuralına dayanır.

<br>


**29. Remark 1: the UCS algorithm is logically equivalent to Djikstra's algorithm.**

&#10230; Not 1: UCS algoritması mantıksal olarak Djikstra algoritması ile aynıdır.

<br>


**30. Remark 2: the algorithm would not work for a problem with negative action costs, and adding a positive constant to make them non-negative would not solve the problem since this would end up being a different problem.**

&#10230; Not 2: Algoritma, negatif eylem maliyetleriyle ilgili bir problem için çalışmaz ve negatif olmayan bir hale getirmek için pozitif bir sabit eklemek problemi çözmez, çünkü problem farklı bir problem haline gelmiş olur.

<br>


**31. Correctness theorem ― When a state s is popped from the frontier F and moved to explored set E, its priority is equal to PastCost(s) which is the minimum cost path from sstart to s.**

&#10230; Doğruluk teoremi ― S durumu sıradaki (frontier) F'den çıkarılır ve daha önceden keşfedilmiş olan E kümesine taşınırsa, önceliği başlangıç durumu olan Sstart'dan, s durumuna kadar asgari (minimum) maliyet yolu olan PastCost(s)'e eşittir.

<br>


**32. Graph search algorithms summary ― By noting N the number of total states, n of which are explored before the end state send, we have:**

&#10230; Çizge arama algoritmaları özeti ― N toplam durumların sayısı, n-bitiş durumu(Send)'ndan önce keşfedilen durum sayısı ise:

<br>


**33. [Algorithm, Acyclicity, Costs, Time/space]**

&#10230; [Algoritma, Asiklik (Çevrimsizlik), Maliyetler, Zaman/arama uzayı]

<br>


**34. [Dynamic programming, Uniform cost search]**

&#10230; [Dinamik programlama, Tek tip maliyet araması]

<br>


**35. Remark: the complexity countdown supposes the number of possible actions per state to be constant.**

&#10230; Not: Karmaşıklık geri sayımı, her durum için olası eylemlerin sayısını sabit olarak kabul eder.

<br>


**36. Learning costs**

&#10230; Öğrenme maliyetleri

<br>


**37. Suppose we are not given the values of Cost(s,a), we want to estimate these quantities from a training set of minimizing-cost-path sequence of actions (a1,a2,...,ak).**

&#10230; Diyelim ki, Cost(s,a) değerleri verilmedi ve biz bu değerleri maliyet yolu eylem dizisini,(a1,a2,...,ak), en aza indiren bir eğitim kümesinden tahmin etmek istiyoruz.

<br>


**38. [Structured perceptron ― The structured perceptron is an algorithm aiming at iteratively learning the cost of each state-action pair. At each step, it:, decreases the estimated cost of each state-action of the true minimizing path y given by the training data, increases the estimated cost of each state-action of the current predicted path y' inferred from the learned weights.]**

&#10230; [Yapılandırılmış algılayıcı ― Yapılandırılmış algılayıcı, her bir durum-eylem çiftinin maliyetini tekrarlı (iteratively) olarak öğrenmeyi amaçlayan bir algoritmadır. Her bir adımda, algılayıcı:, eğitim verilerinden elde edilen gerçek asgari (minimum) y yolunun her bir durum-eylem çiftinin tahmini (estimated) maliyetini azaltır, öğrenilen ağırlıklardan elde edilen şimdiki tahmini(predicted) y' yolununun durum-eylem çiftlerinin tahmini maliyetini artırır.]

<br>


**39. Remark: there are several versions of the algorithm, one of which simplifies the problem to only learning the cost of each action a, and the other parametrizes Cost(s,a) to a feature vector of learnable weights.**

&#10230; Not: Algoritmanın birkaç sürümü vardır, bunlardan biri problemi sadece her bir a eyleminin maliyetini öğrenmeye indirger, bir diğeri ise öğrenilebilir ağırlık öznitelik vektörünü, Cost(s,a)'nın parametresi haline getirir.

<br>


**40. A* search**

&#10230; A* arama

<br>


**41. Heuristic function ― A heuristic is a function h over states s, where each h(s) aims at estimating FutureCost(s), the cost of the path from s to send.**

&#10230; Sezgisel işlev(Heuristic function) ― Sezgisel, s durumu üzerinde işlem yapan bir h fonksiyonudur, burada her bir h(s), s ile send arasındaki yol maliyeti olan FutureCost(s)'yi tahmin etmeyi amaçlar.

<br>


**42. Algorithm ― A∗ is a search algorithm that aims at finding the shortest path from a state s to an end state send. It explores states s in increasing order of PastCost(s)+h(s). It is equivalent to a uniform cost search with edge costs Cost′(s,a) given by:**

&#10230; Algoritma ― A∗, s durumu ile send bitiş durumu arasındaki en kısa yolu bulmayı amaçlayan bir arama algoritmasıdır. Bahse konu algoritma PastCost(s)+h(s)'yi artan sıra ile araştırır. Aşağıda verilenler ışığında kenar maliyetlerini de içeren tek tip maliyet aramasına eşittir:

<br>


**43. Remark: this algorithm can be seen as a biased version of UCS exploring states estimated to be closer to the end state.**

&#10230; Not: Bu algoritma, son duruma yakın olduğu tahmin edilen durumları araştıran tek tip maliyet aramasının taraflı bir sürümü olarak görülebilir.

<br>


**44. [Consistency ― A heuristic h is said to be consistent if it satisfies the two following properties:, For all states s and actions a, The end state verifies the following:]**

&#10230; [Tutarlılık ― Bir sezgisel h, aşağıdaki iki özelliği sağlaması durumunda tutarlıdır denilebilir:, Bütün s durumları ve a eylemleri için, bitiş durumu aşağıdakileri doğrular:]

<br>


**45. Correctness ― If h is consistent, then A∗ returns the minimum cost path.**

&#10230; Doğruluk ― Eğer h tutarlı ise o zaman A∗ algoritması asgari (minimum) maliyet yolunu döndürür.

<br>


**46. Admissibility ― A heuristic h is said to be admissible if we have:**

&#10230; Kabul edilebilirlik ― Bir sezgisel h kabul edilebilirdir eğer:

<br>


**47. Theorem ― Let h(s) be a given heuristic. We have:**

&#10230; Teorem ― h(s) sezgisel olsun ve:

<br>


**48. [consistent, admissible]**

&#10230; [tutarlı, kabul edilebilir]

<br>


**49. Efficiency ― A* explores all states s satisfying the following equation:**

&#10230; Verimlilik ― A* algoritması aşağıdaki eşitliği sağlayan bütün s durumlarını araştırır:

<br>


**50. Remark: larger values of h(s) is better as this equation shows it will restrict the set of states s going to be explored.**

&#10230; Not: h(s)'nin yüksek değerleri, bu eşitliğin araştırılacak olan s durum kümesini kısıtlayacak olması nedeniyle daha iyidir.

<br>


**51. Relaxation**

&#10230; Rahatlama

<br>


**52. It is a framework for producing consistent heuristics. The idea is to find closed-form reduced costs by removing constraints and use them as heuristics.**

&#10230; Bu tutarlı sezgisel için bir altyapıdır (framework). Buradaki fikir, kısıtlamaları kaldırarak kapalı şekilli (closed-form) düşük maliyetler bulmak ve bunları sezgisel olarak kullanmaktır.

<br>


**53. Relaxed search problem ― The relaxation of search problem P with costs Cost is noted Prel with costs Costrel, and satisfies the identity:**

&#10230; Rahat arama problemi (Relaxed search problem) ― Cost maliyetli bir arama probleminin rahatlaması, Costrel maliyetli Prel ile ifade edilir ve kimliği karşılar (satisfies the identity) :

<br>


**54. Relaxed heuristic ― Given a relaxed search problem Prel, we define the relaxed heuristic h(s)=FutureCostrel(s) as the minimum cost path from s to an end state in the graph of costs Costrel(s,a).**

&#10230; Rahat sezgisel (Relaxed heuristic) ― Bir Prel rahat arama problemi verildiğinde, h(s)=FutureCostrel(s) rahat sezgisel eşitliğini Costrel(s,a) maliyet çizgesindeki s durumu ile bir bitiş durumu arasındaki asgari(minimum) maliyet yolu olarak tanımlarız.

<br>


**55. Consistency of relaxed heuristics ― Let Prel be a given relaxed problem. By theorem, we have:**

&#10230; Rahat sezgisel tutarlılığı ― Prel bir rahat problem olarak verilmiş olsun. Teoreme göre:

<br>


**56. consistent**

&#10230; tutarlı

<br>


**57. [Tradeoff when choosing heuristic ― We have to balance two aspects in choosing a heuristic:, Computational efficiency: h(s)=FutureCostrel(s) must be easy to compute. It has to produce a closed form, easier search and independent subproblems., Good enough approximation: the heuristic h(s) should be close to FutureCost(s) and we have thus to not remove too many constraints.]**

&#10230; [Sezgisel seçiminde ödünleşim (tradeoff) ― Sezgisel seçiminde iki yönü dengelemeliyiz:, Hesaplamalı verimlilik: h(s)=FutureCostrel(s) eşitliği kolay hesaplanabilir olmalıdır. Kapalı bir şekil, daha kolay arama ve bağımsız alt problemler üretmesi gerekir., Yeterince iyi yaklaşım: sezgisel h(s), FutureCost(s) işlevine yakın olmalı ve bu nedenle çok fazla kısıtlamayı ortadan kaldırmamalıyız.]

<br>


**58. Max heuristic ― Let h1(s), h2(s) be two heuristics. We have the following property:**

&#10230; En yüksek sezgisel ― h1(s) ve h2(s) aşağıdaki özelliklere sahip iki adet sezgisel olsun:

<br>


**59. Markov decision processes**

&#10230; Markov karar süreçleri

<br>


**60. In this section, we assume that performing action a from state s can lead to several states s′1,s′2,... in a probabilistic manner. In order to find our way between an initial state and an end state, our objective will be to find the maximum value policy by using Markov decision processes that help us cope with randomness and uncertainty.**

&#10230; Bu bölümde, s durumunda a eyleminin gerçekleştirilmesinin olasılıksal olarak birden fazla durum,(s′1,s′2,...), ile sonuçlanacağını kabul ediyoruz. Başlangıç durumu ile bitiş durumu arasındaki yolu bulmak için amacımız, rastgelelilik ve belirsizlik ile başa çıkabilmek için yardımcı olan Markov karar süreçlerini kullanarak en yüksek değer politikasını bulmak olacaktır.

<br>


**61. Notations**

&#10230; Gösterimler

<br>


**62. [Definition ― The objective of a Markov decision process is to maximize rewards. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, transition probabilities T(s,a,s′) from s to s′ with action a, rewards Reward(s,a,s′) from s to s′ with action a, whether an end state was reached IsEnd(s), a discount factor 0⩽γ⩽1]**

&#10230; [Tanım ― Markov karar sürecinin amacı ödülleri en yüksek seviyeye çıkarmaktır. Markov karar süreci aşağıdaki bileşenlerden oluşmaktadır:, başlangıç durumu sstart, s durumunda gerçekleştirilebilecek olası eylemler Actions(s), s durumunda a eyleminin gerçekleştirilmesi ile s′ durumuna geçiş olasılıkları T(s,a,s′), s durumunda a eyleminin gerçekleştirilmesi ile elde edilen ödüller Reward(s,a,s′), bitiş durumuna ulaşılıp ulaşılamadığı IsEnd(s), indirim faktörü 0⩽γ⩽1]

<br>


**63. Transition probabilities ― The transition probability T(s,a,s′) specifies the probability of going to state s′ after action a is taken in state s. Each s′↦T(s,a,s′) is a probability distribution, which means that:**

&#10230; Geçiş olasılıkları ― Geçiş olasılığı T(s,a,s′) s durumundayken gerçekleştirilen a eylemi neticesinde s′ durumuna gitme olasılığını belirtir. Her bir s′↦T(s,a,s′) aşağıda belirtildiği gibi bir olasılık dağılımıdır:

<br>


**64. states**

&#10230; durumlar

<br>


**65. Policy ― A policy π is a function that maps each state s to an action a, i.e.**

&#10230; Politika ― Bir π politikası her s durumunu bir a eylemi ile ilişkilendiren bir işlevdir.

<br>


**66. Utility ― The utility of a path (s0,...,sk) is the discounted sum of the rewards on that path. In other words,**

&#10230; Fayda ― Bir (s0,...,sk) yolunun faydası, o yol üzerindeki ödüllerin indirimli toplamıdır. Diğer bir deyişle,

<br>


**67. The figure above is an illustration of the case k=4.**

&#10230; Yukarıdaki şekil k=4 durumunun bir gösterimidir.

<br>


**68. Q-value ― The Q-value of a policy π at state s with action a, also noted Qπ(s,a), is the expected utility from state s after taking action a and then following policy π. It is defined as follows:**

&#10230; Q-değeri ― S durumunda gerçekleştirilen bir a eylemi için π politikasının Q-değeri, Qπ(s,a) olarak da gösterilir, a eylemini gerçekleştirip ve sonrasında π politikasını takiben s durumundan beklenen faydadır. Q-değeri aşağıdaki şekilde tanımlanmaktadır:

<br>


**69. Value of a policy ― The value of a policy π from state s, also noted Vπ(s), is the expected utility by following policy π from state s over random paths. It is defined as follows:**

&#10230; Bir politikanın değeri ― S durumundaki π politikasının değeri,Vπ(s) olarak da gösterilir, rastgele yollar üzerinde s durumundaki π politikasını izleyerek elde edilen beklenen faydadır. S durumundaki π politikasının değeri aşağıdaki gibi tanımlanır:

<br>


**70. Remark: Vπ(s) is equal to 0 if s is an end state.**

&#10230; Not: Eğer s bitiş durumu ise Vπ(s) sıfıra eşittir.

<br>


**71. Applications**

&#10230; Uygulamalar

<br>


**72. [Policy evaluation ― Given a policy π, policy evaluation is an iterative algorithm that aims at estimating Vπ. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TPE, we have, with]**

&#10230; [Politika değerlendirme ― bir π politikası verildiğinde, politika değerlendirmesini,Vπ, tahmin etmeyi amaçlayan bir tekrarlı (iterative) algoritmadır. Politika değerlendirme aşağıdaki gibi yapılmaktadır:, İlklendirme: bütün s durumları için:, Tekrar: 1'den TPE'ye kadar her t için, ile]

<br>


**73. Remark: by noting S the number of states, A the number of actions per state, S′ the number of successors and T the number of iterations, then the time complexity is of O(TPESS′).**

&#10230; Not: S durum sayısını, A her bir durum için eylem sayısını, S′ ardılların (successors) sayısını ve T yineleme sayısını gösterdiğinde, zaman karmaşıklığı O(TPESS′) olur.

<br>


**74. Optimal Q-value ― The optimal Q-value Qopt(s,a) of state s with action a is defined to be the maximum Q-value attained by any policy starting. It is computed as follows:**

&#10230; En iyi Q-değeri ― S durumunda a eylemi gerçekleştirildiğinde bu durumun en iyi Q-değeri,Qopt(s,a), herhangi bir politika başlangıcında elde edilen en yüksek Q-değeri olarak tanımlanmaktadır. En iyi Q-değeri aşağıdaki gibi hesaplanmaktadır:

<br>


**75. Optimal value ― The optimal value Vopt(s) of state s is defined as being the maximum value attained by any policy. It is computed as follows:**

&#10230; En iyi değer ― S durumunun en iyi değeri olan Vopt(s), herhangi bir politika ile elde edilen en yüksek değer olarak tanımlanmaktadır. En iyi değer aşağıdaki gibi hesaplanmaktadır:

<br>


**76. actions**

&#10230; eylemler

<br>


**77. Optimal policy ― The optimal policy πopt is defined as being the policy that leads to the optimal values. It is defined by:**

&#10230; En iyi politika ― En iyi politika olan πopt, en iyi değerlere götüren politika olarak tanımlanmaktadır. En iyi politika aşağıdaki gibi tanımlanmaktadır:

<br>


**78. [Value iteration ― Value iteration is an algorithm that finds the optimal value Vopt as well as the optimal policy πopt. It is done as follows:, Initialization: for all states s, we have:, Iteration: for t from 1 to TVI, we have:, with]**

&#10230; [Değer tekrarı(iteration) ― Değer tekrarı(iteration) en iyi politika olan πopt, yanında en iyi değeri Vopt'ı, bulan bir algoritmadır. Değer tekrarı(iteration) aşağıdaki gibi yapılmaktadır:, İlklendirme: bütün s durumları için:, Tekrar: 1'den TVI'ya kadar her bir t için:, ile]

<br>


**79. Remark: if we have either γ<1 or the MDP graph being acyclic, then the value iteration algorithm is guaranteed to converge to the correct answer.**

&#10230; Not: Eğer γ<1 ya da Markov karar süreci (Markov Decision Process - MDP) asiklik (çevrimsiz) olursa, o zaman değer tekrarı algoritmasının doğru cevaba yakınsayacağı garanti edilir.

<br>


**80. When unknown transitions and rewards**

&#10230; Bilinmeyen geçişler ve ödüller

<br>


**81. Now, let's assume that the transition probabilities and the rewards are unknown.**

&#10230; Şimdi, geçiş olasılıklarının ve ödüllerin bilinmediğini varsayalım.

<br>


**82. Model-based Monte Carlo ― The model-based Monte Carlo method aims at estimating T(s,a,s′) and Reward(s,a,s′) using Monte Carlo simulation with: **

&#10230; Model-temelli Monte Carlo ― Model-temelli Monte Carlo yöntemi, T(s,a,s′) ve Reward(s,a,s′) işlevlerini Monte Carlo benzetimi kullanarak aşağıdaki formüllere uygun bir şekilde tahmin etmeyi amaçlar:

<br>


**83. [# times (s,a,s′) occurs, and]**

&#10230; [# kere (s,a,s′) gerçekleşme sayısı, ve]

<br>


**84. These estimations will be then used to deduce Q-values, including Qπ and Qopt.**

&#10230; Bu tahminler daha sonra Qπ ve Qopt'yi içeren Q-değerleri çıkarımı için kullanılacaktır.

<br>


**85. Remark: model-based Monte Carlo is said to be off-policy, because the estimation does not depend on the exact policy.**

&#10230; Not: model-tabanlı Monte Carlo'nun politika dışı olduğu söyleniyor, çünkü tahmin kesin politikaya bağlı değildir.

<br>


**86. Model-free Monte Carlo ― The model-free Monte Carlo method aims at directly estimating Qπ, as follows:**

&#10230; Model içermeyen Monte Carlo ― Model içermeyen Monte Carlo yöntemi aşağıdaki şekilde doğrudan Qπ'yi tahmin etmeyi amaçlar:

<br>


**87. Qπ(s,a)=average of ut where st−1=s,at=a**

&#10230; Qπ(s,a)= ortalama ut , st−1=s ve at=a olduğunda

<br>


**88. where ut denotes the utility starting at step t of a given episode.**

&#10230; ut belirli bir bölümün t anında başlayan faydayı ifade etmektedir.

<br>


**89. Remark: model-free Monte Carlo is said to be on-policy, because the estimated value is dependent on the policy π used to generate the data.**

&#10230; Not: model içermeyen Monte Carlo'nun politikaya dahil olduğu söyleniyor, çünkü tahmini değer veriyi üretmek için kullanılan π politikasına bağlıdır.

<br>


**90. Equivalent formulation - By introducing the constant η=11+(#updates to (s,a)) and for each (s,a,u) of the training set, the update rule of model-free Monte Carlo has a convex combination formulation:**

&#10230; Eşdeğer formülasyon - Sabit tanımı η=11+(#güncelleme sayısı (s,a) ) ve eğitim kümesinin her bir (s,a,u) üçlemesi için, model içermeyen Monte Carlo'nun güncelleme kuralı dışbükey bir kombinasyon formülasyonuna sahiptir:

<br>


**91. as well as a stochastic gradient formulation:**

&#10230; olasılıksal bayır formülasyonu yanında:

<br>


**92. SARSA ― State-action-reward-state-action (SARSA) is a boostrapping method estimating Qπ by using both raw data and estimates as part of the update rule. For each (s,a,r,s′,a′), we have:**

&#10230; SARSA ― Durum-eylem-ödül-durum-eylem (State-Action-Reward-State-Action - SARSA), hem ham verileri hem de güncelleme kuralının bir parçası olarak tahminleri kullanarak Qπ'yi tahmin eden bir destekleme yöntemidir. Her bir (s,a,r,s′,a′) için:

<br>


**93. Remark: the SARSA estimate is updated on the fly as opposed to the model-free Monte Carlo one where the estimate can only be updated at the end of the episode.**

&#10230; Not: the SARSA tahmini, tahminin yalnızca bölüm sonunda güncellenebildiği model içermeyen Monte Carlo yönteminin aksine anında güncellenir.

<br>


**94. Q-learning ― Q-learning is an off-policy algorithm that produces an estimate for Qopt. On each (s,a,r,s′,a′), we have:**

&#10230; Q-öğrenme ― Q-öğrenme, Qopt için tahmin üreten politikaya dahil olmayan bir algoritmadır. Her bir (s,a,r,s′,a′) için:

<br>


**95. Epsilon-greedy ― The epsilon-greedy policy is an algorithm that balances exploration with probability ϵ and exploitation with probability 1−ϵ. For a given state s, the policy πact is computed as follows:**

&#10230; Epsilon-açgözlü ― Epsilon-açgözlü politika, ϵ olasılıkla araştırmayı ve 1−ϵ olasılıkla sömürüyü dengeleyen bir algoritmadır. Her bir s durumu için, πact politikası aşağıdaki şekilde hesaplanır:

<br>


**96. [with probability, random from Actions(s)]**

&#10230; [olasılıkla, Actions(s) eylem kümesi içinden rastgele]

<br>


**97. Game playing**

&#10230; Oyun oynama

<br>


**98. In games (e.g. chess, backgammon, Go), other agents are present and need to be taken into account when constructing our policy.**

&#10230; Oyunlarda (örneğin satranç, tavla, Go), başka oyuncular vardır ve politikamızı oluştururken göz önünde bulundurulması gerekir.

<br>


**99. Game tree ― A game tree is a tree that describes the possibilities of a game. In particular, each node is a decision point for a player and each root-to-leaf path is a possible outcome of the game.**

&#10230; Oyun ağacı ― Oyun ağacı, bir oyunun olasılıklarını tarif eden bir ağaçtır. Özellikle, her bir düğüm, oyuncu için bir karar noktasıdır ve her bir kökten (root) yaprağa (leaf) giden yol oyunun olası bir sonucudur.

<br>


**100. [Two-player zero-sum game ― It is a game where each state is fully observed and such that players take turns. It is defined with:, a starting state sstart, possible actions Actions(s) from state s, successors Succ(s,a) from states s with actions a, whether an end state was reached IsEnd(s), the agent's utility Utility(s) at end state s, the player Player(s) who controls state s]**

&#10230; [İki oyunculu sıfır toplamlı oyun ― Her durumun tamamen gözlendiği ve oyuncuların sırayla oynadığı bir oyundur. Aşağıdaki gibi tanımlanır:, bir başlangıç durumu sstart, s durumunda gerçekleştirilebilecek olası eylemler Actions(s), s durumunda a eylemi gerçekleştirildiğindeki ardıllar Succ(s,a), bir bitiş durumuna ulaşılıp ulaşılmadığı IsEnd(s), s bitiş durumunda etmenin elde ettiği fayda Utility(s), s durumunu kontrol eden oyuncu Player(s)]

<br>


**101. Remark: we will assume that the utility of the agent has the opposite sign of the one of the opponent.**

&#10230; Not: Oyuncu faydasının işaretinin, rakibinin faydasının tersi olacağını varsayacağız.

<br>


**102. [Types of policies ― There are two types of policies:, Deterministic policies, noted πp(s), which are actions that player p takes in state s., Stochastic policies, noted πp(s,a)∈[0,1], which are probabilities that player p takes action a in state s.]**

&#10230; [Politika türleri ― İki tane politika türü vardır:, πp(s) olarak gösterilen belirlenimci politikalar , p oyuncusunun s durumunda gerçekleştirdiği eylemler., πp(s,a)∈[0,1] olarak gösterilen olasılıksal politikalar, p oyuncusunun s durumunda a eylemini gerçekleştirme olasılıkları.]

<br>


**103. Expectimax ― For a given state s, the expectimax value Vexptmax(s) is the maximum expected utility of any agent policy when playing with respect to a fixed and known opponent policy πopp. It is computed as follows:**

&#10230; En yüksek beklenen değer(Expectimax) ― Belirli bir s durumu için, en yüksek beklenen değer olan Vexptmax(s), sabit ve bilinen bir rakip politikası olan πopp'a göre oynarken, bir oyuncu politikasının en yüksek beklenen faydasıdır. En yüksek beklenen değer(Expectimax) aşağıdaki gibi hesaplanmaktadır:

<br>


**104. Remark: expectimax is the analog of value iteration for MDPs.**

&#10230; Not: En yüksek beklenen değer(Expectimax), MDP'ler için değer yinelemenin analog halidir.

<br>


**105. Minimax ― The goal of minimax policies is to find an optimal policy against an adversary by assuming the worst case, i.e. that the opponent is doing everything to minimize the agent's utility. It is done as follows:**

&#10230; En küçük-en büyük (minimax) ― En küçük-enbüyük (minimax) politikaların amacı en kötü durumu kabul ederek, diğer bir deyişle; rakip, oyuncunun faydasını en aza indirmek için her şeyi yaparken, rakibe karşı en iyi politikayı bulmaktır. En küçük-en büyük(minimax) aşağıdaki şekilde yapılır:

<br>


**106. Remark: we can extract πmax and πmin from the minimax value Vminimax.**

&#10230; Not: πmax ve πmin değerleri, en küçük-en büyük olan Vminimax'dan elde edilebilir.

<br>


**107. Minimax properties ― By noting V the value function, there are 3 properties around minimax to have in mind:**

&#10230; En küçük-en büyük (minimax) özellikleri ― V değer fonksiyonunu ifade ederse, En küçük-en büyük (minimax) ile ilgili aklımızda bulundurmamız gereken 3 özellik vardır:

<br>


**108. Property 1: if the agent were to change its policy to any πagent, then the agent would be no better off.**

&#10230; Özellik 1: Oyuncu politikasını herhangi bir πagent ile değiştirecek olsaydı, o zaman oyuncu daha iyi olmazdı.

<br>


**109. Property 2: if the opponent changes its policy from πmin to πopp, then he will be no better off.**

&#10230; Özellik 2: Eğer rakip oyuncu politikasını πmin'den πopp'a değiştirecek olsaydı, o zaman rakip oyuncu daha iyi olamazdı.

<br>


**110. Property 3: if the opponent is known to be not playing the adversarial policy, then the minimax policy might not be optimal for the agent.**

&#10230; Özellik 3: Eğer rakip oyuncunun muhalif (adversarial) politikayı oynamadığı biliniyorsa, o zaman en küçük-en büyük(minimax) politika oyuncu için ey iyi (optimal) olmayabilir.

<br>


**111. In the end, we have the following relationship:**

&#10230; Sonunda, aşağıda belirtildiği gibi bir ilişkiye sahip oluruz:

<br>


**112. Speeding up minimax**

&#10230; En küçük-en büyük (minimax) hızlandırma

<br>


**113. Evaluation function ― An evaluation function is a domain-specific and approximate estimate of the value Vminimax(s). It is noted Eval(s).**

&#10230; Değerlendirme işlevi ― Değerlendirme işlevi, alana özgü (domain-specific) ve Vminimax(s) değerinin yaklaşık bir tahminidir. Eval(s) olarak ifade edilmektedir.

<br>


**114. Remark: FutureCost(s) is an analogy for search problems.**

&#10230; Not: FutureCost(s) arama problemleri için bir benzetmedir(analogy).

<br>


**115. Alpha-beta pruning ― Alpha-beta pruning is a domain-general exact method optimizing the minimax algorithm by avoiding the unnecessary exploration of parts of the game tree. To do so, each player keeps track of the best value they can hope for (stored in α for the maximizing player and in β for the minimizing player). At a given step, the condition β<α means that the optimal path is not going to be in the current branch as the earlier player had a better option at their disposal.**

&#10230; Alpha-beta budama ― Alfa-beta budama, oyun ağacının parçalarının gereksiz yere keşfedilmesini önleyerek en küçük-en büyük(minimax) algoritmasını en iyileyen (optimize eden) alana-özgü olmayan genel bir yöntemdir. Bunu yapmak için, her oyuncu ümit edebileceği en iyi değeri takip eder (maksimize eden oyuncu için α'da ve minimize eden oyuncu için β'de saklanır). Belirli bir adımda, β <α koşulu, önceki oyuncunun emrinde daha iyi bir seçeneğe sahip olması nedeniyle en iyi (optimal) yolun mevcut dalda olamayacağı anlamına gelir.

<br>


**116. TD learning ― Temporal difference (TD) learning is used when we don't know the transitions/rewards. The value is based on exploration policy. To be able to use it, we need to know rules of the game Succ(s,a). For each (s,a,r,s′), the update is done as follows:**

&#10230; TD öğrenme ― Geçici fark (Temporal difference - TD) öğrenmesi, geçiş/ödülleri bilmediğimiz zaman kullanılır. Değer, keşif politikasına dayanır. Bunu kullanabilmek için, oyununun kurallarını,Succ (s, a), bilmemiz gerekir. Her bir (s,a,r,s′) için, güncelleme aşağıdaki şekilde yapılır:

<br>


**117. Simultaneous games**

&#10230; Eşzamanlı oyunlar

<br>


**118. This is the contrary of turn-based games, where there is no ordering on the player's moves.**

&#10230; Bu, oyuncunun hamlelerinin sıralı olmadığı sıra temelli oyunların tam tersidir.
 
<br>


**119. Single-move simultaneous game ― Let there be two players A and B, with given possible actions. We note V(a,b) to be A's utility if A chooses action a, B chooses action b. V is called the payoff matrix.**

&#10230; Tek-hamleli eşzamanlı oyun ― Olası hareketlere sahip A ve B iki oyuncu olsun. V(a,b), A'nın a eylemini ve B'nin de b eylemini seçtiği A'nın faydasını ifade eder. V, getiri dizeyi olarak adlandırılır.

<br>


**120. [Strategies ― There are two main types of strategies:, A pure strategy is a single action:, A mixed strategy is a probability distribution over actions:]**

&#10230; [Stratejiler ― İki tane ana strateji türü vardır:, Saf strateji, tek bir eylemdir:, Karışık strateji, eylemler üzerindeki bir olasılık dağılımıdır:]

<br>


**121. Game evaluation ― The value of the game V(πA,πB) when player A follows πA and player B follows πB is such that:**

&#10230; Oyun değerlendirme ― oyuncu A πA'yı ve oyuncu B de πB'yi izlediğinde, Oyun değeri V(πA,πB):

<br>


**122. Minimax theorem ― By noting πA,πB ranging over mixed strategies, for every simultaneous two-player zero-sum game with a finite number of actions, we have:**

&#10230; En küçük-en büyük (minimax) teoremi ― ΠA, πB’nin karma stratejilere göre değiştiğini belirterek, sonlu sayıda eylem ile eşzamanlı her iki oyunculu sıfır toplamlı oyun için:

<br>


**123. Non-zero-sum games**

&#10230; Sıfır toplamı olmayan oyunlar

<br>


**124. Payoff matrix ― We define Vp(πA,πB) to be the utility for player p.**

&#10230; Getiri matrisi ― Vp(πA,πB)'yi oyuncu p'nin faydası olarak tanımlıyoruz.

<br>


**125. Nash equilibrium ― A Nash equilibrium is (π∗A,π∗B) such that no player has an incentive to change its strategy. We have:**

&#10230; Nash dengesi ― Nash dengesi (π ∗ A, π ∗ B) öyle birşey ki hiçbir oyuncuyu, stratejisini değiştirmeye teşvik etmiyor:

<br>


**126. and**

&#10230; ve

<br>


**127. Remark: in any finite-player game with finite number of actions, there exists at least one Nash equilibrium.**

&#10230; Not: sonlu sayıda eylem olan herhangi bir sonlu oyunculu oyunda, en azından bir tane Nash denegesi mevcuttur.

<br>


**128. [Tree search, Backtracking search, Breadth-first search, Depth-first search, Iterative deepening]**

&#10230; [Ağaç arama, Geri izleme araması, Genişlik öncelikli arama, Derinlik öncelikli arama, Tekrarlı (Iterative) derinleşme]

<br>


**129. [Graph search, Dynamic programming, Uniform cost search]**

&#10230; [Çizge arama, Dinamik programlama, Tek tip maliyet araması]

<br>


**130. [Learning costs, Structured perceptron]**

&#10230; [Öğrenme maliyetleri, Yapısal algılayıcı]

<br>


**131. [A star search, Heuristic function, Algorithm, Consistency, correctness, Admissibility, efficiency]**

&#10230; [A yıldız arama, Sezgisel işlev, Algoritma, Tutarlılık, doğruluk, kabul edilebilirlik, verimlilik]

<br>


**132. [Relaxation, Relaxed search problem, Relaxed heuristic, Max heuristic]**

&#10230; [Rahatlama, Rahat arama problemi, Rahat sezgisel, En yüksek sezgisel]

<br>


**133. [Markov decision processes, Overview, Policy evaluation, Value iteration, Transitions, rewards]**

&#10230; [Markov karar süreçleri, Genel bakış, Politika değerlendirme, Değer yineleme, Geçişler, ödüller]

<br>


**134. [Game playing, Expectimax, Minimax, Speeding up minimax, Simultaneous games, Non-zero-sum games]**

&#10230; [Oyun oynama, En yüksek beklenti, En küçük-en büyük, En küçük-en büyük hızlandırma, Eşzamanlı oyunlar, Sıfır toplamı olmayan oyunlar]

<br>


**135. View PDF version on GitHub**

&#10230; GitHub'da PDF sürümünü görüntüleyin

<br>


**136. Original authors**

&#10230; Asıl yazarlar

<br>


**137. Translated by X, Y and Z**

&#10230; X, Y ve Z tarafından tercüme edilmiştir.

<br>


**138. Reviewed by X, Y and Z**

&#10230; X,Y,Z tarafından gözden geçirilmiştir.

<br>


**139. By X and Y**

&#10230; X ve Y ile

<br>


**140. The Artificial Intelligence cheatsheets are now available in [target language].**

&#10230; Yapay Zeka el kitapları artık [hedef dilde] mevcuttur.
