**1. Deep Learning cheatsheet**

&#10230; डीप लर्निंग चीट शीट

<br>

**2. Neural Networks**

&#10230; न्यूरल नेटवर्क

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; न्यूरल नेटवर्क मॉडल की एक श्रेणी है जो परतों के साथ बनाई गई है। आम तौर पर उपयोग किए जाने वाले प्रकार के न्यूरल नेटवर्क में कन्वोल्यूशनल और रेकररेंट  न्यूरल नेटवर्क शामिल होते हैं।

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; आर्किटेक्चर - न्यूरल नेटवर्क आर्किटेक्चर के आसपास शब्दावली नीचे दिए गए आंकड़े में वर्णित है।

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [इनपुट परत, छिपी परत, आउटपुट परत]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;i नेटवर्क की ith परत और j नेटवर्क की jth छिपी परत इकाई को ध्यान में रखते हुए, हमारे पास है:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; जहां हम क्रमशः w, b, z वेट, बायस और आउटपुट नोट करते हैं।

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; एक्टिवेशन फंक्शन - मॉडल के लिए गैर-रैखिक जटिलताओं को पेश करने के लिए एक छिपी इकाई के अंत में एक्टिवेशन फ़ंक्शंस का उपयोग किया जाता है। यहां सबसे आम हैं:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; [सिग्मोइड, तेनएच, रेलेयु, लीकी रेलेयु]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; क्रॉस-एन्ट्रॉपी लॉस  - न्यूरल नेटवर्क के संदर्भ में, क्रॉस-एन्ट्रॉपी लॉस L(z,y) का उपयोग आमतौर पर किया जाता है और इसे निम्नानुसार परिभाषित किया जाता है:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; लर्निंग रेट  - लर्निंग रेट, अक्सर α या कभी-कभी η, जो इंगित करती है कि वेट किस गति से अपडेट होता है। यह तय या अनुकूली रूप से बदला जा सकता है। वर्तमान सबसे लोकप्रिय विधि को एडम कहा जाता है, जो एक तरीका है जो सीखने की दर को अनुकूलित करता है।

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; बैकप्रोपैगेशन - बैकप्रोपैगेशन वास्तविक आउटपुट और वांछित आउटपुट को ध्यान में रखते हुए न्यूरल नेटवर्क में वेट को अपडेट करने का एक तरीका है। वेट डब्ल्यू के संबंध में डेरीवेटिव श्रृंखला नियम का उपयोग करके गणना की जाती है और निम्न रूप में है:
 
<br>

**13. As a result, the weight is updated as follows:**

&#10230;  नतीजतन, वेट निम्नानुसार अद्यतन किया गया है:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; वेट अपडेट करना - एक न्यूरल नेटवर्क में, वेट निम्नानुसार अपडेट किए जाते हैं:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; चरण 1: प्रशिक्षण डेटा का एक बैच लें।

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; चरण 2: संबंधित लॉस प्राप्त करने के लिए फॉरवर्ड प्रोपगेशन करें।

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; चरण 3: ग्रेडिएंट प्राप्त करने के लिए लॉस को बैकप्रोपेगेट करे।

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; चरण 4: नेटवर्क के वेट को अद्यतन करने के लिए ग्रेडियेंट का उपयोग करें।

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; ड्रॉपआउट - ड्रॉपआउट एक तंत्र है जो न्यूरल नेटवर्क में कुछ इकाइयों को छोड़कर प्रशिक्षण डेटा को ओवरफिट करने से रोकने के लिए है। अभ्यास में, न्यूरॉन्स को या तो संभावना पी(p) के साथ गिरा दिया जाता है या संभावना 1-पी(p) के साथ रखा जाता है।

<br>

**20. Convolutional Neural Networks**

&#10230; कनवॉल्यूशनल न्यूरल नेटवर्क

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; कनवॉल्यूशनल लेयर आवश्यकता - डब्ल्यू(w) इनपुट इनपुट वॉल्यूम आकार, एफ(F) कनवॉल्यूशनल लेयर न्यूरॉन्स का आकार, पी(P) शून्य पैडिंग की मात्रा, फिर दिए गए वॉल्यूम में फिट न्यूरॉन्स एन(N) की संख्या ऐसी है कि:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; बैच सामान्यीकरण - यह हाइपरपेरामीटर γ, β का एक कदम है जो बैच को सामान्य करता है {xi}। μB को ध्यान में रखते हुए, σ2B इसका अर्थ और भिन्नता है जिसे हम बैच को सही करना चाहते हैं, यह निम्नानुसार किया जाता है:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; यह आमतौर पर पूरी तरह से जुड़े / कनवॉल्यूशनल परत के बाद और गैर-रैखिकता लेयर से पहले किया जाता है और इसका उद्देश्य उच्च लर्निंग रेट की अनुमति देना और प्रारंभिकता पर मजबूत निर्भरता को कम करना है।

<br>

**24. Recurrent Neural Networks**

&#10230; रेकर्रेंट न्यूरल नेटवर्क
 
<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; गेट के प्रकार - यहां विभिन्न प्रकार के गेट हैं जिन्हें हम एक सामान्य रेकर्रेंट न्यूरला नेटवर्क में सामना करते हैं:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [इनपुट गेट, फॉरगेट गेट, गेट, आउटपुट गेट]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [सेल को लिखें या नहीं ?, एक सेल मिटाएं या नहीं ?, सेल को कितना लिखना है ?, सेल को कितना खुलासा करना है?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; एलएसटीएम - ए लॉन्ग शार्ट टर्म मेमोरी (एलएसटीएम) नेटवर्क एक प्रकार का आरएनएन मॉडल है जो 'फॉरगेट' गेट जोड़कर होने वाली वैनिशिंग ग्रेडिएंट समस्या से बचाता है।

<br>

**29. Reinforcement Learning and Control**

&#10230; रएंफोर्रसमेंट लर्निंग और नियंत्रण

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;रएंफोर्रसमेंट लर्निंग का लक्ष्य एक एजेंट के लिए सीखना है कि पर्यावरण में कैसे विकसित किया जाए।

<br>

**31. Definitions**

&#10230; परिभाषाएं

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; मार्कोव डिसिजन प्रोसेस - एक मार्कोव डिसिजन प्रोसेस (एमडीपी) 5-टुपल (एस(S), ए(A), {Psa}, γ, आर(R)) है जहां:

<br>

**33. S is the set of states**

&#10230; S स्टेट्स का सेट है
 
<br>

**34. A is the set of actions**

&#10230; A एक्शन का सेट है
 
<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; {Psa} s∈S और a∈A के लिए स्टेट ट्रांजीशन प्रॉबब्लिटी हैं
 
<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; γ∈ [0,1 डिस्काउंट फैक्टर  है
 
<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; R:S×A⟶R या R:S⟶R रिवॉर्ड फंक्शन है कि एल्गोरिदम अधिकतम करना चाहता है

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; पालिसी - एक पालिसी π एक फंक्शन π:S⟶A जो एक्शन्स को स्टेट्स को मानचित्र बनाता है।

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; टिप्पणी: हम कहते हैं कि हम किसी दिए गए नीति को निष्पादित करते हैं π अगर स्टेट दिया जाता है तो हम कार्रवाई को a = π (ओं) लेते हैं।
 
<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;

<br>

**44. 1) We initialize the value:**

&#10230;

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;

<br>

**47. times took action a in state s and got to s′**

&#10230;

<br>

**48. times took action a in state s**

&#10230;

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
