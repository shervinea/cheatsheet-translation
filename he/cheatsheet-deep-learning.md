<p dir='rtl' align='left'>
 
 **1. Deep Learning cheatsheet**

&#10230; I'll keep this, not a good choice to translate this sentence to Hebrew.

<br>

**2. Neural Networks**

&#10230; רשתות נוירונים

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; <p dir='rtl' align='left'>רשתות נוירונים בנויות משכבות שונות.
רשתות נוירונים שכיחות כוללות רשתות קונבולוציה (Convolutional Neural Networks)
ורשתות חוזרות (Recurrent Neural Network).</p>

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; ארכיטקטורות - מילון העוסק בארכיטקטורות רשתות נוירונים מתואר בתרשום מטה:

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; [שכבת כניסה, שכבה חבויה, שכבת יציאה]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; <p dir='rtl' align='left'> בסימון i כשכבה ה i במספרה ברשת ו j כנוירון ה j החבויה של השכבה, יש לנו: </p>

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; <p dir='rtl' align='left'> כאשר אנו מגדירים את w, b, z כמשקולת, הטיה ופלט בהתאמה.</p>
w = weight,
b = bias,
z = output

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; <p dir='rtl' align='left'> פונקציית איפשור - פונקציות אלו משמשות בסוף יחידה חבויה על מנת להציג למודל מורכבות לא לינארית, אלו הפונקציות השכיחות: </p>

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230; Nothing to translate

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; <p dir='rtl' align='left'> Cross-entropy loss - בהקשר של רשתות נוירונים זוהי פונקציה שכיחה שמוגדרת כדלהלן: </p>

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; <p dir='rtl' align='left'>  קצב למידה - לעיתים מוצג כ η ומצביע על קצב העדכון של המשקולות. קצב הלמידה יכול להשתנות. השיטה הפופולרית ביותר כרגע לשיפור ותיקון קצב הלמידה היא Adam. </p>

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; פעפוע לאחור - פעפוע לאחור זוהי שיטה לעדכון המשקולות ברשת נוירונים שבה נלקחים בחשבון הפלט הרצוי והפלט המצוי. 
Had some trouble with the rest of the sentence: " The derivative with respect to ... ", wait for some feedback

<br>

**13. As a result, the weight is updated as follows:**

&#10230; כתוצאה מכך, המשקולת מעודכנת בצורה הבאה:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; עדכון משקולות - ברשתות נוירונים, משקולות מעודכנות בצורה הבאה:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; שלב 1: קח/י מקבץ מסט המידע.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; <p dir='rtl' align='left'> שלב 2: בצעו פעפוע קדימה על מנת להשיג את האיבוד (loss) התואם. </p>

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; <p dir='rtl' align='left'> שלב 3: פעפוע לאחור של האיבוד (loss) על מנת לקבל את הגרדיאנט </p>

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; שלב 4: שימוש בגרדיאנט על מנת לעדכן את המשקולות של הרשת

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; <p dir='rtl' align='left'> Dropout - טכניקה שנועדה למנוע אימון-יתר על סט המידע בעזרת שיבוש נוירונים ברשת. בצורה פרקטית הנוירונים "צונחים" עם הסתברות p או נשארים עם הסתברות 1-p </p>

<br>

**20. Convolutional Neural Networks**

&#10230; Nothing to translate

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; <p dir='rtl' align='left'> דרישות של שכבת קונבולוציה - בהנחה ש W מוגדר כגודל הקלט, F כמספר הנוירונים בשכבת הקונבולוציה, P כריפוד באפסים, אז מספר הנוירונים N הרצויים הוא:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; <p dir='rtl' align='left'> נורמליזציית מקבצים - נירמול המקבץ {x,i} בעזרת ההיפטר-פרמטרים γ,β . </p>
I had some trouble with the rest of the sentence: "By noting μB,σ2B the mean... "

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; נעשה אחרי שכבת חיבור/קונבולוציה ולפני שכבה לא לינארית, מכוון לאפשור של קצב למידה גבוה יותר ומקטין את התלות באתחול

<br>

**24. Recurrent Neural Networks**

&#10230; רשתות חוזרות

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; סוגי שערים - להלן סוגים שונים של שערים שבהם נוכל להתקל ברשת חוזרת טיפוסית:

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; [שער כניסה, שער שיכחה, שער, שער יציאה]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; [האם לכתוב לתא?, האם למחוק את התא?, כמה לכתוב לתא?, כמה לחשוף תא?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;  <p dir='rtl' align='left'> LSTM - בעברית זכרון ארוך לטווח קצר - רשת LSTM היא סוג של מודל RNN (Recurrent Neural Network) שנמנעת מבעיות של היעלמות גרדיאנט על ידי הוספה של שערי שיכחה. </p>

<br>

**29. Reinforcement Learning and Control**

&#10230; למידה באמצעות חיזוקים

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; המטרה של למידה באמצעות חיזוקים היא להתפתח בסביבה מסוימת

<br>

**31. Definitions**

&#10230; הגדרות

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; <p dir='rtl' align='left'> תהליך החלטה מרקובי - MDP הוא רשומה של 5 משתנים (S,A,{Psa},γ,R) כאשר: </p> 

<br>

**33. S is the set of states**

&#10230; <p dir='rtl' align='left'> S קבוצת המצבים </p>

<br>

**34. A is the set of actions**

&#10230; <p dir='rtl' align='left'> A קבוצת הפעולות </p>

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; <p dir='rtl' align='left'> {Psa} הם ההסתברות לשינויים עבור s∈S ו a∈A </p>

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; 

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; <p dir='rtl' align='left'> R:S×A⟶R או R:S⟶R היא הפונקציה שהאלגוריתם רוצה למקסם</p>

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230; <p dir='rtl' align='left'> מדיניות - המדיניות π היא פונקציה π:S⟶A שממפה מצבים לפעולות </p> 

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230; <p dir='rtl' align='left'> הערה: נאמר כי נממש מדיניות π אם ניתן מצד s ואנחנו מבצעים פעולה a=π(s)  </p>

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230; <p dir='rtl' align='left'> פונקציית ערך - עבור מדיניות π ומצב s, נגדיר את הערך Vπ כדלהלן: </p>

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230; <p dir='rtl' align='left'> משוואת בלמן - משוואות בלמן האפוטימליות מתארות את פונקציית הערך Vπ* של המדיניות האופטימלית π* </p>

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230; <p dir='rtl' align='left'> הערה: נבחין כי המדיניות האופטימלית π* עבור מצב s היא:  </p>

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230; <p dir='rtl' align='left'> Value iteration algorithm - ממומש בשני צעדים: </p>

<br>

**44. 1) We initialize the value:**

&#10230; איתחול המשתנה:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230; מעבר על המשתנה בהתבסס על המשתנים הקודמים:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230; סבירות מקסימלית משוערכת - מצביעה על הסתברות שינוי המצב כדלהלן:

<br>

**47. times took action a in state s and got to s′**

&#10230; Need to clarify

<br>

**48. times took action a in state s**

&#10230; Need to clarify

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230; <p dir='rtl' align='left'> Q-learning - שיערוך Q ללא מודל, נעשה בצורה הבאה: </p>

<br>

**50. View PDF version on GitHub**

&#10230; <p dir='rtl' align='left'> הצג גרסאת PDF בגיטהאב </p>

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230; [רשתות נוירונים, ארכיטקטורה, פונקציית אקטיבציה, פעפוע לאחור, Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [רשתות קונבולוציה, שכבת קונבולוציה, נירמול מקבצים]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [רשתות חוזרות, שערים, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; <p dir='rtl' align='left'> [למידה באמצעות חיזוקים, תהליך החלטה מרקובי, Value/Policy iteration, תכנות דינאמי , מדיניות חיפוש] </p>
</p>
