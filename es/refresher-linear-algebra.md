**1. Revisión rápida de Algebra Linea y Cálculo **

&#10230; 

<br> 

**2. Notificaciones Generales**

&#10230; 

<br> 

**3. Definiciones**

&#10230; 

<br> 

**4. Vector ― Sea x∈Rn un vector con n entradas, donde xi∈R es la enésima entrada: **

&#10230; 

<br> 

**5. Matriz ― Sea A∈Rm×n una matriz con n filas y m columnas; donde Ai, j∈R es el valor alocado en la i-ésima fila y la n-ésima columna:**

&#10230;   

<br>

**6. Nota: el vector x definido arriba puede ser visto como una matriz de n×1 y es particularmente llamado vector-columna.**

&#10230; 

<br>

**7. Matrices principales**

&#10230; 

<br>

**8. Matriz identidad - La matriz identidad I∈Rn×n es una matriz cuadrada con valor 1 en su diagonal y cero en el resto:**

&#10230; 

<br>

**9. Nota: para todas las matrices A∈Rn×n, tenemos A×I=I×A=A.**

&#10230; 

<br>

**10. Matriz diagonal ― Una matriz diagonal D∈Rn×n es una matriz cuadrada con valores diferentes de zero en su diagonal y cero en el resto:**

&#10230;

<br>

**11. Nota: Sea D una diag(d1,...,dn).**

&#10230;

<br>

**12. Operaciones de matrices**

&#10230;

<br>

**13. Multiplicación**

&#10230;

<br>

**14. Vector-vector ― Hay dos tipos de multiplicaciones vector-vector:**

&#10230;

<br>

**15. producto interno: for x,y∈Rn, se tiene:**

&#10230;

<br>

**16. producto externo: for x∈Rm,y∈Rn, we have:**

&#10230;

<br>

**17. Matriz-vector ― El producto de la matriz A∈Rm×n y el vector x∈Rn, es un vector de tamaño Rn; tal que:**

&#10230;

<br>

**18. donde aTr,i son las filas del vector and ac,j son las columnas del vector A, y xi son las entradas de x.**

&#10230;

<br>

**19. Matriz-matriz ― El producto de las matrices A∈Rm×n y B∈Rn×p es una matriz de tamaño Rn×p, tal que:**

&#10230;

<br>

**20. donde aTr,i,bTr,i son las filas del vector and ac,j,bc,j las columnas de A y B respectivamente**

&#10230;

<br>

**21. Otras operaciones**

&#10230;

<br>

**22. Transpuesta ― La transpuesta de la matriz A∈Rm×n, con notacion AT, es tal que sus entradas son volteadas:**

&#10230;

<br>

**23. Nota: para matrices A,B, se tiene (AB)T=BTAT**

&#10230;

<br>

**24. Inversa ― La inversa de una matriz cuadrada invertible A, llamada A−1 y es la única matriz tal que:**

&#10230;

<br>

**25. Nota: no todas las matrices se pueden invertir. Además, para las matrices A,B, se tiene que (AB)−1=B−1A−1**

&#10230;

<br>

**26. Traza ― La traza de una matriz cuadrada A, tr(A), es la suma de sus elementos en la diagonal:**

&#10230;

<br>

**27. Nota: para matrices A,B, se tiene tr(AT)=tr(A) y tr(AB)=tr(BA)**

&#10230;

<br>

**28. Determinanate ― El determinante de una matriz cuadrada A∈Rn×n, llamado |A| or det(A) es expresado recursivamente en términos de A∖i,∖j, que es la matriz A en su i-ésima fila y j-ésima columna, como se muestra:**

&#10230;

<br>

**29. Nota: A es tiene inversa si y solo si |A|≠0. Además, |AB|=|A||B| y |AT|=|A|.**

&#10230;

<br>

**30. Propiedades de matrices**

&#10230;

<br>

**31. Definiciones**

&#10230;

<br>

**32. Descomposición Simétrica ― Una matriz A puede ser expresada en términos de sus partes simétricas y asimetricas, como se muestra:**

&#10230;

<br>

**33. [Simétrica, Asimétrica]**

&#10230;

<br>

**34. Norma ― La norma o módulo es una función N:V⟶[0,+∞[ donde V es un vector espacial, y tal que para todos los x,y∈V, se tiene:**

&#10230;

<br>

**35. N(ax)=|a|N(x) para un escalar**

&#10230;

<br>

**36. si N(x)=0, entonces x=0**

&#10230;

<br>

**37. Para x∈V, los modulos o normas más comunmente usadas están descritas en la tabla de abajo:**

&#10230;

<br>

**38. [Norma, Notacion, Definición, Caso de uso]**

&#10230;

<br>

**39. Dependencia Lineal ― Un conjunto de vectores se dice linearmente dependiente si uno de los vectores en el grupo puede ser definido como la combinación de los otros.**

&#10230;

<br>

**40. Nota: si no se puede escribir el vector de esta manera, entonces el vector se dice que es linealmente independiente**

&#10230;

<br>

**41. Rango matricial ― El rango de una matriz A, nombrado rank(A), que es la dimensioón del vector espacial generado por sus columnas. Lo que es equivalente al número máximo de columnas linealmente independientes de A.**

&#10230;

<br>

**42. Matriz semi-definida positiva ― Una matriz A∈Rn×n es semi-defininda positivamente (PSD) y se tiene que A⪰0 si:**

&#10230;

<br>

**43. Nota: de igual forma, una matriz A se dice positiva y definida, A≻0, si esa una matriz PSD que satisface para todos los vectores diferentes de cero x, xTAx>0.**

&#10230;

<br>

**44. Eigenvalor, eigenvector ― Dado una matriz A∈Rn×n, λ se dice que es el eigenvalor de A si existe un vector z∈Rn∖{0}, llamado eigenvector, tal que se tiene:**

&#10230;

<br>

**45. Teoréma espectrasl ― Sea A∈Rn×n. si A es simétrica, entonces A es diagonalizable por una matriz real ortogonal U∈Rn×n.
Notando Λ=diag(λ1,...,λn), se tiene que:**

&#10230;

<br>

**46. diagonal**

&#10230;

<br>

**47. Descomposición de valores singulares ― Para una mtraiz A de dimensiones m×n, la descomposición en valores singulares (SVD) es una técnica de factorizacion que garantiza que existen matrices U m×m unitaria, Σ m×n diagonal y V n×n unitaria, tal que:**

&#10230;

<br>

**48. Calculo de matrices**

&#10230;

<br>

**49. Gradiente ― Sea f:Rm×n→R una función y A∈Rm×n una matriz. El gradiente de f con respecto a A es una matriz de m×n, notando que ∇Af(A), tal que:**

&#10230;

<br>

**50. Nota: el gradiente de f esta solo definido cuando f es una función cuyo resultado es un escalar.**

&#10230;

<br>

**51. Matriz Hessiana ― Sea f:Rn→R una función y x∈Rn un vector. La matriz hessiana o hessiano de f con recpecto a x
es una matriz simétrica de n×n, para ∇2xf(x), tal que:**

&#10230;

<br>

**52. Nota: la matriz hessiana de f solo esta definida cuando f es una función que regresa un escalar**

&#10230;

<br>

**53. Operaciones de gradiente ― Para matrices A,B,C, las siguientes propiedades del gradiente vale la pena tener en cuenta:**

&#10230;
