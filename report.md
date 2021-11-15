---
title: "Projet de Conception Optimale"
author: "Antonin Bavoil"
geometry: margin=2.5cm
date: "12 novembre 2021"
---

[comment]: # (To generate the HTML, use `pandoc --toc --standalone --katex report.md -o report.html`)

\newpage{}

# Introduction

L'objectif de ce projet est de coupler simulation numérique et optimisation : il faut minimiser les dégâts d'un feu de forêt en défrichant un rectangle de forêt.
L'application dans le monde réel serait la prévention des feux de forêt à l'aide d'une carte de la densité de la forêt ainsi qu'une carte des vents les plus fréquents. A partir (d'une version plus sophistiquée) de notre programme, on pourrait identifier les zones intéressantes à défricher pour arrêter les incendies.
Dans un premier temps, on va modéliser le feu de forêt par un couple d'équations différentielles partielles non-linéaires du second ordre à une dimension de temps et deux d'espace. Pour résoudre ce système d'équation, on va utiliser la méthode des différences finies (espace) et la méthode d'Euler explicite (temps).
Dans un second temps, on s'intéressera à l'algorithme du simplex pour optimiser la position et la taille d'un rectangle de coupe-feu.

# Simulation

La première étape de ce projet est d'avoir une simulation de feu de forêt. On commence par créer un modèle régit par une équation différentielle partielle, ensuite on discrétise ce modèle, et enfin on l'implémente. J'ai choisi d'utiliser Python pour ce projet car je connais bien ce langage.

## Modèle continu

### Champs considérés

On considère deux champs scalaires :

 - la température $T(x, y, t)$
 - la matière combustible $c(x, y, t)$
avec $t \in I = [0, 2.5]$ et $(x, y) \in \Omega = [0, 1]^2$.

On considère aussi un champ vectoriel :

 - le vent $\overrightarrow{V}(x, y) = (u(x, y), v(x, y))$ avec $u(x, y) = \cos(\pi y)$ et $v(x, y) = 0.6 \sin(\pi (x + 0.2) / 2)$.


### Conditions initiales

À l'instant initial, on a
 - $T(x, y, 0) = 0.2$ si $(x, y) \in \mathcal{B}(x_0, y_0, r_0)$, $T(x, y, 0) = 0$ ailleurs. $x_0 = y_0 = 0.1$, $r_0 = 0.05$
 - $c(x, y, 0) = 5 + r(x, y)$ où $r$ est une fonction aléatoire de $\Omega$.

 À l'origine, on avait $r_0 = 0.01$ et $T = 5$ dans le cercle, mais $r_0$ était de l'ordre de $\Delta x$ (voir la partie Discrétisation) alors j'ai pris $r_0 = 0.05 = 0.01 \times 5$ et $T = 0.2 = 5 \times 5^{-2}$ dans le cercle pour compenser. One devrait avoir une simulation plus précise.

### Dynamique

Il y a combustion lorsque $T \geq 0.05$ et $c \geq 0$.

On a affaire à un modèle convection-diffusion-réaction.

#### Conservation d'énergie (température)

$$\dfrac{\partial T}{\partial t} + \overrightarrow{V} \cdot \overrightarrow{\operatorname{grad}}(T) = \mu \Delta T + R(T, c) T$$

On choisit $\mu = 0.005$, et on définit la loi de réaction $R$ comme suit :

$$R(T, c) =
\begin{cases}
10 &\text{si $T \geq 0.05$ et $c \geq 0$} \\
-5 &\text{si $T \geq 0.05$ et $c < 0 $} \\
0  &\text{si $T < 0.05$}
\end{cases}$$

#### Conservation de la masse (combustible)

$$\dfrac{\partial c}{\partial t} = \tilde{R}(T, c) c$$

avec :

$$\tilde{R} =
\begin{cases}
-100 &\text{si $T \geq 0.05$} \\
0  &\text{si $T < 0.05$}
\end{cases}$$

#### Conditions aux limites

On impose des conditions de Neumann à $T$ sur $\partial \Omega$ : $\frac{\partial T}{\partial \vec{n}} = 0$

On ne donne pas de conditions aux limites pour $c$ car son équation ne comporte pas de dérivée spatiale.

## Discrétisation

### Espace

On utilise la méthode des différences finies, donc on discrétise $\Omega$ (carré unité) en $(N+1)^2$ points, $N = 100$. On note $\Delta x = \Delta y = \frac{1}{N}$ et on obtient les points :

$$(x_i, y_j)_{0 \leq i \leq N, 0 \leq j \leq N} = (i \Delta x, j \Delta y)_{0 \leq i \leq N, 0 \leq j \leq N}$$

On note $T_{i,j} = T(x_i, y_j)$ à temps un temps donné.

$$\left. \Delta T \right|_{i,j} \approx
\frac{T_{i+1,j} - 2 T_{i,j} + T_{i-1,j}}{{\Delta x}^2}
+ \frac{T_{i,j+1} - 2 T_{i,j} + T_{i,j-1}}{{\Delta y}^2}$$

$$\left. \overrightarrow{\operatorname{grad}}(T) \right|_{i,j} =
\left. \begin{pmatrix}
\frac{\partial T}{\partial x} \\[6pt]
\frac{\partial T}{\partial y}
\end{pmatrix} \right|_{i,j}$$
Avec :
$$\left. \frac{\partial T}{\partial x} \right|_{i,j} \approx
\begin{cases}
\frac{T_{i,j} - T_{i-1,j}}{\Delta x} \text{ si $u_{i,j} \geq 0$} \\[6pt]
\frac{T_{i+1,j} - T_{i,j}}{\Delta x} \text{ si $u_{i,j} < \ 0$}
\end{cases}$$

$$\left. \frac{\partial T}{\partial y} \right|_{i,j} \approx
\begin{cases}
\frac{T_{i,j} - T_{i,j-1}}{\Delta y} \text{ si $v_{i,j} \geq 0$} \\[6pt]
\frac{T_{i,j+1} - T_{i,j}}{\Delta y} \text{ si $v_{i,j} < \ 0$}
\end{cases}$$

### Temps

On utilise la méthode d'Euler explicite. On calculera le pas de temps $\Delta t$ et le nombre de pas de temps $N_t$ lors de l'étude de stabilité.

On se donne tout de même $N_t + 1$ points $(t_n)_{0 \leq n \leq N_t} = (n \Delta t)_{0 \leq n \leq N_t}$, et on a :

$$\left. \frac{\partial T}{\partial x} \right|_{i,j}^n \approx
\frac{T_{i,j}^{n+1} - T_{i,j}^n}{\Delta t}$$

donc, pour $1 \leq i \leq N_x - 1$, $1 \leq j \leq N_y - 1$ et $1 \leq n \leq N_t$ :

$$T_{i,j}^{n+1} = T_{i,j}^n + \Delta t \, \phi(T_{i,j}^n, T_{i-1,j}^n, T_{i+1,j}^n, T_{i,j-1}^n, T_{i,j+1}^n, c_{i,j}^n)$$

### Conditions aux limites

Pour $0 \leq i \leq N_x$ :

- en $y = 0$, $T_{i,0} = T_{i,1}$
- en $y = 1$, $T_{i,N_y} = T_{i,Ny-1}$

Pour $0 \leq j \leq N_y$

 - en $x = 0$, $T_{0,j} = T_{1,j}$
 - en $x = 1$, $T_{N_y,j} = T_{N_y-1,j}$

### Mise à jour du combustible

On utilise aussi le schéma d'Euler explicite, soit pour $0 \leq i \leq N_x$, $0 \leq j \leq N_y$ et $1 \leq n \leq N_t$ :

$$c_{i,j}^{n+1} = c_{i,j}^n + \Delta t \, \tilde{\phi}(T_{i,j}^n, c_{i,j}^n)$$

### Stabilité

Le schéma d'Euler explicite est stable lorsque $\Delta t \geq \min(\Delta t_c, \Delta t_d) / 4$ où :

 - $\Delta t_c = \dfrac{h}{\max \left(\| \overrightarrow{V} \|\right)}$, $h = \min(\Delta x, \Delta y)$ (convection)
 - $\Delta t_d = \dfrac{h}{2 \mu}$ (diffusion)

## Implémentation

L'algorithme d'implémentation est le suivant.

````
Initialisation de T, c, u, v
Calcul de delta_t
Boucle temporelle (n)
    Boucle spatiale (i, j)
        Calcul de Laplacien_T[i,j] et grad_T[i,j]
        Calcul de T[i,j] au temps suivant
    Boucle (i)
        Conditions aux limites en y=0 et y=1
    Boucle (j)
        Conditions aux limites en x=0 et x=1
    Boucle spatiale (i, j)
        Calcul de c au temps suivant
    Si pour tout (i, j), T[i,j] < 0.05 alors
        Quitter la boucle temporelle
````

Le critère d'arrêt sur la température sert à éviter une simulation inutilement trop longue. En effet, si elle est inférieure au point d'inflammation partout, la loi de réaction ne peut plus se faire, donc la température ne peut plus repasser au-dessus du point d'inflammation, est donc plus aucun combustible ne sera consommé.

## Résultats

J'ai utilisé Python pour l'implémentation. En vectorisant le code de la simulation, j'arrive à faire un pas de temps en environ 400 &mu;s. Il faut entre 500 et 1000 pas de temps pour finir la simulation donc une simulation prend entre 250 ms et 500 ms. Voici la visualisation de cette simulation :

<video src="https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/simulation.mp4" controls></video>


La température du feu chute bien à t=0.3 s lorsqu'il arrive sur une zone moins dense en combustible. De plus, il suit bien le vent représenté par les flèches.

La vidéo s'arrête à 90% lorsqu'il reste du feu mais il s'agit d'un bug du côté de l'enregistrement de la vidéo car on voit la simulation se terminer si on la visualise directement lors de l'exécution.

# Optimisation

Maintenant que l'on a une simulation, on peut retirer une zone de combustible, un rectangle dans notre cas, et voir ce que cela change à l'état final de la simulation. En particulier, on veut faire un coupe-feu pas trop grand qui réduise au maximum la quantité de combustible consommée par le feu. On aimerait aussi qu'il ne soit pas trop prêt du feu, sinon il suffit de couper le combustible sous le feu, mais le résultat ainsi obtenu n'est pas intéressant.

## Problème d'optimisation

La variable que l'on cherche à optimiser est $X = (x_{min}, x_{max}, y_{min}, y_{max})$ qui définit le rectangle (possiblement vide) :

$$\{(x, y) \in \Omega \ |\ x_{min} \leq x \leq x_{max} \ \land\ y_{min} \leq y \leq y_{max}\} \subset \Omega$$

Pour évaluer la performance d'une valeur de $X$, on définit la fonction de cout suivante :

$$f(X)
= \Delta x \Delta y \sum_{i,j} \left(c_{i,j}^0 - c_{i,j}^n\right)
+ 10 |(x_{max} - x_{min})(y_{max} - y_{min})|
+ 100 \max(0, 0.2 - y_{min})$$

On cherche :

$$X^* = \argmin_{X \in \Omega} f(X)$$

## Algorithmes du simplex

Les algorithmes du simplex sont une classe d'algorithmes d'optimisation sans gradient. Pour effectuer une optimisation dans $\mathbb{R}^n$, on génère un simplex de $n+1$ points, on évalue itérativement la fonction cout en chacun de ses sommets, et on le modifie selon ces valeurs pour qu'il se rapproche du minimum recherché.

Comme ils n'utilisent pas de gradient, ces algorithmes sont peu sensibles au bruit, et ils ne demandent pas des calculs mathématiques compliqués. Cependant, ils demandent beaucoup d'appels à la fonction cout, qui nous prend de l'ordre de la demi-seconde à calculer.

### Nerlder-Mead

L'algorithme de Nelder-Mead est l'algorithme du simplex le plus connu.

L'algorithme est le suivant :

````
Initialisation du simplex (x0, ..., xn)
alpha, beta, gamma = 1, 2, 1/2
Boucle d'optimisation
    Calculer les valeurs de f en x0, ..., xn
    xbest <- le meilleur point
    xworst <- le pire point
    xbar <- barycentre des points privés de xworst
    # mouvement de réflexion
    xr = xbar + alpha * (xbar - xworst)
    Si f(xr) < f(xbest) alors
        # mouvement d'expansion
        xe = xbar + beta * (xbar - xworst)
        Si f(xe) < f(xr) alors
            xworst <- xr
    Sinon
        Si f(xr) < f(xworst) alors
            xworst <- xr
        Sinon
            # mouvement de contraction
            xc = xbar + gamma * (xworst - xbar)
            Si f(xc) < f(xworst) alors
                xworst <- xc
            Sinon
                # mouvement de réduction
                Boucle sur les sommets du simplex (xi)
                    xi <-  xbest + gamma * (xi - xbest)
````

 Lors de la majorité des itérations, le pire point est modifié. Il a été prouvé que cet algorithme pouvait converger vers un point non-stationnaire ; de plus, il est difficilement parallélisable.

### Torczon

Une variante a été proposée, l'algorithme de Torczon, accompagnée d'une preuve de convergence vers un point stationnaire ainsi que d'un algorithme parallélisé.

````
Initialisation du simplex (x0, ..., xn)
alpha, beta, gamma = 1, 2, 1/2
Boucle d'optimisation
    Calculer les valeurs de f en x0, ..., xn
    xbest <- le meilleur point
    # mouvement de réflexion
    Boucle sur i=0..n
        xri <-  (1 + alpha) * xbest - alpha * xi
    Si au moins un f(xri) < f(xi) alors
        Boucle sur i=0..n
            xi <-  (1 - beta) * xbest + beta * xri
    Sinon
        Boucle sur i=0..n
            xi <-  (1 + gamma) * xbest - gamma * xri
````

### Critères d'arrêt

J'ai choisi d'utiliser quatre critères d'arrêt :

 - le nombre maximal d'itérations, fixé à 400
 - le nombre maximal d'appels à la fonction, aussi fixé à 400
 - la valeur minimale du maximum sur `i` de la norme infini des `xi - xbest`, $\displaystyle \max_{i,j} \left| {x^*}_j - \text{simplex[i]}_j \right|$, fixée à 0.01 ($\Delta x$)
 - la valeur minimale du maximum sur `i` de la valeur absolue de `f(xi) - f(xbest)`, aussi fixée à 0.01.

### Initialisation du simplex

Le moyen le plus simple pour initialiser le simplex est de prendre une matrice aléatoire 5 par 4 dont chaque coefficient est tiré d'une loi uniforme dans [0, 1]. Bien que cela puisse fonctionner dans certains cas, on se retrouve souvent avec de très mauvaises solutions car :

 - la probabilité de n'avoir AUCUN rectangle (pour tous les points du simplex, $x_{min} > x_{max}, y_{min} > y_{max}$) est de $\left(\frac{3}{4}\right)^5 \approx 23.7\%$. Dans ce cas on ne peut espérer que l'algorithme converge vers une solution intéressante. Avoir au moins un rectangle ne garantit pas le contraire. L'expérience montre qu'on se retrouve souvent avec un rectangle qui est de hauteur ou de largeur nulle. En effet, s'il est retourné, la fonction cout le force seulement à réduire son aire.
 - ses proportions peuvent être très mauvaises. La probabilité qu'il soit dégénéré (au moins trois points alignés) est nulle, mais il peut être très fin selon une ou plusieurs dimensions. Cela pose notamment problème à Torczon car les proportions du simplex ne changent pas au cours de l'algorithme, seulement sa position, son orientation et sa taille.

Une alternative, déterministe, est de construire un simplex à partir d'un point initial $x_0$ et d'une distance $d$, tous deux donnés par l'utilisateur. Le premier sommet est le point initial $x_0$, les $n$ suivants sont donnés par $x_i = x_0 + d \, e_i, \ i = 1, \dots, n$ avec $e_i$ le vecteur dont la i-ème composante est $1$ et les autres sont nulles.

### Influence de la position de l'initilisation

La première initialisation est près du départ du feu mais seulement une partie affecte le feu, la deuxième se trouve dans le virage du feu, et la troisième à la fin du feu. Pour voir plus précisément où se trouve l'initialisation, on peut se réferrer à la vidéo de la simulation.

On fixe $d = 0.1$.

|Initialisation près du départ du feu|Initialisation au milieu du parcours du feu|Initialisation à la fin du parcours feu|
:--:|:--:|:--:
![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_x0%3D%5B0.1%2C0.3%2C0.2%2C0.4%5D.png)|![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_x0%3D%5B0.4%2C0.8%2C0.4%2C0.6%5D.png)|![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_x0%3D%5B0.4%2C0.6%2C0.6%2C0.7%5D.png)

Dans le premier cas, les deux algorithmes n'arrivent pas à se rapprocher du feu, dans le deuxième cas, les deux réduisent l'aire du rectangle mais ne parviennent pas à remonter le feu pour l'éteindre plus tôt, et dans le troisième cas, ils semblent avoir du mal à trouver une configuration qui arrête le feu car le rectangle est trop petit et une forêt dense ravive le feu juste après le coupe-feu.

Pour ce problème, la solution donnée par les deux algorithmes est donc largement influencée par l'initialisation.


### Influence de la taille initiale du simplex

On va maintenant essayer plusieurs tailles d'initialisation du simplex. D'abord un petit simplex de longueur d'arête $d = 0.01 = \Delta x$, ensuite un moyen de longueur d'arête $d = 0.1$, puis un grand de longueur d'arête $d = 0.3$.

On fixe $x_0 = (0.45, 0.85, 0.4, 0.6)$

|Petit simplex|Simplex moyen|Grand simplex|
:--:|:--:|:--:
![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_d%3D0.01.png)|![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_d%3D0.1.png)|![](https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/benchmark_d%3D0.3.png)

Alors qu'on s'attendrait à ce que le plus petit simplex ait du mal à sortir de son voisinage, c'est celui qui trouve la meilleure solution. Je suppose que c'est surtout lié au hasard, et que lors des itérations, il a obtenu un bon point, ce qui n'est pas arrivé aux deux autres. La taille initiale a peut-être une importance mais il est très improbable qu'une longueur d'arête de l'ordre de $\Delta x$ soit idéale.

### Une simulation

Voici d'abord les différents meilleurs rectangles trouvés par Nelder-Mead en partant de $x_0 = (0.4, 0.8, 0.4, 0.6)$ avec $d = 0.1$ :


<video src="https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/demo_nelder_mead_optimization_steps.mp4" controls></video>

Et la simulation de la solution trouvée :

<video src="https://raw.githubusercontent.com/abavoil/wildfire-simulation/master/report_media/demo_nelder_mead_simulation.mp4" controls></video>

La solution trouvée n'est pas optimale car on pourrait réduire $x_max$ pour avoir une aire plus petite (la partie toute à droite ne sert pas à éteindre le feu), ou encore mieux, rapprocher le coupe-feu du départ de feu pour l'arrêter plus tôt, si cela ne nécessite pas une augmentation de la hauteur qui rende la pénalisation sur l'aire trop importante.


# Conclusion

Ces méthodes peinent globalement à trouver le minimum, selon l'initialisation, on trouve des solutions très variées qui ont des valeurs de fonction cout très différentes.

Il est possible que modifier la fonction cout puisse donner des meilleurs résultats, comme jouer sur le coefficient de la pénalisation sur l'aire, ou bien rajouter des termes pour le pousser dans la bonne direction lorsque le rectangle sort du domaine de simulation ou bien est retourné (peut-être éviter si on peut fournir une bonne initialisation).

Pour tirer des conclusions définitives sur l'influence de l'initialisation du simplex, il faudrait faire des tests en plus grands nombres, mais cela prend beaucoup de temps.

Il ressort globalement que Nelder-Mead trouve plus vite une meilleure solution que Torczon, mais il ne faut pas oublier que Torczon peut être paralléliser, et donc exécuter les simulations cinq par cinq, voire plus si l'on a assez de processeurs.
