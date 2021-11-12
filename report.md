---
title: "Optimization du placement et du dimensionnement d'un coupe-feu"
author: "Antonin Bavoil"
date: "12 novembre 2021"
---

# Introduction

L'objectif de ce projet est de coupler simulation numérique et optimisation : il faut minimiser les dégats d'un feu de fôret en défrichant un rectangle de forêt.
Dans un premier temps, on va modéliser le feu de forêt par un couple d'équations différentielles partielles non-linéaires du second ordre à une dimension de temps et deux d'espace. Pour résoudre ce système d'équation, on va utiliser la méthode des différences finies (espace) et la méthode d'Euler explicite (temps). Dans un second temps, on s'intéressera à l'algorithme du simplex pour optimiser la position et la taille d'un rectangle de coupe-feu.

# Simulation

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
 - $T(x, y, 0) = 5$ si $(x, y) \in \mathcal{B}(x_0, y_0, r_0)$, $T(x, y, 0) = 0$ ailleurs. $x_0 = y_0 = 0.1$, $r_0 = 0.01$
 - $c(x, y, 0) = 5 + r(x, y)$ où $r$ est une fonction aléatoire de $\Omega$.

### Dynamique

Il y a combustion lorsque $T \geq 0.05$ et $c \geq 0$.

On a affaire à un modèle convection-diffusion-réaction.

#### Conservation d'énergie (température)

$$\dfrac{\partial T}{\partial t} + \overrightarrow{V} \cdot \overrightarrow{\operatorname{grad}}(T) = \mu \Delta T + \operatorname{R}(T, c) T$$

On choisit $\mu = 0.005$, et on définit la loi de réaction $\operatorname{R}$ comme suit :

$$\operatorname{R}(T, c) =
\begin{cases}
10 &\text{si $T \geq 0.05$ et $c \geq 0$} \\
-5 &\text{si $T \geq 0.05$ et $c < 0 $} \\
0  &\text{si $T < 0.05$}
\end{cases}$$

#### Conservation de la masse (combustible)

$$\dfrac{\partial c}{\partial t} = \tilde{\operatorname{R}}(T, c) c$$

avec :

$$\tilde{\operatorname{R}} =
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

On se donne tout de même $N_t + 1$ points $(t_n)_{0 \leq n \leq N_t} = (n \Delta t)_{0 \leq n \leq N_t}$, et on a

$$\left. \frac{\partial T}{\partial x} \right|_{i,j}^n \approx
\frac{T_{i,j}^{n+1} - T_{i,j}^n}{\Delta t}$$

donc, pour $1 \leq i \leq N_x - 1$, $1 \leq j \leq N_y - 1$ et $1 \leq n \leq N_t$

$$T_{i,j}^{n+1} = T_{i,j}^n + \Delta t \, \phi(T_{i,j}^n, T_{i-1,j}^n, T_{i+1,j}^n, T_{i,j-1}^n, T_{i,j+1}^n, c_{i,j}^n)$$

### Conditions aux limites

Pour $0 \leq i \leq N_x$ :
- en $y = 0$, $T_{i,0} = T_{i,1}$
- en $y = 1$, $T_{i,N_y} = T_{i,Ny-1}$

Pour $0 \leq j \leq N_y$
 - en $x = 0$, $T_{0,j} = T_{1,j}$
 - en $x = 1$, $T_{N_y,j} = T_{N_y-1,j}$

### Mise à jour du combustible

On utilise aussi le schéma d'Euler explicite, soit pour $0 \leq i \leq N_x$, $0 \leq j \leq N_y$ et $1 \leq n \leq N_t$

$$c_{i,j}^{n+1} = c_{i,j}^n + \Delta t \, \tilde{\phi}(T_{i,j}^n, c_{i,j}^n)$$

### Stabilité

Le schéma d'Euler explicite est stable lorsque $\Delta t \geq \min(\Delta t_c, \Delta t_d) / 4$ où :
 - $\Delta t_c = \dfrac{h}{\max \left(\| \overrightarrow{V} \|\right)}$, $h = \min(\Delta x, \Delta y)$ (convection)
 - $\Delta t_d = \dfrac{h}{2 \mu}$ (diffusion)

## Implémentation

L'algorithme d'implémentation est le suivant.

```
Initialisation de T, c, u, v
Calcul de delta_t
Boucle temporelle (n)
    Boucle spatiale (i, j)
        Calcul de Laplacien_T et grad_T
        Calcul de T au temps suivant
    Boucle (i)
        Conditions aux limites en y=0 et y=1
    Boucle (j)
        Conditions aux limites en x=0 et x=1
    Boucle spatiale (i, j)
        Calcul de c au temps suivant
    Si pour tout (i, j), T[i,j] < 0.05 alors
        Quitter la boucle temporelle
```

Le critère d'arrêt sur le température sert à ne pas avoir de simulation inutilement trop longue. En effet, si elle est inférieure au point d'inflammation partout, la loi de réaction ne peut plus se faire, donc la température ne peut plus repasser au dessus du point d'inflammation, est donc plus aucun combustible ne sera consommé.

# Optimisation

## Algorithmes du simplex

### Nerlder-Mead

### Torczon

## Implémentation

# Conclusion