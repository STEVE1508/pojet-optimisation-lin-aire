#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:49:36 2020

@author: steve
"""


'''
J'ai testé mon programme avec l'exercice 4 du TD 2 qui cherche à maximiser
la fonction f(x,y)=3x - 2y (ce maximum vaut 11 ).
Mon programme Fourrier Motzkin étant conçu pour minimiser les fonctions affines,
j'ai donc minimisé -f pour répondre à la question.Tout ceci se trouve dans la 
la fonction test() sans arguments tout en bas. 
Pour le faire, il suffit de compiler le script tout entier et de saisir test()
dans la console. Elle rend l'optimum , le point en lequel cet optimum est atteint,
les différents intervalles pour chaque coordonnée ainsi que les matrices de 
descente de la méthode de FM'.Les fonctions FM1,FM2 et FM3 sont celles faites en cours.
'''


import numpy as np
def FM1(A,b):
    b1 = np.copy(b.reshape(len(b),1))
    B = np.concatenate((A,-b1),axis=1)
    return B

   
def FM2(B):
    C = np.copy(B)
    for i in range(len(C)):
        if C[i][0] != 0:
            C[i] /= np.abs(C[i][0])
 
    return C

def FM3(C):
    (p,q) = np.shape(C)
    E = np.array([]).reshape(0,q-1)
    G = np.array([]).reshape(0,q-1)
    D = np.array([]).reshape(0,q-1)
    for i in range(p):
        ligne = np.copy(C[i][1:]).reshape(1,q-1)
        if C[i][0] > 0:
            D = np.concatenate((D,-ligne))
        elif C[i][0] < 0:
            G = np.concatenate((G,ligne))
        else:
            E = np.concatenate((E,ligne))
    for g in G:
        for d in D:
            E = np.concatenate((E,(g-d).reshape(1,q-1)))
    
    return E

def FourierMotzkin(A,b,c):
    n = len(c)
    E=FM1(A,b)
    E=FM2(E)
    (p,q)=np.shape(E)
    
    # Boucle pour stocker les matrices de descente
    mat = [E]
    for k in range(q-2):
        E=FM3(E)
        E=FM2(E)
        mat= [E] + mat
    # La liste "mat" contient les matrices de descente
        
    #les intervalles de fluctuation des coordonnées, [1,1] pour la constante
    intervalles = [[1,1]]
    
    sol = [] # le vecteur solution
    
    # Boucle de la remontée 
    x = ''
    for i in range(n):
        E = mat[i]
        p,q=np.shape(E)
        mini=-10**12  # equivalent de - l'infini
        maxi= 10**12  # equivalent de + l'infini
        
        for k in range(p):
            if E[k,0]>0:
                M = 0
                for j in range(1,q):
                    I = intervalles[j-1]
                    if -E[k,j] > 0 : M += -E[k,j]*I[0]
                    else : M += -E[k,j]*I[1]
                maxi = min(maxi,M)
            elif E[k,0]<0:
                m = 0
                for j in range(1,q):
                    I = intervalles[j-1]
                    if E[k,j] >0 : m += E[k,j]*I[1]
                    else : m += E[k,j]*I[0]
                mini = max(mini,m)
        intervalles = [[mini,maxi]] + intervalles
           
    for i in range(n):
        I = intervalles[i]
        if I[0] > I[1]:
            x=f"Polyèdre vide car le mini est supérieur au maxi pour la {i+1}ème coordonnée"
        else : 
            if c[i]>0 : sol.append(min(I))
            else : sol.append(max(I))
    sol = np.array(sol)
    if len(x)==0 : 
        print(f"Le minimum de la fonction vaut : {c.dot(sol)}")
        print(f"Le vecteur solution est : {np.array(sol)}")
        print("Les intervalles domaines des coordonnées :")
        for I in intervalles[:-1]:
            print(I)
        print("Les matrices de descente sont :")
        for i in mat:
            print(np.array(i))
            
    else : return x

def test():
    
    A=np.array([[1.,0.],[-1.,2],[1.,1.],[0.,-1.]])
    b=np.array([1.,2.,3.,4.])
    c=np.array([-3.,2.])
    
    return FourierMotzkin(A,b,c)

