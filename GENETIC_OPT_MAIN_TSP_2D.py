
from GENETIC_OPT import Genetic_Opt

from datetime import datetime
import math
import random 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from itertools import permutations

# anlık rotayı ve en iyi rotayı yan yana çizdirmek için 1 satır ve 2 sütundan oluşacak subplot tanımlıyoruz
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
fig.suptitle("Travelling Salesperson Problem \nGenetic Algorithm Solution", fontsize=16)


count = 0
# Rotaları çizecek fonksiyon
def plotFunction(indPath, indDist, bestPath, bestDist, gen_i):
    
    global coordinates, ax, count
    count += 1
    
    # her iterasyonda 2 figure de temizleniyor
    ax[0].clear()
    ax[1].clear()
    
    # anlık rotanın koordinatları
    indvCoor = coordinates.loc[indPath,:].copy()
    indvCoor = indvCoor.append(indvCoor.iloc[0,:])
    
    # en iyi rotanın kooordinatları
    bestCoor = coordinates.loc[bestPath,:].copy()
    bestCoor = bestCoor.append(bestCoor.iloc[0,:])

    # anlık rota çizdiriliyor
    ax[0].plot(indvCoor.iloc[:,0], indvCoor.iloc[:,1],"-o")
    ax[0].plot(indvCoor.iloc[0,0], indvCoor.iloc[0,1], c="red",marker="o",  markersize=10)
    for i in range(len(indvCoor)): ax[0].annotate(str(indvCoor.index[i]),(indvCoor.iloc[i,0],indvCoor.iloc[i,1]) , fontsize=15 )
    
    ax[0].set_title("{}.Generation \n{}.individual  \nIndividual Distance: {}".format(gen_i, count, round(indDist,4)),loc='left')
    
    # en iyi rota çizdiriliyor
    ax[1].plot(bestCoor.iloc[:,0], bestCoor.iloc[:,1],"-o")
    ax[1].plot(bestCoor.iloc[0,0], bestCoor.iloc[0,1], c="red",marker="o",  markersize=10)
    ax[1].set_title("Best Distance: {}".format(round(bestDist,4)),loc='left')
    for i in range(len(bestCoor)): ax[1].annotate(str(bestCoor.index[i]),(bestCoor.iloc[i,0],bestCoor.iloc[i,1]) , fontsize=15 )

    
    # her iki çizim arasında 0.01 saniye bekliyor
    plt.pause(0.01)
    
    

# rotada kaç nokta olacağını tanımlar 
N_points = 10

# X ve Y için -1,+1 aralığında rastgele koordinat üretir
coordinates = []
for i in range(N_points):
    x_random = random.uniform(-1, 1)
    y_random = random.uniform(-1, 1)
    coordinates.append([x_random, y_random])
coordinates = pd.DataFrame(coordinates)
        
# plt.plot(coordinates.iloc[:,0], coordinates.iloc[:,1])
# plt.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1])
# plt.show()

# pop_all = pd.DataFrame(permutations(range(N_points)))


t = datetime.now()
optimizing = Genetic_Opt(coordinates       = coordinates,
                         plotFunction      = plotFunction,
                         Population_Number = 100,
                         best              = 10, 
                         child             = 70,
                         mutation          = 30, 
                         generation        = 100,
                         fitPrefer         = "min")

optimizing.run_serial()
# besties = optimizing.run_paralel(coreCount=2)

print("TOPLAM OPTİMİZASYON SÜRESİ:", datetime.now()-t)
# besties = optimizing.besties
# coord = coordinates.loc[besties.iloc[0,:-1],:]
# coord = coord.append(coord.iloc[0,:])
# plt.plot(coord.iloc[:,0], coord.iloc[:,1])
# plt.scatter(coord.iloc[:,0], coord.iloc[:,1])
# plt.plot(coord.iloc[0,0],coord.iloc[0,1], marker="o",  markersize=10)

 