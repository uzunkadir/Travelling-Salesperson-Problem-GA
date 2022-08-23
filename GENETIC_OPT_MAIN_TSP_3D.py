
from GENETIC_OPT import Genetic_Opt

from datetime import datetime
import math
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import pandas as pd 
import numpy as np
from itertools import permutations

# anlık rotayı ve en iyi rotayı yan yana çizdirmek için 1 satır ve 2 sütundan oluşacak subplot tanımlıyoruz
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,2,1, projection="3d")
ax1 = fig.add_subplot(1,2,2, projection="3d")
fig.suptitle("Travelling Salesperson Problem \nGenetic Algorithm Solution", fontsize=16)

angle=0
count = 0
# Rotaları çizecek fonksiyon
def plotFunction(indPath, indDist, bestPath, bestDist, gen_i):
    
    global coordinates, ax, count, angle
    count += 1
    angle += 1
    # her iterasyonda 2 figure de temizleniyor
    ax.clear()
    ax1.clear()
    # anlık rotanın koordinatları
    indvCoor = coordinates.loc[indPath,:].copy()
    indvCoor = indvCoor.append(indvCoor.iloc[0,:])
    
    # en iyi rotanın kooordinatları
    bestCoor = coordinates.loc[bestPath,:].copy()
    bestCoor = bestCoor.append(bestCoor.iloc[0,:])
    
    # anlık rota çizdiriliyor
    ax.plot(indvCoor.iloc[:,0], indvCoor.iloc[:,1], indvCoor.iloc[:,2],  "-o")
    ax.plot(indvCoor.iloc[0,0], indvCoor.iloc[0,1], indvCoor.iloc[0,2], c="red",marker="o",  markersize=10)
    for i in range(len(indvCoor)): ax.text(indvCoor.iloc[i,0],indvCoor.iloc[i,1], indvCoor.iloc[i,2],str(indvCoor.index[i]), size=15 )
    ax.view_init(10,angle)
    ax.set_title("{}.Generation \n{}.individual  \nIndividual Distance: {}".format(gen_i, count, round(indDist,4)),loc='left')
    
    
    # en iyi rota çizdiriliyor
    ax1.plot(bestCoor.iloc[:,0], bestCoor.iloc[:,1], bestCoor.iloc[:,2], "-o")
    ax1.plot(bestCoor.iloc[0,0], bestCoor.iloc[0,1],bestCoor.iloc[0,2], c="red",marker="o",  markersize=10)
    ax1.set_title("Best Distance: {}".format(round(bestDist,4)),loc='left')
    for i in range(len(bestCoor)): ax1.text(bestCoor.iloc[i,0],bestCoor.iloc[i,1],bestCoor.iloc[i,2],str(bestCoor.index[i]), size=15 )
    ax1.view_init(10,angle)
    

    # her iki çizim arasında 0.01 saniye bekliyor
    plt.pause(0.1)
    


# rotada kaç nokta olacağını tanımlar 
N_points = 15

# X ve Y için -1,+1 aralığında rastgele koordinat üretir
coordinates = []
for i in range(N_points):
    x_random = random.uniform(-1, 1)
    y_random = random.uniform(-1, 1)
    z_random = random.uniform(-1, 1)
    coordinates.append([x_random, y_random, z_random])
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

 