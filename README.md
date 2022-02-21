# Travelling_Sales_Problem_Genetic_Algorithm
Gezgin Satıcı Problemi'nin Genetik Algoritmalarla Çözümü

Gezgin Satıcı Problemi 

Bir seyyar satıcı olduğunuzu düşünün. A şehrindesiniz ve n tane şehire uğramak istiyorsunuz. Her bir şehire yalnızca 1 kez uğrayarak tüm şehirleri dolaşıp tekrar A şehrine gelmek istiyorsunuz. En kısa rota hangisidir? 

Ek bilgi: Bir döngü bir diyagramın tüm düğümlerine bir sefer uğrayıp başlanan noktaya geri dönüyorsa bu döngüye Hamilton Döngüsü denir. 

Hem kodlayarak hem de problemin basamaklarını anlayarak ilerleyelim. Bu problemi genetik algoritmalarla çözeceğimiz için önce bir sınıf oluşturup metotlarını tanımlamakla başlayalım. 

````
import pandas as pd 
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from itertools import product
import random

class Genetic_Opt:
    
    def __init__(self, 
                 coordinates,             # noktaların koordinatlarını tutan liste
                 plotFunction,            # anlık olarak rotayı çizdirmek için gerekli fonksiyon
                 Population_Number = 100, # her nesildeki olması beklenen populasyon sayısı
                 best              = 10,  # her nesilin yüzde kaçı en iyiler listesini oluşturacak
                 child             = 50,  # her nesilin yüzde kaçı crossover ile oluşacak
                 mutation          = 50,  # her nesilin yüzde kaçı mutasyonla oluşacak
                 generation        = 100, # işlemi sonlandırmak için nesil limiti
                 fitPrefer         = "min" ): # en az uzunluğa sahip rotayı seçmek için parametre

        self.plotFunction  = plotFunction 
        self.coordinates   = coordinates
        self.pathLen       = len(coordinates)
                   
        self.tested_comb  = pd.DataFrame(columns=range(self.pathLen))
        self.besties      = pd.DataFrame(columns=range(self.pathLen))

        self.Population_Number = Population_Number
        
        self.best       = best
        self.child      = child
        self.mutation   = mutation
        self.fitPrefer  = fitPrefer

        self.mutasyon_pop_number   = int((self.Population_Number * self.mutation   ) / 100)            
        self.child_pop_number      = int((self.Population_Number * self.child      ) / 100)
        self.best_number           = int((self.Population_Number * self.best       ) / 100)


        self.generation = generation

        self.child_pop_len      = Population_Number*child      / 100
        self.mutasyon_len       = Population_Number*mutation   / 100
        self.child_pop_all_len  = 0
````

Çaprazlama yapacak fonksiyon aşağıdaki gibi olacak. 

````
    def child_def(self):
        
        # besties listesinin her sütunundaki tekrarlanan değerlerini siler
        pop_child_comb = []
        for k in range(self.pathLen):
            pop_child_comb.append(list( set(self.besties.iloc[:,k].values) ))
        
        
        besties_list    = self.besties.iloc[:,:-1].copy().values.tolist()
        tested_list     = self.tested_comb.iloc[:,:-1].copy().values.tolist()
        
        # üretilebilecek tüm çocuklar üzerinden döngüyle geçer
        child_pop_all = []
        for childPath in product(*pop_child_comb):
            
            #  üretilen rotada tekrarlanan istasyonlar varsa atlar
            if len(set(childPath)) != self.pathLen or childPath[0]!=0:
                continue
            
            # üretilen rota besties listesinde ve test edilenler listesinde yoksa populasyona ekler
            if ((list(childPath) not in besties_list) and \
                (list(childPath) not in tested_list) ):
                child_pop_all.append(childPath)
            
            # crossover için koyulan limite eriştiğinde döngüyü erken durdurur
            if len(child_pop_all) == self.child_pop_number:
                break
            
        child_pop = pd.DataFrame(child_pop_all)
        
        # crossover ile üretilen çocukları populasyona ekler
        self.population    = pd.concat([self.population,child_pop],ignore_index=True)
        self.child_pop_len = len(child_pop)
        self.child_pop_all_len = len(child_pop_all)
````

Mutasyon yapacak fonksiyon aşağıdaki gibi olacak

````
    def mutasyon_def(self, sample ):
        
        mutation = []
        # başlangıç rotamız 0'dan nokta sayısına kadar artan sayılar
        path = list(range(self.pathLen))
        
        for i in range(sample):
            # başlangıç noktası hariç diğer noktaları shuffle yapar
            randomPath = random.sample(path[1:], len(path[1:]))
            mutation.append([0]+randomPath)
            
        mutation = pd.DataFrame(mutation)

        # mutasyon ile üretilen bireyleri populasyona ekler
        self.population   = pd.concat([self.population,mutation], ignore_index=True)
        self.mutasyon_len = len(mutation)
````

Her nesil sonunda sonuçları çizdirmek için bir print fonksiyonumuz olsun. 

````
    def print_result(self,gecenSure):

        print("\n",
              f"geçen süre    : {gecenSure} "    ,"\n",
              f"besties       : {len(self.besties)}","\n",
              f"crossover     : {self.child_pop_len} / {self.child_pop_number} / {self.child_pop_all_len}"          ,"\n",
              f"mutasyon      : {self.mutasyon_len} / {self.mutasyon_pop_number}" ,"\n",
              f"populasyon    : {len(self.population)-len(self.besties)}"         ,"\n",
              f"test edilen   : {len(self.tested_comb)}"  ,"\n",
              "BESTIES LİSTESİ :", "\n",
              self.besties.iloc[:5].to_string(index=False), 
              end="\n")
        print("\n","="*60)
````

Her güzergah için toplam mesafeyi hesaplayacak fonksiyon şu şekilde olacak. 
````
    def calculateDistance(self, path ):
        
        # öklid uzaklığı
        def dist(x1,y1,x2,y2): return ((x2-x1)**2 + (y2-y1)**2)**0.5
        
        # koordinatları istenen rotada sıraya sokar
        self.coor = self.coordinates.loc[path,:].copy()
        
        # kapalı bir poligon olması için ilk noktayı listenin sonuna ekler
        self.coor = self.coor.append(self.coor.iloc[0,:])
        
        # rotanın toplam uzunluğunu bulur
        self.distance = 0
        for i in range(1,len(self.coor)):
            dist_index = dist(*self.coor.iloc[i,:], *self.coor.iloc[i-1,:])
            self.distance += dist_index
                                                
        return self.distance
````

Fonksiyonları nesiller boyunca çalıştıracak bir fonksiyon daha yazalım. 
````
    def run_serial(self):
        self.gen_i = 0
        t = datetime.now()
        
        ######################################################################################################
        ############################   0. GENERATION        ##################################################
        ######################################################################################################
        
        self.population = pd.DataFrame(columns=range(self.pathLen))

        # başlangıç populasyonunu %100 mutasyon olacak şekilde oluşturur
        Genetic_Opt.mutasyon_def(self,self.Population_Number)
        self.population.loc[:,"fitness"] = np.nan
        
        print("\n","="*60)
        
        # en iyi rota başlangıçta varsayılan olarak ardışık sıradır
        self.bestPath = list(range(self.pathLen))
        self.bestDist = float("inf")
        
        for j in tqdm(range(len(self.population)), desc=f"{self.gen_i}.NESİL | Fitnesslar Hesaplanıyor: "):
            
            # listedeki her rotanın uzunluklarını hesaplar 
            # yani populasyondaki her bireyin fitness değerlerini bulur
            path  =  self.population.iloc[j,:-1].copy().values.tolist()
            fit = Genetic_Opt.calculateDistance(self, path)
            self.population.iat[j,-1] =  fit
            
            
            # en iyi rotadan daha iyi bir rota bulursa en iyi rotayı günceller
            if fit < self.bestDist:
                self.bestPath =  self.population.iloc[j,:-1].copy().values.tolist()
                self.bestDist = fit
            
            # rotaları çizdirmek için plot fonksiyonu çağrılır
            self.plotFunction( path, self.distance, self.bestPath, self.bestDist, self.gen_i)
            
        # populasyon listesi fitness değerine göre azdan çoğa sıralanır.
        self.population  = self.population.sort_values(by="fitness", ascending = False if self.fitPrefer=="max" else True)
        
        # populasyonun en iyi n bireyi seçilir.
        self.besties     = self.population.iloc[:self.best_number].copy()
        self.tested_comb = self.population.copy()
        
        ####################################################################################################################
        
        ####################################################################################################################
        ####################################       ITERATIONS         ######################################################
        ####################################################################################################################
        
        gecenSure = datetime.now()-t
        self.print_result(gecenSure)
 
        while (self.gen_i <= self.generation) :

            t = datetime.now()

            self.population = pd.DataFrame(columns=range(self.pathLen))
            self.gen_i += 1

            ### CHILD
            Genetic_Opt.child_def(self)

            ### MUTASYON   
            Genetic_Opt.mutasyon_def(self, sample =int(self.Population_Number-self.child_pop_len ))

            self.population.loc[:,"fitness"] = np.nan

            for j in tqdm(range(len(self.population)), desc=f"{self.gen_i}.NESİL | Fitnesslar Hesaplanıyor: "):
                
                path  = self.population.iloc[j,:-1].copy().values.tolist()                
                fit =  Genetic_Opt.calculateDistance(self, path )
                self.population.iat[j,-1] =  fit
                    
                if fit < self.bestDist:
                    self.bestPath =  self.population.iloc[j,:-1].copy().values.tolist()
                    self.bestDist = fit

                self.plotFunction( path, self.distance, self.bestPath, self.bestDist, self.gen_i)

                
            self.tested_comb = pd.concat([ self.tested_comb, self.population])
            self.tested_comb = self.tested_comb.sort_values(by="fitness", ascending = False if self.fitPrefer=="max" else True)

            self.besties     = self.tested_comb.iloc[:self.best_number].copy()

            self.bestPath    = self.besties.iloc[0,:-1]
            self.bestDist    = self.besties.iloc[0,-1]


            ####################################################################################################################
    
            gecenSure = datetime.now()-t
            self.print_result(gecenSure)

        return self.bestieS
````

Genetik algoritma sınıfımızı çalıştıracak başka bir python dosyasında noktaları üretip üzerinden algoritmayı çalıştıralım. 
````

from GENETIC_OPT import Genetic_Opt

from datetime import datetime
import math
import random 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from itertools import permutations

# anlık rotayı ve en iyi rotayı yan yana çizdirmek için 1 satır ve 2 sütundan oluşacak subplot tanımlıyoruz
fig, ax = plt.subplots(nrows=1, ncols=2)
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
    ax[0].set_title("{}.Generation \n{}.individual  \nIndividual Distance: {}".format(gen_i, count, round(indDist,4)),loc='left')
    
    # en iyi rota çizdiriliyor
    ax[1].plot(bestCoor.iloc[:,0], bestCoor.iloc[:,1],"-o")
    ax[1].plot(bestCoor.iloc[0,0], bestCoor.iloc[0,1], c="red",marker="o",  markersize=10)
    ax[1].set_title("Best Distance: {}".format(round(bestDist,4)),loc='left')
    
    # her iki çizim arasında 0.001 saniye bekliyor
    plt.pause(0.001)
    
    

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
                         Population_Number = 50,
                         best              = 10,
                         child             = 50,
                         mutation          = 50, 
                         generation        = 50,
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
````

































