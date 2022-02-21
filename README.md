# Travelling_Sales_Problem_Genetic_Algorithm
Gezgin Satıcı Problemi'nin Genetik Algoritmalarla Çözümü

Gezgin Satıcı Problemi 

Bir seyyar satıcı olduğunuzu düşünün. A şehrindesiniz ve n tane şehire uğramak istiyorsunuz. Her bir şehire yalnızca 1 kez uğrayarak tüm şehirleri dolaşıp tekrar A şehrine gelmek istiyorsunuz. En kısa rota hangisidir? 

Ek bilgi: Bir döngü bir diyagramın tüm düğümlerine bir sefer uğrayıp başlanan noktaya geri dönüyorsa bu döngüye Hamilton Döngüsü denir. 

Hem kodlayarak hem de problemin basamaklarını anlayarak ilerleyelim. Bu problemi genetik algoritmalarla çözeceğimiz için önce bir sınıf oluşturup metotlarını tanımakla başlayalım. 

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
                 coordinates,
                 plotFunction,
                 Population_Number = 100,
                 best              = 10,
                 child             = 50,
                 mutation          = 50, 
                 generation        = 100, 
                 fitPrefer         = "min" ):
                 
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

      pop_child_comb = []
      for k in range(self.pathLen):
          pop_child_comb.append(list( set(self.besties.iloc[:,k].values) ))

      besties_list    = self.besties.iloc[:,:-1].copy().values.tolist()
      tested_list     = self.tested_comb.iloc[:,:-1].copy().values.tolist()

      child_pop_all = []

      for childPath in product(*pop_child_comb):

          if len(set(childPath)) != self.pathLen or childPath[0]!=0:
              continue

          if ((list(childPath) not in besties_list) and \
              (list(childPath) not in tested_list) ):
              child_pop_all.append(childPath)

          if len(child_pop_all) == self.child_pop_number:
              break

      child_pop = pd.DataFrame(child_pop_all)

      self.population    = pd.concat([self.population,child_pop],ignore_index=True)
      self.child_pop_len = len(child_pop)
      self.child_pop_all_len = len(child_pop_all)
````

Mutasyon yapacak fonksiyon aşağıdaki gibi olacak

````
  def mutasyon_def(self, sample ):
      mutation = []
      path = list(range(self.pathLen))
      for i in range(sample):
          randomPath = random.sample(path[1:], len(path[1:]))
          mutation.append([0]+randomPath)
      mutation = pd.DataFrame(mutation)
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
        
        # Öklid uzaklığı
        def dist(x1,y1,x2,y2): return ((x2-x1)**2 + (y2-y1)**2)**0.5

        self.coor = self.coordinates.loc[path,:].copy()
        self.coor = self.coor.append(self.coor.iloc[0,:])
        
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
        
        self.population = pd.DataFrame(columns=range(self.pathLen))

        Genetic_Opt.mutasyon_def(self,self.Population_Number)
        
        self.population.loc[:,"fitness"] = np.nan
        
        print("\n","="*60)
        
        self.bestPath = list(range(self.pathLen))
        self.bestDist = float("inf")

        for j in tqdm(range(len(self.population)), desc=f"{self.gen_i}.NESİL | Fitnesslar Hesaplanıyor: "):

            path  =  self.population.iloc[j,:-1].copy().values.tolist()
            fit = Genetic_Opt.calculateDistance(self, path)
            self.population.iat[j,-1] =  fit
            
            
            if fit < self.bestDist:
                self.bestPath =  self.population.iloc[j,:-1].copy().values.tolist()
                self.bestDist = fit
            
            self.plotFunction( path, self.distance, self.bestPath, self.bestDist, self.gen_i)
            
            
        self.population  = self.population.sort_values(by="fitness", ascending = False if self.fitPrefer=="max" else True)
        self.besties     = self.population.iloc[:self.best_number].copy()
        self.tested_comb = self.population.copy()
        

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

            #########################    BACKTEST    ##############################
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

            self.bestPath = self.besties.iloc[0,:-1]
            self.bestDist = self.besties.iloc[0,-1]

            gecenSure = datetime.now()-t
            self.print_result(gecenSure)
````































