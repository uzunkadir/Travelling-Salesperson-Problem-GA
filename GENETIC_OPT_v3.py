# -*- coding: utf-8 -*-
"""
Created on Sat May  7 10:50:20 2022

@author: kadir
"""



import pandas as pd 
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from itertools import product
import random
import time 

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

############################################################################################################################################


    
    def ccw(self,A,B,C):
    	return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    def intersect(self,A,B,C,D):
    	return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
    
    def controlintersect(self,coord):
        status=False
        
        for i in range(0,len(coord)-1):
            A = coord[i]
            B = coord[i+1]
            
            for j in range(i+1,len(coord)):
                C = coord[j-1]
                D = coord[j]
            
                if self.intersect(A,B,C,D):
                    status=True
                    break
            else:
                continue
            break
                
        
        
        return status


    def child_def(self):
        
        besties_list    = self.besties.iloc[:,:-1].copy().values.tolist()
        tested_list     = self.tested_comb.iloc[:,:-1].copy().values.tolist()
        pop_list        = self.population.copy().values.tolist()    
        
        # besties listesinin her sütunundaki tekrarlanan değerlerini siler
        pop_child_comb = []
        for k in range(self.pathLen):
            pop_child_comb.append(list( set(self.besties.iloc[:,k].values) ))
        child_pop = []
        
        t = datetime.now()
        
        while True:
            # crossover süresi 5 saniyeden fazla sürerse veya 
            # istenen sayıya ulaşılırsa döngüyü durdurur
            if ((len(child_pop)) >= self.child_pop_number) or\
                (datetime.now()-t).total_seconds() > 5: 
                    break
            
            childPath = [0]
            state = True
            
            while state:
                for i in range(1,len(pop_child_comb)):

                    a = random.choice(pop_child_comb[i])
                    

                    if (a not in childPath) :  
                        childPath.append(a)
                        
                    if len(childPath) == self.pathLen:
                        state = False
                        break
            
            coor = self.coordinates.loc[childPath,:].copy().values.tolist()          
                        
            # Hamilton döngüsünde döngünün tersi de kendisidir.
            # childPathR = [0]+childPath[1:][::-1]
            
            if ((childPath  not in besties_list) and \
                (childPath  not in tested_list)  and \
                (childPath  not in child_pop)    and \
                (childPath  not in pop_list)     ):
                
                child_pop.append(childPath)

        self.child_pop = pd.DataFrame(child_pop)
        
        # crossover ile üretilen çocukları populasyona ekler
        self.population    = pd.concat([self.population,self.child_pop],ignore_index=True)
        self.child_pop_len = len(child_pop)



    def mutasyon_def(self, sample ):
        
        mutation = []
        
        # başlangıç rotamız 0'dan nokta sayısına kadar artan sayılar
        path = list(range(self.pathLen))
        
        besties_list    = self.besties.iloc[:,:-1].copy().values.tolist()
        tested_list     = self.tested_comb.iloc[:,:-1].copy().values.tolist()
        pop_list        = self.population.copy().values.tolist()
        
        t= datetime.now()
        state = True
        while state:
            for i in range(sample):
                if len(mutation)>=sample : 
                    state=False
                    break

                # başlangıç noktası hariç diğer noktaları shuffle yapar
                randomPath  = [0] + random.sample(path[1:], len(path[1:]))
                
                coor = self.coordinates.loc[randomPath,:].copy().values.tolist()

                #  Hamilton döngüsünde döngünün tersi de kendisidir.
                # randomPathR = [0] + randomPath[1:][::-1]

                if ((randomPath  not in besties_list) and \
                    (randomPath  not in tested_list)  and \
                    (randomPath  not in mutation)     and \
                    (randomPath  not in pop_list)     and \
                    not(self.controlintersect(coor)) ):
                    mutation.append(randomPath)
                    
            print(len(mutation))
        mutation = pd.DataFrame(mutation)

        # mutasyon ile üretilen bireyleri populasyona ekler
        self.population   = pd.concat([self.population,mutation], ignore_index=True)
        self.mutasyon_len = len(mutation)
        
        
####################################################################################################
    
    def print_result(self,gecenSure):

        print("\n",
              f"geçen süre    : {gecenSure} "    ,"\n",
              f"besties       : {len(self.besties)}","\n",
              f"crossover     : {self.child_pop_len} / {self.child_pop_number} "          ,"\n",
              f"mutasyon      : {self.mutasyon_len} / {self.mutasyon_pop_number}" ,"\n",
              f"populasyon    : {len(self.population)}"         ,"\n",
              f"test edilen   : {len(self.tested_comb)}"  ,"\n",
              # "BESTIES LİSTESİ :", "\n",
              # self.besties.iloc[:5].to_string(index=False), 
              end="\n")

        print("\n","="*60)




    def calculateDistance(self, path ):
        
        if len(self.coordinates.columns)==2:
            #öklid uzaklığı
            def dist(x1,y1,x2,y2): return ((x2-x1)**2 + (y2-y1)**2)**0.5
        
        if len(self.coordinates.columns)==3:
            #öklid uzaklığı
            def dist(x1,y1,z1,x2,y2,z2): return ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
        
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
        
        
        
    def run_serial(self):
        self.gen_i = 0
        t = datetime.now()
        
        ####################################################################################################################
        ############################   0. GENERATION        ################################################################
        ####################################################################################################################
        
        self.population = pd.DataFrame(columns=range(self.pathLen))

        # başlangıç populasyonunu %100 mutasyon olacak şekilde oluşturur
        Genetic_Opt.mutasyon_def(self,self.Population_Number)
        self.population.loc[:,"fitness"] = np.nan
        
        print("\n","="*60)
        
        # en iyi rota başlangıçta varsayılan olarak ardışık sıradır
        self.bestPath = list(range(self.pathLen))
        self.bestDist = float("inf")
        
        for j in tqdm(range(len(self.population)), desc=f"{self.gen_i}.NESİL: "):
            
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

            for j in tqdm(range(len(self.population)), desc=f"{self.gen_i}.NESİL: "):
                
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


            #####################################################################
    
            gecenSure = datetime.now()-t
            self.print_result(gecenSure)


        return self.besties


