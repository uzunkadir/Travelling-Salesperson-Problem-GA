
import pandas as pd 
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from itertools import product
import random

class Genetic_Opt:
    
    def __init__(self, 
                 pop_all,
                 fitnessFunction,
                 param_step        = [],
                 param_step_int    = 2,
                 Population_Number = 100,
                 best       = 10,
                 similarity = 50,
                 child      = 20,
                 mutation   = 30, 
                 generation = 100, 
                 fitTercih  = "max" ):


        self.fitnessFunction  = fitnessFunction 
        self.pop_all          = pop_all
        self.pop_all_len      = len(pop_all)
        
                   
        self.tested_comb  = pd.DataFrame(columns=self.pop_all.columns)
        self.besties      = []


        if param_step == []:
            for i in range(len(pop_all.columns)):
                try:
                    param_step.append((pop_all[pop_all.iloc[:,i] != pop_all.iloc[0,i]].iloc[0,i]-pop_all.iloc[0,i])*param_step_int)
                except:
                    param_step.append(2)
                    
        self.param_step   = param_step
        
        self.param_count  = len(param_step)
        
        self.Population_Number = Population_Number
        self.best       = best
        self.similarity = similarity
        self.child      = child
        self.mutation   = mutation
        self.fitTercih  = fitTercih


        param_min_max = []
        for i in range(len(param_step)):
            param_min_max.append([min(self.pop_all.iloc[:,i]), max(self.pop_all.iloc[:,i]) ])
        
        self.param_min_max = param_min_max
        
        self.similarity_pop_number = int((self.Population_Number * self.similarity ) / 100)
        self.mutasyon_pop_number   = int((self.Population_Number * self.mutation   ) / 100)            
        self.child_pop_number      = int((self.Population_Number * self.child      ) / 100)
        self.best_number           = int((self.Population_Number * self.best       ) / 100)
        self.similarity_per_box    = int(self.similarity_pop_number/self.best_number)


        self.besties    = []
        self.generation = generation

        
        self.child_pop_len      = Population_Number*child      / 100
        self.similarity_pop_len = Population_Number*similarity / 100
        self.mutasyon_len       = Population_Number*mutation   / 100
        self.similarity_pop_all = 0
        self.child_pop_all_len  = 0

############################################################################################################################################


    def child_def(self):

        pop_child_comb = []
        for k in range(self.param_count):
            pop_child_comb.append(list( set(self.besties.iloc[:,k].values) ))

        besties_list    = self.besties.iloc[:,:self.param_count].copy().values.tolist()
        tested_list     = self.tested_comb.iloc[:,:self.param_count].copy().values.tolist()

        child_pop_all = pd.DataFrame(columns=self.pop_all.columns)
        for i in product(*pop_child_comb):
            if len(set(i)) != self.param_count:
                continue
            if ((list(i) not in besties_list) and \
                (list(i) not in tested_list) ):
        
                temp           = self.pop_all.loc[self.pop_all[(self.pop_all.iloc[:,:]==list(i)).all(1)].index].copy()
                child_pop_all  = pd.concat([child_pop_all,temp])
            
            if len(child_pop_all) == self.child_pop_number:
                break
            
        # child_pop = child_pop_all.sample(n= min(self.child_pop_number, len(child_pop_all)) ).copy()
        
        child_pop = child_pop_all.copy()
        
        self.pop_all       = self.pop_all.drop(child_pop.index)
        self.population    = pd.concat([self.population,child_pop])
        self.child_pop_len = len(child_pop)
        self.child_pop_all_len = len(child_pop_all)






    def similarity_def(self):
        
        similarity_pop_all = pd.DataFrame(columns=self.pop_all.columns)
        similarity_pop     = pd.DataFrame(columns=self.pop_all.columns)
        
        
        for i in range(len(self.besties)): 
            sim_min_max = []
            for j in range(self.param_count):
                sim_min_max.append([max( self.param_min_max[j][0], self.besties.iloc[i][j]-self.param_step[j]),
                                    min( self.param_min_max[j][1], self.besties.iloc[i][j]+self.param_step[j])])
            
                
            
            similarity_pop_temp = self.pop_all.copy()
            for i in range(len(sim_min_max)):
                 similarity_pop_temp = similarity_pop_temp[((sim_min_max[i][0] <= similarity_pop_temp.iloc[:,i]) & \
                                                            (sim_min_max[i][1] >= similarity_pop_temp.iloc[:,i]) )]
                
                
            similarity_pop_temp = similarity_pop_temp.loc[np.setdiff1d(similarity_pop_temp.index.values, similarity_pop_all.index.values)].copy()
            similarity_pop_all  = pd.concat([similarity_pop_all, similarity_pop_temp])
            similarity_sample   = similarity_pop_temp.sample(n= min(len(similarity_pop_temp), self.similarity_per_box  )).copy()
            similarity_pop      = pd.concat([similarity_pop,similarity_sample])
    
    
    
        self.pop_all            = self.pop_all.drop(similarity_pop.index)
        self.population         = pd.concat([self.population,similarity_pop])
        self.similarity_pop_len = len(similarity_pop)
        self.similarity_pop_all = len(similarity_pop_all)


    def mutasyon_def(self, sample ):
    
        mutasyon = self.pop_all.sample(n= sample ).copy()
        
        self.pop_all      = self.pop_all.drop(mutasyon.index)
        self.population   = pd.concat([self.population,mutasyon])
        self.mutasyon_len = len(mutasyon)
    
    
####################################################################################################
    
    def print_result(self,gecenSure):

        print("\n",
              f"geçen süre    : {gecenSure} saniye"    ,"\n",
              f"besties       : {len(self.besties)} / {self.best_number}"                ,"\n",
              f"crossover     : {self.child_pop_len} / {self.child_pop_number} / {self.child_pop_all_len}"          ,"\n",
              f"similarity    : {self.similarity_pop_len} / {self.similarity_pop_number} / {self.similarity_pop_all}  ","\n",
              f"mutasyon      : {self.mutasyon_len} / {self.mutasyon_pop_number}" ,"\n",
              f"populasyon    : {len(self.population)-len(self.besties)}"         ,"\n",
              f"toplam ihtimal: {self.pop_all_len}"   ,"\n",
              f"kalan ihtimal : {len(self.pop_all)}"      ,"\n",
              f"test edilen   : {len(self.tested_comb)}"  ,"\n",
              "BESTIES LİSTESİ :", "\n",
              self.besties.iloc[:5].to_string(index=False), 
              end="\n")

        print("\n","="*60)




    def getFitnessSerial(self, paramlist ):
        fit = self.fitnessFunction( paramlist)
                                    
        return fit
        
        
        
    def run_serial(self):
        gen_i = 0
        t = datetime.now()

        self.population = self.pop_all.sample(n=int(min(self.Population_Number, len(self.pop_all)))).copy()

        print("mutasyon:", datetime.now()-t)
        t = datetime.now()

        self.pop_all    = self.pop_all.drop(self.population.index)
        print("drop:", datetime.now()-t)
        t = datetime.now()
        
        self.population.loc[:,"fitness"] = np.nan
        
        print("\n","="*60)

        for j in tqdm(range(len(self.population)), desc=f"{gen_i}.NESİL | Backtestler Yapılıyor: "):

            paramlist  =  self.population.iloc[j,:self.param_count].copy().values.tolist()
            fit = Genetic_Opt.getFitnessSerial(self, paramlist)
            self.population.iat[j,-1] =  fit


        self.population  = self.population.sort_values(by="fitness", ascending = False if self.fitTercih=="max" else True)
        self.besties     = self.population.iloc[:self.best_number].copy()
        self.tested_comb = self.population.copy()
        

        gecenSure = datetime.now()-t
        self.print_result(gecenSure)
 

        while (gen_i <= self.generation and len(self.pop_all) != 0) and (self.child_pop_len != 0 or self.similarity_pop_len != 0):

            t = datetime.now()
            self.population = pd.DataFrame(columns=self.pop_all.columns)
            gen_i += 1
            
            ### CHILD
            if self.child != 0 : Genetic_Opt.child_def(self)

            print("child:", datetime.now()-t)
            t = datetime.now()
        
        
            ### SIMILARITY   
            if self.similarity != 0 : Genetic_Opt.similarity_def(self)
            
            print("similarity:", datetime.now()-t)
            t = datetime.now()
            
            ### MUTASYON   
            if self.mutation != 0:
                Genetic_Opt.mutasyon_def(self, sample =int(min(self.Population_Number-self.child_pop_len-self.similarity_pop_len , 
                                                           len(self.pop_all) )))
                
            print("mutasyon:", datetime.now()-t)
            t = datetime.now()
            #########################    BACKTEST    ##############################
            self.population.loc[:,"fitness"] = np.nan

            for j in tqdm(range(len(self.population)), desc=f"{gen_i}.NESİL | Backtestler Yapılıyor: "):
                
                paramlist  = self.population.iloc[j,:self.param_count].copy().values.tolist()                

                fit =  Genetic_Opt.getFitnessSerial(self, paramlist )
                
                self.population.iat[j,-1] =  fit
                
            self.tested_comb = pd.concat([ self.tested_comb, self.population])
            # self.population  = pd.concat([ self.population,  self.besties])
            # self.population  = self.population.sort_values(by=self.fitMethod, ascending = False if self.fitTercih=="max" else True)
            self.tested_comb = self.tested_comb.sort_values(by="fitness", ascending = False if self.fitTercih=="max" else True)

            self.besties     = self.tested_comb.iloc[:self.best_number].copy()
            
            #####################################################################
    
            gecenSure = datetime.now()-t
            self.print_result(gecenSure)


        return self.besties



####################################################################################################

    def getFitnessParalel( paramlist  ):
        
        selfi = paramlist[-1]
        paramlist = paramlist[:-1]
        fit = selfi.fitnessFunction( paramlist = paramlist )
        
        return fit


    def run_paralel(self, coreCount=2):
        gen_i = 0
        t = datetime.now()
   
        self.population = self.pop_all.sample(n=min(self.Population_Number, len(self.pop_all))).copy()
        # self.pop_all    = self.pop_all.drop(self.population.index)
        self.population.loc[:,"fitness"] = np.nan
        
        print("\n","="*60)
        
        
        paramlist = self.population.iloc[:,:-1].copy().values.tolist()
        pop_fits = []
        
        for i in range(len(paramlist)): paramlist[i].append(self)
        
        with mp.Pool(processes=coreCount) as pool:
            pop_fits = list(tqdm(pool.imap(Genetic_Opt.getFitnessParalel, paramlist ), total=len(paramlist),desc=f"{gen_i}.NESİL | Backtestler Yapılıyor: " ) )
        
        
        for i in range(len(self.population)): 
            self.population.iat[i,-1] = pop_fits[i]


        self.pop_all     = self.pop_all.drop(self.population.index)
        self.population  = self.population.sort_values(by="fitness", ascending = False if self.fitTercih=="max" else True)
        self.besties     = self.population.iloc[:self.best_number].copy()
        self.tested_comb = self.population.copy()
            
        gecenSure = datetime.now()-t
        self.print_result(gecenSure)


        self.besties.to_csv("BestiesStoploss.csv")
        self.tested_comb.to_csv("TestedStoploss.csv")

        
        while (gen_i <= self.generation and len(self.pop_all) != 0) and (self.child_pop_len != 0 or self.similarity_pop_len != 0):

            t = datetime.now()
            self.population = pd.DataFrame(columns=self.pop_all.columns)
            gen_i += 1
            
            ### CHILD
            if self.child != 0 : Genetic_Opt.child_def(self)
            
            
            ### SIMILARITY   
            if self.similarity != 0 : Genetic_Opt.similarity_def(self)
            

            
            ### MUTASYON   
            if self.mutation != 0:
                Genetic_Opt.mutasyon_def(self, sample = int(min(self.Population_Number-self.child_pop_len-self.similarity_pop_len , 
                                                           len(self.pop_all) )))

            #########################    BACKTEST    ##############################
            self.population.loc[:,"fitness"] = np.nan

            
            paramlist = self.population.iloc[:,:-1].values.tolist()
            pop_fits = []
            
            for i in range(len(paramlist)): paramlist[i].append(self)
            
            with mp.Pool(processes=coreCount) as pool:
                pop_fits = list(tqdm(pool.imap(Genetic_Opt.backtest_paralel, paramlist ), total=len(paramlist),desc=f"{gen_i}.NESİL | Backtestler Yapılıyor: " ) )
            
    
            for i in range(len(self.population)): 
                self.population.iloc[i,-1] = pop_fits[i]

                                
            self.tested_comb = pd.concat([ self.tested_comb, self.population])
            self.population  = pd.concat([ self.population,  self.besties])
            self.population  = self.population.sort_values(by="fitness", ascending = False if self.fitTercih=="max" else True)
            self.besties     = self.population.iloc[:self.best_number].copy()
            
            
            self.besties.to_csv("BestiesStoploss.csv")
            self.tested_comb.to_csv("TestedStoploss.csv")      
            #####################################################################
                
            gecenSure = datetime.now()-t
            self.print_result(gecenSure)

        
        return self.besties

