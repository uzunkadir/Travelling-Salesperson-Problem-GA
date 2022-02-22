

if __name__ == '__main__':
    

    from GENETIC_OPT import Genetic_Opt
    
    from datetime import datetime
    import math
    import random 
    import matplotlib.pyplot as plt
    import pandas as pd 
    import numpy as np
    from itertools import permutations

    # öklid uzaklığı
    def dist(x1,y1,x2,y2): return ((x2-x1)**2 + (y2-y1)**2)**0.5
        
    
    def distAll(index):
        global coordinates 
        coor = coordinates.loc[index,:]
        distance = 0
        for i in range(1,len(coor)):
            dist_index = dist(*coor.iloc[i,:], *coor.iloc[i-1,:])
            distance += dist_index
            
        dist_index = dist(*coor.iloc[0,:], *coor.iloc[-1,:])
        distance += dist_index
        
        return distance


    N_points = 8
    coordinates = []
    for i in range(N_points):
        x_random = random.uniform(-1, 1)
        y_random = random.uniform(-1, 1)
        coordinates.append([x_random, y_random])
    coordinates = pd.DataFrame(coordinates)
            
    # plt.plot(coordinates.iloc[:,0], coordinates.iloc[:,1])
    plt.scatter(coordinates.iloc[:,0], coordinates.iloc[:,1])
    plt.show()
    
    pop_all = pd.DataFrame(permutations(range(N_points)))
    
    

    t = datetime.now()
    optimizing = Genetic_Opt(pop_all           = pop_all, 
                             fitnessFunction   = distAll,
                             param_step        = [],
                             Population_Number = 100,
                             best              = 5,
                             similarity        = 0,
                             child             = 50,
                             mutation          = 50, 
                             generation        = 100,
                             fitTercih         = "min")
    
    besties = optimizing.run_serial()
    # besties = optimizing.run_paralel(coreCount=2)
    
    print("TOPLAM OPTİMİZASYON SÜRESİ:", datetime.now()-t)

    coord = coordinates.loc[besties.iloc[0,:-1],:]
    coord = coord.append(coord.iloc[0,:])
    plt.plot(coord.iloc[:,0], coord.iloc[:,1])
    plt.scatter(coord.iloc[:,0], coord.iloc[:,1])


