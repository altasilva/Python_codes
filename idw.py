#Interpolar 4 parâmetros da equação de chuvas intensas. 
#Parâmetros K, a, b, c.

import math
import numpy as np
import pandas as pd


def __harvesine(lon1, lat1, lon2, lat2):
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    d = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return(d)
# ------------------------------------------------------------
# Prediction
def __idw(x, y, z, xi, yi):
    #list_xyzi = []
    list_x = []
    list_y = []
    list_zi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (__harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        list_x.append(xyzi[0])
        list_y.append(xyzi[1])
        list_zi.append(xyzi[2])
 
    return(list_x, list_y, list_zi)


def interpolar():
    #Open File 
    Filename1 = "../data/parametros_tese.csv"
    Filename2 = "../data/parametros_desconhecidos.csv"
    
    known=pd.read_csv(Filename1, sep=',')
    unknown = pd.read_csv(Filename2, sep=',')

    df1 = pd.DataFrame(known)
    df2 = pd.DataFrame(unknown)
    
    df1.iloc[1:,3:].astype(float)
    df2.iloc[1:,1:].astype(float)
   

    x = df1['Latitude'].tolist()
    y = df1['Longitude'].tolist()
    
    K = df1['K'].tolist()
    a = df1['a'].tolist()
    b = df1['b'].tolist()
    c = df1['c'].tolist()
    
    xi = df2['Latitude'].tolist()
    yi = df2['Longitude'].tolist()
    
    local = df2['Local'].tolist()

    # running the function
    param_K = np.array(__idw(x,y,K,xi,yi)).T
    param_a = np.array(__idw(x,y,a,xi,yi)).T
    param_b = np.array(__idw(x,y,b,xi,yi)).T
    param_c = np.array(__idw(x,y,c,xi,yi)).T

    # create file
    Filename3 = '../data/parametros_estimados.csv'
    df3 = pd.DataFrame(param_K, columns=['Latitude', 'Longitude', 'K'])
    df3.insert(0, 'Local', local)   
    df3.insert(4, 'a', param_a[:,2])
    df3.insert(5, 'b', param_b[:,2])
    df3.insert(6, 'c', param_c[:,2])
    
    df3.to_csv(Filename3, header=True, index = False )
    
    return(param_K, param_a, param_b, param_c)

interpolar()
