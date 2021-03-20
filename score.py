# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:14:13 2020

@author: wu
"""

import re



def score_cut(woe,f,c,base,odd,pdo):
    a=list(float(pdo)/np.log(2)*(np.array(woe).dot(f)))
    return [int(x) for x in a]
def score(df,coo,base,odd,pdo,coe):
    return base-float(pdo)/np.log(2)*np.log(odd)-float(pdo)/np.log(2)*(np.array(df[coo]).dot(coe[:-1])+coe[-1])


def score1(base,odd,pdo,c):
    return base-float(pdo)/np.log(2)*np.log(odd)-float(pdo)/np.log(2)*c
def cut1(a,b):
    ddd=dict()
    a=[float(l) for l in a]    
    a=[str(l) for l in a]    
    di=dict(zip(a,b))
#    stt0='a='+a
#    c = compile(stt0,'','exec')   # 编译为字节代码对象  
#    exec(c) 
#    stt1='b='+b
#    c = compile(stt1,'','exec')   # 编译为字节代码对象  
#    exec(c)
    for l in di:
        if l=='-9999990.0':
            if di[l] not in [float('inf'),-float('inf')]:
                ddd['x==-9999990.0']=di[l]
            else:
                ddd['x==-9999990.0']=0
        if l=='-9999999.0':
            if di[l] not in [float('inf'),-float('inf')]:
                ddd['x==-9999999.0']=di[l]
            else:
                ddd['x==-9999999.0']=0
        if l=='-9999990':
            if di[l] not in [float('inf'),-float('inf')]:
                ddd['x==-9999990']=di[l]
            else:
                ddd['x==-9999990']=0
        if l=='-9999999':
            if di[l] not in [float('inf'),-float('inf')]:
                ddd['x==-9999999']=di[l]
            else:
                ddd['x==-9999999']=0
            
        if l=='nan':
            if di[l] not in [float('inf'),-float('inf')]:
                ddd['x==""']=di[l]
            else:
                ddd['x==""']=0
    if 'nan' not in di.keys():
        ddd['x==""']=0            
    di.pop('-9999990.0',0)
    di.pop('-9999999.0',0)
    di.pop('-9999990',0) 
    di.pop('-9999999',0)
    di.pop('nan',0)
    a=list(di.keys())
    a=[float(l) for l in a]    
    a.sort()
    a=[str(l) for l in a]    
    if float(a[0])>0:     
        strr='0<=x<='+str(a[0])
        ddd[strr]=di[a[0]]
    elif float(a[0])==0:
        strr='x=='+str(a[0])
        ddd[strr]=di[a[0]]
    else:
        strr='x<='+str(a[0])
        ddd[strr]=di[a[0]]        
    for i in range(len(a))[1:-1]:
        strr=str(a[i-1])+'<x<='+str(a[i])
        ddd[strr]=di[a[i]]
    strr=str(a[len(a)-2])+'<x'
    ddd[strr]=di[a[len(a)-1]]    
    return ddd


class score():
    def __init__(self, model,col,messa,base,odd,pdo):
        self.model=model
        self.cooo=col
        self.messa=messa
        self.base=base
        self.odd=odd
        self.pdo=pdo
        
    def _score(self):
        dddf2=self.messa
        var=pd.DataFrame()
        for j in self.cooo:    
            var_stat=dddf2[dddf2['var']==j][['WOE','Bin']]
            var_stat['1']=range(len(var_stat))
            var_stat['WOE']=var_stat['WOE']+var_stat['1']/10000
            bin_dic = dict(zip(var_stat['WOE'], var_stat['Bin']))
            for woe in bin_dic:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', bin_dic[woe]).split(', ')
                bin_dic[woe] = end_points    
            kl=[]
            kb=[]
            kn=[]
            for l in bin_dic:
                kb.append(l)
                kl.append(bin_dic[l][-1])
                kn.append(j)
            var_stat['Bin']=kl
            var_stat['WOE']=kb
            var_stat['name']=kn
            var=var.append(var_stat)
        
        var['WOE']=var['WOE'].apply(lambda x:round(x,2))

        ce=[]
        for i in range(len(self.cooo+['1'])):
            cc=np.array([0]*len(self.cooo+['1']))
            cc[i]=1
            c=-np.log(float(1)/self.model.predict(cc)-1)
            ce.append(list(c)[0])
        f=pd.DataFrame()
        f[0]=self.cooo+[1]
        f[1]=ce
        ivvv=pd.DataFrame()
        k1=[]
        k2=[]
        k3=[]
        i=0
        for j in self.cooo:
            i=i+1
            cx=list(var[var['name']==j]['WOE'])
            k1.append(j)
            k2.append(list(var[var['name']==j]['Bin']))
            k3.append(score_cut(cx,list(f[f[0]==j][1])[0],0,self.base,self.odd,-self.pdo))
            
        ivvv[0]=k1
        ivvv[1]=k2
        ivvv[2]=k3


        di=dict()
        
        
        for i in list(ivvv[0]):
            di[i]=cut1(list(ivvv[ivvv[0]==i][1])[0],list(ivvv[ivvv[0]==i][2])[0])
        
        di['interception']=int(score1(self.base,self.odd,self.pdo,f[f[0]==1][1])) 
        self.di=di
        return self






#
#
#ccc=score(result,coooo,ceshi.messa,600,20,20)
#
#         
#
#
#
#
#
#ccc._score()
#
#
#
#
#ccc.di










