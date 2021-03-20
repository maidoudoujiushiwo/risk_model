# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:12:03 2020

@author: wu
"""

#					COPYRIGHT: 
#							AUTH: jian.wu
#                           Email:fengyuguohou2010@hotmail.com

#!pip install varclushi

from varclushi import VarClusHi


def psi_cut(A,DA,x):
    A.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan,float('inf')],0,inplace=True)
    A=A[[x]]
    bins=[-float('inf')]+DA
    bins=list(set(bins))
    bins.sort()    
    a=pd.DataFrame([bins[-1]]).T
    a.columns =A.columns 
    A=pd.concat([A,a],ignore_index=True)        
    if float('inf') in bins:
        bins.remove(float('inf'))
    lab=[i for i in bins[1:]]
    AA=pd.DataFrame()
    AA[x] = pd.cut(A[x], bins, labels = lab)
    AA[x] = AA[x].astype('float64')
    df=pd.DataFrame()
    df=AA[x].groupby(AA[x]).agg(['count'])
    df[x+'_count']=df['count']
    return df[[x+'_count']],AA[[x]]


def psi(A,B,col=[],name='',p='',ll=True,lll=False,l1=400,l2=700):
    if p!='':
        if len(A)<len(B):
            BB=B.sample(len(A))
            AA=A
        else:
            AA=A.sample(len(B))
            BB=B
        p_bin=[l1]+[l1+i*5 for i in range(120)]
        A_1,A_3=psi_cut(A,p_bin,p)
        B_1,B_3=psi_cut(B,p_bin,p)
        A_4,A_2=psi_cut(AA,p_bin,p)
        B_4,B_2=psi_cut(BB,p_bin,p)
        AA_1=A_1.copy()
        BB_1=B_1.copy()
        AA_1['pp']=np.cumsum(A_1/A_1.sum())
        BB_1['pp']=np.cumsum(B_1/B_1.sum())
        plt.figure(figsize = (6, 3))
        plt.title('score'+'_psi')
        if lll:
            AA_1['pp'].plot(linewidth=1.5,xlim=(l1,l2),label='A',color='red')  
            BB_1['pp'].plot(linewidth=1.5,xlim=(l1,l2),label='B',color='blue')              
        else:    
            A_2[p].plot(kind='density',linewidth=1.5,xlim=(l1,l2),label='A',color='red')  
            B_2[p].plot(kind='density',linewidth=1.5,xlim=(l1,l2),label='B',color='blue')  
        plt.legend(loc='upper right')
        plt.show()
        BX=B_1-A_1
        BX=BX.fillna(0)
        BX=BX-BX
        A_1=A_1-BX
        A_1=A_1.fillna(0)
        B_1=B_1-BX
        B_1=B_1.fillna(0)        
        A_1=A_1+1
        B_1=B_1+1        
        ss=(-A_1/A_1.sum()+B_1/B_1.sum())*np.log((B_1/B_1.sum())/(A_1/A_1.sum()))
        ss.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan,float('inf')],0,inplace=True)
        s=float(ss.sum())
        mmx=list(A_1[A_1[p+'_count']==float(np.max(A_1))].index).pop()    
        A_1_1=A_1[A_1.index>=mmx]    
        A_1_2=A_1[A_1.index<mmx]
        B_1_1=B_1[B_1.index>=mmx]    
        B_1_2=B_1[B_1.index<mmx]                
        ps1=(A_1_1/A_1_1.sum()-B_1_1/B_1_1.sum())
        ps1.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan,float('inf')],0,inplace=True)
        pss1=float(ps1.sum())
        ps2=(B_1_2/B_1_2.sum()-A_1_2/A_1_2.sum())
        ps2.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan,float('inf')],0,inplace=True)
        pss2=float(ps2.sum())        
        return s,pss1+pss2
    elif name!='':
        col=[float(min(A[name]))]+[float(min(A[name]))+(float(max(A[name]))-float(min(A[name])))*i/20 for i in range(21)]
        col1=[float(min(B[name]))]+[float(min(B[name]))+(float(max(B[name]))-float(min(B[name])))*i/20 for i in range(21)]
        col=col+col1+[max(B[name])]+[max(A[name])]
        a=(float(max(A[name]))-float(min(A[name])))/30
        b=(float(max(B[name]))-float(min(B[name])))/30
        x=min(a,b)
        col.sort()
        ccc=[]
        ccc.append(col[-1])
        for i in range(len(col)-1):
            if col[-i]-col[-i-1]>x:
                ccc.append(col[-i-1])
        A_1,A_2=psi_cut(A,ccc,name)
        B_1,B_2=psi_cut(B,ccc,name)
        BX=B_1-A_1
        BX=BX.fillna(0)
        BX=BX-BX
        A_1=A_1-BX
        A_1=A_1.fillna(0)
        B_1=B_1-BX
        B_1=B_1.fillna(0)        
        A_1=A_1+1
        B_1=B_1+1        
        A_1_1=A_1/np.float(A_1.sum())
        B_1_1=B_1/np.float(B_1.sum())      
        ccc=list(A_1_1.index)+list(B_1_1.index)
        ccc=list(set(ccc))
        c=pd.DataFrame(index=ccc)
        c[0]=A_1_1+0.00001
        c[1]=B_1_1+0.00001
        c=c.fillna(0.00001)
        if ll:
            plt.figure(figsize = (8, 4))
            plt.subplot(1, 1, 1)
            plt.title(name+'_psi')
            bar_width = 0.35
            plt.bar(np.arange(len(A_1_1)),list(A_1_1[name+'_count']),bar_width,label='A',color='red')
            plt.bar(np.arange(len(B_1_1))+bar_width, list(B_1_1[name+'_count']),bar_width ,label='B')            
            if len(A_1_1)>=len(B_1_1):
                plt.xticks(np.arange(len(A_1_1))+bar_width/2,tuple(A_1_1.index))
            else:
                plt.xticks(np.arange(len(B_1_1))+bar_width/2,tuple(B_1_1.index))
            plt.legend(loc='upper center', fancybox=True, ncol=5)     
            plt.show()
        ss=(c[0]-c[1])*np.log(c[0]/c[1])
        ss.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan,float('inf')],0,inplace=True)
        s=float(ss.sum())
        return s



def ksss(df, score_name, y_name):
    df=df.sort_values(by=score_name)
    grouped = df[y_name].groupby(df[score_name])
    df_agg = grouped.agg(['count', 'sum'])
    df_agg = df_agg.sort_index(ascending = False)
    df_agg['good'] = df_agg['count'] - df_agg['sum']
    df_agg['cum_bad_rate'] = np.cumsum(df_agg['sum']) / sum(df_agg['sum'])
    df_agg['cum_good_rate'] = np.cumsum(df_agg['good']) / sum(df_agg['good'])
    return np.max(abs(df_agg['cum_bad_rate'] - df_agg['cum_good_rate']))
    
class auswahle():
    def __init__(self,data,test_data,y):        
        self.data = data
        self.test_data=test_data
        self.y = y

    def psi_ceshi(self,col):
        cc=[]
        for l in col:
            cc.append(psi(self.data ,self.test_data,col=[],name=l,p='',ll=False,lll=False,l1=400,l2=700))
        ppsi=pd.DataFrame(index=range(len(col)))
        ppsi['var']=col
        ppsi['psi']=cc
        return ppsi
    def del_high_corr_ks(self,model,col,p_cri,y):    
        corr = self.data[col].corr(method = 'pearson')    
        corr_vars = corr.sum()    
        corr_vars=corr_vars.sort_values()    
        var = list(corr_vars.index)    
        var_del = []    
        for i in range(len(var) - 1):    
            var_i = var[i]    
            if var_i not in var_del:      
                for j in range(i+1,len(var)):
                    if j!=1:
                        var_j = var[j]    
                        if var_j not in var_del:    
                            if (abs(corr[var_i][var_j]) > p_cri):
                                if (ksss(self.data,var_i,y) >= ksss(self.data,var_j,y)):    
                                    var_del.append(var_j)
                                else:
                                    var_del.append(var_i)
        col1=list(set(var) - set(var_del))
        return col1
    def varclust(self,col):
        varclust = VarClusHi(self.data[col])
        return varclust,demo1_vc.info,demo1_vc.rsquare



