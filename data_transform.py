# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:32:30 2020

@author: wu
"""


#					COPYRIGHT: 
#							AUTH: jian.wu
#                           Email:fengyuguohou2010@hotmail.com
#							2020-08-02
'''
Life is short, You need Python~
'''

from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

from sklearn.decomposition import NMF  
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from time import time
import seaborn as sns


from sklearn import ensemble

from sklearn.preprocessing import OneHotEncoder 



def diss(a,b):
    '''
    define distance
    '''
    return np.sum((a-b)**2,axis=1)

def scatter(x,colors,n):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    if type(colors)!=type(None):    
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
    else:
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
        
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    if type(colors)!=type(None):
        for i in range(n):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
#            txt.set_path_effects([
#                PathEffects.Stroke(linewidth=5, foreground="w"),
#                PathEffects.Normal()])
            txts.append(txt)

    return None



class Unsupervised():
    """Unsupervised feature engineering(Kmeans, PCA, NMF and Tsen)
    Parameters
    ----------    
    Data :Import pandas table include the original  features.
    
    y : target, can be None.
    
    col: Name of the original  features.
    
    
    """
    
    def __init__(self,Data,col=None,y=None):        
        self.Data = Data
        self.y = y
        self.col = col
        self.uk_p=None
        self.kpre=None
        self.pca=None
        self.nmf=None
        self.digits_proj=None
        self.tsne_p=None
        self.tpre=None
        
    def _check(self,col,data):
        if col==None:
            raise ValueError('No import features')
        else:
            return data[col].values
    def unkmeans(self):
        """create the kmeans
        """
        X=self._check(self.col,self.Data)
        estimator = KMeans(n_clusters=2)#build Kmeans
        estimator.fit(X)#kmeans
        centroids = estimator.cluster_centers_ #center of it
        self.kpre=estimator.predict(X)
#        ac0,ac1=diss(X,centroids[0]),diss(X,centroids[1])#distance 
#        aw0,aw1=int(np.where(ac0==np.min(ac0))[0]),int(np.where(ac1==np.min(ac1))[0])
        self.uk_p=centroids
        return self
        
    def _var(self,var,Data,col):
        """create the feature
        """
        X=self._check(col,Data)
        var0,var1=diss(X,var[0]),diss(X,var[1])
        return var0,var1
    
    def _unk_var(self):
        """create the feature of kmeans
        """
        if type(self.uk_p)==type(None):
            raise ValueError('NO Kmeans have been built')
        else:
            self.Data['k0'],self.Data['k1']=self._var(self.uk_p,self.Data,self.col) 
        return self
        
    def unk_var(self,Data):
        """create the feature of kmeans
        """
        if type(self.uk_p)==type(None):
            raise ValueError('NO Kmeans have been built')
        else:
            Data['k0'],Data['k1']=self._var(self.uk_p,Data,self.col) 
        return Data
        
    def unPca(self):
        """create the Pca
        """
        X=self._check(self.col,self.Data)
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        return self
        
    def unNmf(self):
        """create the Nmf
        """

        X=self._check(self.col,self.Data)
        for l in self.col:
            self.Data[l+str('e')]=self.Data[l].apply(np.exp)
        self.coll=[l+str('e') for l in self.col]   
        X=self._check(self.coll,self.Data)
        self.nmf = NMF(n_components=2)
        self.nmf.fit(X)
        return self


    def untsne(self):
        """create the Tsne
        """
        X=self._check(self.col,self.Data)
        self.digits_proj = TSNE(random_state=2018).fit_transform(X)
        estimator = KMeans(n_clusters=2)#构造聚类器
        estimator.fit(self.digits_proj)#聚类
        self.tpre=estimator.predict(self.digits_proj)
        centroids = estimator.cluster_centers_ #获取聚类中心
        ac0,ac1=diss(self.digits_proj,centroids[0]),diss(self.digits_proj,centroids[1])#distance 
        aw0,aw1=int(np.where(ac0==np.min(ac0))[0]),int(np.where(ac1==np.min(ac1))[0])
        self.tsne_p=[X[aw0],X[aw1]]

    def _unt_var(self):
        """create the feature of kmeans
        """
        if type(self.tsne_p)==type(None):
            raise ValueError('NO tsen have been built')
        else:
            self.Data['t0'],self.Data['t1']=self._var(self.tsne_p,self.Data,self.col) 

        return self
        
    def unt_var(self,Data):
        """create the feature of kmeans
        """
        if type(self.tsne_p)==type(None):
            raise ValueError('NO tsen have been built')
        else:
            Data['k0'],Data['k1']=self._var(self.tsne_p,Data,self.col) 
        return Data

    def unsum(self):
        X=self._check(self.col,self.Data)
        kmeans=self.unkmeans()
        pca=self.unPca()
        nmf=self.unNmf()
        tsne=self.untsne()
        self._unk_var()
        self._unt_var()
        X1=self._check(self.coll,self.Data)
        pcae=self.pca.transform(X)
        nmfe=self.nmf.transform(X1)
        self.Data['p0'],self.Data['p1']=pcae[:,0],pcae[:,1]
        self.Data['n0'],self.Data['n1']=nmfe[:,0],nmfe[:,1]
        if type(self.y)==type(None):
            return self.Data[['k0','k1','p0','p1','n0','n1','t0','t1']]
        else:
            return self.Data[['k0','k1','p0','p1','n0','n1','t0','t1',self.y]] 
    def make_re(self,da):
        X=self._check(self.col,da)
        for l in self.col:
            da[l+str('e')]=da[l].apply(np.exp)
        X1=self._check(self.coll,da)
        pcae=self.pca.transform(X)
        nmfe=self.nmf.transform(X1)
        da['p0'],da['p1']=pcae[:,0],pcae[:,1]
        da['n0'],da['n1']=nmfe[:,0],nmfe[:,1]
        da['t0'],da['t1']=self._var(self.tsne_p,da,self.col) 
        da['k0'],da['k1']=self._var(self.uk_p,da,self.col) 

        if type(self.y)==type(None):
            return da[['k0','k1','p0','p1','n0','n1','t0','t1']]
        else:
            return da[['k0','k1','p0','p1','n0','n1','t0','t1',self.y]] 
    def t_plot(self):
        """plot the Tsen with y
        """
        return scatter(self.digits_proj,self.Data[self.y].values,2)

    def t_plot_1(self):
        """plot the Tsen with itself
        """
        return scatter(self.digits_proj,self.tpre,2)
     
    def k_plot(self):
        """plot the kmeans with y
        """
        return scatter(self.Data[['k0','k1']].values,self.Data[self.y].values,2)
     
    def k_plot_1(self):
        """plot the kmeasn with itself
        """
        return scatter(self.Data[['k0','k1']].values,self.kpre,2)
        


class feature_eng(object):
    def __init__(self,col1,col2,y, Data,Test_data=None):
        '''
        input:
            Data 
            col1 数值型变量的list
            col2 非数值型变量list
        '''
        self.col1=col1
        self.col2=col2

        self.col=col1+col2
        self.y=y

        self.data=Data[self.col+[y]]
        if self.check(Test_data):
            self.test_data=Test_data[self.col+[y]]
        else:
            self.test_data=None
        
    def check(self,x):
        if x is not None:
            return True
        else:
            return False
        
    def drop_missing(self,axis=0):
        '''
        drop missing col
        '''
        self.data = self.data.dropna(axis=axis,inplace=False)
        if self.check(self.test_data):
            self.test_data = self.test_data.dropna(axis=axis,inplace=False)
        return self
#数值型变量填充
    def na_fill(self,drop=True):
        '''
        col is the feature list want to fill na
        drop=Ture drop the org feature
        '''

#            data_copy = self.data.copy(deep=True)
        for i in self.col:
            if self.data[i].isnull().sum()>0:
                self.data[i+'_mean'] = self.data[i].fillna(self.data[i].mean())
                self.data[i+'_median'] = self.data[i].fillna(self.data[i].median())
                if self.test_data is not None:
                    self.test_data[i+'_mean'] = self.test_data[i].fillna(self.test_data[i].mean())
                    self.test_data[i+'_median'] = self.test_data[i].fillna(self.test_data[i].median())
                if drop:
                    del self.data[l]
                    del self.test_data[l]                
        return self
#离散变量处理，label_encode
    def label_encode(self):
        
        def la(da,test,l):
            l_encoder = LabelEncoder()
            l_encoder.fit(da[l].values)
            da[l+'l_encode'] = l_encoder.transform(da[l])
                
            if test is not None:
                test[+'l_encode'] = l_encoder.transform(test[l])
                return da[l+'l_encode'],test[l+'l_encode']
            else:
                return da[l+'l_encode']
        for i in self.col2:
            if self.test_data is not None:
                self.data[i+'_l_encode'],self.test_data[i+'_l_encode']=la(self.data,self.test_data,i)
            else:
                self.data[i+'_l_encode']=la(self.data,self.test_data,i)
        return self
#离散变量按照计算woe处理
    def woe_encode(self):
        def Cal_WOE(df, x, y):
            grouped = df[y].groupby(df[x])
            df_agg = grouped.agg(['sum', 'count'])
            t1 = df_agg['sum'].sum()
            t0 = df_agg['count'].sum() - t1
            n_c = len(df_agg['sum'])
            x_woe = list(range(n_c))
            for i in range(n_c):
                t1_i = float(df_agg.iloc[i, 0])
                t0_i = float(df_agg.iloc[i, 1] - t1_i)
                if t1_i == 0 and t0_i != 0:
                    x_woe[i] = -1
                elif t1_i != 0 and t0_i == 0:
                    x_woe[i] = 1
                elif t1_i == 0 and t0_i == 0:
                    x_woe[i] = 0
                else:
                    x_woe[i] = np.log( (t1_i/t1) / (t0_i/t0) )
            df_agg[x + '_woe'] = x_woe
            return df_agg
        for x in self.col2:
            
          df_agg = Cal_WOE(self.data, x, self.y)
          if self.check(self.test_data):
              nn=list(self.test_data[x].unique())
              inde=list(df_agg.index)
              na=[l for l in nn if l not in inde]
              inde=inde+na
              ll=list(df_agg[x + '_woe'])+[np.nan]*len(na)              
              self.data[x+'_woe_encode'] = self.data[x].replace(inde, ll)
              self.test_data[x+'_woe_encode'] = self.test_data[x].replace(inde, ll)
          else:
              self.data[x+'_woe_encode'] = self.data[x].replace(df_agg.index, df_agg[x + '_woe'])
        return self
#根据逾期密度的tif-idf，有时候很好用，慎用
    def y_tf_idf(self):
        def cal_tfidf(df,x,y):
            grouped = df[y].groupby(df[x])
            df_agg = grouped.agg(['sum', 'count'])
            t1 = df_agg['sum'].sum()
            t0 = df_agg['count'].sum()
            
            df_agg['tf']=df_agg['sum']/float(t1)        
            df_agg['idf']=(float(t0)/df_agg['count']).apply(np.log)
            df_agg['tf_idf']=df_agg['tf']*df_agg['idf']
            return df_agg
        for x in self.col2:
            
          df_agg = cal_tfidf(self.data, x, self.y)
          if self.check(self.test_data):
              nn=list(self.test_data[x].unique())
              inde=list(df_agg.index)
              na=[l for l in nn if l not in inde]
              inde=inde+na
              ll=list(df_agg['tf_idf'])+[np.nan]*len(na)              
              self.data[x+'_y_tfidf'] = self.data[x].replace(inde, ll)
              self.test_data[x+'_y_tfidf'] = self.test_data[x].replace(inde, ll)
          else:
              self.data[x+'_y_tfidf'] = self.data[x].replace(df_agg.index, df_agg['tf_idf'])
        return self
#根据四种无监督变量衍生方法（基于Kmeans, PCA, NMF and T-sne）
    def uns(self,col):
        self.unsp=   Unsupervised(self.data,col,self.y)
        self.data[['k0','k1','p0','p1','n0','n1','t0','t1']]=self.unsp.unsum()[['k0','k1','p0','p1','n0','n1','t0','t1']]
        if self.check(self.test_data):
            self.test_data[['k0','k1','p0','p1','n0','n1','t0','t1']]=self.unsp.make_re(self.test_data)[['k0','k1','p0','p1','n0','n1','t0','t1']]
        return self
#gbdt embeding
    def gbdt_e(self,col,nEst = 20,depth = 2,learnRate = 0.005,subSamp = 0.5,onehot=False):
       
        MModel = ensemble.GradientBoostingRegressor(n_estimators=nEst,max_depth=depth,learning_rate=learnRate,subsample = subSamp,loss='ls')
        
        MModel.fit(self.data[col],self.data['y'])
        mm=MModel.apply(self.data[col])
        if onehot:
            enc = OneHotEncoder()  
            enc.fit(MModel.apply(self.data[col]))#将位置码转化为01码  
            new_feature_train=enc.transform(MModel.apply(self.data[col]))  
            shap=new_feature_train.shape[1]
            ccol=[str(i)+'gbdt_onehot' for i in range(shap)]
            for i in range(len(ccol)):
                self.data[ccol[i]]=new_feature_train[:,i]
            if self.check(self.test_data):  
                nnew=nc.transform(MModel.apply(self.test_data[col])) 
                for i in range(len(ccol)):
                    self.test_data[ccol[i]]=nnew[:,i]
        else:

            new_feature_train=MModel.apply(self.data[col])  
            shap=new_feature_train.shape[1]
            ccol=[str(i)+'gbdt' for i in range(shap)]
            for i in range(len(ccol)):
                self.data[ccol[i]]=new_feature_train[:,i]
            if self.check(self.test_data):  
                nnew=MModel.apply(self.test_data[col])  
                for i in range(len(ccol)):
                    self.test_data[ccol[i]]=nnew[:,i]
            

        
        

        
        





        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        





        


        
        
        
