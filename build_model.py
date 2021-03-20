# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:07:51 2020

@author: wu
"""


import copy
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score


from bin_new import *

class stepwise(object):
    def __init__(self, Data,Data1, col,y):
        # ...
#        self.max_num = 15
#        self.max_actions = 3
        self.col=col
#        self.tree=dict()
        self.col_set=[]
#        self.col_dic=dict()
        self.Data=Data.copy()
        self.Data1=Data1.copy()
#        self.baseline=baseline
        self.y=y
#        self.aselect=pd.DataFrame()
        self.best=[]
#        self.bbest=[]
        self.best_auc=0
#        self.bbest_auc=0
    def s_interfect(self):
        # ...        
        self.Data['1']=1
        self.Data1['1']=1
        return self
    def generate(self):
        # ...        
        cx=list(set(self.col)-set(self.best))
        mm='NA'
        for l in cx:
            try:
                ccc=self.best+[l]
                logit = sm.Logit(self.Data[self.y],self.Data[ccc+['1']])
                result = logit.fit()
                res=result.summary()
                if self.test(res):
                    rre=result.predict(self.Data[ccc+['1']])
                    self.Data['p']=rre    
    #                self.col_set.append(set(ccc))               
                    rrre=result.predict(self.Data1[ccc+['1']])
                    self.Data1['p']=rrre    
                    au=roc_auc_score(np.array(self.Data[self.y]),np.array(self.Data['p']))*0.5+\
                            roc_auc_score(np.array(self.Data1[self.y]),np.array(self.Data1['p']))*0.5
                    if au>self.best_auc:
                        mm=l
                        self.best_auc=au
            except:
                pass
#                return ccc,au
#            else:
#                return self.generate(ccc[:-1])
        if mm!='NA':
            self.best=self.best+[mm]
            return 'verbesset'
        else:
            return 'break'
    def test(self,res): 
        # ...            
        sd=str(res).split('\n')
        cs=[]
        for i in sd[12:-1]:
            cn=i.split(' ')
            cc=copy.copy(cn)
            for i in cn:
                if i=='':
                    cc.remove(i)
            cs.append(cc)
        dic=dict()
        for i in cs:
            dic[i[0]]=i                    
        a0=[]
        a1=[]
        for i in dic:
            if i!='1':
                a0.append(float(dic[i][4]))
                a1.append(float(dic[i][1]))
        if max(a0)>0.05 or min(a1)<0:
            return False
        else:
            return True


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy



class MonteCarlo(object):
    def __init__(self, Data,Data1, col,y,baseline=0):
        # ...
        self.max_num = 15
        self.max_actions = 3
        self.col=col
        self.tree=dict()
        self.col_set=[]
        self.col_dic=dict()
        self.Data=Data.copy()
        self.Data1=Data1.copy()
        self.baseline=baseline
        self.y=y
#        self.aselect=pd.DataFrame()
        self.best=[]
        self.bbest=[]
        self.best_auc=0
        self.bbest_auc=0
    def s_interfect(self):
        # ...        
        self.Data['1']=1
        self.Data1['1']=1
        return self
    def first_action(self,asc=[]):
        # ...
        pii=pd.DataFrame()
        au=[]
        co=[]
        for l in self.col:
            if l not in asc:
                ccc=asc+[l]+['1']
                try:
                    logit = sm.Logit(self.Data[self.y],self.Data[ccc])
                    result = logit.fit()
                    rre=result.predict(self.Data[ccc])
                    self.Data['p']=rre
                    rrre=result.predict(self.Data1[ccc])
                    self.Data1['p']=rrre
                    co.append(l)               
                    au.append(roc_auc_score(np.array(self.Data[self.y]),np.array(self.Data['p']))*0.5+\
                              roc_auc_score(np.array(self.Data1[self.y]),np.array(self.Data1['p']))*0.5)
                except:
                    pass
        pii[0]=co
        pii[1]=au
        pii=pii.sort_values(by=1)
        pii=pii.sort_values(by=1,ascending=False)
        self.col_set=list(pii.iloc[:20][0])
#        self.aselect=self.aselect.append(pii.iloc[:10])
        if asc==[]:
            self.baseline=np.mean(pii.iloc[:20][1])
        self.col_dic=dict()    
        for l in self.col_set:
            a=dict()
            a[tuple([l])]=list(pii[pii[0]==l][1])[0]            
            self.col_dic[l]=a
        self.col_set=[]
        return self    
    def update_leaves(self,x,l=3):
        # ...
        ca=list(set(self.col)-set(x))
        k=randint(0, len(ca)-1)   
        l=l-1
        if l>=0:
            return self.update_leaves(x+[ca[k]],l)
        else:
            return x
        
    def generate(self,ccc):
        # ...        
        if set(ccc) not in self.col_set:
            logit = sm.Logit(self.Data[self.y],self.Data[ccc+['1']])
            result = logit.fit()
            res=result.summary()
            if self.test(res):
                rre=result.predict(self.Data[ccc+['1']])
                self.Data['p']=rre    
                rrre=result.predict(self.Data1[ccc+['1']])
                self.Data1['p']=rrre    
                self.col_set.append(set(ccc))               
                au=roc_auc_score(np.array(self.Data[self.y]),np.array(self.Data['p']))*0.5+\
                        roc_auc_score(np.array(self.Data1[self.y]),np.array(self.Data1['p']))*0.5
                return ccc,au
            else:
                return self.generate(ccc[:-1])
        else:
            return ccc,0
    def test(self,res): 
        # ...            
        sd=str(res).split('\n')
        cs=[]
        for i in sd[12:-1]:
            cn=i.split(' ')
            cc=copy.copy(cn)
            for i in cn:
                if i=='':
                    cc.remove(i)
            cs.append(cc)
        dic=dict()
        for i in cs:
            dic[i[0]]=i                    
        a0=[]
        a1=[]
        for i in dic:
            if i!='1':
                a0.append(float(dic[i][4]))
                a1.append(float(dic[i][1]))
        if max(a0)>0.05 or min(a1)<0:
            return False
        else:
            return True
    def random_spread(self,x,k,epoch=20):
#        epoch=10
        auc=[]
        for l in range(epoch):
            try:
                c=self.update_leaves(x)
                a,b=self.generate(c)
                if b>np.log((self.baseline+0.05)/self.baseline):
                    auc.append(b)
                if b>self.bbest_auc:
                    self.bbest_auc=b
                    self.bbest=c
            except:
                pass
        if len(auc)>0:
            return np.mean(auc)
        else:
            return 0
    def deep_spread(self):
        better='Na'
        self.bbest=[]
        self.bbest_auc=0
        for l in self.col_dic:
            ccc=self.best+[l]
            score=self.random_spread(ccc,len(ccc))
            if score>self.best_auc:
                better=l
                self.best_auc=score
        if better!='Na':
            self.best=self.best+[better]
            self.first_action(self.best)
            print (len(self.best))
            return self
        else:
            print ('Done')
            return 'Done'            
    def deep_spread_1(self):
        better='Na'
#        self.bbest=[]
#        self.bbest_auc=0
        ca=list(set(self.col)-set(self.best))
        cacol=[]
        for l in range(15):
            k=randint(0, len(ca)-1)
            if k not in cacol:
                cacol.append(k)
            else:
                k=randint(0, len(ca)-1)                
            ccc=self.best+[ca[k]]
            score=self.random_spread(ccc,len(ccc),15)
            if score>self.best_auc:
                better=ca[k]
                self.best_auc=score
        if better!='Na':
            self.best=self.best+[better]
            self.col_set=[]
#            self.first_action(self.best)
            return self
        else:
            print ('Done')
            return 'Done'

import lightgbm as lgb
import gc
import pandas as pd
import numpy as np
#import time
from sklearn.model_selection import KFold,StratifiedKFold
from datetime  import datetime

class model_lgbm():
    def __init__(self):        
        self.models=None
        self.importances=None
        self.prediction=None
        self.feature_names=None
    def predict_proba(self,Test):
        prediction= np.zeros(len(Test))
        X_test=Test[self.feature_names].values
        for model in self.models:
            prediction += model.predict(X_test, num_iteration=model.best_iteration)/5
        return prediction
                
    def validation_prediction_lgb(self,X,y,feature_names, ratio =1, X_test = None,istest = False):
        self.feature_names=feature_names    
        n_fold = 5
        folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        params = {
        'bagging_freq': 5,     
        'boost_from_average':'false',    
        'boost': 'gbdt',   
        'learning_rate': 0.01,
        'max_depth': 8,
        'metric':'auc',
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 10.0,
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': 1}
        importances = pd.DataFrame()     
        if istest:
            prediction = np.zeros(len(X_test))
        models = []
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
            print('Fold', fold_n, 'started at', time.ctime())    
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            weights = [ratio  if val == 1 else 1 for val in y_train]                
            train_data = lgb.Dataset(X_train, label=y_train,  weight=weights)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            model = lgb.train(params,train_data,num_boost_round=5000,
                            valid_sets = [train_data, valid_data],verbose_eval=400,early_stopping_rounds = 1000)
            imp_df = pd.DataFrame() 
            imp_df['feature']  = feature_names
            imp_df['split']    = model.feature_importance()
            imp_df['gain']     = model.feature_importance(importance_type='gain')    
            imp_df['fold']     = fold_n + 1    
            importances = pd.concat([importances, imp_df], axis=0)
            models.append(model)    
            if istest == True:    
                prediction += model.predict(X_test, num_iteration=model.best_iteration)/5    
        if istest == True:   
            self.models=models
            self.importances=importances
            self.prediction=prediction
            return self
        else:    
            self.models=models
            self.importances=importances 
            return self



import shap

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

class shap_woe_explain():
    def __init__(self,model,data,col,y):        
        self.model=model
        self.data=data
        self.col=col
        self.y=y
    def woee(self):
        a=ff_bin_woe(self.y,self.data[col+['y']], pd.DataFrame(),10, 0.05, 200, [], [], [self.y],True,False)
        a.woe()
        self.d1=a.d1

        self.explainer=shap.TreeExplainer(self.model)
        self.shap_values= self.explainer.shap_values(self.data[self.col].values)
        
        for i in range(len(self.col)):
            try:
                gc.collect()
            
                self.data[self.col[i]+'_shap']=self.shap_values[1][:,i]
            
                self.data[self.col[i]+'_woe']=self.d1[self.col[i]]
            except:
                pass
        return self
    def _plot(self,l):
        if l not in self.col:
            print ('error : not in list')
            return None
        else:
            try:
                sns.set() #恢复seaborn的默认主题
                fig,axes = plt.subplots()
                sns.scatterplot(x=l,y=l+'_shap',data=self.data,ax=axes)
                sns.scatterplot(x=l,y=l+'_woe',data=self.data,ax=axes)
                axes.set_title(col[i])
            except:
                pass

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers, losses, optimizers, Sequential,metrics

from sklearn.model_selection import train_test_split

class MyRNN(keras.Model):
    # Cell方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__() 


        self.rnn = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.3, return_sequences=True),
            layers.SimpleRNN(units, dropout=0.3)
        ])
        self.outlayer = Sequential([
        	layers.Dense(128),
        	layers.Dropout(rate=0.5),
        	layers.ReLU(),
        	layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs # [b, 80]

        x = self.rnn(x)

        prob = self.outlayer(x)

        prob = tf.sigmoid(x)

        return prob                
    
    










