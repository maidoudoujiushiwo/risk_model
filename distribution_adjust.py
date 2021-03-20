# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:07:09 2020

@author: wu
"""


import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  


class gan_psi:
    def __init__(self,l1,l2,LR_G = 0.001, LR_D = 0.001,ART_COMPONENTS =15):
        self.LR_G=LR_G
        self.LR_D=LR_D
        self.ART_COMPONENTS=ART_COMPONENTS
        self.l1=l1
        self.l2=l2
        self.init_placeholders()
        self.init_graph()

    def init_placeholders(self):
#        self.art_labels = tf.placeholder(tf.float32,[None,1])                 
#        self.art_labels0 = tf.placeholder(tf.float32,[None,1])                
        self.G_in = tf.placeholder(tf.float32,[None,self.ART_COMPONENTS])             
        self.real_in = tf.placeholder(tf.float32,[None,self.ART_COMPONENTS],name='real_in')  
    def init_graph(self):
        with tf.variable_scope(self.l1):  
#            self.G_art = tf.concat((self.G_in,self.art_labels0),1)                      
            self.D_l = tf.layers.dense(self.G_in,20,tf.nn.relu,name='0')          
            self.pb = tf.layers.dense(self.D_l,1,tf.nn.sigmoid)
        
        with tf.variable_scope(self.l2):  
#            self.real_art = tf.concat((self.real_in,self.art_labels),1)                 
            self.D_l0 = tf.layers.dense(self.real_in,20,tf.nn.relu,name='1')  
            self.prob_artist0 = tf.layers.dense(self.D_l0,1,tf.nn.sigmoid,name='out')  
            #fake art  
            self.D_l1 = tf.layers.dense(self.G_in,20,tf.nn.relu,name='1',reuse=True)  
            self.prob_artist1 = tf.layers.dense(self.D_l1,1,tf.nn.sigmoid,name='out',reuse=True)      
            self.prob_artist2=tf.multiply(self.prob_artist1,self.pb,name="real_distance")
        self.D_loss = -tf.reduce_mean(tf.log(self.prob_artist0)+tf.log(1-self.prob_artist2))  
        self.G_loss = tf.reduce_mean(tf.log(1-self.prob_artist2))  
          
        self.train_D = tf.train.AdamOptimizer(self.LR_D).minimize(  
               self.D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.l2))            
        self.train_G = tf.train.AdamOptimizer(self.LR_G).minimize(  
               self.G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.l1))          
        return self
        
    def train_step(self, G_ideas,real_in):
        for i in range(10):
            D1,D2 = sess.run([self.D_loss,self.G_loss,self.train_D],  
                                          {self.G_in:G_ideas,self.real_in:real_in})[:2]        
        D1,D2 = sess.run([self.D_loss,self.G_loss,self.train_D,self.train_G],  
                                          {self.G_in:G_ideas,self.real_in:real_in})[:2]        
        return D1,D2

    def eval_step(self, G_ideas):
        pb = sess.run([self.prob_artist2],{self.G_in:G_ideas})[0]        
        return pb

#import gc
#gc.collect()
#
#
#cc=del_high_corr_ks(m2,col,0.85)
#
#
#
#col=cc
#
#
#tf.reset_default_graph()
#a=gan_psi('G1','D1',LR_G = 0.001, LR_D = 0.001,ART_COMPONENTS =len(col))
#
#
#sess= tf.Session()  
#sess.run(tf.global_variables_initializer())  
#
#
#
#bachsize=len(m3)
#
#
#
#
#
#
#data1=m2[col].values
#data2=m3[col].values
#
#for i in range(500):
#    ax=np.random.randint(len(m3),size=bachsize)
#    d1=data1[ax]
#    D1,D2=a.train_step(d1,data2)
#    if i%50==0:  
#        print D1,D2


