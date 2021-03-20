# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:59:07 2020

@author: wu
"""


#factor analysis

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


class factor_analysis():
    def __init__(self,data,col,n):
        self.data=data
        self.col=col
        self.n=n
    def test(self):
        self.chi_square_value,self.p_value=calculate_bartlett_sphericity(self.data[self.col])
        self.kmo_all,self.kmo_model=calculate_kmo(self.data[self.col])
        return self.chi_square_value,self.p_value,self.kmo_all,self.kmo_model
    def analysis(self):
        self.fa = FactorAnalyzer(self.n, rotation=None)
        self.fa.fit(self.data[self.col])
        return self
    def _plot(self):
        ev, v = self.fa.get_eigenvalues()
        plt.scatter(range(1,self.data[self.col].shape[1]+1),ev)
        plt.plot(range(1,self.data[self.col].shape[1]+1),ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()
        df_cm = pd.DataFrame(np.abs(self.fa.loadings_), index=self.col)
        plt.figure(figsize = (14,14))
        ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
        # 设置y轴的字体的大小l
        ax.yaxis.set_tick_params(labelsize=15)
        plt.title('Factor Analysis', fontsize='xx-large')
        # Set y-axis label
        plt.ylabel('Sepal Width', fontsize='xx-large')
#        plt.savefig('factorAnalysis.png', dpi=500)

    def _transform(self):
        return  self.fa(self.data[self.col])
        

        
        

