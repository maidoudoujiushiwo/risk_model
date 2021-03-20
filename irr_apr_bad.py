# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:26:40 2020

@author: wu
"""


def irr_apr(rate,n):
#    n=12
    a=10000
    f=a*rate/12    
    f1=0
    ff=[f1+(f-f1)/10000*k for k in range(10000)]
    for l in ff:
        b=[float(a)/n+l]*n        
        r=np.irr([-10000]+b)
        irr=(r+ 1)** 12- 1
#        print(irr-rate)
        if  abs(irr-rate)<0.0001:
            return l/10000.0,r




def macd(rate,m,n):
    r=irr_apr(rate,n)[0]
    pv=0
    npv=0
    for l in range(m+1)[1:-1]:
#        print(l)
        pv=pv+1.0/n/(1+r)**(l)*(l)
        npv=npv+1.0/n/(1+r)**(l)
    pv=pv+(n-m)/n/(1+r)**(m)*(m)
    npv=npv+float(n-m)/n/(1+r)**(m)
    return pv/npv
    

def loss(rate,bad,m,n):
    mad=macd(rate,m,n)
    br=bad*12/mad
    ra=rate-br
    for i in range(10):
        mad=macd(ra,m,n)
        br=bad*12/mad
#        print(br,mad)
        ra=rate-br
    return br


def rrt(rate,m,n):
    ll=0
    for l in range(int(m)):
        ll=ll+(n-l)/float(n)
    j=m-int(m)
    ll=ll+j*(n-int(m))/float(n)
    ll=ll/float(m)
    return 12/m*rate/ll
