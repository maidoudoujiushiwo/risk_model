# encoding:utf-8
import re
from itertools import combinations
import pandas as pd
import numpy as np
import datetime



#					COPYRIGHT: 
#							AUTH: jian.wu
#                           Email:fengyuguohou2010@hotmail.com
#							2016-12-29

def group_by_df(data, x, y):

    grouped = data[y].groupby(data[x])

    df_agg = grouped.agg(['sum', 'count'])    

    df_agg[x]=df_agg.index
    df_agg['go']=df_agg['count']-df_agg['sum']

    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.sort_values(by=[x], ascending=True)

    return df_agg




   
def ks_best(data,start,end):
    temp_df = data.loc[start:end]
    temp_sum = sum(temp_df['sum'])
    temp_go = sum(temp_df['go'])
    d1 = np.cumsum(temp_df['sum']) / temp_sum
    d2 = np.cumsum(temp_df['go']) / temp_go
    d3=abs(d1 - d2)
    ks_point = list(d3).index((max(d3)))

    return temp_df.index[ks_point]


def iv_best(data,start,end):
    def DF_Bina_Disc(df, t1, t0): 
        max_iv = 0
        df_iv = 0
        df_t1 = float(df['sum'].sum())
        df_t0 = float(df['count'].sum() - df_t1)
        df_iv_0 = (df_t1/t1 - df_t0/t0) * np.log((df_t1/t1) / (df_t0/t0))
        iv_point = 0
        for i in range(len(df.index)-1):
            df1 = df[df.index <= df.index[i]]
    
            df2 = df[df.index > df.index[i]]
    
            df1_t1 = float(df1['sum'].sum())
    
            df1_t0 = float(df1['count'].sum() - df1_t1)
    
            df2_t1 = float(df2['sum'].sum())
    
            df2_t0 = float(df2['count'].sum() - df2_t1)
            if (df1_t1+df1_t0) / (t1+t0) > 0.1 and (df2_t1+df2_t0) / (t1+t0) > 0.1:
                if df1_t1 * df1_t0 == 0:
                    df1_iv = 1
                else:
                    df1_iv = (df1_t1/t1 - df1_t0/t0) * np.log((df1_t1/t1) / (df1_t0/t0))
                if df2_t1 * df2_t0 == 0:
                    df2_iv = 1
                else:
                    df2_iv = (df2_t1/t1 - df2_t0/t0) * np.log((df2_t1/t1) / (df2_t0/t0))
                df_iv = df1_iv + df2_iv
            if df_iv > max_iv:
                max_iv = df_iv
                iv_point = df.index[i]
        df_iv_inc = max_iv - df_iv_0
        return [max_iv, df_iv_inc, iv_point]
    temp_df = data.loc[start:end]
    temp_sum = sum(temp_df['sum'])
    temp_go = sum(temp_df['go'])

    return DF_Bina_Disc(temp_df,temp_sum,temp_go)[2]



def best_point(data,start,end,rate,all_len,ks_c=True):

    temp_df = data.loc[start:end]
    temp_len = sum(temp_df['count'])

    start_l = sum(
        np.cumsum(temp_df['go'] + temp_df['sum']) < rate * all_len)
    end_r = sum(
        np.cumsum(temp_df['go'] + temp_df['sum']) <= temp_len - rate * all_len)
    start_new = start + start_l
    end_new = start + end_r - 1
    if end_new >= start_new:
        if sum(temp_df.iloc[start_new:end_new]['sum']) != 0 and sum(temp_df.iloc[start_new:end_new]['go']) != 0:            
            if ks_c:
                return ks_best(data,start_new,end_new)
            else:
                return iv_best(data,start_new,end_new)
        else:
            return None
    else:
        return None    
    
def best_cut(data, total_len, max_ti,rate, start, end, current,ks):
    temp_df = data.loc[start:end]
    temp_len = sum(temp_df['count'])

    if temp_len < rate * total_len * 2 or current >= max_ti:
        return []
    new_po = best_point(data,start,end,rate,total_len,ks_c=ks)
    if new_po is not None:

        l_list = best_cut(data, total_len, max_ti,rate, start, new_po, current+1,ks)
        r_list = best_cut(data, total_len, max_ti,rate, new_po+1, end, current+1,ks)
    else:
        l_list = []
        r_list = []
    new_list=l_list + [new_po] + r_list
    return list(filter(lambda x: x is not None, new_list))

def urteil(li):
    if len(li)<4:
        return 1
    else:
        lii=[li[i]-li[i-1] for i in range(len(li))[1:]]
        lii=list(map(lambda x:x if x!=0 else 1, lii))
#        print lii
        zz=np.sign([lii[i]/lii[i-1] for i in range(len(lii))[1:]]).sum()
        if zz in [len(li)-2,len(li)-4]:
            return 1
        else:
            return 0
        
def IV_choose(data,new_list,ur=False):

    temp_list = []
    for i in range(1, len(new_list)):
        if i == 1:

            temp_list.append(data.loc[new_list[i - 1]:new_list[i]])
        else:
            temp_list.append(data.loc[new_list[i - 1] + 1:new_list[i]])
    total_good = sum(data['go'])
    total_bad = sum(data['sum'])
    good_percent_series = pd.Series(list(map(lambda x: float(sum(x['go'])) / total_good, temp_list)))
    bad_percent_series = pd.Series(list(map(lambda x: float(sum(x['sum'])) / total_bad, temp_list)))

    woe_list = list(np.log(good_percent_series / bad_percent_series))

    IV_series = (good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series)
    if np.inf in list(IV_series) or -np.inf in list(IV_series):
        return None
    if ur:
        if urteil(woe_list)==0:
            return None    
        else:
            return sum(IV_series)    

    if sorted(woe_list)==woe_list or sorted(woe_list,reverse=True)==woe_list:
        return sum(IV_series)
    else:
        return None


def _combination(data,piece_num, cut_off_list,ur):
    point_list = list(combinations(cut_off_list, piece_num - 1))
#避免向下不是最优（注意）
    point_list = list(combinations(cut_off_list, piece_num - 2))+point_list
    point_list = list( map(lambda x: sorted(x + (0, len(data) - 1)), point_list))
    print (len(point_list))
    bins = list(map(lambda x: IV_choose(data,x,ur), point_list))
    bins_IV = list(filter(lambda x: x is not None, bins))
    if len(bins_IV) == 0:
        print('no suitbale bins for ' + str(piece_num) + ' pieces')
        return None,None
    else:
        inde=bins.index(max(bins_IV))
        return point_list[inde],bins[inde]


def bins_out(data, max_piece, cut_off_list,ur):
    piece_num = min(max_piece, len(cut_off_list) + 1)
    if piece_num == 1:
        return cut_off_list
    for c_piece_num in sorted(range(2, piece_num + 1), reverse=True):
        result,iv = _combination(data, c_piece_num, cut_off_list,ur)
        if c_piece_num==2 and iv is not None:
            if iv<0.03:
                return None
        if result is not None:
            return result,iv
    return None


#campare two mthode of cut:mononie and U.
def bins_out_result(data,max_piece,cut_off_list,ur):
    
    if ur:
        x0=bins_out(data, max_piece, cut_off_list,ur)
        x1=bins_out(data, max_piece, cut_off_list,False)        
        if x0 is None and x1 is None:
            print("no suitbale")
            return [0, len(data) - 1]
        elif x0 is None  and x1 is not None:
            return x1[0]
        elif x0 is not None  and x1 is  None:
            return x0[0]
        else:
            if x0[1]/(x1[1]+0.001)>2:
                return x0[0]
            else:
                return x1[0]
    else:
        x1=bins_out(data, max_piece, cut_off_list,ur)
        if x1 is None:
            print("no suitbale ")
            return [0, len(data) - 1]
        else:
            return x1[0]        


def calculator(data_df, x, new_list, na_df):
    if len(na_df) != 0:
        total_good = sum(data_df['go']) + sum(na_df['go'])
        total_bad = sum(data_df['sum']) + sum(na_df['sum'])
        na_good_percent = na_df['go'] / float(total_good)
        na_bad_percent = na_df['sum'] / float(total_bad)
        na_indicator = pd.DataFrame({'Bin': list(na_df[[x]].iloc[:,0]), 'KS': [None]*len(na_df), 'WOE': list(np.log(na_bad_percent/na_good_percent)),
                                     'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                                     'total_count': list(na_df['go'] + na_df['sum']),
                                     'bad_rate': list(na_df['sum'] /(na_df['go'] + na_df['sum']))})
    else:
        total_good = sum(data_df['go'])
        total_bad = sum(data_df['sum'])
        na_indicator = pd.DataFrame()
    default_CDF = np.cumsum(data_df['sum']) / total_bad
    undefault_CDF = np.cumsum(data_df['go']) / total_good
    ks_list = list(abs(default_CDF - undefault_CDF).loc[new_list[:len(new_list) - 1]])
    temp_df_list = []
    bin_list = []
    for i in range(1, len(new_list)):
        if i == 1:
            temp_df_list.append(data_df.loc[new_list[i - 1]:new_list[i]])
            bin_list.append('(-inf, ' + str(data_df[x][new_list[i]]) + ']')
        else:
            temp_df_list.append(data_df.loc[new_list[i - 1] + 1:new_list[i]])
            if i == len(new_list) - 1:
                bin_list.append('(' +str( data_df[x][new_list[i - 1]]) + ', inf)')
            else:
                bin_list.append(
                    '(' + str(data_df[x][new_list[i - 1]]) + ', ' + str(
                        data_df[x][new_list[i]]) + ']')
    good_percent_series = pd.Series(list(map(lambda x: float(sum(x['go'])) / total_good, temp_df_list)))
    bad_percent_series = pd.Series(list(map(lambda x: float(sum(x['sum'])) / total_bad, temp_df_list)))
    woe_list = list(np.log(bad_percent_series/good_percent_series))   
    IV_list = list((good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series))
    total_list = list(map(lambda x: sum(x['go']) + sum(x['sum']), temp_df_list))
    bad_rate_list = list(map(lambda x: float(sum(x['sum'])) / (sum(x['go']) + sum(x['sum'])), temp_df_list))
    non_na_indicator = pd.DataFrame({'Bin': bin_list, 'KS': ks_list, 'WOE': woe_list, 'IV': IV_list,
                                     'total_count': total_list, 'bad_rate': bad_rate_list})
    result_indicator = pd.concat([non_na_indicator, na_indicator], axis=0).reset_index(drop=True)
    return result_indicator



def all_get(data, na_df, total, piece, rate, x,out_in_list,ks,ur):

    cut_off_list = best_cut(data, total, piece,rate, 0, len(data), 0,ks)
    print (cut_off_list)
    best_knots = bins_out_result(data,piece,cut_off_list,ur)
    if best_knots==[] and (min(data['sum'])>0 and min(data['go'])>0):
        na_df=na_df.append(data) 
        out_in_list=out_in_list+list(data.iloc[:,0])  
    return calculator(data,x, best_knots, na_df),out_in_list,best_knots


 
    


def Bin_best(y, x, data=pd.DataFrame(), piece=5, rate=0.05, min_=50, out_in_list=[],ks=True,ur=False):

    if len(data) == 0:
        print ('no data')
        return pd.DataFrame()
    data = data.loc[data.index, [x, y]]


    if len(data) == 0:
        return pd.DataFrame()

    data[x] = data[x].astype(str)

    out_in_list = out_in_list + ['None', 'nan',np.nan,np.inf,-np.inf,'inf','-inf']

    na_df = data.loc[data[x].apply(lambda x: x in out_in_list)]


    non_na_df = data.loc[data[x].apply(lambda x: x not in out_in_list)]

#    try:
#        na_df[x] = na_df[x].astype(float)
#    
#    except:
#        pass

    try:
        non_na_df[x] = non_na_df[x].astype(float)    
    except:
        pass


    # generate the grouped_by format which is used for the later process
    na_df = group_by_df(na_df, x,y)
    non_na_df = group_by_df(non_na_df, x,y)
#    print factor_name
    if len(non_na_df) == 0:
        print('sry, missing x')
        return pd.DataFrame(),out_in_list
    total = len(data)
    min_rate = min_/float(total)
    rate = max(rate, min_rate)
    result,out_in_list,best_s = all_get(non_na_df, na_df, total, piece, rate, x,out_in_list,ks,ur)
    # print(time.localtime(time.time()))
    if len(best_s)==2:
        print('sry, no suitable')
        return pd.DataFrame(),out_in_list
    return result,out_in_list



def var_woe(x, bin_dic, out_in_list):
    val = None
    if str(x) in out_in_list and pd.isnull(x) is False:
        for woe in bin_dic:
            if float(bin_dic[woe][0].lstrip().rstrip()) == x:
                val = woe
    elif pd.isnull(x):
        for woe in bin_dic:
            if bin_dic[woe] == ['nan']:
                val = woe                
    else:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            if end_points[0].lstrip().rstrip() not in out_in_list:
                if end_points[0].lstrip().rstrip() == '-inf':
                    if x <= float(end_points[1].lstrip().rstrip()):
                        val = woe
                elif end_points[1].lstrip().rstrip() == 'inf':
                    if x > float(end_points[0].lstrip().rstrip()):
                        val = woe
                elif (x > float(end_points[0].lstrip().rstrip())) & (x <= float(end_points[1].lstrip().rstrip())):
                    val = woe
    return val


#y='y', data=data[['sum_hry_nof_m2_days']+['y']], data1=pd.DataFrame(), piece=5, rate=0.05, min_size=50, out_in_list=[], not_var_list=[], flag_list=[],kkks=False,ur=False

def df_woe(y, data=pd.DataFrame(), data1=pd.DataFrame(), piece=5, rate=0.05, min_size=50, out_in_list=[], not_var_list=[], flag_list=[],kkks=True,ur=False):
    data_woe = data[flag_list]
    if len(data1)>0:
        data_woe1 = data1[flag_list]        
 
    data_bin = pd.DataFrame()
    if len(data) == 0:
        print ('Original input data is empty')
        return pd.DataFrame()
    var_list = data.columns
    not_var_list.extend([y])
    not_var_list.extend(out_in_list)
    out_in_list.extend(['None', 'nan'])
    not_max_var = []
    for var in data.columns:
        percent = data[var].value_counts(normalize=True, dropna=False)

        if percent.max() >= 1-rate:
            not_max_var.append(var)
    target = list(set(var_list) - set(not_var_list)-set(not_max_var))
    iv_list = []
    ks_list = []
    target_1=[]
    if len(target) == 0:
        print ('No variable')
        return pd.DataFrame()
    iter = 0
    for var in target: 
        
        print (var)
        try:
            var_stat,out_in_list_1=Bin_best(y, var, data, piece, rate, min_size, out_in_list,kkks,ur) 

#            var_stat,not_in_list_1= Best_KS_Bin(y, var, data, 'bad','good', piece, rate,50, [])

            if len(var_stat) > 0:
                   if len(var_stat['WOE']) != len(set(var_stat['WOE'])):
                    var_stat.iloc[var_stat['Bin']=='NA','WOE'] = var_stat.iloc[var_stat['Bin']=='NA','WOE']+0.0000001
                var_stat['var'] = var
                
                var_stat['WOE']=var_stat[['total_count','WOE']].apply(lambda x: 0 if x[0]<len(data)*0.05 else x[1],axis=1)            
                bin_dic = dict(zip(var_stat['WOE'], var_stat['Bin']))
                for woe in bin_dic:
                    match_case = re.compile("\(|\)|\[|\]")
                    end_points = match_case.sub('', bin_dic[woe]).split(', ')
                    bin_dic[woe] = end_points
                data_woe[var] = list(map(lambda x: var_woe(x, bin_dic, out_in_list_1), data[var].map(lambda x: float(x))))
                if len(data1)>0:   
                    data_woe1[var] = list(map(lambda x: var_woe(x, bin_dic, out_in_list_1), data1[var].map(lambda x: float(x))))
                ivv=list(var_stat['IV'])
                while float('inf') in ivv:
                    ivv.remove(float('inf'))
                iv = sum(ivv)
                ks = max(var_stat['KS'])
                data_bin = pd.concat([data_bin, var_stat])
                # info_dic.update({var: [iv, ks]})
                iv_list.append(iv)
                ks_list.append(ks)
                iter += 1
                print (iter)
                target_1.append(var)
            else:
    #                iv_list.append('nan')
    #                ks_list.append('nan')
                print (var, 'checked')
        except:
            print (var+'--error')
            pass

    data_stat = pd.DataFrame({'var': target_1, 'iv': iv_list, 'ks': ks_list}).sort_values(by='iv', ascending=False)
    if len(data1)>0:
        data_woe.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        data_woe1.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        return data_woe,data_woe1, data_bin, data_stat
    else:
        data_woe.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        return data_woe, data_bin, data_stat



class ff_bin_woe:
    def __init__(self,y,data,data1,piece,rate,min_size,out_in_list,not_var_list,
                 flag_list,ks,ur): 
        self.y=y
        self.data=data
        self.data1=data1
        self.piece=piece
        self.rate=rate
        self.min_size=min_size
        self.out_in_list=out_in_list
        self.not_var_list=not_var_list
        self.flag_list=flag_list
        self.ks=ks
        self.ur=ur
        self.d0=None
        self.d1=None
        self.messa=None
        self.iv=None
    
    def woe(self):    
        if len(self.data1)>0:
            self.d0,self.d1,self.messa,self.iv=df_woe(
                self.y
                ,self.data
                ,self.data1
                ,self.piece
                ,self.rate
                ,self.min_size
                ,self.out_in_list
                ,self.not_var_list
                ,self.flag_list
                ,self.ks
                ,self.ur)
        else:
            self.d0,self.messa,self.iv=df_woe(
            self.y
            ,self.data
            ,self.data1
            ,self.piece
            ,self.rate
            ,self.min_size
            ,self.out_in_list
            ,self.not_var_list
            ,self.flag_list
            ,self.ks
            ,self.ur)
        return self
    def dwoe(self,m):

        if self.messa is None:
            print('error：未定义woe')
            return None
        else:
            col=list(self.iv['var'])
            
            for k in col:
                out_in_list=self.out_in_list
                cd=self.messa[self.messa['var']==k].copy()
                cd['1']=range(len(cd))
                cd['WOE']=cd['WOE']+cd['1']/100000
                dc=dict()
                for l in range(len(cd)):                
                    dc[list(cd['WOE'])[l]]=cd['Bin'][l].replace(')','').replace('(','').replace(']','').split(',')
                    if len(dc[list(cd['WOE'])[l]])==1:
                        dc[list(cd['WOE'])[l]]=dc[list(cd['WOE'])[l]][0]
                for woe in dc:
                    if len(dc[woe])!=2:
                        match_case = re.compile("\(|\)|\[|\]")
                        end_points = match_case.sub('', dc[woe]).split(', ')
                        dc[woe] = end_points
                    
                for l in dc:
                    if len(dc[l])==1:
                        out_in_list.append(dc[l][0])
        
                m[k]= list(map(lambda x: var_woe(x, dc, out_in_list), m[k].map(lambda x: float(x))))
            return m    

    def woe_cal(self,m):
        if self.messa is None:
            print('error：未定义woe')
            return None
        else:
            col=list(self.iv['var'])
            iv=[]
            for fac in col:
                na_df = group_by_df(m,fac,self.y)
                total_good = sum(na_df['go'])
                total_bad = sum(na_df['sum'])
                na_good_percent = (na_df['go']+1) / float(total_good)
                na_bad_percent = (na_df['sum']+1) / float(total_bad)
                na_indicator = pd.DataFrame({'Bin': list(na_df.iloc[:,0]), 'KS': [None]*len(na_df), 'WOE': list(np.log(na_bad_percent/na_good_percent)),
                                             'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                                             'total_count': list(na_df['go'] + na_df['sum']),
                                             'bad_rate': list(na_df['sum'] /(na_df['go'] + na_df['sum']))})    
                na_indicator['var'] = fac
                iv.append(na_indicator['IV'].sum())
        
            return pd.DataFrame({'var':col,'iv':iv})

#
#
#class f_bin_woe:
#    def __init__(self): 
#        self.flag=''
#        self.model=pd.DataFrame()
#        self.col=[]
#        self.cor_method='pearson'
#        self.p_cri=0.5
#        self.ivv=pd.DataFrame({'v':[],'iv':[],'i_index':[],'woe_t':[]})
#        self.id=''
#        self.ce=[]
#        self.obj=pd.DataFrame({'n':[],'obj':[],'re':[]})
#        self.N=5
#        self.coo=[]
#
#    def load(self,df,a,b,c,d):
#        a=copy.copy(a)
#        b=copy.copy(b)
#        c=copy.copy(c)
#        if b in a:
#            a.remove(b)
#        if c in a:
#            a.remove(c)            
#        self.model=df[a+[b]+[c]]
#        self.flag=b
#        self.col=list(set(a))        
#        self.id=c
#        self.model.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu'],np.nan,inplace=True)
#        self.N=d
##计算iv最大的woe 
#    def df_dis(self,df,y):
#        df=copy.copy(df)
#        for x in self.col:
#            if df[x].dtype=='object':
#                df_agg = Cal_WOE(df,x, y)
#                ob=pd.DataFrame({'n':[x],'obj':[list(df_agg.index)],'re':[list(df_agg[x + '_woe'])]})
#                self.obj=self.obj.append(ob,ignore_index=True)
#                df[x] = df[x].replace(df_agg.index, df_agg[x + '_woe'])
#        return df

#def dwoe(m,dddf2,col,not_in_list=[]):
#    for k in col:
#        cd=dddf2[dddf2['var']==k].copy()
#        cd['1']=range(len(cd))
#        cd['WOE']=cd['WOE']+cd['1']/100000
#        dc=dict()
#        for l in range(len(cd)):                
#            dc[list(cd['WOE'])[l]]=cd['Bin'][l].replace(')','').replace('(','').replace(']','').split(',')
#            if len(dc[list(cd['WOE'])[l]])==1:
#                dc[list(cd['WOE'])[l]]=dc[list(cd['WOE'])[l]][0]
#        for woe in dc:
#            if len(dc[woe])!=2:
#                match_case = re.compile("\(|\)|\[|\]")
#                end_points = match_case.sub('', dc[woe]).split(', ')
#                dc[woe] = end_points
#            
#        for l in dc:
#            if len(dc[l])==1:
#                not_in_list.append(dc[l][0])
#
#        m[k]= list(map(lambda x: var_woe(x, dc, not_in_list), m[k].map(lambda x: float(x))))
#    return m
#
#def woe_cal(m,col,y):
#    iv=[]
#    for fac in col:
#        na_df = group_by_df(m, y, fac, 'go', 'sum', False)
#        total_good = sum(na_df['go'])
#        total_bad = sum(na_df['sum'])
#        na_good_percent = (na_df['go']+1) / float(total_good)
#        na_bad_percent = (na_df['sum']+1) / float(total_bad)
#        na_indicator = pd.DataFrame({'Bin': list(na_df.iloc[:,0]), 'KS': [None]*len(na_df), 'WOE': list(np.log(na_bad_percent/na_good_percent)),
#                                     'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
#                                     'total_count': list(na_df['go'] + na_df['sum']),
#                                     'bad_rate': list(na_df['sum'] /(na_df['go'] + na_df['sum']))})    
#        na_indicator['var'] = fac
#        iv.append(na_indicator['IV'].sum())
#
#    return pd.DataFrame({'var':col,'iv':iv})


    
#
#ceshi=ff_bin_woe('y',Data[col+['y']], pd.DataFrame(), 5, 0.05, 50, [], [], ['y'],False,False)
##
##
#ceshi.woe()
##ceshi0=ceshi.d0
##
##ceshi1=ceshi.d1
##
##ceshi2=ceshi.d2
##
##
##ceshi.d3
##
#dd=ceshi.dwoe(Data.copy())
##
##
##ddd=ceshi.woe_cal(dd)
##    
##
##
#iv=ceshi.iv
##
##iv==ceshi2
##
##iv['3']=ceshi2['iv']
##
##
##
##
##iv['4']=iv['3']-iv['iv']
#
#dir(ceshi)
#
#
#
#ceshi.d1
#
