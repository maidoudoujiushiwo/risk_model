# 风控模型（risk_model）自动建模代码

   由于一些原因吧，跳坑太频繁了，所以难以回到一个大厂做风控模型，因此就把这几年积攒的一些代码公开出来，
如果有人能有所裨益，把风险做好，也算是种安慰，真的挺喜欢作风控的哈哈哈。当然可能存在一些bug，等有心情在
好好修改添加吧。

理论上这套代码连起来是可以实现评分卡自动变量处理，分箱，跑模型打分的。

如果发现bug或者使用问题欢迎发邮件给我：fengyuguohou2010@hotmail.com。

# 目录：

1、一些非传统衍生类的的数据处理，包括缺失值处理，label_encode,离散值woe赋值，基于y信息的tf-idf，基于四种无监督聚类（kmeans，pca，tsne，nmf）变量衍生，基于gbdt的变量衍生
有时间（后续会把基于rnn之类的衍生加上）。存在data_transform.py文件。

2、自动woe最优分箱（单调或者单峰且满足iv等条件），接近sas-em效果，不太需要手动介入调整。bin_new.py

3、变量选择：基于psi，相关性，iv，ks和变量聚类。auswahlen.py

4、因子分析：无监督的因子分析。factor_analysis.py

5、样本分布对齐，可以做类似样本的迁移学习，在大样本中召回更接近新业务小样本进行学习。distribution_adjust.py

6、建模代码，包括stewise选变量和蒙特卡洛选变量的逻辑回归，lgbm，rnn。以及基于shap和分箱对比的机器学习变量解释性代码。build_model.py

7、评分卡打分代码。score.py

# 应用解释：

# 1 data_transform.py 
里边就一个主要的类：feature_eng

ceshi=feature_eng(col1,col2,y, Data,Test_data)

col1:数值型变量的list

col2:离散型变量的list

y是label

Data 是训练集

Test_data是测试集

dir(feature_eng)查看方法 

drop_missing是处理缺失值，

gbdt_e是生成gbdt衍生变量，

label_encode是生成label_encode变量，

woe_encode用来使用woe给离散变量赋值，

y_tf_idf基于y的tf-idf衍生，

uns是生成四种无监督衍生（详见：https://github.com/maidoudoujiushiwo/Unsupervised ）

![image](https://github.com/maidoudoujiushiwo/risk_model/blob/master/image/tsne.png)

可以看到根据german_credit的数据画出图的标记，虽然是无监督生成的结果，但是对y的0和1区分能力很强很契合，可能有一定运气的成分。

# 2 bin_new.py

这是一个自动完成分箱几乎不用手动调整的代码，只需调用类：ff_bin_woe。

输入项中：
y label

data 输入的训练集

data1 可以同时输入测试集，没有测试集输入pd.DataFrame()

piece 期望分bin的最大箱数

rate 每一个分箱样本数最小希望占比

min_size 每一个分箱最小的样本数

out_in_list 特殊值，单独一箱

not_var_list 不需要分箱的变量
flag_list=[y] 存放lanel

ks 希望开始粗分箱根据ks还是iv，Ture为ks，False为iv。（智能输入这两个选项）

ur 希望分箱中有单峰还是希望所有变量都分为单调。只能选择True or False。

示例 

ceshi=ff_bin_woe('y',factor0[col+['y']], pd.DataFrame(), 5, 0.05, 50, [], [], ['y'],True,False)

ceshi.woe()

生成的结果中 ceshi.iv是存放iv结果，ceshi.d0存放分箱之后的数据集，ceshi.messa（比较重要，后续生成打分的文件也要用到）存放分箱的规则。

如果新来的一个测试集data，只需调用：dd=ceshi.dwoe(Data.copy())

既可以对新测试集完成woe变换。

注：ks=False 即采用iv做粗分箱，有可能会取得最终更高的iv的分bin，但是分箱速度回比ks粗分箱慢一点。

# 3 auswahlen.py

auswahle这个类主要实现三个功能，

1、psi计算，

2、根据相关性和单变量ks，去除相关性高的两个变量中ks较低的一个。

3、varclust变量聚类分析，类似sas中的 proc varclus

# 4 factor_analysis.py

实现无监督因子分析，可以结合业务经验预先判断维度的和衍生逻辑，比较适用一些简单的无监督打分场景。

示例：

ad=factor_analysis(Data,col,5)

data是训练集，col是变量list，5是指定的因子数目。

ad.test()

检测是否可以做因子分析，显著情况。

ad.analysis()

运行结果

ad._plot()

画图


german_credit的数据结果如下
![image](https://github.com/maidoudoujiushiwo/risk_model/blob/master/image/fa0.png)
![image](https://github.com/maidoudoujiushiwo/risk_model/blob/master/image/fa1.png)

# 5 build_model.py

stepwise,MonteCarlo是基于statsmodels中逻辑回归的两个评分卡模型，具体打分需要在调用score.py。

model_lgbm是一个简单lgbm的模型训练kernel，需要再加上grid-search.

MyRNN是一个简单的rnn模型示例。准备加入贷中数据，把类别embeding一下，concat上交易数据，做个rnn的b卡，看以后能去哪家在继续尝试吧。

shap_woe_explain是用来解释机器学习模型中变量是否合理的，如果woe和shap是相反的两种趋势，那么说明变量虽然看起来进入了模型，但是没有发挥应有的作用。

如图所示两种情况：

![image](https://github.com/maidoudoujiushiwo/risk_model/blob/master/image/微信图片_20200804221500.jpg)
![image](https://github.com/maidoudoujiushiwo/risk_model/blob/master/image/微信图片_20200804221506.jpg)

# 6 distribution_adjust.py

这部分可以看我之前写的文章，https://mp.weixin.qq.com/s/722dDYdKf2sMqP8wGwVvZA 。

利用CGAN（有监督对抗网络）选择建模样本

你也可以简单用个lgbm定义好label做这个，就是一个从大样本数据中召回和新业务样本类似数据的功能。


# 7 score.py

这一部分是按照odds，pdo基准和翻倍打分的代码，比较流程话。

score类的输入项：

model:logit = sm.Logit(m1[y],m1[col+['1']])          model = logit.fit() 是这里边的拟合之后的结果

col 是模型的入模变量。

messa 是代码2部分中ff_bin_woe生成的woe分箱的信息，就是类中的messa的信息。

base 是基准比如600.

odd是odd比如 20

pdo是pdo。

# 附录：根据vintage 风险水平，计算年化风险

大家经常irr，apr的聊着，从收息角度还能算的清楚，但是年化坏账能算清楚的同学就不多了，尤其是从vintage推到出来，年化坏账率。

这极大的制约了我们在资金资产谈判中快速判断风险收益，甚至很多项目启动了还不知道预期风险成本。要想在这种关键场合成功装逼

，下面的公式和代码可能会帮助到你。

提供两种计算思路

1、年化坏账率=vintage坏账率*12/久期

2、年化坏账率=年累计坏账金额/年月均余额=（12/平均在贷时长）*（vintage坏账/年月均余额）

下面上代码：https://github.com/maidoudoujiushiwo/risk_model 
中，irr_apr_bad.py


方法一：
rrt(0.03,6,12) vintage3+ badrate=0.03,平均在贷期限=6，产品期数12

方法二：

loss(0.18,0.07,18,36) irr利率18，vintage3+ badrate=0.07,平均在贷期限=18，产品期数36







