# -*- coding: utf-8 -*-

###以西瓜书76为例决策树
import pandas as pd
import numpy as np
import math
import copy
# pip install openpyxl


#计算信息熵
def Information_Entropy(plist):
    ent = 0
    for p in plist:
        ent += p*math.log(p,2)
    ent = -ent
    return ent

def get_p(feature,data):
    idx = data[feature].value_counts().index.tolist()
    val = data[feature].value_counts().tolist()
    sum_val = sum(val)
    plist = []
    for i in val:
        plist.append(i/sum_val)
    return idx,plist

class Node:
    def __init__(self, tol, feature_index, condition=None, next=None, result=None):
        self.leaf = True if condition == None else False
        # if self.leaf == True:
            # self.result = 
        self.label_set = torch.unique(self.label) # 0, 1
        self.labels_full = labels_full
        self.max_depth = max_depth
        self.decide = {}

def Decision_Tree(data, y_list,  root_node,feature_list,entD):
    #data:根节点未划分的数据
    #root_node:根节点
    #feature:当前属性集合（分支节点）
    #entD:根节点的信息熵
    #example for西瓜数据集2.0:
    #对于初始时，data为整个数据集，root_node为好瓜，feature为要考察的属性（分支节点）,entD为好瓜坏瓜信息熵
#        total_num = data.shape[0]##根节点D的样本个数    
    features_list = [i for i in range(data.shape[1])]
    y_index = features_list[-1]
    print('输入根节点:',root_node)
    print('取值:',data[root_node].values[0])
    
    ####判断递归结束
    # if entD==0:
    #     print('Wrong!') if data['Churn'].values[0]==y_list[0] else print('Right!')
    #     return
    # 如果全属于同一类别 C：
    if len(data[:,y_index].unique()) == 1:
        return Node(condition=None, result=data[0, y_index])
    
    if len(feature_list) == 0 or len(data) == 0:
        vals,counts = np.unique(data[:,y_index], return_counts=True)
        max_index = np.argmax(counts)
        return Node(result=vals[max_index])
    
    gain_step1 = []###所有属性的增益
    gain_ratio_step = [] ###所有属性的信息增益率
    ent_whole_step1 = [] ###所有属性的各个取值的信息熵
    ent_whole_name = []  ###所有属性的各个取值的名字
    
    #注意：如果直接将features_list输出，那么在这个分支remove了这个属性，对于其他分支也无法使用，将产生错误影响!!!!
    #因此输出和输入不是同一个列表!!!!!    
    features_list = copy.deepcopy(old_features_list)
#    features_list = old_features_list
    
    for feature in features_list: #对于不同属性
        grouped = data.groupby(feature)
        fe = data[feature].value_counts().index.tolist()
#        print(fe)
        subclass = []
        for f in fe:
            subclass.append(grouped.get_group(f))#按分支节点划分子类
        idx,plist = [],[]
        sub_num = []##每个子类的样本个数
        for sub in subclass:
            i,p = get_p('好瓜',sub) #计算每个子类好坏瓜的概率
            idx.append(i)
            plist.append(p)
            sub_num.append(sub.shape[0])##每个子类的样本个数
        ##多个分支节点的信息熵
        ent_whole = []  
        for per_plist in plist:
            per_ent = Information_Entropy(per_plist)
            ent_whole.append(per_ent)
            
        gain = get_Gain(sub_num,ent_whole,entD)
        gain_step1.append(gain)  #每个属性的信息增益,每个属性一个数  
        
        gain_ratio = get_Gain_Ratio(gain,sub_num,ent_whole,entD)#每个属性的信息增益率,每个属性一个数  
        gain_ratio_step.append(gain_ratio)    
            
        ent_whole_step1.append(ent_whole)          #这个属性几个子类的信息熵
        ent_whole_name.append(fe)                  #这个属性几个子类的信息熵对应的名字，如‘清晰’
        
        print('属性:',feature,'增益:',gain,'增益率:',gain_ratio)
   #####保存不同属性的增益，根据信息增益准则，找到信息增益最大的，选为划分属性
    re1 = save_re(features_list,gain_step1)
    re1.to_excel(str(root_node)+str(data[root_node].values[0])+'.xlsx')
    
    idmax = gain_step1.index(max(gain_step1))   
    root_node = features_list[idmax]  ##新的根节点“纹理”  
    entD = ent_whole_step1[idmax]
    entD_name = ent_whole_name[idmax]
    
    print('产生新的节点:',root_node,'下一阶段可取值:',entD_name)
    print('--------------------------------')    
    features_list.remove(root_node)  ##属性中 去除新的根节点
    
    die_data = data.groupby(root_node)
    
#    new_depart = data[root_node].value_counts().index.tolist()
    for i in range(len(entD_name)):
        print('根节点:',root_node,'------------','取值:',entD_name)
        Decision_Tree(die_data.get_group(entD_name[i]),root_node,features_list,entD[i])

#    return die_data,root_node,features_list,entD,entD_name
        
#        ent_whole_2 = ent_whole_step1[idmax]
        
def get_Gain(sub_num,ent_whole,entD):
    ##信息增益的计算
    #sub_num:该属性各个取值的个数D_v，求和为D
    #ent_whole:各个取值信息熵Ent(D_v)
    #entD:根节点信息熵Ent(D)
    add = 0
    total_num = sum(sub_num)
    for i in range(len(ent_whole)):
        add += (sub_num[i]/total_num)*ent_whole[i]
    return entD - add
#        Gain = get_Gain(sub_num,ent_whole,entD)
    
def get_Gain_Ratio(gain,sub_num,ent_whole,entD):
    #信息增益率的计算
    add = 0  #IV
    total_num = sum(sub_num)
    for i in range(len(ent_whole)):
        add += (sub_num[i]/total_num)*math.log((sub_num[i]/total_num))
    if add==0:
        add = add+0.00001
    return gain/(-add)
        
        
##保存每个属性的增益结果
def save_re(features_list,gain_step):
    features_listcp,gain_stepcp = features_list.copy(),gain_step.copy()
    features_listcp = np.array(features_listcp)
    gain_stepcp = np.array(gain_step)
    
    re = np.hstack((features_listcp.reshape((features_listcp.shape[0],1)),gain_stepcp.reshape((features_listcp.shape[0],1))))
    return pd.DataFrame(re,columns=['属性','信息增益'])


def get_data(path):
    data = pd.read_csv(path)
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    # data['tenure']=pd.to_numeric(data['tenure'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    data = data[['MultipleLines','tenure','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    columns = data.columns.to_list()
    for key in columns:
        if key != 'TotalCharges':
            data[key] = pd.factorize(data[key])[0]
    
    target = data['Churn'].values # 取得所有的 y
    # data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    data = data.to_numpy()
    return data, target

def main():
    # data = pd.read_excel('watermelon20.xlsx')
    data, target = get_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # print()
    #data.isnull().sum()
    #根节点
    features_list = [i for i in range(data.shape[1])]
    print(features_list)
    root_node = features_list[-1]
    # print(data[:, root_node])
    vals,counts = np.unique(data[:, root_node], return_counts=True)
    max_index = counts.argmax()
    
    # index = np.argmax(counts)
    # return Node(result=vals[index])
    print(vals)
    #y = data['好瓜'].value_counts().shape[0]        
    ##以好瓜为根节点，计算各个属性的信息增益
    # features_list = ['色泽','根蒂','敲声','纹理','脐部','触感']
    print()
    # y_list = list(set(target))
    # # print(y_list)
    # features_list = data.columns.to_list()
    # # print(data)
    # # features_list = list(set(target))
    # root_node = features_list[-1]
    
    # idx,plist = get_p(root_node,data)
    # entD = Information_Entropy(plist)
    
    # ##############
    # ##根节点为好瓜,进行递归
    # Decision_Tree(data,y_list,root_node,features_list,entD)

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-


