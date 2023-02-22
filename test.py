import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# tree = dtc()
def get_watermalon():
    data = pd.read_excel('watermelon20.xlsx')
    features_list = ['色泽','根蒂','敲声','纹理','脐部','触感','好瓜']
    for key in features_list:
        data[key] = pd.factorize(data[key])[0]
    target = data['好瓜'].values
    data = data.drop('好瓜',axis=1)
    return data, target

def get_data(path):
    data = pd.read_csv(path)
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    # data['tenure']=pd.to_numeric(data['tenure'],errors='coerce')
    data.loc[data['TotalCharges'].isnull().values==True,'TotalCharges'] = data[data['TotalCharges'].isnull().values==True]['MonthlyCharges']
    # data = data[['MultipleLines','tenure','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    data = data[['MonthlyCharges', 'TotalCharges', 'Churn']]
    columns = data.columns.to_list()
    for key in columns:
        if key != 'TotalCharges':
            data[key] = pd.factorize(data[key])[0]
    
    target = data['Churn'].values # 取得所有的 y
    # data = data.drop('Churn', axis=1).to_numpy() # 取得 X
    data = data.drop('Churn', axis=1)
    return data, target


class AdaBoost:

    def __init__(self, K, T, max_dep):
        # 注意这里的分类器可以不是同一个模型：可以是m个Logistic，也可以是一些朴素贝叶斯+Logistic
        self.clf = [dtc(criterion='gini', max_depth=max_dep) for _ in range(K)]
        self.K = K
        self.T = 50 if T == None else T
        # 缓存基分类器和权重参数
        self.clf_arr = []
        self.alpha_arr = []
        self.y_label = None
    def train_predict(self, X):
        y_pred = np.zeros((self.K, X.shape[0]))
        for i in range(self.K):
            y_pred[i] = self.clf[i].predict(X)
        res = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            vals,counts = np.unique(y_pred[:,i], return_counts=True)
            max_index = np.argmax(counts)
            res[i] = vals[max_index]
        return res

    def fit(self, X, y):
        self.y_label = list(set(y))
        # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.30, stratify = target, random_state = 1)
        num = X.shape[0]
        self.w = np.ones(num) / num
        self.beta = np.zeros(self.T)
        res = np.zeros(X.shape[1])

        for t in range(1, self.T):
            for i in range(self.K):
                # 这里默认每个模型都有 sklearn 风格的 model.fit() 函数
                # 并且将 W 作为已知权重代入其中
                # D(i) 以 取1的概率大小 为准
                self.clf[i].fit(X, y, sample_weight = self.w)
            # self.clf_arr.extend([self.clf])
            # 预测结果、预测概率
            y_pred = self.train_predict(X)
            # y_pred = self.clf[t].predict(X)
            print("目前的精确度：{}".format(np.sum(y_pred==y.reshape(-1))/len(y_pred)))
            not_equal = (y_pred != y).reshape(-1)
            epsilon = np.dot(self.w, not_equal)
            if epsilon > 0.5:
                self.T = t - 1
                return
            beta_t = 1 / (1 - epsilon)
            self.beta[t] = beta_t
            temp = np.ones(len(not_equal)) * not_equal
            self.w = self.w * (beta_t ** (1 - temp))
            self.w /= np.sum(self.w)
        self.beta = np.log(1 / self.beta)
        # for i in range(X.shape[1]):
        #     res[i] = 
        return

    def predict(self, X):
        res = np.zeros((len(self.y_label), X.shape[0]))
        self.beta = np.log(1 / self.beta) # (1, (len.self.T))
        for ans in range(len(self.y_label)):
            pred_array = np.zeros((self.T, X.shape[0]))
            for i in range(self.K):
                pred_array[i] = (self.clf[i].predict(X) == self.y_label[ans]).reshape(-1)
            res[ans] = np.dot(self.beta, pred_array)
        res = np.argmax(res, axis=0)
        return res
    
if __name__ == '__main__':
    
    data, target = get_data(path='WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # print(data)
    # iris = load_iris()
    # data, target = iris.data, iris.target
    # 在外面转成numpy送进去
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size= 0.35, stratify = target, random_state = 30)
    # LDA_predict(train_x, test_x, train_y, test_y)
    # bayes_predict(train_x, test_x, train_y, test_y)
    # print()
    ada = AdaBoost(K=10, T=30, max_dep=3)
    ada.fit(train_x, train_y)
    y_pred = ada.train_predict(test_x)
    print("(AdaBoost)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    another = dtc(criterion='gini', max_depth=3)
    another.fit(train_x, train_y)
    y_pred = another.predict(test_x)
    print("(AdaBoost2)测试集预测精度为acc=",np.sum(y_pred==test_y.reshape(-1))/len(y_pred))
    print([*zip([i for i in range(train_x.shape[1])], another.feature_importances_)])
    dot_data = tree.export_graphviz(another, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('graph')