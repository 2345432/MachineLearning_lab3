import pickle
import numpy as np
class AdaBoostClassifier:
    def __init__(self, weak_classifier, n_weakers_limit):
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        pass
    def is_good_enough(self):
        '''Optional'''
        pass
    def fit(self,X,y):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        w=np.ones(X.shape[0])
        w=w/X.shape[0]
        alpha=0
        for i in range(self.n_weakers_limit):
            print(i)
            self.weak_classifier.fit(X,y,w)
            predict_y=self.weak_classifier.predict(X)
            error=0
            for i in range(len(predict_y)):
                if(predict_y[i]!=y[i]):
                    error+=1
            error_rate=error/len(predict_y)
            print("error_rate",error_rate)
            #错误率大于0.5时跳过该弱分类器
            #if(error_rate>=0.5):
            #    break
            #防止错误率为0时报错
            if(error_rate<np.exp (-16)):
                error_rate=np.exp (-16)

            alpha=0.5*np.log((1-error_rate)/error_rate)
            z=0
            for j in range(w.shape[0]):
                z+=w[j]*(np.exp(-(alpha*y[j]*predict_y[j])))
            for j in range(w.shape[0]):
                w[j]=w[j]*(np.exp(-(alpha*y[j]*predict_y[j])))/z
            #将该分类器与其权重alpha保存至文件中
            self.save([self.weak_classifier,alpha],"C:\\Users\\Administrator\\Desktop\\data\\"+str(i)+".txt")
    def predict_scores(self, X):
        alpha_list = []
        weak_classification_list = []
        y = np.zeros(X.shape[0])
        for i in range(self.n_weakers_limit):
            #读取保存的弱分类器文件
            weak_classification,alpha  = self.load("C:\\Users\\Administrator\\Desktop\\data\\" + str(i) + ".txt")
            alpha_list.append(alpha)
            weak_classification_list.append(weak_classification)
        for i in range(self.n_weakers_limit):
            #进行预测
            y += weak_classification_list[i].predict(X) * alpha_list[i]
        return y
    def predict(self, X, threshold=0):
        y=self.predict_scores(X)
        for i in range(y.shape[0]):
            if(y[i]>=threshold):
                y[i]=1
            else:
                y[i]=-1
        return y
    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)
    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
