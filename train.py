from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 读取训练集
    x = pickle.load(open("C:\\Users\\Administrator\\Desktop\\data\\savex.txt", "rb"))
    y = pickle.load(open("C:\\Users\\Administrator\\Desktop\\data\\savey.txt", "rb"))



    x = np.mat(x)
    y = np.mat(y).T
    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    #划分30%为测试集
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=2)

    #初始化
    num_weak_classifier=5
    num_list=[]
    pred_score_list=[]
    max_score=0
    report=''

    #弱分类器数目从5到50进行实验
    while(num_weak_classifier<=50):
        print('弱分类器数量：',num_weak_classifier)
        num_list.append(num_weak_classifier)

        #定义弱分类器
        b = DecisionTreeClassifier(splitter='random',max_depth=4)
        #定义adaboost分类器
        a = AdaBoostClassifier(b, num_weak_classifier)
        #使用训练集进行训练
        a.fit(X_train, y_train)
        #对测试集进行分类
        y_pred = a.predict(X_val)
        #计算准确率
        correct = 0
        for i in range(y_pred.shape[0]):
            if (y_pred[i] == y_val[i]):
                correct += 1
        score=correct/y_val.shape[0]
        print('准确率：',score)
        pred_score_list.append(score)

        #生成准确率最高时的报告
        if(score>max_score):
            max_score=score
            report = classification_report(y_val, y_pred)
        num_weak_classifier+=1


    print(report)
    #画图
    plt.xlabel('number of weak classifier')
    plt.ylabel('classification accuracy')
    plt.plot(num_list, pred_score_list)
    plt.show()
    #保存report
    file = open('C:\\Users\\Administrator\\Desktop\\data\\report.txt', 'w')
    file.write(report)
    file.close()

