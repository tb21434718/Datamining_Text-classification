import pickle
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法包

def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


# 导入训练集向量空间
trainpath = "train_word_bag/tfidfspace.dat"
train_set = readbunchobj(trainpath)
# d导入测试集向量空间
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
# 应用贝叶斯算法
# alpha:0.001 alpha 越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
total = len(predicted);
rate = 0
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        rate += 1
        print(file_name, ": 实际类别：", flabel, "-->预测分类：", expct_cate)
# 精度
print("error_rate:", float(rate) * 100 / float(total), "%")

from sklearn import metrics


def metrics_result(actual,predict):
    print("精度：{0:.3f}".format(metrics.precision_score(actual,predict,average='weighted')))
    print("召回：{0:0.3f}".format(metrics.recall_score(actual,predict,average='weighted')))
    print("f1-score:{0:.3f}".format(metrics.f1_score(actual,predict,average='weighted')))


metrics_result(test_set.label, predicted)