import os
from sklearn.datasets.base import Bunch
import pickle#持久化类
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer#TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer#TF-IDF向量生成类




def readbunchobj(path):
    file_obj=open(path,"rb")
    bunch=pickle.load(file_obj)
    file_obj.close()
    return bunch
def writebunchobj(path,bunchobj):
    file_obj=open(path,"wb")
    pickle.dump(bunchobj,file_obj)
    file_obj.close()
def readfile(path):
    fp = open(path, "r", encoding='gb2312', errors='ignore')
    content = fp.read()
    fp.close()
    return content

#导入分词后的词向量bunch对象
path="test_word_bag/test_set.dat"
bunch=readbunchobj(path)

#停用词
stopword_path="stop_words.txt"
stpwrdlst=readfile(stopword_path).splitlines()

#构建测试集TF-IDF向量空间
testspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})

#导入训练集的词袋
trainbunch=readbunchobj("train_word_bag/tfidfspace.dat")

#使用TfidfVectorizer初始化向量空间
vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,vocabulary=trainbunch.vocabulary)
transformer=TfidfTransformer();
testspace.tdm=vectorizer.fit_transform(bunch.contents)
testspace.vocabulary=trainbunch.vocabulary

#创建词袋的持久化
space_path="test_word_bag/testspace.dat"
writebunchobj(space_path,testspace)