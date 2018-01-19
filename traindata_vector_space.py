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


path="train_word_bag/train_set.dat"
bunch=readbunchobj(path)

#停用词
stopword_path="stop_words.txt"
stpwrdlst=readfile(stopword_path).splitlines()
#构建TF-IDF词向量空间对象
tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
#使用TfidVectorizer初始化向量空间模型
vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
transfoemer=TfidfTransformer()#该类会统计每个词语的TF-IDF权值

#文本转为词频矩阵，单独保存字典文件
tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary=vectorizer.vocabulary_

#创建词袋的持久化
space_path="train_word_bag/tfidfspace.dat"
writebunchobj(space_path,tfidfspace)