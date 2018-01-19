import os

import pickle
from sklearn.datasets.base import Bunch
#Bunch 类提供了一种key，value的对象形式
#target_name 所有分类集的名称列表
#label 每个文件的分类标签列表
#filenames 文件路径
#contents 分词后文件词向量形式
def readfile(path):
    fp = open(path, "r", encoding='gb2312', errors='ignore')
    content = fp.read()
    fp.close()
    return content

bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])

# wordbag_path="train_word_bag/train_set.dat"
# seg_path="train_seg/"

wordbag_path="test_word_bag/test_set.dat"#test_word_bag文件夹已经自己创建
seg_path="test_seg/"

catelist=os.listdir(seg_path)
bunch.target_name.extend(catelist)#将类别信息保存到Bunch对象

for mydir in catelist:
    class_path=seg_path+mydir+"/"
    file_list=os.listdir(class_path)
    for file_path in file_list:
        fullname=class_path+file_path
        bunch.label.append(mydir)#保存当前文件的分类标签
        bunch.filenames.append(fullname)#保存当前文件的文件路径
        bunch.contents.append(readfile(fullname).strip())#保存文件词向量

#Bunch对象持久化
file_obj=open(wordbag_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()

print("构建文本对象结束")