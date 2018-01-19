# -*- coding: utf-8 -*-
import os
import jieba


def savefile(savepath, content):
    fp = open(savepath, "w",encoding='gb2312', errors='ignore')
    fp.write(content)
    fp.close()
def readfile(path):
    fp = open(path, "r", encoding='gb2312', errors='ignore')
    content = fp.read()
    fp.close()
    return content

corpus_path = "test_small/"  # 未分词分类预料库路径
seg_path = "test_seg/"  # 分词后分类语料库路径

catelist = os.listdir(corpus_path)  # 获取改目录下所有子目录
print(catelist)
for mydir in catelist:
    class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
    seg_dir = seg_path + mydir + "/"  # 拼出分词后预料分类目录
    if not os.path.exists(seg_dir):  # 是否存在，不存在则创建
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)
    for file_path in file_list:
        fullname = class_path + file_path
        content = readfile(fullname).strip()  # 读取文件内容
        content = content.replace("\r\n", "").strip()  # 删除换行和多余的空格
        content_seg = jieba.cut(content)
        savefile(seg_dir + file_path, " ".join(content_seg))

print("分词结束")