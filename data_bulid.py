import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import pandas as pd
import numpy as np

root = tk.Tk()
root.withdraw()

# Ask user to choose PCA or Entropy
method = None

method = simpledialog.askstring("选择方法", "请选择PCA还是熵权法")

if method == 'PCA':
    if_Entropy = False
elif method == '熵权法':
    if_Entropy = True
else:
    messagebox.showwarning("警告", "请输入PCA或者熵权法，不要打错字哦！")
    quit()

# Ask user to choose the input file
input_file_path = simpledialog.askstring("选择文件", "输入数据集")
if not input_file_path:
    messagebox.showwarning("警告", "您未填写地址哦")
    quit()

# Ask user to choose the file for positive data
pos_data = []

pos_data = simpledialog.askstring("选择文件", '输入正向数据，如1，2，3')

# Ask user to choose the file for negative data
neg_data = []

neg_data = simpledialog.askstring("选择文件", '输入负向数据，如1，2，3')

# Ask user to choose the output输出文件的地址
output_file_path = simpledialog.askstring("选择文件", "输出数据集")
if not output_file_path:
    messagebox.showwarning("警告", "您未填写地址哦")
quit()


print(f"您选择的方法是 {method}")
print(f"数据集文件地址是 {input_file_path}")
print(f"正向数据是 {pos_data}")
print(f"负向数据是 {neg_data}")
print(f"输出文件地址是 {output_file_path}")

def Entropy_weight(data, negative_dictionary, return_weight=True, Add_Linear_weight=False):
    try:
        type_dictionary = dict(zip(list(data),['P'] * (len(data.columns))))
    except:
        data = pd.DataFrame(data)
        type_dictionary = dict(zip(list(data),['P'] * (len(data.columns))))
    for i in negative_dictionary:
        type_dictionary[i] = 'N'
    for i in list(data):
        if type_dictionary[i] == 'P':
            data[i] = (data[i] - data[i].min())/(data[i].max() - data[i].min())
        else:
            data[i] = (data[i].max() - data[i])/(data[i].max() - data[i].min())
    #计算k
    m,n = data.shape  #m行k列
    data1 = data.values    
    k = 1/np.log(m)
    yij = data1.sum(axis = 0)

    #计算pij
    pij = data1/yij
    test = pij*np.log(pij)
    test = np.nan_to_num(test)

    #计算每种指标的信息熵
    ej = -k*(test.sum(axis = 0))
    #计算每种指标的权重
    wi = (1 - ej)/np.sum(1 - ej)
    wi_list = list(wi)   #权重列表
    
    if return_weight:
        print('weight:\n',wi_list)
        
    if Add_Linear_weight:
        for i in range(len(wi_list)):
            data[list(data)[i]] = wi_list[i] * data[list(data)[i]]
        return data

def PCA(df):
    # 将DataFrame数据转化为NumPy数组
    X = df.values

    # 对数据进行中心化
    X = X - np.mean(X, axis=0)

    # 计算协方差矩阵
    cov_mat = np.cov(X, rowvar=False)

    # 计算特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # 将特征向量按特征值大小进行排序
    eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(reverse=True)

    # 选择最大的特征值对应的特征向量作为转换矩阵
    w = eig_pairs[0][1].reshape(-1,1)

    # 使用转换矩阵将数据降维
    X_pca = X.dot(w)

    # 将降维后的数据转化为DataFrame格式
    df_pca = pd.DataFrame(X_pca, columns=['PCA'])

    return df_pca
    
df = pd.read_excel(f'{input_file_path}')
cols = pos_data.split('，') + neg_data.split('，')
df = df.filter(cols, axis=1)

if method == '熵权法':
    output = Entropy_weight(df,negative_dictionary = neg_data, return_weight=False, Add_Linear_weight=True)
    output = pd.DataFrame({method:output.apply(lambda x: x.sum(), axis=1)})
    output.to_excel(f'{output_file_path}')
else:
    output = PCA(df)
    output.to_excel(f'{output_file_path}')