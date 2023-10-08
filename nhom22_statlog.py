import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score #Độ chính xác

#Đọc dữ liệu Train - Test
data_train = pd.read_csv("data_statlog/sat.trn",encoding='utf-8',header=None,sep=' ')
data_test = pd.read_csv("data_statlog/sat.tst",encoding='utf-8',header=None,sep=' ')

#Dữ liệu bên tập dữ liệu huấn luyện
x_train = data_train.values[:,:-1]     #Lấy tất cả các hàng và Các cột không phải cột lớp
y_train = data_train.values[:,-1]      #Lấy tất cả các hàng của cột cuối

#Dữ liệu bên tập dữ liệu kiểm tra
x_test = data_test.values[:,:-1]       #Lấy tất cả các hàng và Các cột không phải cột lớp
y_test = data_test.values[:,-1]        #Lấy tất cả các hàng của cột cuối

'''     ---  1.KNN  ---   '''
from sklearn.neighbors import KNeighborsClassifier

#0. Lặp Tầm 10 lần
lap = 10

#1.Khởi tạo mô hình knn với 5 phần tử liền kề
model_KNN = KNeighborsClassifier(n_neighbors=100)

#2. Huấn luyện mô hình
model_KNN.fit(x_train,y_train)

#3. Dự đoán nhãn cho mô hình
y_pred = model_KNN.predict(x_test)

print('Nhãn dự đoán: ',y_pred)
