import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score #Độ chính xác
from sklearn.metrics import f1_score #F1

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

#0. Các biến tham gia
lap = 5        #Cho biết là sẽ lặp 5 lần
kqLap = []      #Mảng này dùng để gom kết quả
tongKQ = float(0)

#Vòng lặp
for i in range(lap):
    print("Lặp lần thứ: ",i+1)
    #1.Khởi tạo mô hình knn với sự random 70 - 140  phần tử liền kề
    model_KNN = KNeighborsClassifier(n_neighbors=random.randint(70,140))

    #2. Huấn luyện mô hình
    model_KNN.fit(x_train,y_train)

    #3. Dự đoán nhãn cho mô hình
    y_pred = model_KNN.predict(x_test)

    #4. Đánh giá 
    f1 = f1_score(y_test,y_pred,average="macro")
    print('Độ chính xác: ',accuracy_score(y_test,y_pred)*100)
    print(f'F1-score: {f1*100:.2f}%')

    #5.Lưu vào mảng
    kqLap.append(f1)

#Vòng lặp tổng kết quả
for i in range(lap):
    tongKQ += kqLap[i]

#Trung bình cộng của F1 khi 5 lần lặp
print(f"Trung Bình cộng của F1 khi thực hiện 5 lần lặp: {(tongKQ/lap)*100:.2f}%")
