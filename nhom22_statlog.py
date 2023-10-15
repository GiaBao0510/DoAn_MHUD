#Tắt cảnh báo
import warnings
warnings.filterwarnings('ignore')

#Thêm một số thư viện và lớp
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score #Sự chính xác
from sklearn.metrics import precision_score #Độ chính xác
from sklearn.metrics import recall_score #Gọi lại
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
lap = 10        #Cho biết là sẽ lặp 5 lần
tongKQ = float(0)

#Vòng lặp
print('\tDự đoán nhãn bằng phương pháp K nearest neighbor')
for i in range(lap):
    langGieng = random.randint(100,150)
    print("=========================")
    print("Lặp lần thứ: {0} - K: {1}".format(i+1,langGieng))
    #1.Khởi tạo mô hình knn với sự random 70 - 140  phần tử liền kề
    model_KNN = KNeighborsClassifier(n_neighbors=langGieng)

    #2. Huấn luyện mô hình
    model_KNN.fit(x_train,y_train)

    #3. Dự đoán nhãn cho mô hình
    y_pred = model_KNN.predict(x_test)

    #4. Đánh giá 
    f1 = f1_score(y_test,y_pred,average="macro")   #Gồm Tập nhãn thực tế, tập nhãn dữ đoán và Tính F1 score theo từng nhãn và sau đó lấy trung bình cộng của các F1 score này. Phương pháp này là phương pháp mặc định.
    print(f'Độ chính xác: {accuracy_score(y_test,y_pred)*100:.2f}%')
    print(f'F1-score: {f1*100:.2f}%')

    #5.Lưu vào mảng
    tongKQ += f1

#Trung bình cộng của F1 khi 5 lần lặp
print(f"Trung Bình cộng của F1 khi thực hiện 5 lần lặp bằng phương pháp KNN: {(tongKQ/lap)*100:.2f}%")

''' 3. Cây quyết định bằng gini'''

from sklearn.tree import DecisionTreeClassifier
#Các biến tham gia
TongKQ_DTC = float(0)
SoLuongNgauNhien = random.randint(100,150)

#Vòng lặp
print('\tDự đoán nhãn bằng phương pháp cây quyết định')
for i in range(lap):
    print('==========================')
    print(f'Lần lặp thứ {i}')
    #1.Khởi tạo mô hình
    indexGiNi = DecisionTreeClassifier(criterion="gini",max_depth=5)

    #2.Huấn luyện mô hình
    indexGiNi.fit(x_train,y_train)

    #3.Dự đoán nhãn cho tập dữ liệu kiểm tra
    y_predict = indexGiNi.predict(x_test)

    #4. Đánh giá
    F1 = f1_score(y_test,y_predict, average="macro")    #Gồm Tập nhãn thực tế, tập nhãn dữ đoán và Tính F1 score theo từng nhãn và sau đó lấy trung bình cộng của các F1 score này. Phương pháp này là phương pháp mặc định.
    Recall = recall_score(y_test,y_predict,average="macro")
    Accuracy = accuracy_score(y_test,y_predict)
    Precision = precision_score(y_test,y_predict,average="macro")
    print(f'F1_score: {F1*100:.3f}%')
    print(f'Độ chính xác: {Precision*100:.3f}%')
    print(f'Sự chính xác: {Accuracy*100:.3f}%')
    print(f'Recall: {Recall*100:.3f}%')

    #Tổng kết quả F1 sau mỗi lần lặp
    TongKQ_DTC+=F1

#Tính trung bình cộng của F1 sau 5 lần lặp
print(f"Trung Bình cộng của F1 khi thực hiện 5 lần lặp bằng phương pháp Cây quyết định: {(TongKQ_DTC/lap)*100:.3f}%")
