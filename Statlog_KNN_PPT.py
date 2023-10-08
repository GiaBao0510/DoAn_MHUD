import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt

num_loop = 5
average = 0

# Đọc dữ liệu từ tệp
url = './data_statlog/sat.trn'
sat_train = pd.read_csv(url, header=None, sep=' ')

url = './data_statlog/sat.tst'
sat_test = pd.read_csv(url, header=None, sep=' ')

# Phân chia dữ liệu
X_train = sat_train.iloc[:, :-1]
y_train = sat_train.iloc[:, -1]
X_test = sat_test.iloc[:, :-1]
y_test = sat_test.iloc[:, -1]

# Chuyển đổi sang NumPy array
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# Đặt lại tên cột Nhãn
sat_train.rename(columns={36: 'Class'}, inplace=True)
sat_test.rename(columns={36: 'Class'}, inplace=True)

for ringloop in range(num_loop):
    k = random.randint(80, 150)
    print('-------------------------------')
    print(f'Lần lặp thứ {ringloop + 1}: Sử dụng K = {k}')

    # Tạo Mô Hình KNN
    knn_model = KNeighborsClassifier(n_neighbors=k)

    # Huấn luyện KNN
    knn_model.fit(X_train, y_train)

    # Dự đoán nhãn
    y_pred = knn_model.predict(X_test)

    # Đo lường hiệu suất tỷ lệ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Do chinh xac: {accuracy*100:.2f}%')
    print('-------------------------------')
    # Cập nhật biến
    average += accuracy

# Trung bình tổng
average /= num_loop
print(f'Trung binh tong cua giai thuat KNN la : {average*100:.2f}%')

# Vẽ biểu đồ cho accuracy
iterations = range(1, num_loop + 1)
accuracies = [average * 100] * num_loop

plt.bar(iterations, accuracies, color='green',)
plt.xlabel('Lần lặp')
plt.ylabel('Độ chính xác (%)')
plt.title('Biểu đồ độ chính xác của KNN')
plt.grid(True)
plt.show()
