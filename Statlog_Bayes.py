import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

num_iterations = 10
accuracies = []  # Danh sách để lưu trữ độ chính xác từ mỗi lần huấn luyện

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

print('----------Bayes----------')
for i in range(num_iterations):
    from sklearn.naive_bayes import MultinomialNB

    # Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    # Dự đoán
    y_pred_mnb = mnb.predict(X_test)

    # Đánh giá mô hình
    from sklearn.metrics import accuracy_score, classification_report

    accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
    accuracies.append(accuracy_mnb)  # Thêm độ chính xác vào danh sách accuracies
    print(f"Độ chính xác của Multinomial Naive Bayes (Lần {i + 1}): {accuracy_mnb}")

    # Tạo báo cáo phân loại
    report_mnb = classification_report(y_test, y_pred_mnb)
    print(f"Báo cáo phân loại cho Multinomial Naive Bayes (Lần {i + 1}):")
    print(report_mnb)

# Vẽ biểu đồ hoặc thực hiện các thao tác khác dựa trên giá trị trong accuracies
plt.plot(range(1, num_iterations + 1), accuracies, marker='o', linestyle='-')
plt.title('Độ chính xác của Multinomial Naive Bayes')
plt.xlabel('Lần huấn luyện')
plt.ylabel('Độ chính xác')
plt.grid(True)
plt.show()
