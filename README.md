
#Isa723-DataAnalysis
## 📘 Glossary: Regression and Time Series Analysis (Song ngữ Anh - Việt)

### 1. Linear Regression Analysis (Phân tích hồi quy tuyến tính)

| Tiếng Việt                                        | English Term                                | Giải thích |
|--------------------------------------------------|---------------------------------------------|------------|
| Mô hình hồi quy tuyến tính hai biến             | Simple Linear Regression                    | Mô hình giữa 1 biến phụ thuộc và 1 biến độc lập |
| Các kiểm định thống kê                           | Statistical Tests                           | t-test, F-test, kiểm định giả thuyết hệ số |
| Hồi quy tuyến tính đa biến                       | Multiple Linear Regression                  | Mô hình mở rộng với nhiều biến độc lập |
| Đa cộng tuyến và giải pháp                       | Multicollinearity and Remedies              | Các biến độc lập tương quan mạnh gây sai lệch |
| Phương sai & kiểm định giả thuyết OLS            | OLS Variance and Hypothesis Testing         | Đánh giá độ tin cậy hệ số OLS qua t, F-test |

---

### 2. Regression with Dummy and Qualitative Variables

| Tiếng Việt                                        | English Term                                | Giải thích |
|--------------------------------------------------|---------------------------------------------|------------|
| Hồi quy với biến định tính và định lượng         | Regression with Quantitative and Qualitative Variables | Sử dụng cả biến số và biến giả |
| Hồi quy với biến giả mùa vụ                      | Regression with Seasonal Dummies            | Dùng biến giả cho mùa xuân, hè, thu, đông |
| Hồi quy logistic nhị phân                        | Binary Logistic Regression                  | Biến phụ thuộc có 2 trạng thái (0,1) |
| Hồi quy đa lựa chọn (nominal >2)                 | Multinomial Logistic Regression             | Biến phụ thuộc dạng phân loại nhiều giá trị |

---

### 3. Time Series Analysis (Phân tích chuỗi thời gian)

| Tiếng Việt                                        | English Term                                | Giải thích |
|--------------------------------------------------|---------------------------------------------|------------|
| Thành phần chuỗi thời gian                       | Components of Time Series                   | Xu hướng, mùa vụ, chu kỳ, nhiễu |
| Phân tích chuỗi thời gian                        | Time Series Analysis Methods                | ACF/PACF, decomposition, ARIMA... |
| Làm trơn hàm mũ bậc 1                             | Simple Exponential Smoothing (SES)         | Không có xu hướng/mùa vụ |
| Làm trơn hàm mũ bậc 2                             | Holt’s Linear Exponential Smoothing         | Có xu hướng tuyến tính |

---

### 4. Time Series Regression (Hồi quy chuỗi thời gian)

| Tiếng Việt                                        | English Term                                | Giải thích |
|--------------------------------------------------|---------------------------------------------|------------|
| Hàm tự tương quan, toán tử trễ                   | Autocorrelation Function, Lag Operator      | Dùng trong ACF, PACF và AR(p) |
| Chuỗi dừng / không dừng                          | Stationary / Non-stationary Time Series     | Dựa vào kiểm định ADF |
| Biến đổi chuỗi không dừng                        | Differencing for Stationarity               | Lấy sai phân để làm dừng chuỗi |
| Hồi quy AR(p)                                     | Autoregressive Model of order p             | Hồi quy giá trị hiện tại theo p giá trị quá khứ |
| Mô hình ARDL                                      | Autoregressive Distributed Lag Model        | Kết hợp biến trễ của Y và X |
| Hồi quy chuỗi thời gian dừng                      | Regression with Stationary Variables        | Dùng OLS an toàn |
| Hồi quy chuỗi không dừng                          | Regression with Non-stationary Variables    | Có thể dẫn đến hồi quy giả |
| Quan hệ nhân quả và đồng tích hợp                | Granger Causality and Cointegration         | Kiểm định mối quan hệ dài hạn giữa các chuỗi |
| Mô hình hiệu chỉnh sai số (ECM)                   | Error Correction Model                      | Áp dụng khi có quan hệ đồng tích hợp |
| Hồi quy sai lệch giả                              | Spurious Regression                         | Khi hồi quy chuỗi không dừng, không đồng tích hợp |

---

📎 **Ghi chú sử dụng**: Bạn có thể thêm bảng này vào phụ lục tài liệu kỹ thuật, hoặc đính kèm dưới dạng tệp `glossary.md`.

