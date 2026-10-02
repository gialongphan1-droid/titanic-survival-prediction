# Titanic Survival Prediction

Dự đoán hành khách trên tàu Titanic sống sót hay không — bài toán **binary classification**
trên bộ dữ liệu Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

## 1. Mục tiêu

- **Input:** thông tin hành khách (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, ...).
- **Output:** `Survived` = 0 (không sống sót) hoặc 1 (sống sót).
- **Metric chính:** ROC-AUC.
- **Metric phụ:** F1 (Survived=1), Precision/Recall, Confusion Matrix.
- **Không dùng Accuracy làm metric chính** vì bài toán có mất cân bằng nhãn (~62% / 38%).

## 2. Dữ liệu

| File                         | Số dòng | Số cột | Vai trò                         |
| ---------------------------- | ------- | ------ | ------------------------------- |
| `data/train.csv`             | 891     | 12     | Huấn luyện (có nhãn `Survived`) |
| `data/test.csv`              | 418     | 11     | Dự đoán (không có nhãn)         |
| `data/gender_submission.csv` | 418     | 2      | Mẫu định dạng file nộp Kaggle   |

Nguồn: [Kaggle Titanic](https://www.kaggle.com/c/titanic/data).

## 3. Cấu trúc project

```
titanic-survival-prediction/
├── data/              # Dataset CSV (không commit lên Git)
├── models/            # Pipeline đã train, lưu bằng joblib
├── notebooks/         # EDA, thử nghiệm
├── reports/           # Báo cáo, figures
│   └── figures/
├── scripts/           # Script tiện ích (download, ...)
├── src/               # Code chính: preprocess, train, predict
├── test/              # Unit test cho code trong src/
├── requirements.txt   # Thư viện cần cài
└── README.md
```

## 4. Cài đặt

Yêu cầu: Python >= 3.10, Git.

```bash
# 1. Clone repo
git clone <repo-url>
cd titanic-survival-prediction

# 2. Tạo môi trường ảo
python -m venv venv

# 3. Kích hoạt venv
source venv/Scripts/activate    # Windows + Git Bash
# source venv/bin/activate      # macOS / Linux

# 4. Cài thư viện
pip install -r requirements.txt
```

Kiểm tra cài đặt thành công:

```bash
python -c "import pandas, numpy, sklearn, xgboost, shap, joblib, streamlit; print('ALL OK')"
```

## 5. Cách chạy

> ⚠️ **Trạng thái:** đang phát triển. Các lệnh dưới đây sẽ được hoàn thiện theo từng bước.

### 5.1. Huấn luyện model

```bash
python src/train.py
```

Kết quả: pipeline tốt nhất được lưu vào `models/best_model.joblib`.

### 5.2. Dự đoán

```bash
python src/predict.py --input data/test.csv --output reports/submission.csv
```

Kết quả: file `reports/submission.csv` theo định dạng Kaggle.

### 5.3. Chạy demo Streamlit

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## 6. Quy trình 8 bước

- [x] **Bước 1 — Setup:** venv, cấu trúc thư mục, `requirements.txt`, data
- [ ] **Bước 2 — EDA:** thống kê mô tả, phân bố, tương quan, missing values
- [ ] **Bước 3 — Tiền xử lý:** `Pipeline` + `ColumnTransformer`, chia train/valid **trước** khi fit
- [ ] **Bước 4 — Feature Engineering:** tạo feature mới từ `Name`, `Ticket`, `FamilySize`, ...
- [ ] **Bước 5 — Modeling:** Logistic Regression → Random Forest → XGBoost/LightGBM
- [ ] **Bước 6 — SHAP:** giải thích mô hình, feature importance
- [ ] **Bước 7 — Streamlit:** app demo dự đoán tương tác
- [ ] **Bước 8 — README hoàn chỉnh:** kết quả, hướng dẫn, hình ảnh

## 7. Quy tắc code

Các quy tắc dưới đây là **bắt buộc** — vi phạm sẽ dẫn đến data leakage hoặc kết quả không tái lập được.

1. **`random_state=42`** xuyên suốt mọi bước (chia data, model, SMOTE).
2. **Chia train/valid TRƯỚC khi fit scaler/encoder.**
   - Dùng `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
   - Scaler/encoder **chỉ fit trên train**, rồi transform cả train và valid.
3. **Dùng `Pipeline` + `ColumnTransformer`** — không tiền xử lý rời rạc ngoài pipeline.
4. **Bỏ cột `PassengerId`, `Name`, `Ticket`, `Cabin`** ở bước đầu (có thể trích xuất feature từ `Name`/`Ticket` ở Bước 4).
5. **Lưu pipeline cuối bằng `joblib`** — không lưu riêng model.
6. **Không dùng Accuracy làm metric chính** (data mất cân bằng nhẹ).
7. **Không commit** `data/`, `models/*.joblib`, `venv/`, `__pycache__/`.

## 8. Kết quả

> ⚠️ **Chưa có kết quả.** Bảng dưới đây sẽ được điền sau Bước 5 (Modeling).

| Model               | ROC-AUC | F1 (Survived=1) | Precision | Recall |
| ------------------- | ------- | --------------- | --------- | ------ |
| Logistic Regression | —       | —               | —         | —      |
| Random Forest       | —       | —               | —         | —      |
| XGBoost / LightGBM  | —       | —               | —         | —      |

**Ngưỡng đạt:** ROC-AUC >= 0.85 (mục tiêu), >= 0.80 (tối thiểu).

## 9. Tác giả

- **Họ tên:** Phan Gia Long
- **Ngày bắt đầu:** 2026-10-03
- **Mục đích:** học Data Science end-to-end (EDA → modeling → giải thích → deploy).
