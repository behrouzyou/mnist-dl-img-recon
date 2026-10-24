# 📊 Human Activity Recognition with PCA, LDA & Feature Selection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)

A complete machine learning pipeline for **Human Activity Recognition (HAR)** using smartphone accelerometer and gyroscope data.


---

## 🎯 Project Goal

Compare the impact of different **dimensionality reduction** and **feature selection** techniques on classification performance:

- ✅ Baseline (no dimensionality reduction)
- ✅ Feature Selection with Mutual Information
- ✅ PCA (Principal Component Analysis)
- ✅ LDA (Linear Discriminant Analysis)

Along with data visualization using **t-SNE** and **UMAP**.

---

## 📁 Dataset

[Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) from Kaggle
(Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones))

| Set | Samples | Features |
|---|---|---|
| **Train** | 7,352 | 561 + subject + Activity |
| **Test** | 2,947 | 561 + subject + Activity |

### 🏃 Activity Classes (6 classes)

| Class | Train Count |
|---|---|
| LAYING | 1,407 |
| STANDING | 1,374 |
| SITTING | 1,286 |
| WALKING | 1,226 |
| WALKING_UPSTAIRS | 1,073 |
| WALKING_DOWNSTAIRS | 986 |

> 21 subjects are present in the training set.

---

## 🛠️ Pipeline Steps

### 1️⃣ ETL / EDA
- Load train and test data
- Inspect shapes, dtypes, class distribution, and subjects

### 2️⃣ Standardization
- Normalize features with `StandardScaler`
- **Important:** fit only on train, transform on test (avoid data leakage)

### 3️⃣ Baseline Classification
- `LogisticRegression(max_iter=2000)` on all 562 features

### 4️⃣ Feature Selection
- **Variance Threshold** with `threshold=0.01`
- **SelectKBest** with `mutual_info_classif` and `k=100`

### 5️⃣ PCA (Principal Component Analysis)
- Choose number of components based on cumulative explained variance
- **95% variance → 102 components**
- **99% variance → 179 components**

### 6️⃣ LDA (Linear Discriminant Analysis)
- Reduce to **5 components** (number of classes minus one)

### 7️⃣ Visualization
- **LDA** — 2D projection with label-based separation
- **t-SNE** — non-linear projection (perplexity=30, learning_rate=200)
- **UMAP** — non-linear projection (n_neighbors=15, min_dist=0.1)

---

## 📈 Results

| Method | # Features | Accuracy |
|---|---|---|
| **Baseline (Logistic Regression)** | 562 | **95.11%** |
| Feature Selection (Mutual Info, k=100) | 100 | 90.46% |
| PCA (102 components, 95% variance) | 102 | 93.08% |
| **LDA (5 components)** | **5** | **96.37%** 🏆 |

### 💡 Key Insights

- **LDA with only 5 components achieved the best performance** (96.37%) — because unlike PCA which only preserves variance, LDA also uses label information to maximize between-class separation.
- **PCA** with 102 components reached 93% — dimensionality reduction without losing key information.
- **Feature Selection** with Mutual Information performed worse (90.46%) — likely because k=100 was not sufficient for this dataset.
- **Baseline** has high accuracy with all features but higher computational cost.

---

## 📊 Visualizations

| Method | Observation |
|---|---|
| **LDA (2D)** | Reasonably good class separation, especially for static activities (LAYING, SITTING, STANDING) |
| **t-SNE** | Clear clustering but with overlap between WALKING_UPSTAIRS and WALKING_DOWNSTAIRS |
| **UMAP** | Better global structure with distinct clusters |

---

## ⚙️ Installation & Usage

### Requirements

```bash
pip install -r requirements.txt

git clone https://github.com/behrouzyou/Human-Activity-PCA.git
cd Human-Activity-PCA
jupyter notebook human-activity-recognition-fs-fe-pca-lda.ipynb
/kaggle/input/datasets/organizations/uciml/human-activity-recognition-with-smartphones

🔑 Libraries Used
Library	Purpose
numpy, pandas	Data processing
scikit-learn	Models, dimensionality reduction, feature selection
matplotlib	Visualization
umap-learn	UMAP projection
Key scikit-learn classes:
StandardScaler · LogisticRegression · VarianceThreshold · SelectKBest · mutual_info_classif · PCA · LinearDiscriminantAnalysis · TSNE

📂 Project Structure
text
Human-Activity-PCA/
│
├── human-activity-recognition-fs-fe-pca-lda.ipynb   # Main notebook
├── requirements.txt                                 # Dependencies
├── LICENSE                                          # MIT License
└── README.md                                        # This file
📚 Resources

📗 UCI HAR Dataset

📙 Kaggle Dataset

👤 Author
Behrouz Yousefi

GitHub: @behrouzyou

⭐ Support
If this project was useful to you, please give it a ⭐! 🙏

text

---

## 📄 فایل ۲: `requirements.txt`

```txt
numpy>=2.0.2
pandas>=2.2.0
scikit-learn>=1.6.1
matplotlib>=3.10.0
umap-learn>=0.5.11
jupyter>=1.0.0
📄 فایل ۳: LICENSE (MIT License)
txt
MIT License

Copyright (c) 2026 Behrouz Yousefi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

