# 📧 Email Classifier — Phase 1 (Core ML)

This is Phase 1 of a multi-phase project that automatically sorts emails
into categories so users don't have to dig through a cluttered inbox.

The model classifies any email's text into one of **4 categories**:

- 🛑 `spam`
- ⭐ `important`
- 🏷️ `promotions`
- 🔐 `otp`

---

## 🎯 Project Idea

Most inboxes mix important mail (college notices, job updates), OTPs,
promotional offers, and spam all in one place — making it easy to miss
something important. This project trains a machine learning model to
automatically label each email, so it can later be sorted into folders
based on the category.

This is Phase 1 of a larger plan:

1. ✅ **Phase 1 (this repo/folder):** Train and test a core ML classifier
2. 🔜 **Phase 2:** Wrap the model in a Flask API
3. 🔜 **Phase 3:** Connect to a Node.js + MongoDB backend
4. 🔜 **Phase 4:** Build a frontend dashboard showing sorted inbox folders
5. 🔜 **Phase 5:** Personalize categories based on user type (student/professional)

---

## 📁 Files

| File | Description |
|---|---|
| `dataset.csv` | Starter labeled dataset (56 sample emails across 4 categories) |
| `train_classifier_simple.py` | Script that loads data, trains, tests, and saves the model |
| `requirements.txt` | Python libraries needed |
| `email_classifier_model.pkl` | Trained model (generated after running the script) |
| `tfidf_vectorizer.pkl` | Text-to-numbers converter (generated after running the script) |

---

## 🧠 How It Works

1. **Load data** — emails + their correct category from `dataset.csv`
2. **Convert text to numbers** — using `TfidfVectorizer`, which scores
   words by how meaningful they are to a specific email (e.g. "OTP",
   "discount", "exam" are strong signals for their categories)
3. **Train a model** — `DecisionTreeClassifier` learns patterns between
   word-scores and categories
4. **Test on unseen emails** — 20% of the data is hidden during training
   and used afterward to measure real accuracy
5. **Save the trained model** — using `joblib`, so it can be reused later
   without retraining

---

## ▶️ How to Run

### Option A — Google Colab

1. Open [Google Colab](https://colab.research.google.com) → New Notebook
2. **Cell 1** — upload the dataset:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
   Click the upload button and select `dataset.csv`.

3. **Cell 2** — paste and run the contents of `train_classifier_simple.py`

4. **Optional — download the trained model files** to your computer
   (needed for Phase 2):
   ```python
   from google.colab import files
   files.download("email_classifier_model.pkl")
   files.download("tfidf_vectorizer.pkl")
   ```

### Option B — Local Python

```bash
pip install -r requirements.txt
python train_classifier_simple.py
```

---

## 📊 Sample Output

```
Accuracy: 91.67%
Your OTP for net banking login is 482910, do not share with anyone. -> otp
Flat 60% off on all shoes, sale ends today, shop now! -> promotions
Dear student, your internal exam marks have been uploaded to the portal. -> important
You have won a brand new car, click here to claim your prize immediately. -> spam
```

> **Note:** Accuracy will vary slightly depending on dataset size. With
> only 56 starter examples, 70–90%+ is expected. Accuracy improves as
> more real-world examples are added (see below).

---

## 🔧 Improving the Model

This starter dataset was written by hand, since public spam datasets
only label "spam vs not spam" and don't match these 4 categories. To
improve real-world accuracy:

1. Collect 10–20 real emails per category from your own inbox
   (remove personal info if needed)
2. Add them to `dataset.csv` in the same `text,category` format
3. Re-run the script and compare accuracy

You can also swap the model with one line of code:

```python
# Decision Tree (current)
model = DecisionTreeClassifier(random_state=42)

# Random Forest (usually more accurate with more data)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

---

## 🛣️ What's Next — Phase 2

The trained model (`email_classifier_model.pkl` +
`tfidf_vectorizer.pkl`) will be wrapped inside a small **Flask API**.
This lets a Node.js backend send raw email text to an endpoint and
receive back a predicted category — exactly like calling any other
REST API.

---

## 🛠️ Tech Stack

- Python
- pandas
- scikit-learn (`TfidfVectorizer`, `DecisionTreeClassifier`)
- joblib

---

## 👤 Author

Built as a self-directed learning project combining full-stack
development (Node.js, Express, MongoDB, Flask) with machine learning.
