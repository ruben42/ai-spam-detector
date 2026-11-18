# 🤖 AI Spam Detector

A Machine Learning project that classifies text messages as Spam or Ham using Python and Naive Bayes.

## 📌 About the Project

AI Spam Detector is a simple Artificial Intelligence application designed to detect unwanted messages.

The user enters a message and the model predicts whether it is Spam or Ham.

## 🚀 Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## 🧠 Machine Learning

The project uses:

- TF-IDF for text vectorization.
- Multinomial Naive Bayes for classification.
- Train/Test Split for model evaluation.

## 📂 Project Structure

```text
ai-spam-detector/
│
├── data/
│   └── spam.csv
│
├── main.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Enter the project folder:

```bash
cd ai-spam-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Train the model:

```bash
python train.py
```

Start the application:

```bash
streamlit run app.py
```

## 🎯 Features

- Spam and Ham classification.
- Machine Learning predictions.
- Estimated class probabilities.
- Interactive Streamlit interface.

## 📚 Learning Objectives

This project was created to learn:

- Python programming.
- Machine Learning fundamentals.
- Text classification.
- Model evaluation.
- Web application development.

## 👨‍💻 Author

Rúben Pádua

## 📄 License

This project is available under the MIT License.

## 📸 Application Preview

![AI Spam Detector](screenshots/spam-detector(1).png)
![AI Spam Detector](screenshots/spam-detector(2).png)