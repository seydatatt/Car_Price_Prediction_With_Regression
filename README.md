
# Car_Price_Prediction_With_Regression

This project is a **machine learning regression model** built to **predict car prices** based on various features such as year, mileage, engine size, and more. It uses deep learning with TensorFlow and Keras, and is developed in **Python** using **Jupyter Notebook**.

## 📌 Overview

The goal of this project is to create a predictive model that accurately estimates used car prices using historical data. The dataset was preprocessed to remove outliers and irrelevant columns, and a neural network was trained to make predictions.

---

## 📂 Dataset

- The dataset used is an Excel file: `merc.xlsx`.
- It includes features like:
  - `year`
  - `mileage`
  - `engineSize`
  - `fuelType`
  - and other relevant information.
  
Outliers and anomalies (such as cars with extreme prices) were removed to improve model accuracy.

---

## ⚙️ Technologies & Libraries Used

- Python
- Jupyter Notebook
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- TensorFlow / Keras

---

## 🧪 Model

- Model type: **Feedforward Neural Network**
- Layers: 4 hidden layers with ReLU activation
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Scaler: MinMaxScaler (to normalize feature values)
- Train/test split: 70/30

---

## 📈 Training & Evaluation

- The model was trained for **300 epochs** with a batch size of **250**.
- Loss and validation loss were tracked and visualized.
- Evaluation metrics:
  - **Mean Absolute Error (MAE)**
  - Visual comparison between actual and predicted prices via scatter plot.

---

## 💡 Example Prediction

The model can be used to predict the price of a new car by feeding it the car’s features after scaling:

```python
newCarSeries = dataFrame.drop("price", axis=1).iloc[2]
newCarSeries = scaler.transform(newCarSeries.values.reshape(-1, 5))
model.predict(newCarSeries)
```

---

## 🚀 How to Run

1. Make sure you have Python and Jupyter installed.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow openpyxl
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run the notebook containing the project code.
5. Make sure `merc.xlsx` is in the same directory.

---

## 📌 To Do

- Add a web interface (using Flask or Streamlit)
- Include more car features
- Improve model accuracy with hyperparameter tuning

---

## 📃 License

This project is open-source and available for any educational or personal use.
