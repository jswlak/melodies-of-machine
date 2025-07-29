# 📘 Linear Regression - Theory Notes

Linear Regression is one of the most fundamental algorithms in supervised machine learning. It is used for **predicting continuous outcomes** based on the input features.

---

## 🧠 What is Linear Regression?

Linear Regression tries to model the relationship between a dependent variable `y` and one or more independent variables `x` using a straight line.

- **Simple Linear Regression**: One input feature  
- **Multiple Linear Regression**: Multiple input features

---

## 🧮 Mathematical Equation

### 📍 Simple Linear Regression:

\[
y = mx + b
\]

- \( y \): predicted value  
- \( x \): input feature  
- \( m \): slope (weight)  
- \( b \): intercept (bias)

### 📍 Multiple Linear Regression:

\[
y = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b
\]

Or in vector form:

\[
\hat{y} = \mathbf{w}^T \mathbf{x} + b
\]

---

## 🎯 Objective

To find the best-fit line that minimizes the difference between actual and predicted values.

---

## 🧾 Cost Function

We use **Mean Squared Error (MSE)** to measure the model's performance:

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Our goal is to minimize the MSE.

---

## 📉 Gradient Descent (Optimization)

To minimize the cost function, we update the weights using **Gradient Descent**:

\[
w := w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}
\]

\[
b := b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]

- \( \alpha \): learning rate  
- \( \partial \): partial derivative

---

## ✅ Assumptions of Linear Regression

1. **Linearity**: Linear relationship between features and target
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

---

## ⚖️ Advantages

- Easy to understand and implement
- Fast training
- Interpretable model

---

## ⚠️ Limitations

- Assumes linear relationships
- Sensitive to outliers
- Poor performance on complex, non-linear datasets

---

## 🔍 Applications

- Predicting house prices
- Forecasting stock prices
- Estimating costs or demand
- Analyzing trends in business and economics

---


