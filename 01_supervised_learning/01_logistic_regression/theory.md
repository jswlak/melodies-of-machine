# Logistic Regression

## 🔹 What is Logistic Regression?
- A classification algorithm used to predict binary outcomes (0 or 1).
- Output is a probability score between 0 and 1 using the **sigmoid function**.

## 🔹 Sigmoid Function
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

## 🔹 Hypothesis
\[
\hat{y} = \sigma(w^T x + b)
\]

## 🔹 Cost Function
Binary cross-entropy loss:
\[
J(\theta) = -\frac{1}{m} \sum \left[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right]
\]

## 🔹 Optimization
- Use **Gradient Descent** to minimize cost.

## 🔹 Applications
- Spam detection
- Disease prediction
- Customer churn classification