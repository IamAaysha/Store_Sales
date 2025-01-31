# Sales Prediction Using Linear Regression in Python  

## Project Overview  
This project focuses on predicting **Item Outlet Sales** using a **Linear Regression model**. The dataset undergoes preprocessing, feature transformation, and model training to forecast sales performance.  

## Dataset  
The project utilizes two CSV files:  
- **train.csv** – Used for training the regression model.  
- **test.csv** – Used to evaluate model performance.  

## Key Features  
- **Categorical variables**: Item type, outlet location, outlet size, etc.  
- **Numerical variables**: Item price (MRP), visibility, weight, and outlet sales.  

## Key Steps  

### 1. Data Preprocessing  
- Checked for and removed missing values.  
- Separated features (X) and target variable (Y) (**Item Outlet Sales**).  

### 2. Model Training & Evaluation  
- Built a **Linear Regression** model using `sklearn`.  
- Trained the model on cleaned data.  
- Computed **Root Mean Square Error (RMSE)** for both train and test datasets to measure model performance.  

### 3. Feature Encoding & Scaling  
- Encoded categorical variables using **LabelEncoder**.  
- Normalized numerical features using **MinMaxScaler** to improve model performance.  

## Results & Insights  
- The trained model predicts sales values based on input features.  
- **RMSE scores** provide insights into the model’s accuracy.  
- Preprocessing steps (**handling missing values, encoding, and scaling**) significantly impact prediction performance.  

## Technologies Used  
- **Python** (`Pandas`, `NumPy`, `Scikit-Learn`)  
- **Linear Regression** for prediction  
- **Label Encoding** for categorical data transformation  
- **Min-Max Scaling** for feature normalization  

## How to Run the Project  

### Clone the repository:  
```bash
git clone <repo-link>
```

### Install dependencies:  
```bash
pip install pandas scikit-learn numpy
```

### Update dataset paths in the script:  
```python
train_data = pd.read_csv("path/to/train.csv")
test_data = pd.read_csv("path/to/test.csv")
```

### Run the script to train the model and generate predictions.  

## Next Steps & Improvements  
- Implement **feature selection** to improve model accuracy.  
- Try **Random Forest** or **Gradient Boosting** models for better sales predictions.  
- Perform **hyperparameter tuning** to optimize the regression model.  

## Disclaimer  
This project is for **educational purposes only** and may require additional tuning for real-world deployment.  
