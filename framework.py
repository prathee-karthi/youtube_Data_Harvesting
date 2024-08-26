import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
# Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

# Display the data
st.write("Data Preview:")
st.write(data.head())

# Select features and target variable
st.write("Feature Selection:")
features = st.multiselect("Select Features", data.columns.tolist(), default=data.columns.tolist()[:-1])
target = st.selectbox("Select Target", data.columns.tolist(), index=len(data.columns.tolist()) - 1)

if len(features) > 0 and target:
   X = data[features]
   y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select algorithm
algorithm = st.selectbox(
"Select Machine Learning Algorithm",
("Decision Tree", "Random Forest", "SVM", "K-Nearest Neighbors", "Logistic Regression")
)

# Train the selected model
if algorithm == "Decision Tree":
   model = DecisionTreeClassifier()
elif algorithm == "Random Forest":
   model = RandomForestClassifier()
elif algorithm == "SVM":
   model = SVC()
elif algorithm == "K-Nearest Neighbors":
   model = KNeighborsClassifier()
elif algorithm == "Logistic Regression":
   model = LogisticRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Allow user to make predictions
st.write("Make Predictions:")
input_data = {feature: st.number_input(f"Input {feature}", value=float(X_train[feature].mean())) for feature in features}
input_df = pd.DataFrame([input_data])
if st.button("Predict"):
   prediction = model.predict(input_df)
   st.write(f"Prediction: {prediction[0]}")
