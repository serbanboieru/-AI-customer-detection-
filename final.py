import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv('C:/Users/Boieru/Desktop/ML/final/BankChurners.csv')

# Map target variable
data['Attrition_Flag'] = data['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant columns
relevant_columns = [
    'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
    'Marital_Status', 'Income_Category', 'Card_Category',
    'Months_on_book', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
    'Credit_Limit'
]

X = data[relevant_columns]
y = data['Attrition_Flag']

# Map income ranges to numerical values
income_map = {
    'Less than $40K': 20000,
    '$40K - $60K': 50000,
    '$60K - $80K': 70000,
    '$80K - $120K': 100000,
    '$120K +': 120000,
    'Unknown': 0
}
X['Income_Category'] = X['Income_Category'].map(income_map)

# Categorical and numerical columns
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Card_Category']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-5)

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[lr_reduction], verbose=1)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred)}")
print(confusion_matrix(y_test,y_pred))

# GUI for Customer Classification
class CustomerAttritionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Attrition Prediction")
        self.inputs = {}
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Customer Attributes").grid(row=0, column=0, columnspan=2)

        # Dropdowns for categorical columns
        for i, col in enumerate(categorical_cols):
            tk.Label(self.root, text=col).grid(row=i+1, column=0)
            options = list(data[col].unique())
            self.inputs[col] = ttk.Combobox(self.root, values=options)
            self.inputs[col].grid(row=i+1, column=1)

        # Dropdown for Income Category
        tk.Label(self.root, text="Income_Category").grid(row=len(categorical_cols) + 1, column=0)
        income_options = list(income_map.keys())
        self.inputs['Income_Category'] = ttk.Combobox(self.root, values=income_options)
        self.inputs['Income_Category'].grid(row=len(categorical_cols) + 1, column=1)

        # Input fields for numerical columns
        start_row = len(categorical_cols) + 2
        for i, col in enumerate(numerical_cols):
            if col == 'Income_Category':  # Already handled
                continue
            tk.Label(self.root, text=col).grid(row=start_row + i, column=0)
            self.inputs[col] = tk.Entry(self.root)
            self.inputs[col].grid(row=start_row + i, column=1)

        # Button for classification
        tk.Button(self.root, text="Classify Customer", command=self.classify_customer).grid(
            row=start_row + len(numerical_cols), column=0, columnspan=2
        )

    def classify_customer(self):
        try:
            # Collect input data
            input_data = []
            for col in categorical_cols:
                value = self.inputs[col].get()
                if value == "":
                    raise ValueError(f"Please select a value for {col}.")
                input_data.append(value)

            # Handle Income Category separately
            income_value = self.inputs['Income_Category'].get()
            if income_value == "":
                raise ValueError("Please select a value for Income_Category.")
            input_data.append(income_map[income_value])  # Map to numerical value

            # Add numerical inputs
            for col in numerical_cols:
                if col == 'Income_Category':  # Skip as already handled
                    continue
                value = self.inputs[col].get()
                try:
                    input_data.append(float(value))
                except ValueError:
                    raise ValueError(f"Please enter a valid number for {col}.")

            # Create DataFrame for preprocessing
            input_df = pd.DataFrame([input_data], columns=categorical_cols + numerical_cols)
            input_preprocessed = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(input_preprocessed)
            result = "Attrited Customer" if prediction[0][0] > 0.5 else "Existing Customer"
            messagebox.showinfo("Prediction Result", f"The customer is predicted to be: {result}")

        except Exception as e:
            messagebox.showerror("Input Error", str(e))


# Run the GUI
root = tk.Tk()
app = CustomerAttritionApp(root)
root.mainloop()
