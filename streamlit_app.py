import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Healthcare Data Analytics",
    page_icon="🏥",
    layout="wide"
)

# App title
st.title("🏥 Healthcare Data Analytics Dashboard")
st.markdown("Upload and analyze healthcare data")

# Sidebar
st.sidebar.header("Data Input")

# File upload - CSV only
uploaded_file = st.sidebar.file_uploader("Upload your health dataset", type=["csv"])

# Create a demo dataset
def get_demo_data():
    # Sample healthcare costs data
    data = pd.DataFrame({
        'PatientID': range(1, 101),
        'Age': np.random.randint(18, 85, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Department': np.random.choice(['Cardiology', 'Neurology', 'Oncology', 'Orthopedics'], 100),
        'LengthOfStay': np.random.randint(1, 30, 100),
        'TotalCost': np.random.uniform(500, 50000, 100),
        'Readmission': np.random.choice([0, 1], 100, p=[0.85, 0.15])
    })
    return data

# Use demo data button
use_demo = st.sidebar.button("Use Demo Data")

# Load data
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully!")
elif use_demo:
    df = get_demo_data()
    st.sidebar.success("Demo data loaded!")

# Main content
if df is not None:
    # Data overview
    st.header("Data Overview")
    st.dataframe(df.head())
    
    # Basic statistics
    st.header("Data Statistics")
    st.dataframe(df.describe())
    
    # Simple visualizations using Matplotlib
    st.header("Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Age'], bins=10, edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        if 'Gender' in df.columns:
            st.subheader("Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)
    
    # Length of Stay Analysis (if available)
    if 'LengthOfStay' in df.columns and 'Department' in df.columns:
        st.header("Length of Stay by Department")
        
        dept_los = df.groupby('Department')['LengthOfStay'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(dept_los['Department'], dept_los['LengthOfStay'])
        ax.set_xlabel('Department')
        ax.set_ylabel('Average Length of Stay (days)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Cost Analysis (if available)
    if 'TotalCost' in df.columns:
        st.header("Cost Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['Age'], df['TotalCost'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Age vs. Total Cost')
        st.pyplot(fig)
    
    # Simple predictive modeling
    if 'LengthOfStay' in df.columns and 'Age' in df.columns:
        st.header("Predictive Modeling")
        
        # Choose target variable
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        target_variable = st.selectbox("Select target variable", 
                                      [col for col in numeric_cols if col not in ['PatientID']])
        
        # Choose features
        features = st.multiselect("Select features for prediction", 
                                [col for col in numeric_cols if col != target_variable and col != 'PatientID'],
                                default=[col for col in numeric_cols if col != target_variable and col != 'PatientID'][:2])
        
        if features and st.button("Train Model"):
            # Prepare data
            X = df[features]
            y = df[target_variable]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Display results
            st.subheader("Model Performance")
            st.write(f"Training R² Score: {train_score:.4f}")
            st.write(f"Testing R² Score: {test_score:.4f}")
            
            # Display coefficients
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df)
else:
    st.info("Please upload a CSV file or use the demo data to get started.")

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:** This healthcare analytics platform helps analyze patient data,
monitor clinical outcomes, and build simple predictive models.
""")
