import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from PIL import Image

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

BETA = 3
THRESHOLD = 0.1452
MODEL_PATH = "notebooks/output/churn_tree_model.pkl"

EDGES = {
    "tenure": [0, 2, 6, 12, 19, 29, 40, 50, 61, 69, float("inf")],
    "MonthlyCharges": [0, 20.05, 25.05, 45.40, 59.00, 70.55, 79.25, 85.74, 94.55, 103.28, float("inf")],
    "TotalCharges": [0, 84.44, 269.81, 543.95, 928.88, 1372.45, 2071.61, 3248.81, 4509.76, 5969.91, float("inf")]
}

RISK_SEGMENTS = {
    "top_priority": {"min": 80, "max": 100, "label": "Top-Priority", "color": "#4a4a4a"},
    "high": {"min": 50, "max": 80, "label": "High-Risk", "color": "#6b6b6b"},
    "medium": {"min": 15, "max": 50, "label": "Medium-Risk", "color": "#8c8c8c"},
    "low": {"min": 0, "max": 15, "label": "Low-Risk", "color": "#b0b0b0"}
}

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .segment-top-priority {
        background-color: #3d3d3d;
        border: 2px solid #1a1a1a;
        color: #ffffff;
    }
    .segment-high {
        background-color: #5a5a5a;
        border: 2px solid #3d3d3d;
        color: #ffffff;
    }
    .segment-medium {
        background-color: #a0a0a0;
        border: 2px solid #7a7a7a;
        color: #1a1a1a;
    }
    .segment-low {
        background-color: #d4d4d4;
        border: 2px solid #b0b0b0;
        color: #1a1a1a;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def get_model_features(model):
    """Get the exact feature names from the trained model"""
    return list(model.feature_names_in_)

def preprocess_data(df, model):
    """Preprocess input data according to training pipeline"""
    data = df.copy()
    
    # Store IDs and target for later
    has_id = 'customerID' in data.columns
    has_target = 'Churn' in data.columns
    
    # Handle TotalCharges
    if 'TotalCharges' in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        if 'tenure' in data.columns and 'MonthlyCharges' in data.columns:
            data["TotalCharges"] = data["TotalCharges"].fillna(
                data["MonthlyCharges"] * data["tenure"]
            ).astype(float)
    
    # Encode gender
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    
    # Encode Contract
    if 'Contract' in data.columns:
        data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    # Encode Yes/No columns
    for col in data.columns:
        if col in ['Churn', 'customerID']:
            continue
        unique_vals = data[col].dropna().unique()
        if set(unique_vals) <= {'Yes', 'No'}:
            data[col] = data[col].map({'No': 0, 'Yes': 1})
    
    # Encode service-specific columns
    for col in data.columns:
        if col in ['Churn', 'customerID']:
            continue
        unique_vals = data[col].dropna().unique()
        if 'No phone service' in unique_vals or 'No internet service' in unique_vals:
            mapping = {'No phone service': -1, 'No internet service': -1, 'No': 0, 'Yes': 1}
            data[col] = data[col].map(mapping)
    
    # Binning continuous features
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in data.columns:
            data[f"{col}_bin"] = pd.cut(
                data[col],
                bins=EDGES[col],
                labels=False,
                include_lowest=True
            )
    
    # Drop original continuous columns and ID/target BEFORE one-hot encoding
    columns_to_drop = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if has_id:
        columns_to_drop.append('customerID')
    if has_target:
        columns_to_drop.append('Churn')
    
    data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)
    
    # ONE-HOT ENCODE categorical features
    data = pd.get_dummies(data, drop_first=False)
    
    # Get exact feature names from the model (most reliable!)
    training_columns = get_model_features(model)
    
    # Add missing columns with 0s
    for col in training_columns:
        if col not in data.columns:
            data[col] = 0
    
    # Remove extra columns and reorder to match training EXACTLY
    data = data[training_columns]
    
    return data

def predict_churn(model, data):
    """Make predictions with calibration"""
    # Get raw probabilities
    probs_uncalibrated = model.predict_proba(data)[:, 1]
    
    # Apply calibration
    probs_calibrated = ((1/BETA) * probs_uncalibrated) / ((1/BETA) * probs_uncalibrated + (1 - probs_uncalibrated))
    
    # Apply threshold
    predictions = (probs_calibrated >= THRESHOLD).astype(int)
    predictions = np.where(predictions == 1, "Yes", "No")
    
    return predictions, probs_calibrated * 100

def get_risk_segment(probability):
    """Classify customer into risk segment based on churn probability"""
    if probability >= 80:
        return "top_priority", "Top-Priority", "Immediate action required. Often month-to-month or low-tenure customers."
    elif probability >= 50:
        return "high", "High-Risk", "Proactive retention strategies recommended."
    elif probability >= 15:
        return "medium", "Medium-Risk", "Monitor and consider targeted engagement."
    else:
        return "low", "Low-Risk", "No immediate action needed. Continue quality service."

def create_input_form():
    """Create input form for single prediction"""
    st.subheader("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
    with col2:
        st.markdown("**Account Information**")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
    with col3:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", 
            ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
    
    col4, col5 = st.columns(2)
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    with col5:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges * tenure, step=0.1)
    
    # Create dataframe from inputs
    customer_data = pd.DataFrame({
        'customerID': ['CUST_001'],
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    return customer_data

def display_prediction_result(prediction, probability):
    """Display prediction result with 4-level risk segmentation"""
    prob_value = probability[0]
    segment_key, segment_label, segment_action = get_risk_segment(prob_value)
    
    st.markdown(f"""
    <div class="prediction-box segment-{segment_key}">
        <h2 style="margin: 0;">{segment_label.upper()} SEGMENT</h2>
        <h3 style="margin-top: 1rem;">Churn Probability: {prob_value:.2f}%</h3>
        <p style="font-size: 1.1rem; margin-top: 1rem;">{segment_action}</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 2))
    colors = ['#d4d4d4', '#a0a0a0', '#5a5a5a', '#3d3d3d']
    bounds = [0, 15, 50, 80, 100]
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=ax, orientation='horizontal')
    cb.set_label('Risk Segment', fontsize=12, weight='bold')
    cb.set_ticks([7.5, 32.5, 65, 90])
    cb.set_ticklabels(['Low', 'Medium', 'High', 'Top-Priority'])
    
    ax.axvline(x=prob_value/100, color='#e74c3c', linewidth=3, linestyle='--')
    
    st.pyplot(fig)
    plt.close()

def show_feature_importance(model, feature_names):
    """Display feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 15 Feature Importances', fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()

def show_shap_explanation(model, data, feature_names):
    """Display SHAP explanation for predictions"""
    try:
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
            
            # Handle different SHAP output formats
            # For binary classification, shap_values is a list [class_0_values, class_1_values]
            if isinstance(shap_values, list):
                # Use values for positive class (index 1 = "Yes"/Churn)
                shap_values_to_use = shap_values[1]
                expected_value = explainer.expected_value[1]
            elif len(shap_values.shape) == 3:
                # 3D array format: (n_samples, n_features, n_classes)
                shap_values_to_use = shap_values[:, :, 1]
                expected_value = explainer.expected_value[1]
            else:
                # Single output format
                shap_values_to_use = shap_values
                expected_value = explainer.expected_value
            
            # SHAP summary plot
            st.subheader("Feature Impact on Prediction")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values_to_use, data, feature_names=feature_names, 
                            plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()
            
            if len(data) == 1:
                st.subheader("Detailed Prediction Breakdown")
                fig, ax = plt.subplots(figsize=(12, 3))
                shap.force_plot(expected_value, shap_values_to_use[0], 
                              data.iloc[0], matplotlib=True, show=False, 
                              feature_names=feature_names)
                st.pyplot(fig)
                plt.close()
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts customer churn using a Decision Tree model trained on telecom customer data.
    Customers are segmented into four risk categories for prioritized retention actions.
    """)
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    with st.sidebar:
        st.title("Navigation")
        mode = st.radio("Select Mode:", 
                    ["Single Prediction", 
                        "Batch Prediction"])
        
        st.markdown("---")
        st.markdown("### Risk Segments")
        st.markdown("""
        **Top-Priority** (>=80%)  
        **High-Risk** (50-80%)  
        **Medium-Risk** (15-50%)  
        **Low-Risk** (<15%)
        """)
        st.info(f"""
        **Threshold:** {THRESHOLD}  
        **Beta (Calibration):** {BETA}  
        **Max Depth:** 10  
        **Class Weight (Churn):** 3:1
        """)
        
    
    # Main content based on mode
    if mode == "Single Prediction":
        customer_data = create_input_form()
        
        if st.button("Predict Churn", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                processed_data = preprocess_data(customer_data, model)
                
                prediction, probability = predict_churn(model, processed_data)
                
                st.markdown("---")
                display_prediction_result(prediction, probability)
                
                with st.expander("View Customer Details"):
                    st.dataframe(customer_data.T, use_container_width=True)
                
                # SHAP explanation
                with st.expander("View Feature Explanations (SHAP)"):
                    show_shap_explanation(model, processed_data, processed_data.columns.tolist())
    
    elif mode == "Batch Prediction":
        st.subheader("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Your CSV should have the same format as the training data (Telco-Customer-Churn.csv)")
        with col2:
            if st.button("View Sample Format"):
                sample_df = pd.read_csv("data/Telco-Customer-Churn.csv").head(3)
                st.dataframe(sample_df, use_container_width=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} customers")
                
                with st.expander("Preview Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("Predict All", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Keep IDs if present
                        customer_ids = df['customerID'] if 'customerID' in df.columns else [f"CUST_{i}" for i in range(len(df))]
                        
                        processed_data = preprocess_data(df, model)
                        
                        predictions, probabilities = predict_churn(model, processed_data)
                        
                        results_df = pd.DataFrame({
                            'CustomerID': customer_ids,
                            'Churn_Prediction': predictions,
                            'Churn_Probability': probabilities
                        })
                        
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        def assign_segment(prob):
                            if prob >= 80:
                                return "Top-Priority"
                            elif prob >= 50:
                                return "High-Risk"
                            elif prob >= 15:
                                return "Medium-Risk"
                            else:
                                return "Low-Risk"
                        
                        results_df['Risk_Segment'] = results_df['Churn_Probability'].apply(assign_segment)
                        
                        top_priority_count = (results_df['Risk_Segment'] == 'Top-Priority').sum()
                        high_risk_count = (results_df['Risk_Segment'] == 'High-Risk').sum()
                        medium_risk_count = (results_df['Risk_Segment'] == 'Medium-Risk').sum()
                        low_risk_count = (results_df['Risk_Segment'] == 'Low-Risk').sum()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Top-Priority (>=80%)", top_priority_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("High-Risk (50-80%)", high_risk_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Medium-Risk (15-50%)", medium_risk_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Low-Risk (<15%)", low_risk_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        def style_segment(val):
                            if val == 'Top-Priority':
                                return 'background-color: #3d3d3d; color: white'
                            elif val == 'High-Risk':
                                return 'background-color: #5a5a5a; color: white'
                            elif val == 'Medium-Risk':
                                return 'background-color: #a0a0a0; color: black'
                            else:
                                return 'background-color: #d4d4d4; color: black'
                        
                        st.dataframe(results_df.style.applymap(
                            style_segment, subset=['Risk_Segment']
                        ), use_container_width=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(probabilities, bins=30, color='#5a6c7d', edgecolor='#2c3e50', alpha=0.8)
                        ax.axvline(x=15, color='#7f8c8d', linestyle='--', linewidth=1.5, label='15%')
                        ax.axvline(x=50, color='#5a5a5a', linestyle='--', linewidth=1.5, label='50%')
                        ax.axvline(x=80, color='#2c3e50', linestyle='--', linewidth=1.5, label='80%')
                        ax.set_xlabel('Churn Probability (%)', fontsize=12)
                        ax.set_ylabel('Number of Customers', fontsize=12)
                        ax.set_title('Distribution of Churn Probabilities by Risk Segment', fontsize=14, weight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    

if __name__ == "__main__":
    main()