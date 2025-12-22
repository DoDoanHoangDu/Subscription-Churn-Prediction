import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BETA = 3
THRESHOLD = 0.1452
MODEL_PATH = "notebooks/output/churn_tree_model.pkl"
TREE_PLOT_PATH = "notebooks/output/churn_tree_plot.png"

EDGES = {
    "tenure": [0, 2, 6, 12, 19, 29, 40, 50, 61, 69, float("inf")],
    "MonthlyCharges": [0, 20.05, 25.05, 45.40, 59.00, 70.55, 79.25, 85.74, 94.55, 103.28, float("inf")],
    "TotalCharges": [0, 84.44, 269.81, 543.95, 928.88, 1372.45, 2071.61, 3248.81, 4509.76, 5969.91, float("inf")]
}

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-yes {
        background-color: #ffcccc;
        border: 2px solid #cc0000;
    }
    .churn-no {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
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
    """Display prediction result with styling"""
    is_churn = prediction[0] == "Yes"
    
    if is_churn:
        st.markdown(f"""
        <div class="prediction-box churn-yes">
            <h2 style="color: #cc0000; margin: 0;">HIGH CHURN RISK</h2>
            <h3 style="margin-top: 1rem;">Churn Probability: {probability[0]:.2f}%</h3>
            <p style="font-size: 1.1rem; margin-top: 1rem;">
                This customer is likely to churn. Consider retention strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box churn-no">
            <h2 style="color: #00cc00; margin: 0;">LOW CHURN RISK</h2>
            <h3 style="margin-top: 1rem;">Churn Probability: {probability[0]:.2f}%</h3>
            <p style="font-size: 1.1rem; margin-top: 1rem;">
                This customer is likely to stay. Continue providing quality service.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauge
    fig, ax = plt.subplots(figsize=(8, 2))
    colors = ['#00cc00', '#ffcc00', '#cc0000']
    bounds = [0, 30, 70, 100]
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=ax, orientation='horizontal')
    cb.set_label('Churn Risk Level', fontsize=12, weight='bold')
    cb.set_ticks([15, 50, 85])
    cb.set_ticklabels(['Low', 'Medium', 'High'])
    
    # Add marker for current probability
    ax.axvline(x=probability[0]/100, color='black', linewidth=3, linestyle='--')
    
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
            
            # Individual force plot for first prediction
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
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts customer churn using a Decision Tree model trained on telecom customer data.
    The model uses **calibrated probabilities** (β=3) and an **optimized threshold** (0.1452) to maximize recall.
    """)
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=80)
        st.title("Navigation")
        mode = st.radio("Select Mode:", 
                       ["Single Prediction", 
                        "Batch Prediction", 
                        "Model Info",
                        "Decision Tree"])
        
        st.markdown("---")
        st.markdown("### Model Parameters")
        st.info(f"""
        **Threshold:** {THRESHOLD}  
        **Beta (Calibration):** {BETA}  
        **Max Depth:** 10  
        **Class Weight (Churn):** 3:1
        """)
        
        st.markdown("---")
        st.markdown("### Performance Metrics")
        st.success("""
        **F2 Score:** 0.7482  
        **Accuracy:** ~80%  
        **Recall:** High (prioritized)
        """)
    
    # Main content based on mode
    if mode == "Single Prediction":
        customer_data = create_input_form()
        
        if st.button("Predict Churn", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Preprocess
                processed_data = preprocess_data(customer_data, model)
                
                # Predict
                prediction, probability = predict_churn(model, processed_data)
                
                # Display results
                st.markdown("---")
                display_prediction_result(prediction, probability)
                
                # Show input summary
                with st.expander("📋 View Customer Details"):
                    st.dataframe(customer_data.T, use_container_width=True)
                
                # SHAP explanation
                with st.expander("View Feature Explanations (SHAP)"):
                    show_shap_explanation(model, processed_data, processed_data.columns.tolist())
    
    elif mode == "Batch Prediction":
        st.subheader("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")
        
        # Sample file download
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 Your CSV should have the same format as the training data (Telco-Customer-Churn.csv)")
        with col2:
            if st.button("View Sample Format"):
                sample_df = pd.read_csv("data/Telco-Customer-Churn.csv").head(3)
                st.dataframe(sample_df, use_container_width=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} customers")
                
                # Show preview
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("Predict All", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Keep IDs if present
                        customer_ids = df['customerID'] if 'customerID' in df.columns else [f"CUST_{i}" for i in range(len(df))]
                        
                        # Preprocess
                        processed_data = preprocess_data(df, model)
                        
                        # Predict
                        predictions, probabilities = predict_churn(model, processed_data)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'CustomerID': customer_ids,
                            'Churn_Prediction': predictions,
                            'Churn_Probability': probabilities
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Total Customers", len(results_df))
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            churn_count = (results_df['Churn_Prediction'] == 'Yes').sum()
                            st.metric("Predicted Churners", churn_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            churn_rate = (churn_count / len(results_df)) * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Results table
                        st.dataframe(results_df.style.apply(
                            lambda x: ['background-color: #ffcccc' if v == 'Yes' else 'background-color: #ccffcc' 
                                     for v in x], 
                            subset=['Churn_Prediction']
                        ), use_container_width=True)
                        
                        # Probability distribution
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(probabilities, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                        ax.axvline(x=THRESHOLD*100, color='red', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD*100:.2f}%)')
                        ax.set_xlabel('Churn Probability (%)', fontsize=12)
                        ax.set_ylabel('Number of Customers', fontsize=12)
                        ax.set_title('Distribution of Churn Probabilities', fontsize=14, weight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Download results
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
    
    elif mode == "Model Info":
        st.subheader("Model Information & Performance")
        
        tab1, tab2, tab3 = st.tabs(["Model Details", "Feature Importance", "Methodology"])
        
        with tab1:
            st.markdown("### Decision Tree Classifier")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Hyperparameters")
                st.code("""
Max Depth: 10
Min Samples Split: 80
Min Samples Leaf: 20
Criterion: Gini
Splitter: Best
Class Weight: {No: 1, Yes: 3}
                """)
            
            with col2:
                st.markdown("#### Model Performance")
                st.code("""
Test F2 Score: 0.7482
Test Accuracy: ~80%
Best Threshold: 0.1452
Calibration Beta: 3
AUC-ROC: High
                """)
            
            st.markdown("#### Training Data Split")
            st.markdown("""
            - **Training Set:** 60% (4,225 customers)
            - **Validation Set:** 20% (1,408 customers)
            - **Test Set:** 20% (1,409 customers)
            - **Stratification:** Yes (maintains churn ratio)
            """)
            
            st.markdown("#### Feature Engineering")
            st.markdown("""
            1. **Categorical Encoding:**
               - Binary features: 0/1
               - Services without subscription: -1
               - Contract types: 0/1/2
            
            2. **Continuous Feature Binning:**
               - Tenure: 10 bins based on training quantiles
               - Monthly Charges: 10 bins based on training quantiles
               - Total Charges: 10 bins based on training quantiles
            
            3. **Calibration:**
               - Applied post-hoc probability calibration (β=3)
               - Adjusts for class imbalance
               - Improves probability estimates
            """)
        
        with tab2:
            st.markdown("### Feature Importance Analysis")
            show_feature_importance(model, model.feature_names_in_)
            
            st.markdown("#### Top Contributing Features")
            st.markdown("""
            Based on the decision tree's split importance, the most influential features for predicting churn are:
            
            - **Tenure (binned):** How long the customer has been with the company
            - **Contract Type:** Month-to-month customers churn more
            - **Internet Service:** Fiber optic users show different patterns
            - **Monthly Charges (binned):** Higher charges correlate with churn
            - **Payment Method:** Electronic check users churn more
            - **Tech Support:** Customers without support churn more
            """)
        
        with tab3:
            st.markdown("### Methodology")
            
            st.markdown("#### 1. Probability Calibration")
            st.latex(r'''
            p_{calibrated} = \frac{\frac{1}{\beta} \cdot p_{raw}}{\frac{1}{\beta} \cdot p_{raw} + (1 - p_{raw})}
            ''')
            st.markdown("""
            Where β = 3 (ratio of class weights). This calibration:
            - Adjusts for class imbalance
            - Improves probability estimates
            - Makes the model more conservative (higher recall)
            """)
            
            st.markdown("#### 2. Threshold Optimization")
            st.markdown("""
            The threshold (0.1452) was selected by:
            - Optimizing F2 score on validation set
            - F2 score weighs recall 2× more than precision
            - Business priority: Catch potential churners (high recall)
            - Acceptable trade-off: Some false positives
            """)
            
            st.markdown("#### 3. Why This Matters")
            st.markdown("""
            **High Recall Strategy:**
            - Better to incorrectly flag some loyal customers (false positive)
            - Than to miss actual churners (false negative)
            - Allows proactive retention efforts
            - Reduces revenue loss from churn
            """)
    
    else:  # Decision Tree visualization
        st.subheader("Decision Tree Visualization")
        
        st.markdown("""
        This is a visual representation of the trained decision tree. Each node shows:
        - **Split condition:** The feature and threshold used
        - **Gini impurity:** Measure of node purity
        - **Samples:** Number of training samples
        - **Value:** Distribution of [No Churn, Churn]
        - **Class:** Majority class at that node
        """)
        
        try:
            if Path(TREE_PLOT_PATH).exists():
                image = Image.open(TREE_PLOT_PATH)
                st.image(image, caption="Decision Tree Structure", use_container_width=True)
            else:
                st.warning("Tree plot not found. Generate it from the notebook first.")
                
                if st.button("Generate Tree Plot Now"):
                    with st.spinner("Generating tree visualization..."):
                        from sklearn.tree import plot_tree
                        fig, ax = plt.subplots(figsize=(200, 40))
                        plot_tree(
                            model,
                            feature_names=model.feature_names_in_,
                            class_names=[str(c) for c in model.classes_],
                            filled=True,
                            rounded=True,
                            fontsize=12,
                            ax=ax
                        )
                        plt.savefig(TREE_PLOT_PATH, dpi=100, bbox_inches='tight')
                        plt.close()
                        st.success("Tree plot generated!")
                        st.rerun()
        except Exception as e:
            st.error(f"Error displaying tree: {str(e)}")
        
        st.markdown("---")
        st.info("""
        **How to Read the Tree:**
        - Start at the top (root) node
        - Follow the branches based on feature values
        - Each split separates customers into groups
        - Leaf nodes (bottom) contain the final predictions
        - Color intensity indicates class confidence
        """)

if __name__ == "__main__":
    main()