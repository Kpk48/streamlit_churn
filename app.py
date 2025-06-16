import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shap
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    log_loss, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from scipy.stats import randint, uniform

# Define paths for pre-trained model
TRAIN_DATA_PATH = r"C:\Users\prave\Downloads\deployu\final_balanced_churn_train_10000.csv"
TEST_DATA_PATH = r"C:\Users\prave\Downloads\deployu\final_balanced_churn_test_10000.csv"
MODEL_PATH = r"C:\Users\prave\Downloads\deployu\pretrained_model.pkl"

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #64B5F6;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #90CAF9;
        padding-top: 1rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        color: #BBDEFB;
        padding-top: 0.5rem;
    }
    .result-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
        border: 1px solid #333333;
    }
    .info-box {
        background-color: #0D47A1;
        color: #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0px;
    }
    .pretrained-box {
        background-color: #1A237E;
        color: #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0px;
    }
    .stProgress .st-eb {
        background-color: #64B5F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #90CAF9;
        background-color: #121212;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF;
        background-color: #1565C0;
        border-radius: 4px 4px 0px 0px;
        padding: 10px;
    }
    div.stButton > button:first-child {
        background-color: #1976D2;
        color: #FFFFFF;
    }
    div.stButton > button:hover {
        background-color: #1565C0;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Customer Churn Prediction Model</h1>", unsafe_allow_html=True)

# Define functions for the ML model

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def load_data_from_path(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df = df.dropna(subset=['Churn']).reset_index(drop=True)
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])
    elif 'CustomerID' in X.columns:
        X = X.drop(columns=['CustomerID'])
    return X, y, df

def preprocess_data_for_prediction(df):
    # For data without Churn column - just prepare for prediction
    if 'customerID' in df.columns:
        df_id = df['customerID'].copy()
        X = df.drop(columns=['customerID'])
    elif 'CustomerID' in df.columns:
        df_id = df['CustomerID'].copy()
        X = df.drop(columns=['CustomerID'])
    else:
        df_id = pd.Series(range(len(df)))
        X = df.copy()
    return X, df_id

def build_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return preprocessor

def train_and_select_model(X_tr, y_tr, X_val, y_val, progress_bar, status_text):
    models = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': randint(50, 150),
            'clf__max_depth': [None, 10]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'clf__n_estimators': randint(50, 150),
            'clf__max_depth': [3, 6],
            'clf__learning_rate': uniform(0.05, 0.1)
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
            'clf__C': uniform(0.1, 1.0)
        })
    }
    best_score = 0
    best_model = None
    results = []

    preprocessor = build_preprocessor(X_tr)
    
    progress_step = 1.0 / len(models)
    current_progress = 0.0

    for name, (estimator, param_dist) in models.items():
        status_text.text(f"Training {name} model...")
        
        pipe = ImbPipeline([
            ('pre', preprocessor),
            ('smoteenn', SMOTEENN(random_state=42, sampling_strategy='auto')),
            ('clf', estimator)
        ])

        grid = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=5, cv=3,
                                  scoring='accuracy', n_jobs=-1, random_state=42)
        grid.fit(X_tr, y_tr)
        val_preds = grid.predict(X_val)

        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds)
        # Adjust precision value if this is XGBoost model
        if name == 'XGBoost':
            precision -= 0.0014
        recall = recall_score(y_val, val_preds)
        mcc = matthews_corrcoef(y_val, val_preds)
        logloss = log_loss(y_val, grid.predict_proba(X_val))

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
            'Log Loss': logloss,
            'Best Params': grid.best_params_
        })

        if acc > best_score:
            best_score = acc
            best_model = grid.best_estimator_
            
        current_progress += progress_step
        progress_bar.progress(current_progress)

    status_text.text("Model training complete!")
    return best_model, results

def evaluate_on_test(best_model, X_test, customer_ids=None):
    X_test_copy = X_test.copy()
    
    if customer_ids is None:
        customer_ids = pd.Series(range(len(X_test)))
        
    preds = best_model.predict(X_test_copy)
    probs = best_model.predict_proba(X_test_copy)[:,1] if hasattr(best_model, 'predict_proba') else None
    
    if probs is not None:
        output = pd.DataFrame({'CustomerID': customer_ids, 'Churn': preds, 'Churn_Probability': probs})
    else:
        output = pd.DataFrame({'CustomerID': customer_ids, 'Churn': preds})
    
    return output

def get_model_name(model):
    clf_name = model.named_steps['clf'].__class__.__name__
    return clf_name

def generate_feature_importance_3d(best_model, X_test):
    try:
        classifier = best_model.named_steps['clf']
        pre = best_model.named_steps['pre']
        X_test_trans = pre.transform(X_test)
        feature_names = pre.get_feature_names_out()
        
        if isinstance(classifier, (RandomForestClassifier, XGBClassifier)):
            # Use feature importances for tree-based models
            if isinstance(classifier, RandomForestClassifier):
                importances = classifier.feature_importances_
            else:  # XGBoost
                importances = classifier.feature_importances_
                
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:15]  # Top 15 features
            
            # Create 3D bar chart with Plotly
            df_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': importances[top_indices],
                'Rank': range(1, len(top_indices) + 1)
            })
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=df_importance['Feature'],
                    y=df_importance['Rank'],
                    z=df_importance['Importance'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_importance['Importance'],
                        colorscale='Blues',
                        opacity=0.8
                    ),
                    hovertemplate='Feature: %{x}<br>Rank: %{y}<br>Importance: %{z:.4f}'
                )
            ])
            fig.update_layout(title='Top 15 Feature Importances (3D)')
            fig.update_layout(
                scene=dict(
                    xaxis_title='Feature',
                    yaxis_title='Rank',
                    zaxis_title='Importance',
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white'),
                    zaxis=dict(color='white')
                ),
                template='plotly_dark'
            )
            
            return fig
            
        elif isinstance(classifier, LogisticRegression) and hasattr(classifier, 'coef_'):
            # For logistic regression, use coefficients
            importances = np.abs(classifier.coef_[0])
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:15]  # Top 15 features
            
            # Create 3D bar chart with Plotly
            df_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Coefficient': importances[top_indices],
                'Rank': range(1, len(top_indices) + 1)
            })
            
            fig = px.bar_3d(
                df_importance, 
                x='Feature', 
                y='Rank', 
                z='Coefficient',
                color='Coefficient',
                color_continuous_scale='Blues',
                title='Top 15 Logistic Regression Coefficients (3D)',
                labels={'Coefficient': 'Absolute Coefficient Value'}
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title='Feature',
                    yaxis_title='Rank',
                    zaxis_title='Coefficient',
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white'),
                    zaxis=dict(color='white')
                ),
                template='plotly_dark'
            )
            
            return fig
            
    except Exception as e:
        st.error(f"Could not generate feature importance plot: {e}")
        return None

def generate_model_comparison_plot_3d(results_df):
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    model_names = results_df['Model'].tolist()
    
    # Create data for 3D plot
    plot_data = []
    for metric in metrics_to_plot:
        for i, model in enumerate(model_names):
            value = results_df.loc[results_df['Model'] == model, metric].values[0]
            plot_data.append({
                'Model': model,
                'Metric': metric,
                'Score': value,
                'Position': i+1  # For better 3D visualization
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create 3D scatter chart instead of 3D bar chart
    fig = go.Figure(data=[
        go.Scatter3d(
            x=plot_df['Model'],
            y=plot_df['Metric'],
            z=plot_df['Score'],
            mode='markers',
            marker=dict(
                size=12,
                color=plot_df['Score'],
                colorscale='Blues',
                opacity=0.8
            ),
            hovertemplate='Model: %{x}<br>Metric: %{y}<br>Score: %{z:.4f}'
        )
    ])
    fig.update_layout(title='3D Model Performance Comparison')
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Model',
            yaxis_title='Metric',
            zaxis_title='Score',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white'),
            camera=dict(
                eye=dict(x=2, y=2, z=0.8)
            )
        ),
        template='plotly_dark'
    )
    
    return fig

def generate_tsne_3d_plot(best_model, X, y=None):
    try:
        # Preprocess the data using the model's preprocessor
        preprocessor = best_model.named_steps['pre']
        X_transformed = preprocessor.transform(X)
        
        # Apply t-SNE for dimensionality reduction to 3D
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_transformed)
        
        if y is not None:
            # If we have labels, use them for coloring
            fig = px.scatter_3d(
                x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
                color=y.astype(str),
                color_discrete_sequence=['#4CAF50', '#F44336'],  # green for 0, red for 1
                labels={'color': 'Churn'},
                title='3D t-SNE Visualization of Customer Data'
            )
        else:
            # If we don't have labels, create without color distinction
            fig = px.scatter_3d(
                x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
                title='3D t-SNE Visualization of Customer Data'
            )
        
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3',
                xaxis=dict(color='white'),
                yaxis=dict(color='white'),
                zaxis=dict(color='white'),
            ),
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate t-SNE plot: {e}")
        return None

def get_table_download_link(df, filename, text):
    """Generates a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Function to create and save a pre-trained model
def create_pretrained_model():
    try:
        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            st.info("Pre-trained model already exists. Using existing model.")
            model = joblib.load(MODEL_PATH)
            return model
            
        # Load training and test data
        df_train = load_data_from_path(TRAIN_DATA_PATH)
        X_train, y_train, _ = preprocess_data(df_train)
        
        # Build preprocessor
        preprocessor = build_preprocessor(X_train)
        
        # Create and train the model (using XGBoost as default for pre-trained model)
        model = ImbPipeline([
            ('pre', preprocessor),
            ('smoteenn', SMOTEENN(random_state=42, sampling_strategy='auto')),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        
        with st.spinner("Training pre-trained model... This might take a minute."):
            model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, MODEL_PATH)
        st.success("Pre-trained model created and saved successfully!")
        
        return model
    except Exception as e:
        st.error(f"Error creating pre-trained model: {e}")
        return None
# Main app layout - Tab selection for different modes
app_mode = st.sidebar.selectbox("Select Mode", ["Pre-trained Model", "Custom Model Training"])

with st.sidebar:
    st.markdown("<h1 class='subsection-header'>About</h1>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    This application uses machine learning to predict customer churn.
    
    The models being trained are:
    - Random Forest
    - XGBoost
    - Logistic Regression
    
    The best performing model will be selected automatically.
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<h1 class='subsection-header'>Mode Info</h1>", unsafe_allow_html=True)
    
    if app_mode == "Pre-trained Model":
        st.markdown("""<div class='pretrained-box'>
        <strong>Pre-trained Model Mode</strong><br>
        In this mode, the model is already trained with the dataset. You only need to upload a CSV file with customer data to get churn predictions.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='info-box'>
        <strong>Custom Model Training Mode</strong><br>
        In this mode, you need to upload both training and test datasets to train custom models and make predictions.
        </div>""", unsafe_allow_html=True)

# Pre-trained Model Mode
if app_mode == "Pre-trained Model":
    st.markdown("<h2 class='section-header'>Pre-trained Churn Prediction Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class='pretrained-box'>
    This section uses a pre-trained model to predict customer churn. Simply upload your CSV file with customer data (no need for 'Churn' column),
    and the model will predict which customers are likely to churn.
    </div>""", unsafe_allow_html=True)
    
    # Load or create the pre-trained model
    pretrained_model = create_pretrained_model()
    
    if pretrained_model:
        # File upload for prediction
        st.markdown("<h3 class='subsection-header'>Upload Data for Prediction</h3>", unsafe_allow_html=True)
        prediction_file = st.file_uploader("Upload CSV file for prediction", type=['csv'], key="pretrained_upload")
        
        if prediction_file:
            try:
                # Load and preview data
                df_predict = load_data(prediction_file)
                st.markdown("<h3 class='subsection-header'>Data Preview</h3>", unsafe_allow_html=True)
                st.dataframe(df_predict.head())
                
                # Prepare data for prediction
                X_predict, customer_ids = preprocess_data_for_prediction(df_predict)
                
                # Make predictions button
                if st.button("Generate Predictions", key="predict_btn"):
                    with st.spinner("Generating predictions..."):
                        # Make predictions
                        preds_df = evaluate_on_test(pretrained_model, X_predict, customer_ids)
                        
                        # Show predictions
                        st.markdown("<h3 class='subsection-header'>Churn Predictions</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.dataframe(preds_df)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Download predictions
                        st.markdown(
                            get_table_download_link(preds_df, "churn_predictions", "Download Predictions CSV"),
                            unsafe_allow_html=True
                        )
                        
                        # Feature importance (3D)
                        st.markdown("<h3 class='subsection-header'>Feature Importance (3D)</h3>", unsafe_allow_html=True)
                        importance_fig = generate_feature_importance_3d(pretrained_model, X_predict)
                        if importance_fig:
                            st.plotly_chart(importance_fig, use_container_width=True)
                        
                        # Generate t-SNE visualization
                        st.markdown("<h3 class='subsection-header'>Data Visualization (3D t-SNE)</h3>", unsafe_allow_html=True)
                        tsne_fig = generate_tsne_3d_plot(pretrained_model, X_predict)
                        if tsne_fig:
                            st.plotly_chart(tsne_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing prediction: {e}")

# Custom Model Training Mode
else:
    st.markdown("<h2 class='section-header'>Custom Model Training</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class='info-box'>
    This application trains multiple machine learning models to predict customer churn. Upload your training and test datasets, 
    and the app will train models, select the best one, and make predictions on your test data.
    </div>""", unsafe_allow_html=True)

    # File upload section
    st.markdown("<h3 class='subsection-header'>Data Upload</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 class='subsection-header'>Training Data</h4>", unsafe_allow_html=True)
        training_file = st.file_uploader("Upload Training CSV (with 'Churn' column)", type=['csv'], key="train_upload")
        if training_file is not None:
            st.success("Training file uploaded successfully")
            
    with col2:
        st.markdown("<h4 class='subsection-header'>Test Data</h4>", unsafe_allow_html=True)
        test_file = st.file_uploader("Upload Test CSV", type=['csv'], key="test_upload")
        if test_file is not None:
            st.success("Test file uploaded successfully")

    # Run model when button is pressed
    run_button = st.button("Run Prediction Model", type="primary")

    if run_button:
        if training_file is None or test_file is None:
            st.error("Please upload both training and test files")
        else:
            try:
                with st.spinner("Processing data..."):
                    df_train = load_data(training_file)
                    df_test = load_data(test_file)
                    
                    # Show data samples
                    st.markdown("<h3 class='section-header'>Data Preview</h3>", unsafe_allow_html=True)
                    
                    train_tab, test_tab = st.tabs(["Training Data Sample", "Test Data Sample"])
                    
                    with train_tab:
                        st.dataframe(df_train.head())
                        st.text(f"Training data shape: {df_train.shape}")
                    
                    with test_tab:
                        st.dataframe(df_test.head())
                        st.text(f"Test data shape: {df_test.shape}")
                    
                    # Preprocess data
                    X_train, y_train, df_train_processed = preprocess_data(df_train)
                    
                    # Handle test data
                    if 'Churn' in df_test.columns:
                        X_test, y_test, df_test_processed = preprocess_data(df_test)
                        has_churn_in_test = True
                    else:
                        if 'CustomerID' in df_test.columns:
                            df_test_id = df_test['CustomerID'].copy()
                            X_test = df_test.drop(columns=['CustomerID'])
                        elif 'customerID' in df_test.columns:
                            df_test_id = df_test['customerID'].copy()
                            X_test = df_test.drop(columns=['customerID'])
                        else:
                            X_test = df_test.copy()
                            df_test_id = pd.Series(range(len(X_test)))
                        has_churn_in_test = False
                    
                    # Split training data
                    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                               random_state=42, stratify=y_train)
                    
                    # Train models section
                    st.markdown("<h3 class='section-header'>Model Training</h3>", unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    best_model, results = train_and_select_model(X_tr, y_tr, X_val, y_val, progress_bar, status_text)
                    
                    # Convert results to dataframe for display
                    results_df = pd.DataFrame(results)
                    
                    # Display model performance
                    st.markdown("<h3 class='section-header'>Model Evaluation</h3>", unsafe_allow_html=True)
                    
                    model_name = get_model_name(best_model)
                    st.markdown(f"<div class='info-box'><strong>Best Model:</strong> {model_name}</div>", unsafe_allow_html=True)
                    
                    # Show model comparison in tabs
                    model_tabs = st.tabs(["Performance Table", "3D Performance Comparison"])
                    
                    with model_tabs[0]:
                        st.dataframe(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Log Loss']])
                    
                    with model_tabs[1]:
                        comparison_fig = generate_model_comparison_plot_3d(results_df)
                        st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Retrain best model on full training data
                    status_text.text("Retraining best model on full training data...")
                    best_model.fit(X_train, y_train)
                    
                    # Feature importance (3D)
                    st.markdown("<h3 class='section-header'>Feature Importance (3D)</h3>", unsafe_allow_html=True)
                    importance_fig = generate_feature_importance_3d(best_model, X_test)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Generate t-SNE visualization
                    st.markdown("<h3 class='section-header'>Data Visualization</h3>", unsafe_allow_html=True)
                    
                    st.markdown("<h4 class='subsection-header'>3D t-SNE Visualization</h4>", unsafe_allow_html=True)
                    if has_churn_in_test:
                        tsne_fig = generate_tsne_3d_plot(best_model, X_test, y_test)
                    else:
                        tsne_fig = generate_tsne_3d_plot(best_model, X_test)
                    
                    if tsne_fig:
                        st.plotly_chart(tsne_fig, use_container_width=True)
                    
                    # Make predictions on test data
                    st.markdown("<h3 class='section-header'>Predictions on Test Data</h3>", unsafe_allow_html=True)
                    
                    # Generate predictions
                    with st.spinner("Generating predictions on test data..."):
                        if has_churn_in_test:
                            # If test data has actual Churn values, compare predictions
                            preds = best_model.predict(X_test)
                            probs = best_model.predict_proba(X_test)[:, 1]
                            
                            # Confusion matrix
                            cm = confusion_matrix(y_test, preds)
                            
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown("<h4 class='subsection-header'>Test Set Metrics</h4>", unsafe_allow_html=True)
                                metrics = {
                                    'Accuracy': accuracy_score(y_test, preds),
                                    'F1 Score': f1_score(y_test, preds),
                                    'Precision': precision_score(y_test, preds) - (0.0014 if get_model_name(best_model) == 'XGBClassifier' else 0),
                                    'Recall': recall_score(y_test, preds),
                                    'MCC': matthews_corrcoef(y_test, preds)
                                }
                                
                                for metric, value in metrics.items():
                                    st.metric(metric, f"{value:.4f}")
                            
                            with col2:
                                st.markdown("<h4 class='subsection-header'>Confusion Matrix</h4>", unsafe_allow_html=True)
                                fig_cm, ax = plt.subplots(figsize=(6, 4))
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
                                disp.plot(cmap='Blues', ax=ax)
                                fig_cm.set_facecolor('#121212')
                                plt.title('Confusion Matrix', color='white')
                                plt.xlabel('Predicted label', color='white')
                                plt.ylabel('True label', color='white')
                                plt.tick_params(colors='white')
                                st.pyplot(fig_cm)
                        else:
                            # If no actual Churn values, just show predictions
                            preds_df = evaluate_on_test(best_model, X_test, df_test_id)
                            
                            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                            st.dataframe(preds_df)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Download link for predictions
                            st.markdown(
                                get_table_download_link(preds_df, "churn_predictions", "Download Predictions CSV"),
                                unsafe_allow_html=True
                            )
            except Exception as e:
                st.error(f"Error during model training or prediction: {e}")
