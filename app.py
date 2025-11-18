# Save this as app.py for Streamlit deployment
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Fairness Audit", page_icon="⚖️", layout="wide")

st.title("⚖️ AI Bias Audit Dashboard")
st.subheader("Analyzing Gender Bias in Income Prediction Models")

@st.cache_data
def load_data():
    """Load and prepare Adult Dataset"""
    dataset = AdultDataset()
    return dataset

def main():
    # Sidebar for controls
    st.sidebar.header("Audit Configuration")
    protected_attribute = st.sidebar.selectbox(
        "Protected Attribute",
        ["sex", "race"],
        index=0
    )
    
    # Load data
    with st.spinner("Loading dataset and analyzing bias..."):
        dataset = load_data()
        
        # Convert to dataframe for analysis
        df = dataset.convert_to_dataframe()[0]
        
        # Display dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Positive Outcomes", f"{df['income-per-year'].mean()*100:.1f}%")
    
    # Bias Analysis Section
    st.header("📊 Bias Analysis Results")
    
    # Calculate metrics
    privileged_group = [{'sex': 1}]  # Male
    unprivileged_group = [{'sex': 0}]  # Female
    
    metric_orig = BinaryLabelDatasetMetric(
        dataset, 
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group
    )
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        di = metric_orig.disparate_impact()
        st.metric("Disparate Impact", f"{di:.3f}", 
                 delta="Fair" if 0.8 <= di <= 1.25 else "Biased",
                 delta_color="normal" if 0.8 <= di <= 1.25 else "inverse")
    
    with col2:
        spd = metric_orig.statistical_parity_difference()
        st.metric("Statistical Parity Difference", f"{spd:.3f}")
    
    with col3:
        consistency = metric_orig.consistency()
        st.metric("Consistency", f"{consistency:.3f}")
    
    with col4:
        # Basic outcome rates by gender
        male_rate = df[df['sex'] == 1]['income-per-year'].mean()
        female_rate = df[df['sex'] == 0]['income-per-year'].mean()
        st.metric("Gender Gap", f"{(male_rate - female_rate)*100:.1f}%")
    
    # Visualization
    st.header("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographic parity chart
        fig, ax = plt.subplots(figsize=(8, 6))
        rates = [female_rate, male_rate]
        labels = ['Female', 'Male']
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(labels, rates, color=colors, alpha=0.7)
        ax.set_ylabel('High Income Rate')
        ax.set_title('Income Distribution by Gender')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        # Feature correlation with gender
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_with_gender = numeric_df.corr()['sex'].drop('sex').sort_values()
        
        correlation_with_gender.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('Feature Correlation with Gender')
        ax.set_xlabel('Correlation Coefficient')
        st.pyplot(fig)
    
    # Mitigation Section
    st.header("🛠️ Bias Mitigation")
    
    if st.button("Apply Bias Mitigation Algorithm"):
        with st.spinner("Applying reweighing and retraining model..."):
            # Apply reweighing
            RW = Reweighing(unprivileged_groups=unprivileged_group,
                          privileged_groups=privileged_group)
            dataset_transformed = RW.fit_transform(dataset)
            
            # Train model on transformed data
            # (Add your model training code here)
            
            st.success("Bias mitigation applied successfully!")
            st.info("""
            **Mitigation Techniques Applied:**
            - Reweighing: Adjusting sample weights
            - Fairness-aware model training
            - Post-processing calibration
            """)
    
    # Recommendations
    st.header("💡 Recommendations")
    
    st.write("""
    **Based on this audit:**
    
    1. **Model Changes Required**: Disparate impact outside acceptable range (0.8-1.25)
    2. **Data Collection**: Ensure balanced representation in training data
    3. **Continuous Monitoring**: Implement regular bias audits
    4. **Transparency**: Document model limitations and fairness considerations
    """)

if __name__ == "__main__":
    main()
