import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Fairness Audit Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def generate_synthetic_hiring_data(n_samples=1000):
    """Generate synthetic hiring data with intentional bias"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        age = np.random.randint(22, 55)
        experience = np.random.randint(0, 20)
        education_level = np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.2, 0.1])
        skills_score = np.random.normal(75, 15)
        
        # Introduce gender bias
        gender = np.random.choice(['Male', 'Female'])
        
        if gender == 'Female':
            cultural_fit = np.random.normal(70, 12)
            technical_score = np.random.normal(72, 14)
        else:
            cultural_fit = np.random.normal(78, 10)
            technical_score = np.random.normal(80, 12)
        
        # Biased hiring decision
        base_score = (skills_score * 0.3 + experience * 2 + cultural_fit * 0.2 + technical_score * 0.3)
        if gender == 'Female':
            base_score -= 8  # Explicit bias
            
        hired = 1 if base_score > 160 else 0
        
        data.append({
            'age': age,
            'experience': experience,
            'education_level': education_level,
            'skills_score': max(0, min(100, skills_score)),
            'cultural_fit': max(0, min(100, cultural_fit)),
            'technical_score': max(0, min(100, technical_score)),
            'gender': gender,
            'hired': hired
        })
    
    return pd.DataFrame(data)

def main():
    st.markdown('<h1 class="main-header">⚖️ AI Bias Audit Dashboard</h1>', unsafe_allow_html=True)
    st.subheader("Analyzing Gender Bias in Synthetic Hiring Data")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    sample_size = st.sidebar.slider("Sample Size", 500, 2000, 1000)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", False)
    
    # Generate data
    with st.spinner("Generating synthetic hiring data..."):
        df = generate_synthetic_hiring_data(sample_size)
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(df))
    
    with col2:
        female_count = len(df[df['gender'] == 'Female'])
        st.metric("Female Candidates", female_count)
    
    with col3:
        male_count = len(df[df['gender'] == 'Male'])
        st.metric("Male Candidates", male_count)
    
    with col4:
        hiring_rate = df['hired'].mean() * 100
        st.metric("Overall Hiring Rate", f"{hiring_rate:.1f}%")
    
    # Bias Analysis Section
    st.header("📊 Bias Analysis")
    
    # Calculate hiring rates by gender
    hiring_by_gender = df.groupby('gender')['hired'].mean()
    female_rate = hiring_by_gender.get('Female', 0)
    male_rate = hiring_by_gender.get('Male', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Female Hiring Rate", f"{female_rate*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Male Hiring Rate", f"{male_rate*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        disparity = (male_rate - female_rate) * 100
        st.metric("Gender Disparity", f"{disparity:.1f}%", 
                 delta="Biased" if disparity > 5 else "Fair",
                 delta_color="inverse" if disparity > 5 else "normal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.header("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hiring rate comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        genders = ['Female', 'Male']
        rates = [female_rate, male_rate]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(genders, rates, color=colors, alpha=0.7)
        ax.set_ylabel('Hiring Rate')
        ax.set_title('Hiring Rate by Gender')
        ax.set_ylim(0, max(rates) * 1.2)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        # Score distributions
        fig, ax = plt.subplots(figsize=(8, 6))
        score_columns = ['skills_score', 'technical_score', 'cultural_fit']
        
        female_scores = df[df['gender'] == 'Female'][score_columns].mean()
        male_scores = df[df['gender'] == 'Male'][score_columns].mean()
        
        x = np.arange(len(score_columns))
        width = 0.35
        
        ax.bar(x - width/2, female_scores, width, label='Female', alpha=0.7)
        ax.bar(x + width/2, male_scores, width, label='Male', alpha=0.7)
        
        ax.set_xlabel('Score Type')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Gender')
        ax.set_xticks(x)
        ax.set_xticklabels(['Skills', 'Technical', 'Cultural Fit'])
        ax.legend()
        
        st.pyplot(fig)
    
    # Model Training Section
    st.header("🤖 ML Model Analysis")
    
    if st.button("Train Model & Analyze Bias"):
        with st.spinner("Training model and analyzing bias..."):
            # Prepare data for modeling
            df_encoded = df.copy()
            df_encoded['gender'] = df_encoded['gender'].map({'Female': 0, 'Male': 1})
            
            X = df_encoded.drop(['hired'], axis=1)
            y = df_encoded['hired']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
                
                # Feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['feature'], feature_importance['importance'])
                ax.set_title('Feature Importance in Hiring Decisions')
                ax.set_xlabel('Importance Score')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Bias Analysis Results")
                
                # Calculate disparate impact
                disparate_impact = female_rate / male_rate if male_rate > 0 else 0
                
                st.write(f"**Disparate Impact Ratio**: {disparate_impact:.3f}")
                if 0.8 <= disparate_impact <= 1.25:
                    st.success("✅ Within acceptable fairness range (0.8-1.25)")
                else:
                    st.error("❌ Outside acceptable fairness range")
                
                st.write(f"**Statistical Parity Difference**: {male_rate - female_rate:.3f}")
    
    # Recommendations
    st.header("💡 Recommendations & Insights")
    
    st.markdown("""
    **Based on this audit:**
    
    - 🔍 **Monitor Gender Impact**: Regular bias audits are essential
    - ⚖️ **Fairness Thresholds**: Set acceptable ranges for bias metrics
    - 📊 **Transparent Reporting**: Document model limitations
    - 🔄 **Continuous Improvement**: Implement bias mitigation strategies
    
    **Next Steps:**
    1. Implement reweighing techniques
    2. Add adversarial debiasing
    3. Regular model monitoring
    4. Stakeholder education
    """)
    
    # Raw data display (optional)
    if show_raw_data:
        st.header("📋 Raw Data Sample")
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
