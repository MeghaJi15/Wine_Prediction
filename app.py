import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Woodbridge Style - Wine Predictor", page_icon="🍷", layout="wide")

# ----------------------------------------------------------
# PREMIUM EDITORIAL CSS
# ----------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #F9F7F2;
        color: #2D2D2D;
        font-family: 'Inter', sans-serif;
    }

    .brand-header {
        text-align: center;
        padding: 60px 0;
        border-bottom: 1px solid #E0DCD0;
        margin-bottom: 60px;
    }
    
    .brand-logo {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: #1a1a1a;
        margin-bottom: 15px;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(20px, 6vw, 40px);
        font-weight: 200;
        line-height: 1.1;
        color: #1a1a1a;
        margin: 15px 0;
        text-transform: uppercase;
    }

    .spec-card {
        background: transparent;
        border: 1px solid #E0DCD0;
        padding: 50px;
        position: relative;
        margin-bottom: 40px;
    }

    /* Subtle Grid Overlay */
    .spec-card::after {
        content: "";
        position: absolute;
        top: 0; left: 10%; right: 10%; bottom: 0;
        border-left: 1px dashed rgba(224, 220, 208, 0.6);
        border-right: 1px dashed rgba(224, 220, 208, 0.6);
        pointer-events: none;
    }

    /* Input Styling */
    .stNumberInput label {
        font-family: 'Playfair Display', serif !important;
        text-transform: uppercase;
        font-size: 12px !important;
        letter-spacing: 1.5px;
        color: #888 !important;
    }

    .stNumberInput div div input {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid #2D2D2D !important;
        border-radius: 0px !important;
        font-size: 22px !important;
        padding: 12px 0px !important;
    }

    .stButton>button {
        background-color: #1a1a1a;
        color: #ffffff;
        border-radius: 0px;
        border: none;
        padding: 25px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 14px;
        margin-top: 40px;
        transition: 0.4s;
    }

    .stButton>button:hover {
        background-color: #8B0000;
        color: white;
    }

    .result-box {
        margin-top: 80px;
        padding: 80px;
        text-align: center;
        border: 1px solid #1a1a1a;
        background-color: #fffdfa;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# DATA & MODEL LOGIC
# ----------------------------------------------------------
@st.cache_resource
def get_model():
    try:
        df = pd.read_csv("winequality-red.csv")
        df.columns = [c.replace(' ', '_').lower().strip() for c in df.columns]
    except Exception as e:
        # Create dummy data if file is missing for demonstration
        cols = ['alcohol', 'ph', 'sulphates', 'citric_acid', 'residual_sugar', 'fixed_acidity', 'quality']
        df = pd.DataFrame(np.random.rand(100, 7), columns=cols)
        df['quality'] = np.random.randint(3, 9, 100)

    X = df.drop("quality", axis=1)
    y_class = [1 if q >= 7 else 0 for q in df["quality"]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y_class)
    
    return model, scaler, X.columns, df

model, scaler, feature_names, df = get_model()

# ----------------------------------------------------------
# UI LAYOUT
# ----------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <p class="brand-logo">Woodbridge Heritage</p>
    <h1 class="main-title">Predict the Wine Quality</h1>
    
</div>
""", unsafe_allow_html=True)

# Main Form Area
col_left, col_spacer, col_right = st.columns([1.5, 0.2, 2])

with col_left:
    # Use a reliable direct link for the image
    st.image("wine.jpg", 
             
             width=300)

with col_right:
    
    st.markdown("<p style='font-family: Playfair Display; font-size: 26px; margin-bottom: 40px; border-bottom: 1px solid #E0DCD0; padding-bottom: 10px;'>Laboratory Specs</p>", unsafe_allow_html=True)
    
    display_features = ['alcohol', 'ph', 'sulphates', 'citric_acid', 'residual_sugar', 'fixed_acidity']
    input_values = {}
    
    c1, c2 = st.columns(2)
    for i, feat in enumerate(display_features):
        target_col = c1 if i % 2 == 0 else c2
        with target_col:
            nice_label = feat.replace("_", " ").title()
            val = st.number_input(
                nice_label, 
                value=float(df[feat].mean()),
                step=0.01,
                format="%.2f",
                key=f"input_{feat}"
            )
            input_values[feat] = val
            
    # Prepare data for prediction
    full_input_row = []
    for feat in feature_names:
        full_input_row.append(input_values.get(feat, df[feat].mean()))

    predict_btn = st.button("Generate Tasting Profile")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# RESULTS & OUTPUT (Explicitly "Premium" vs "Basic")
# ----------------------------------------------------------
if predict_btn:
    input_array = np.array([full_input_row])
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    # Custom Score 
    final_score = int(75 + (prob * 20)) if prediction == 1 else int(60 + (prob * 14))

    st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True) # Extra space

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box">
            <p style="letter-spacing: 5px; text-transform: uppercase; font-size: 14px; color: #8B0000; font-weight: bold;">Result: Premium Quality Wine</p>
            <h2 class="main-title" style="margin: 25px 0; font-size: 90px; color: #1a1a1a;">{final_score}</h2>
            <p style="font-family: Inter; letter-spacing: 2px; text-transform: uppercase; font-size: 12px; color: #7a7a7a; margin-bottom: 30px;">Excellent Balance & Character</p>
            <p style="font-family: Playfair Display; font-size: 24px; font-style: italic; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                "This vintage demonstrates exceptional varietal character. Expect deep flavors of blackberry and rich cedar with an elegant, toasty finish."
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box" style="border-color: #D0CCC0;">
            <p style="letter-spacing: 5px; text-transform: uppercase; font-size: 14px; color: #7a7a7a; font-weight: bold;">Result: Basic Quality Wine</p>
            <h2 class="main-title" style="margin: 25px 0; font-size: 90px; color: #7a7a7a;">{final_score}</h2>
            <p style="font-family: Inter; letter-spacing: 2px; text-transform: uppercase; font-size: 12px; color: #999; margin-bottom: 30px;">Standard Winemaker Selection</p>
            <p style="font-family: Playfair Display; font-size: 24px; font-style: italic; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                "A soft and approachable wine. Aromas of juicy plums marry with notes of herb, making it a versatile choice for everyday pairing."
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("""
<div style="margin-top: 120px; padding: 60px 0; text-align: center; border-top: 1px solid #E0DCD0;">
    <p style="font-size: 12px; letter-spacing: 4px; color: #aaa; text-transform: uppercase; font-family: 'Inter';"> 
        Enology AI Assistant • Proprietary Dataset Analysis • 2026
    </p>
</div>
""", unsafe_allow_html=True)