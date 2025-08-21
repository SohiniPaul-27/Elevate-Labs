import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="AegisShield - Cybersecurity Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -------------------- HEADER --------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        font-weight: bold;
        color: #00ADB5;
        text-align: center;
    }
    .sub-title {
        font-size:20px;
        color: #EEEEEE;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🛡️ AegisShield: Cybersecurity Threat Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time monitoring & anomaly detection powered by ML</p>', unsafe_allow_html=True)
st.divider()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🔍 Navigation")
options = st.sidebar.radio("Choose a section:", ["Upload Data", "Visualize", "Threat Detection", "About", "Honeypot"])

# -------------------- UPLOAD DATA --------------------
if options == "Upload Data":
    st.subheader("📂 Upload Network Traffic / Log Dataset")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file, on_bad_lines="skip")
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())
        st.session_state['df'] = df
        # Define target column globally
        st.session_state['target_col'] = df.columns[-1]

# -------------------- VISUALIZATION --------------------
elif options == "Visualize":
    st.subheader("📊 Data Visualization")
    if 'df' in st.session_state:
        df = st.session_state['df']

        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Features Distribution")
            st.bar_chart(df.iloc[:, :-1].sum())

        with col2:
            st.write("Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=df.iloc[:, -1], ax=ax, palette="viridis")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Upload a dataset first!")

# -------------------- THREAT DETECTION & ADAPTIVE LEARNING --------------------
elif options == "Threat Detection":
    st.subheader("🧠 Threat Detection & Adaptive Learning")
    if 'df' in st.session_state:
        df = st.session_state['df']
        target_col = st.session_state['target_col']

        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        for col in categorical_cols:
            le_col = LabelEncoder()
            df[col] = le_col.fit_transform(df[col].astype(str))

        # Encode target column
        if df[target_col].dtype == 'object':
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col].astype(str))
            st.session_state['label_encoder'] = le_target

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Load or initialize adaptive model
        if "adaptive_model" in st.session_state:
            model = st.session_state["adaptive_model"]
        else:
            model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
            model.partial_fit(X, y, classes=np.unique(y))
            st.session_state["adaptive_model"] = model

        st.success("✅ Adaptive model ready!")

        # Run predictions
        y_pred = model.predict(X)
        st.subheader("📈 Detection Report")
        st.text(classification_report(y, y_pred))

        # Confusion Matrix
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # -------------------- Real-time Adaptive Update --------------------
        st.markdown("### ⚡ Upload new labeled data to update IDS")

        new_file = st.file_uploader("Upload new labeled CSV", type=["csv"], key="adaptive")
        if new_file:
            new_df = pd.read_csv(new_file, on_bad_lines="skip")
            
            # Encode categorical columns in new data
            new_cat_cols = new_df.select_dtypes(include=['object']).columns.tolist()
            if target_col in new_cat_cols:
                new_cat_cols.remove(target_col)
            for col in new_cat_cols:
                le_col = LabelEncoder()
                new_df[col] = le_col.fit_transform(new_df[col].astype(str))

            # Encode target column using original LabelEncoder
            if 'label_encoder' in st.session_state:
                le_target = st.session_state['label_encoder']
                new_df[target_col] = le_target.transform(new_df[target_col].astype(str))

            new_X = new_df.iloc[:, :-1]
            new_y = new_df.iloc[:, -1]

            if st.button("Update Model"):
                model.partial_fit(new_X, new_y)
                st.session_state["adaptive_model"] = model
                joblib.dump(model, "cyber_model.pkl")
                st.success(f"✅ Model updated with {len(new_df)} new samples and saved!")

    else:
        st.warning("⚠️ Upload a dataset first!")

# -------------------- HONEYPOT INTEGRATION --------------------
elif options == "Honeypot":
    st.subheader("🕵️ Honeypot Feed Integration")
    st.markdown(
        """
        - AegisShield can connect to a honeypot server or logs.
        - Honeypot captures live attack attempts, which can be fed to the adaptive model.
        - This enables **automatic learning of new attack patterns**.
        """
    )

    honeypot_file = st.file_uploader("Upload Honeypot Logs (CSV)", type=["csv"])
    if honeypot_file:
        logs_df = pd.read_csv(honeypot_file, on_bad_lines="skip")
        st.dataframe(logs_df.head())

        target_col = logs_df.columns[-1]  # define target column for honeypot
        if "adaptive_model" in st.session_state:
            model = st.session_state["adaptive_model"]
            le = st.session_state.get('label_encoder', None)

            # Encode categorical columns in honeypot logs
            honeypot_cat_cols = logs_df.select_dtypes(include=['object']).columns.tolist()
            if target_col in honeypot_cat_cols:
                honeypot_cat_cols.remove(target_col)
            for col in honeypot_cat_cols:
                le_col = LabelEncoder()
                logs_df[col] = le_col.fit_transform(logs_df[col].astype(str))

            # Encode target column using original LabelEncoder
            if le:
                logs_df[target_col] = le.transform(logs_df[target_col].astype(str))

            X_logs = logs_df.iloc[:, :-1]
            y_logs = logs_df.iloc[:, -1]

            if st.button("Update Model with Honeypot Logs"):
                model.partial_fit(X_logs, y_logs)
                st.session_state["adaptive_model"] = model
                joblib.dump(model, "cyber_model.pkl")
                st.success(f"✅ Model updated with {len(logs_df)} honeypot samples!")
        else:
            st.warning("⚠️ Train or load adaptive model first in Threat Detection.")

# -------------------- ABOUT --------------------
elif options == "About":
    st.subheader("ℹ️ About AegisShield")
    st.markdown(
        """
        **AegisShield** is a lightweight cybersecurity monitoring dashboard  
        that allows you to:
        - 📂 Upload and visualize traffic/log data  
        - 🧠 Detect anomalies and attacks using ML with **adaptive learning**  
        - 🕵️ Feed honeypot logs to learn **new attack patterns automatically**  
        - 🔎 Explore patterns with visual dashboards  

        Built with **Streamlit** for interactive real-time use.  
        Future integrations: **Zeek IDS**, **Wazuh-SOAR**, and more.
        """
    )
