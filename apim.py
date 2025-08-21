import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System (IDS)")

# -----------------------------
# Dataset Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, on_bad_lines="skip")
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Drop unnamed/empty cols
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Encode categorical cols
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le  # save encoders

    # Target column detection
    if "Label" not in df.columns:
        st.error("No 'Label' column found. Please check dataset.")
    else:
        X = df.drop("Label", axis=1)
        y = df["Label"]

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Light RandomForest
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
        st.subheader("📑 Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.success("✅ IDS model trained successfully (fast mode).")

        # -----------------------------
        # Live intrusion test
        # -----------------------------
        st.subheader("🔍 Live Intrusion Test")
        test_input = {}
        for col in X.columns:
            if col in le_dict:  # categorical
                options = list(le_dict[col].classes_)
                val = st.selectbox(f"{col}", options)
                test_input[col] = le_dict[col].transform([val])[0]
            else:  # numeric
                val = st.number_input(f"{col}", value=float(X[col].median()))
                test_input[col] = val

        if st.button("Predict Intrusion"):
            input_df = pd.DataFrame([test_input])
            pred = model.predict(input_df)[0]
            pred_label = "Benign ✅" if pred == 0 else "Attack 🚨"
            st.subheader(f"🔎 Prediction: {pred_label}")

        # -----------------------------
        # Batch intrusion test
        # -----------------------------
        st.subheader("📂 Batch Intrusion Test")
        batch_file = st.file_uploader("Upload CSV of new packets for batch prediction", type="csv", key="batch")

        if batch_file:
            batch_df = pd.read_csv(batch_file, on_bad_lines="skip")
            batch_df = batch_df.loc[:, ~batch_df.columns.str.contains('^Unnamed')]

            # Safe encoding for categorical columns
            for col, le in le_dict.items():
                if col in batch_df.columns:
                    batch_df[col] = batch_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    st.warning(f"Column {col} missing in batch CSV. Filling with median values.")
                    batch_df[col] = X[col].median()

            # Ensure same columns as training
            missing_cols = set(X.columns) - set(batch_df.columns)
            for col in missing_cols:
                batch_df[col] = X[col].median()
            batch_df = batch_df[X.columns]

            if st.button("Predict Batch Intrusions", key="batch_button"):
                batch_preds = model.predict(batch_df)
                batch_preds_labels = ["Benign ✅" if p == 0 else "Attack 🚨" for p in batch_preds]
                batch_df["Prediction"] = batch_preds_labels
                st.subheader("📊 Batch Predictions")
                st.dataframe(batch_df)
                st.success(f"✅ Predicted {len(batch_df)} rows successfully!")

        # -----------------------------
        # Visual Insights
        # -----------------------------
        st.subheader("📈 Visual Insights")

        # Attack vs Benign
        if "Label" in df.columns:
            st.markdown("**Attack vs Benign distribution:**")
            attack_counts = df["Label"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(attack_counts, labels=attack_counts.index, autopct='%1.1f%%',
                    startangle=90, colors=['#4CAF50','#F44336'])
            ax1.axis('equal')
            st.pyplot(fig1)

        # Protocol distribution
        if "Protocol" in df.columns:
            st.markdown("**Protocol distribution:**")
            protocol_counts = df["Protocol"].value_counts()
            fig2, ax2 = plt.subplots()
            sns.barplot(x=protocol_counts.index, y=protocol_counts.values,
                        palette='coolwarm', ax=ax2)
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

        # Numeric feature histogram
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.markdown("**Numeric Feature Distribution:**")
        selected_col = st.selectbox("Select numeric feature", numeric_cols)
        fig3, ax3 = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax3, color='skyblue')
        st.pyplot(fig3)
