from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      f1_score,
      confusion_matrix,
      classification_report
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
      page_title="Student Mental Health",
      page_icon="🧠",
      layout="wide",
      initial_sidebar_state="expanded"
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
REFERENCE_DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"

TARGET = "Mental_Health_Risk"
ID_COLUMN = "Student_ID"

# Variables présentes dans le dataset
DATASET_COLUMNS = [
      "Student_ID",
      "Age",
      "Gender",
      "Year_of_Study",
      "Sleep_Hours",
      "Study_Hours_Per_Week",
      "GPA",
      "Social_Media_Hours",
      "Screen_Time_Hours",
      "Physical_Activity",
      "Financial_Stress",
      "Family_Support",
      "Attendance_Rate",
      "Part_Time_Job",
      "Diet_Quality",
      "Previous_Mental_Health_History",
      "Mental_Health_Risk"
]


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0B1020;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151D35 0%, #101629 100%);
    border-right: 1px solid #26314D;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 4px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 28px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 18px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #19233D, #141C32);
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #F8FAFC;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-icon {
    font-size: 23px;
}

.info-card {
    background: #151F36;
    border: 1px solid #2A3858;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #192B45, #17243A);
    border: 1px solid #35496B;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7667F5, #4F9CF9);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 22px;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #6554E8, #3688ED);
    color: white;
}

.stTextInput input, .stNumberInput input {
    background-color: #17223A;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #17223A;
    border-color: #344563;
}

hr {
    border-color: #293653;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CHARGEMENT DU MODELE ET DU DATASET
# =========================================================

@st.cache_resource
def load_model():
      return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
      if REFERENCE_DATA_PATH.exists():
            return pd.read_csv(REFERENCE_DATA_PATH)
      return None


try:
      model = load_model()
except Exception as e:
      st.error(f"Impossible de charger le modèle : {e}")
      st.stop()


reference_df = load_reference_data()

# Le modèle reste la source de vérité pour les variables attendues.
if hasattr(model, "feature_names_in_"):
      FEATURES = list(model.feature_names_in_)
else:
      st.error(
            "Le modèle ne possède pas feature_names_in_. "
            "Vérifie les variables utilisées lors de son entraînement."
      )
      st.stop()

if TARGET in FEATURES or ID_COLUMN in FEATURES:
      st.error(
            "Vérifie le pipeline : Mental_Health_Risk et Student_ID "
            "ne doivent pas être des variables d'entrée du modèle."
      )
      st.stop()

if reference_df is not None:
      missing_columns = [
            col for col in FEATURES
            if col not in reference_df.columns
      ]

      if missing_columns:
            st.error(
                  f"Colonnes manquantes dans le dataset de référence : {missing_columns}"
            )
            st.stop()


# =========================================================
# SIDEBAR / NAVIGATION
# =========================================================

with st.sidebar:
      st.markdown("## 🧠 Student Mental Health")
      st.caption("Student Mental Health Analytics")

      st.divider()

      page = st.radio(
            "NAVIGATION",
            [
                  "🏠  Overview",
                  "🔎  Student Prediction",
                  "📁  Batch Prediction"
            ],
            label_visibility="collapsed"
      )

      st.divider()

      st.markdown("### Model information")
      st.markdown("**Algorithm**")
      st.caption(type(model).__name__)

      st.markdown("**Input features**")
      st.caption(f"{len(FEATURES)} variables")

      st.divider()
      st.caption("MindScope · Student Mental Health Project")


# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================

def metric_card(label, value, icon):
      st.markdown(f"""
            <div class="metric-card">
                  <div class="metric-icon">{icon}</div>
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
            </div>
      """, unsafe_allow_html=True)


def prepare_input(data):
      """Respecte l'ordre exact des colonnes attendues par le modèle."""
      return pd.DataFrame([data], columns=FEATURES)


def get_student_input():
      """Construit les champs à partir des données réelles."""
      if reference_df is None:
            st.error(
                  "Dataset de référence introuvable. "
                  "Vérifie REFERENCE_DATA_PATH dans app.py."
            )
            return None

      values = {}

      sections = [
            ("👤 Personal information", ["Age", "Gender", "Year_of_Study"]),
            ("📚 Academic lifestyle", [
                  "Study_Hours_Per_Week",
                  "GPA",
                  "Attendance_Rate",
                  "Part_Time_Job"
            ]),
            ("🌙 Daily habits", [
                  "Sleep_Hours",
                  "Social_Media_Hours",
                  "Screen_Time_Hours",
                  "Physical_Activity",
                  "Diet_Quality"
            ]),
            ("💜 Wellbeing indicators", [
                  "Financial_Stress",
                  "Family_Support",
                  "Previous_Mental_Health_History"
            ])
      ]

      for section_title, section_features in sections:
            active_features = [
                  feature for feature in section_features
                  if feature in FEATURES
            ]

            if not active_features:
                  continue

            st.markdown(f"### {section_title}")

            columns = st.columns(2)

            for i, feature in enumerate(active_features):
                  with columns[i % 2]:
                        series = reference_df[feature].dropna()
                        unique_values = sorted(series.unique().tolist())

                        if (
                              pd.api.types.is_object_dtype(series)
                              or pd.api.types.is_bool_dtype(series)
                              or len(unique_values) <= 10
                        ):
                              if not unique_values:
                                    st.warning(f"Aucune valeur disponible pour {feature}")
                                    continue

                              values[feature] = st.selectbox(
                                    feature.replace("_", " "),
                                    unique_values,
                                    key=f"input_{feature}"
                              )

                        elif pd.api.types.is_numeric_dtype(series):
                              min_value = float(series.min())
                              max_value = float(series.max())
                              median_value = float(series.median())

                              if pd.api.types.is_integer_dtype(series):
                                    values[feature] = st.number_input(
                                          feature.replace("_", " "),
                                          min_value=int(min_value),
                                          max_value=int(max_value),
                                          value=int(median_value),
                                          step=1,
                                          key=f"input_{feature}"
                                    )
                              else:
                                    values[feature] = st.number_input(
                                          feature.replace("_", " "),
                                          min_value=min_value,
                                          max_value=max_value,
                                          value=median_value,
                                          step=0.1,
                                          key=f"input_{feature}"
                                    )

      return values


# =========================================================
# PAGE 1 : OVERVIEW
# =========================================================

if page == "🏠  Overview":

      st.markdown(
            '<div class="main-title">Student Mental Health Dashboard</div>',
            unsafe_allow_html=True
      )
      st.markdown(
            '<div class="subtitle">Explore student wellbeing, '
            'understand patterns and monitor model performance.</div>',
            unsafe_allow_html=True
      )

      if reference_df is not None:
            total_students = len(reference_df)
            total_features = len(FEATURES)

            if TARGET in reference_df.columns:
                  risk_count = (
                        reference_df[TARGET].astype(str).str.lower()
                        .isin(["1", "high", "yes", "true", "at risk", "risk"])
                        .sum()
                  )
                  risk_rate = risk_count / total_students * 100 if total_students else 0
            else:
                  risk_rate = 0

            c1, c2, c3 = st.columns(3)

            with c1:
                  metric_card("Total students", f"{total_students:,}", "👥")
            with c2:
                  metric_card("Model input features", total_features, "🧩")
            with c3:
                  metric_card("Dataset risk rate", f"{risk_rate:.1f}%", "💜")

            st.markdown(
                  '<div class="section-title">Student data exploration</div>',
                  unsafe_allow_html=True
            )

            if TARGET in reference_df.columns:
                  left, right = st.columns(2)

                  with left:
                        distribution = (
                              reference_df[TARGET]
                              .astype(str)
                              .value_counts()
                              .reset_index()
                        )
                        distribution.columns = ["Risk", "Students"]

                        fig = px.pie(
                              distribution,
                              names="Risk",
                              values="Students",
                              hole=0.65,
                              title="Mental health risk distribution"
                        )
                        fig.update_layout(
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E2E8F0"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                  with right:
                        numeric_features = [
                              col for col in FEATURES
                              if col in reference_df.columns
                              and pd.api.types.is_numeric_dtype(reference_df[col])
                        ]

                        if numeric_features:
                              selected_feature = st.selectbox(
                                    "Explore a numerical variable",
                                    numeric_features
                              )

                              fig = px.histogram(
                                    reference_df,
                                    x=selected_feature,
                                    color=TARGET,
                                    barmode="overlay",
                                    opacity=0.75,
                                    title=f"Distribution of {selected_feature}"
                              )
                              fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#E2E8F0"
                              )
                              st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                  '<div class="section-title">Model performance</div>',
                  unsafe_allow_html=True
            )

            st.info(
                  "Importe un fichier de test contenant les variables d'entrée "
                  "et la colonne Mental_Health_Risk pour calculer les métriques."
            )

            test_file = st.file_uploader(
                  "Upload your labelled test dataset",
                  type=["csv"],
                  key="dashboard_test"
            )

            if test_file is not None:
                  test_df = pd.read_csv(test_file)

                  required = FEATURES + [TARGET]
                  missing = [col for col in required if col not in test_df.columns]

                  if missing:
                        st.error(f"Colonnes manquantes : {missing}")
                  else:
                        X_test = test_df[FEATURES]
                        y_test = test_df[TARGET]
                        y_pred = model.predict(X_test)

                        m1, m2, m3, m4 = st.columns(4)

                        with m1:
                              metric_card(
                                    "Accuracy",
                                    f"{accuracy_score(y_test, y_pred):.2%}",
                                    "🎯"
                              )
                        with m2:
                              metric_card(
                                    "Precision",
                                    f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.2%}",
                                    "🔍"
                              )
                        with m3:
                              metric_card(
                                    "Recall",
                                    f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.2%}",
                                    "📡"
                              )
                        with m4:
                              metric_card(
                                    "F1-score",
                                    f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.2%}",
                                    "⚖️"
                              )

                        labels = sorted(pd.Series(y_test).unique().tolist())
                        cm = confusion_matrix(y_test, y_pred, labels=labels)

                        fig = px.imshow(
                              cm,
                              x=[str(x) for x in labels],
                              y=[str(x) for x in labels],
                              text_auto=True,
                              color_continuous_scale="Purples",
                              title="Confusion matrix"
                        )
                        fig.update_layout(
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#E2E8F0"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Detailed classification report"):
                              report = classification_report(
                                    y_test,
                                    y_pred,
                                    output_dict=True,
                                    zero_division=0
                              )
                              st.dataframe(
                                    pd.DataFrame(report).transpose(),
                                    use_container_width=True
                              )

      else:
            st.warning(
                  "Dataset de référence introuvable. "
                  "Vérifie le chemin configuré dans REFERENCE_DATA_PATH."
            )


# =========================================================
# PAGE 2 : INDIVIDUAL PREDICTION
# =========================================================

elif page == "🔎  Student Prediction":

      st.markdown(
            '<div class="main-title">Individual Student Assessment</div>',
            unsafe_allow_html=True
      )
      st.markdown(
            '<div class="subtitle">Enter student information to obtain '
            'a model prediction.</div>',
            unsafe_allow_html=True
      )

      with st.form("student_prediction_form"):

            st.markdown(
                  '<div class="info-card">Complete the fields below using '
                  'the available student information.</div>',
                  unsafe_allow_html=True
            )

            student_data = get_student_input()

            submitted = st.form_submit_button(
                  "✨  Analyze student profile"
            )

      if submitted and student_data is not None:

            missing = [col for col in FEATURES if col not in student_data]

            if missing:
                  st.error(f"Certaines variables n'ont pas été renseignées : {missing}")
            else:
                  input_df = prepare_input(student_data)

                  prediction = model.predict(input_df)[0]

                  st.markdown(
                        '<div class="section-title">Prediction result</div>',
                        unsafe_allow_html=True
                  )

                  st.markdown(
                        f"""
                        <div class="result-card">
                              <h3 style="color:#A5B4FC;">Student assessment</h3>
                              <h1 style="color:white;">{prediction}</h1>
                              <p style="color:#94A3B8;">
                                    Predicted mental health risk category
                              </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                  )

                  if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]
                        classes = model.classes_

                        probability_df = pd.DataFrame({
                              "Category": classes,
                              "Probability": probabilities
                        })

                        st.markdown("### Prediction probabilities")

                        fig = px.bar(
                              probability_df,
                              x="Category",
                              y="Probability",
                              text=probability_df["Probability"].map(
                                    lambda x: f"{x:.1%}"
                              ),
                              title="Model confidence by category"
                        )
                        fig.update_layout(
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E2E8F0",
                              yaxis_tickformat=".0%"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                  st.caption(
                        "Cette prédiction est une estimation produite par un modèle "
                        "de machine learning. Elle ne constitue pas un diagnostic médical."
                  )


# =========================================================
# PAGE 3 : BATCH PREDICTION
# =========================================================

elif page == "📁  Batch Prediction":

      st.markdown(
            '<div class="main-title">Batch Student Prediction</div>',
            unsafe_allow_html=True
      )
      st.markdown(
            '<div class="subtitle">Analyze multiple student profiles '
            'with one CSV upload.</div>',
            unsafe_allow_html=True
      )

      st.markdown("""
    <div class="info-card">
        <h4 style="color:#A5B4FC;">📌 How it works</h4>
        <p style="color:#CBD5E1;">
            Upload a CSV file containing all the model input variables.
            The application will generate a prediction for each student.
        </p>
    </div>
      """, unsafe_allow_html=True)

      uploaded_file = st.file_uploader(
            "Choose a student CSV file",
            type=["csv"],
            key="batch_file"
      )

      if uploaded_file is not None:

            input_df = pd.read_csv(uploaded_file)

            missing_features = [
                  col for col in FEATURES
                  if col not in input_df.columns
            ]

            if missing_features:
                  st.error(
                        f"Les colonnes suivantes sont manquantes : {missing_features}"
                  )
            else:
                  st.markdown("### Dataset preview")
                  st.dataframe(input_df.head(), use_container_width=True)

                  if st.button("🚀 Generate predictions"):

                        X_batch = input_df[FEATURES]
                        predictions = model.predict(X_batch)

                        results_df = input_df.copy()
                        results_df["Predicted_Mental_Health_Risk"] = predictions

                        if hasattr(model, "predict_proba"):
                              probabilities = model.predict_proba(X_batch)
                              results_df["Prediction_Confidence"] = (
                                    probabilities.max(axis=1)
                              )

                        st.success(
                              f"{len(results_df)} student profiles analyzed successfully."
                        )

                        st.markdown("### Prediction results")
                        st.dataframe(
                              results_df,
                              use_container_width=True
                        )

                        st.download_button(
                              label="⬇️ Download predictions",
                              data=results_df.to_csv(index=False).encode("utf-8"),
                              file_name="student_mental_health_predictions.csv",
                              mime="text/csv"
                        )

                        prediction_counts = (
                              pd.Series(predictions)
                              .astype(str)
                              .value_counts()
                              .reset_index()
                        )
                        prediction_counts.columns = ["Risk category", "Students"]

                        fig = px.bar(
                              prediction_counts,
                              x="Risk category",
                              y="Students",
                              color="Risk category",
                              title="Predicted risk distribution"
                        )
                        fig.update_layout(
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E2E8F0",
                              showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
