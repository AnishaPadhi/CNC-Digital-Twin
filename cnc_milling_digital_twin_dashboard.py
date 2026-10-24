import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="CNC Milling Machine Digital Twin",
    page_icon="⚙️",
    layout="wide",
)

# -------------------- Styling --------------------
st.markdown("""
<style>
    .main {background-color: #f5f7fb;}
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e4e8f0;
        padding: 14px 16px; border-radius: 12px;
    }
    .small-note {color: #667085; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ CNC Milling Machine — Digital Twin")
st.caption("Software prototype for machine-condition monitoring, AI-based health classification, and maintenance support.")

# -------------------- Data generation --------------------
FEATURES = [
    "Spindle_Temperature_C",
    "Vibration_mm_s",
    "Spindle_RPM",
    "Power_kW",
    "Tool_Wear_min",
    "Feed_Rate_mm_min",
]

@st.cache_data
def make_sample_data(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    rpm = rng.normal(2400, 350, n).clip(800, 5000)
    temp = (42 + (rpm - 1800) * 0.006 + rng.normal(0, 4, n)).clip(25, 100)
    vibration = (1.2 + (rpm / 3000) * 0.7 + rng.normal(0, 0.45, n)).clip(0.1, 12)
    power = (1.8 + rpm * 0.0011 + rng.normal(0, 0.35, n)).clip(0.2, 12)
    wear = rng.uniform(0, 300, n)
    feed = rng.normal(650, 130, n).clip(100, 1200)

    # Illustrative labels for a classroom prototype; not validated CNC safety limits.
    risk_score = (
        (temp > 70).astype(int)
        + (vibration > 3.2).astype(int)
        + (power > 5.2).astype(int)
        + (wear > 220).astype(int)
        + (rpm > 3800).astype(int)
    )
    labels = np.where(risk_score >= 3, "Critical",
             np.where(risk_score >= 1, "Warning", "Normal"))
    df = pd.DataFrame({
        "Timestamp": pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min"),
        "Spindle_Temperature_C": temp.round(2),
        "Vibration_mm_s": vibration.round(2),
        "Spindle_RPM": rpm.round(0).astype(int),
        "Power_kW": power.round(2),
        "Tool_Wear_min": wear.round(1),
        "Feed_Rate_mm_min": feed.round(0).astype(int),
        "Health_Label": labels,
    })
    return df

def add_health_labels(df):
    # Only used to train a demo model when a user-uploaded CSV lacks labels.
    temp = df["Spindle_Temperature_C"]
    vib = df["Vibration_mm_s"]
    power = df["Power_kW"]
    wear = df["Tool_Wear_min"]
    rpm = df["Spindle_RPM"]
    score = (
        (temp > 70).astype(int)
        + (vib > 3.2).astype(int)
        + (power > 5.2).astype(int)
        + (wear > 220).astype(int)
        + (rpm > 3800).astype(int)
    )
    return np.where(score >= 3, "Critical", np.where(score >= 1, "Warning", "Normal"))

def recommendations(row, predicted):
    advice = []
    if row["Spindle_Temperature_C"] > 70:
        advice.append("Check spindle cooling and lubrication; verify temperature against the machine manual.")
    if row["Vibration_mm_s"] > 3.2:
        advice.append("Inspect tool condition, tool-holder tightness, workpiece clamping, and spindle bearings.")
    if row["Power_kW"] > 5.2:
        advice.append("Check cutting load, feed/speed settings, tool wear, and motor load.")
    if row["Tool_Wear_min"] > 220:
        advice.append("Inspect the cutting tool and consider a tool change based on the manufacturer's guidance.")
    if row["Spindle_RPM"] > 3800:
        advice.append("Confirm spindle speed is suitable for the tool and material.")
    if not advice:
        advice.append("No demo thresholds exceeded. Continue routine monitoring and follow the machine maintenance schedule.")
    if predicted == "Critical":
        advice.insert(0, "CRITICAL DEMO ALERT: pause and inspect the machine according to site safety procedures.")
    elif predicted == "Warning":
        advice.insert(0, "WARNING: inspect the highlighted parameters and verify against actual machine limits.")
    return advice

# -------------------- Sidebar --------------------
st.sidebar.header("Dashboard Controls")
data_mode = st.sidebar.radio("Data source", ["Built-in simulated data", "Upload CSV"])
uploaded = None
if data_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CNC readings (.csv)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Demo alert thresholds")
st.sidebar.caption("Illustrative values for demonstration only. Replace with limits from your CNC machine/tool manuals.")
temp_limit = st.sidebar.number_input("Spindle temperature (°C)", min_value=20.0, max_value=150.0, value=70.0, step=1.0)
vib_limit = st.sidebar.number_input("Vibration (mm/s)", min_value=0.1, max_value=30.0, value=3.2, step=0.1)
power_limit = st.sidebar.number_input("Power (kW)", min_value=0.1, max_value=50.0, value=5.2, step=0.1)
wear_limit = st.sidebar.number_input("Tool wear (min)", min_value=1.0, max_value=1000.0, value=220.0, step=10.0)
rpm_limit = st.sidebar.number_input("Spindle speed (RPM)", min_value=100.0, max_value=20000.0, value=3800.0, step=100.0)

# -------------------- Load / validate data --------------------
if data_mode == "Upload CSV" and uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    raw_df = make_sample_data()

required = FEATURES
missing = [c for c in required if c not in raw_df.columns]
if missing:
    st.error("CSV is missing required columns: " + ", ".join(missing))
    st.markdown("Required column names: `" + "`, `".join(required) + "`")
    st.stop()

df = raw_df.copy()
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEATURES).reset_index(drop=True)
if df.empty:
    st.error("No usable numeric records found in the required columns.")
    st.stop()

if "Timestamp" not in df.columns:
    df["Timestamp"] = np.arange(1, len(df) + 1)
if "Health_Label" not in df.columns:
    df["Health_Label"] = add_health_labels(df)

valid_labels = {"Normal", "Warning", "Critical"}
df["Health_Label"] = df["Health_Label"].astype(str).str.title()
if not set(df["Health_Label"].unique()).issubset(valid_labels):
    df["Health_Label"] = add_health_labels(df)

# Use the selected demo thresholds for the live rule status.
def rule_label(row):
    score = sum([
        row["Spindle_Temperature_C"] > temp_limit,
        row["Vibration_mm_s"] > vib_limit,
        row["Power_kW"] > power_limit,
        row["Tool_Wear_min"] > wear_limit,
        row["Spindle_RPM"] > rpm_limit,
    ])
    return "Critical" if score >= 3 else ("Warning" if score >= 1 else "Normal")

df["Rule_Status"] = df.apply(rule_label, axis=1)

# Train a demonstration Random Forest from available labels.
try:
    X = df[FEATURES]
    y = df["Health_Label"]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("At least two health classes with two or more records each are needed.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, test_predictions)
    latest = df.iloc[-1]
    prediction = str(model.predict(latest[FEATURES].to_frame().T)[0])
    probabilities = model.predict_proba(latest[FEATURES].to_frame().T)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
except Exception as exc:
    st.warning(f"Model could not be trained from this dataset: {exc}")
    st.stop()

# -------------------- Main dashboard --------------------
latest = df.iloc[-1]
rule_status = latest["Rule_Status"]
status_to_show = "Critical" if "Critical" in [prediction, rule_status] else (
    "Warning" if "Warning" in [prediction, rule_status] else "Normal"
)
color = {"Normal": "🟢", "Warning": "🟠", "Critical": "🔴"}[status_to_show]

st.markdown(f"### Current virtual machine state: {color} **{status_to_show}**")
st.caption("This is a demonstration dashboard. It does not connect to a CNC controller or provide certified machine-safety monitoring.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spindle temperature", f"{latest['Spindle_Temperature_C']:.1f} °C")
c2.metric("Vibration", f"{latest['Vibration_mm_s']:.2f} mm/s")
c3.metric("Spindle speed", f"{latest['Spindle_RPM']:.0f} RPM")
c4.metric("Power", f"{latest['Power_kW']:.2f} kW")
c5.metric("Tool wear", f"{latest['Tool_Wear_min']:.0f} min")

st.markdown("---")
left, right = st.columns([1.5, 1])

with left:
    st.subheader("Machine parameter trends")
    chart_metric = st.selectbox("Choose parameter", FEATURES, index=0)
    trend = df.tail(min(200, len(df)))
    fig = px.line(trend, x="Timestamp", y=chart_metric, title=f"{chart_metric} over recent records")
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=330)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Health distribution")
    counts = df["Health_Label"].value_counts().reindex(["Normal", "Warning", "Critical"], fill_value=0)
    health_df = counts.rename_axis("Health").reset_index(name="Records")
    fig_health = px.pie(health_df, names="Health", values="Records", hole=0.48)
    fig_health.update_layout(margin=dict(l=5, r=5, t=10, b=10), height=330, legend_title_text="")
    st.plotly_chart(fig_health, use_container_width=True)

st.markdown("---")
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("AI prediction")
    st.metric("Random Forest prediction", prediction)
    st.metric("Test-set accuracy", f"{model_accuracy * 100:.1f}%")
    st.caption("Accuracy is calculated against the labels in the selected dataset. If labels were generated by demo rules, this measures agreement with those rules—not real-world CNC failure accuracy.")
    prob_df = pd.DataFrame({"Health class": list(class_probabilities.keys()), "Model probability": list(class_probabilities.values())})
    fig_prob = px.bar(prob_df, x="Health class", y="Model probability", range_y=[0, 1])
    fig_prob.update_layout(height=250, margin=dict(l=5, r=5, t=10, b=5))
    st.plotly_chart(fig_prob, use_container_width=True)

with col_b:
    st.subheader("Maintenance support")
    st.write(f"**Rule-based status:** {rule_status}")
    advice = recommendations(latest, status_to_show)
    for item in advice:
        st.write(f"- {item}")

st.markdown("---")
st.subheader("Recent machine records")
display_cols = ["Timestamp"] + FEATURES + ["Health_Label", "Rule_Status"]
st.dataframe(df[display_cols].tail(20), use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download processed machine data as CSV",
    data=csv,
    file_name="cnc_milling_machine_dashboard_data.csv",
    mime="text/csv",
)

with st.expander("Model evaluation details"):
    st.write("Confusion matrix (test set)")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, test_predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
    st.dataframe(cm_df)
    st.text("Classification report")
    st.code(classification_report(y_test, test_predictions, zero_division=0))

st.markdown("""
<div class="small-note">
<strong>Academic prototype note:</strong> The built-in dataset and thresholds are synthetic examples for demonstrating the workflow.
They must not be used as real CNC operating limits. For a real machine, use verified sensor data and manufacturer/site-approved limits.
</div>
""", unsafe_allow_html=True)
