import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# Styling
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
}
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
    margin: auto;
}
.title {
    text-align: center;
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Expanded skills database
SKILLS_DB = [
    "python","java","c++","javascript","typescript",
    "html","css","react","angular","node","express",
    "mongodb","sql","mysql","postgresql",
    "machine learning","ml","deep learning","nlp",
    "tensorflow","pandas","numpy","scikit",
    "data analysis","data science",
    "docker","kubernetes","aws","azure",
    "api","rest api","backend","frontend",
    "git","github","linux",
    "testing","selenium"
]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Extract PDF text
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# Predict role
def predict_role(text):
    vec = vectorizer.transform([clean_text(text)])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

# Refine prediction
def refine_prediction(text, base_pred):
    text = text.lower()
    roles = {
        "Full Stack Developer": ["full stack","mern","node","express"],
        "Web Developer": ["html","css","javascript","react"],
        "ML Engineer": ["machine learning","ml"],
        "Data Scientist": ["data science","nlp"],
        "Python Developer": ["python","django"],
        "Java Developer": ["java","spring"]
    }
    scores = {r: sum(k in text for k in v) for r, v in roles.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else base_pred

# JD similarity
def match_score(resume, jd):
    if not jd.strip():
        return 0
    vec = vectorizer.transform([clean_text(resume), clean_text(jd)])
    return round(cosine_similarity(vec[0], vec[1])[0][0] * 100, 2)

# Improved skill extraction
def extract_skills(text):
    text = text.lower()

    # Normalize variations
    text = text.replace("machine learning", "ml")
    text = text.replace("deep learning", "dl")
    text = text.replace("data science", "data")

    found = []
    for skill in SKILLS_DB:
        if re.search(rf"\b{skill}\b", text):
            found.append(skill)

    return list(set(found))

# Improved skill matching
def skill_match(resume, jd):

    jd = jd.lower()

    # Smart JD enrichment
    if "developer" in jd:
        jd += " python java javascript html css react node sql"
    if "data" in jd:
        jd += " pandas numpy ml"
    if "cloud" in jd:
        jd += " aws docker kubernetes"
    if jd.strip() == "":
        jd += " python java html css javascript react sql"

    resume_skills = extract_skills(resume)
    jd_skills = extract_skills(jd)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    if len(jd_skills) == 0:
        return 0, [], []

    score = (len(matched) / len(jd_skills)) * 100

    return round(score, 2), matched, missing

# ATS score
def ats_score(jd, skill, text):
    wc = len(text.split())
    quality = 100 if wc > 500 else 70 if wc > 250 else 40
    return round((0.5*jd)+(0.3*skill)+(0.2*quality),2)

# UI
st.markdown('<div class="title">AI Resume Screening System</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("Paste Job Description")

# Analyze
if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Upload a resume")
        st.stop()

    resume_text = extract_text_from_pdf(uploaded_file)

    base = predict_role(resume_text)
    role = refine_prediction(resume_text, base)

    jd_score = match_score(resume_text, jd_text)
    skill_score, matched, missing = skill_match(resume_text, jd_text)
    ats = ats_score(jd_score, skill_score, resume_text)

    st.markdown("---")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Role", role)
    c2.metric("ATS Score", f"{ats}%")
    c3.metric("Skill Match", f"{skill_score}%")

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ats,
        title={'text': "ATS Score"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00c6ff"}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # Donut
    donut = go.Figure(data=[go.Pie(
        labels=["Matched", "Missing"],
        values=[skill_score, 100 - skill_score],
        hole=0.6
    )])
    donut.update_layout(template="plotly_dark")
    st.plotly_chart(donut, use_container_width=True)

    # Skills
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Matched Skills")
        if matched:
            for s in matched:
                st.markdown(f"<span style='background:#16a34a;padding:6px 10px;border-radius:8px;margin:4px;display:inline-block'>{s}</span>", unsafe_allow_html=True)
        else:
            st.info("No matching skills")

    with col5:
        st.subheader("Missing Skills")
        if missing:
            for s in missing:
                st.markdown(f"<span style='background:#dc2626;padding:6px 10px;border-radius:8px;margin:4px;display:inline-block'>{s}</span>", unsafe_allow_html=True)
        else:
            st.success("No missing skills")