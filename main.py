
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="수학 성취도 예측기", page_icon="📚")

st.title("📚 수학 성취도 예측기")
st.write("학생의 학습 습관을 바탕으로 수학 성취도 수준을 예측합니다.")
st.info("이 결과는 참고용 예측이며 실제 성적을 보장하지 않습니다.")

uploaded_file = st.file_uploader(
    "student-mat.csv 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("먼저 student-mat.csv 파일을 업로드해 주세요.")
    st.stop()

df = pd.read_csv(uploaded_file, sep=";")

required = [
    "studytime", "absences", "failures",
    "schoolsup", "famsup", "G3"
]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"필요한 데이터 열이 없습니다: {missing}")
    st.stop()

data = df[required].copy()

data["성취도"] = data["G3"].apply(
    lambda x: "높음" if x >= 10 else "낮음"
)

st.subheader("1. 데이터 확인")
st.write("전체 학생 수:", len(data))
st.dataframe(data.head())

st.subheader("2. 데이터 시각화")

fig1 = px.histogram(
    data,
    x="studytime",
    color="성취도",
    barmode="group",
    title="학습 시간과 수학 성취도"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(
    data,
    x="성취도",
    y="absences",
    color="성취도",
    title="성취도별 결석 횟수"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 예측 모델 학습")

X = data[
    ["studytime", "absences", "failures",
     "schoolsup", "famsup"]
]
y = data["성취도"]

numeric_features = [
    "studytime", "absences", "failures"
]
categorical_features = ["schoolsup", "famsup"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

st.write(
    "정확도:",
    f"{accuracy_score(y_test, pred):.2%}"
)

st.write(
    "정밀도:",
    f"{precision_score(y_test, pred, pos_label='높음', zero_division=0):.2%}"
)

st.write(
    "재현율:",
    f"{recall_score(y_test, pred, pos_label='높음', zero_division=0):.2%}"
)

cm = confusion_matrix(
    y_test, pred,
    labels=["낮음", "높음"]
)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["예측 낮음", "예측 높음"],
    y=["실제 낮음", "실제 높음"],
    title="혼동 행렬"
)
st.plotly_chart(fig_cm)

st.subheader("4. 나의 성취도 예측하기")

with st.form("prediction_form"):
    studytime = st.selectbox(
        "주간 학습 시간 수준",
        [1, 2, 3, 4]
    )

    absences = st.number_input(
        "결석 횟수", min_value=0, value=2
    )

    failures = st.number_input(
        "과거 낙제 횟수", min_value=0, value=0
    )

    schoolsup = st.selectbox(
        "학교의 추가 교육 지원",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "가족의 교육 지원",
        ["yes", "no"]
    )

    submitted = st.form_submit_button("성취도 예측하기")

if submitted:
    new_student = pd.DataFrame([{
        "studytime": studytime,
        "absences": absences,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup
    }])

    result = model.predict(new_student)[0]
    probabilities = model.predict_proba(new_student)[0]
    classes = list(model.named_steps["classifier"].classes_)

    high_probability = probabilities[classes.index("높음")]

    st.success(f"예측 결과: {result}")
    st.write(
        f"높은 성취도 그룹일 확률: {high_probability:.1%}"
    )
    st.caption("예측 결과는 참고용이며 실제 성적과 다를 수 있습니다.")
