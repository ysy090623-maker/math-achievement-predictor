
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="수학 성취도 예측기",
    page_icon="📚",
    layout="wide"
)

st.title("📚 수학 성취도 예측기")
st.write("학생의 학습 습관을 바탕으로 수학 성취도 수준을 예측합니다.")
st.info("예측 결과는 참고용이며 실제 성적을 보장하지 않습니다.")

# GitHub 저장소에 있는 CSV 파일 자동 불러오기
try:
    df = pd.read_csv("student-mat.csv", sep=";")
except Exception as e:
    st.error("데이터 파일을 불러오지 못했습니다.")
    st.write(e)
    st.stop()

# 필요한 데이터 확인
required = [
    "studytime",
    "absences",
    "failures",
    "schoolsup",
    "famsup",
    "G3"
]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"데이터에 필요한 열이 없습니다: {missing}")
    st.stop()

data = df[required].copy()

# 성취도 분류
data["성취도"] = data["G3"].apply(
    lambda x: "높음" if x >= 10 else "낮음"
)

st.subheader("1. 데이터 확인")

col1, col2 = st.columns(2)

col1.metric("전체 학생 수", f"{len(data)}명")
col2.metric("사용 변수 수", "5개")

st.dataframe(data.head(), use_container_width=True)

with st.expander("변수 설명 보기"):
    st.write("studytime: 주간 학습 시간 수준")
    st.write("absences: 결석 횟수")
    st.write("failures: 과거 낙제 횟수")
    st.write("schoolsup: 학교의 교육 지원 여부")
    st.write("famsup: 가족의 교육 지원 여부")
    st.write("G3: 최종 수학 성적")

st.subheader("2. 데이터 시각화")

fig1 = px.histogram(
    data,
    x="studytime",
    color="성취도",
    barmode="group",
    title="주간 학습 시간과 수학 성취도"
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

st.subheader("3. 예측 모델 학습 및 평가")

X = data[
    ["studytime", "absences", "failures", "schoolsup", "famsup"]
]

y = data["성취도"]

if y.value_counts().min() < 2:
    st.error("모델 학습을 위한 데이터가 부족합니다.")
    st.stop()

numeric_features = [
    "studytime",
    "absences",
    "failures"
]

categorical_features = [
    "schoolsup",
    "famsup"
]

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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

col1, col2, col3 = st.columns(3)

col1.metric(
    "정확도",
    f"{accuracy_score(y_test, pred):.2%}"
)

col2.metric(
    "정밀도",
    f"{precision_score(y_test, pred, pos_label='높음', zero_division=0):.2%}"
)

col3.metric(
    "재현율",
    f"{recall_score(y_test, pred, pos_label='높음', zero_division=0):.2%}"
)

cm = confusion_matrix(
    y_test,
    pred,
    labels=["낮음", "높음"]
)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["예측 낮음", "예측 높음"],
    y=["실제 낮음", "실제 높음"],
    title="혼동 행렬"
)

st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("4. 나의 수학 성취도 예측하기")

with st.form("prediction_form"):

    studytime = st.selectbox(
        "주간 학습 시간 수준",
        [1, 2, 3, 4]
    )

    absences = st.number_input(
        "결석 횟수",
        min_value=0,
        value=2
    )

    failures = st.number_input(
        "과거 낙제 횟수",
        min_value=0,
        value=0
    )

    schoolsup = st.selectbox(
        "학교의 교육 지원 여부",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "가족의 교육 지원 여부",
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
        f"높은 성취도 그룹으로 예측될 확률: {high_probability:.1%}"
    )

    st.caption(
        "이 결과는 참고용 예측이며 실제 성적과 다를 수 있습니다."
    )
