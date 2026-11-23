import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import kruskal, chi2_contingency, fisher_exact, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

# ============================================================
# 1. 페이지 설정 및 디자인
# ============================================================
st.set_page_config(
    page_title="MATH INSIGHT | 수학 성취도 분석",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background:#F4F7FB;}
.block-container {max-width:1450px;padding-top:1.7rem;padding-bottom:3rem;}
div[data-testid="stMetric"] {
    background:#fff;
    border:1px solid #E1E8F0;
    border-radius:13px;
    padding:15px 17px;
}
div[data-testid="stTabs"] button {font-weight:650;}
.hero {
    background:linear-gradient(115deg,#132B4A 0%,#20577D 62%,#347D9D 100%);
    padding:28px 30px;
    border-radius:17px;
    color:white;
    margin-bottom:16px;
}
.hero h2 {color:white;margin:0 0 8px 0;font-size:1.7rem;}
.hero p {color:#E7F0F8;margin:0;font-size:1rem;}
.muted {color:#64748B;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.title("📘 MATH INSIGHT")

st.markdown("""
<div class="hero">
<h2>학생의 학습 습관과 수학 성취도 분석</h2>
<p>데이터 탐색 · 통계 분석 · 선형 회귀 모델 평가 · 변수 중요도 · 점수 예측</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "UCI Student Performance 데이터셋 · 수학 과목(student-mat.csv) · "
    "선형 회귀 기반 최종 성적 G3 예측"
)

# ============================================================
# 2. 데이터 불러오기 및 전처리
# ============================================================
with st.sidebar:
    st.header("분석 설정")
    st.markdown("**예측에 사용하는 변수**")
    st.write("• 주간 학습 시간(studytime)")
    st.write("• 결석 횟수(absences)")
    st.write("• 과거 낙제 횟수(failures)")
    st.write("• 학교 교육 지원(schoolsup)")
    st.write("• 가족 교육 지원(famsup)")
    st.divider()
    st.caption("G1·G2는 최종 성적과 밀접하므로 예측 변수에서 제외합니다.")
    st.caption("성취도 단계는 예측 점수를 해석하기 위한 프로젝트 내부 기준입니다.")

uploaded = st.file_uploader(
    "student-mat.csv 파일 업로드 (선택)",
    type=["csv"]
)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), sep=";")


@st.cache_data(show_spinner=False)
def read_local_csv(path, modified):
    return pd.read_csv(path, sep=";")


try:
    if uploaded is not None:
        raw = read_uploaded_csv(uploaded.getvalue())
        source_label = "직접 업로드한 CSV"

    elif os.path.exists("student-mat.csv"):
        raw = read_local_csv(
            "student-mat.csv",
            os.path.getmtime("student-mat.csv")
        )
        source_label = "GitHub 저장소의 student-mat.csv"

    else:
        st.info(
            "분석을 시작하려면 UCI Student Performance의 "
            "student-mat.csv 파일을 업로드하거나 "
            "main.py와 같은 저장소 위치에 넣어 주세요."
        )
        st.stop()

except Exception as exc:
    st.error(
        "CSV 파일을 읽지 못했습니다. "
        "UCI 원본 파일은 세미콜론(;)으로 구분되어 있습니다."
    )
    st.exception(exc)
    st.stop()


required = [
    "studytime",
    "absences",
    "failures",
    "schoolsup",
    "famsup",
    "G3"
]

missing = [c for c in required if c not in raw.columns]

if missing:
    st.error(f"CSV에 필요한 열이 없습니다: {', '.join(missing)}")
    st.stop()

data = raw[required].copy()

for c in ["studytime", "absences", "failures", "G3"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

original_rows = len(data)

data = data.dropna(subset=["G3"]).copy()
data = data[data["G3"].between(0, 20)].copy()

if len(data) < 30:
    st.error("회귀 모델을 학습하기에는 유효한 데이터가 너무 적습니다.")
    st.stop()


# 세 단계는 회귀 예측 점수를 이해하기 위한 보조 해석 기준
STAGE_BINS = [-0.01, 9.99, 14.99, 20.01]
STAGE_LABELS = ["낮음", "보통", "높음"]

data["성취도 단계"] = pd.cut(
    data["G3"],
    bins=STAGE_BINS,
    labels=STAGE_LABELS
)

features = [
    "studytime",
    "absences",
    "failures",
    "schoolsup",
    "famsup"
]

numeric_features = [
    "studytime",
    "absences",
    "failures"
]

categorical_features = [
    "schoolsup",
    "famsup"
]

X = data[features]
y = data["G3"].astype(float)


# ============================================================
# 3. 회귀 모델 및 기준 모델 구성
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

regression_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

baseline_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", DummyRegressor(strategy="mean"))
])

with st.spinner("선형 회귀 모델을 학습하고 평가하고 있습니다..."):

    regression_model.fit(X_train, y_train)
    baseline_model.fit(X_train, y_train)

    pred_test = np.clip(
        regression_model.predict(X_test),
        0,
        20
    )

    baseline_pred = np.clip(
        baseline_model.predict(X_test),
        0,
        20
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_mae = -cross_val_score(
        regression_model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )


metrics = {
    "MAE": mean_absolute_error(y_test, pred_test),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_test)),
    "R²": r2_score(y_test, pred_test)
}

baseline_metrics = {
    "MAE": mean_absolute_error(y_test, baseline_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
    "R²": r2_score(y_test, baseline_pred)
}


# ============================================================
# 4. 상단 요약 대시보드
# ============================================================
st.subheader("프로젝트 분석 현황")

a, b, c, d = st.columns(4)

a.metric("분석 학생 수", f"{len(data):,}명")
b.metric("예측 변수 수", "5개")
c.metric("평가용 데이터", f"{len(y_test)}명")
d.metric("모델", "선형 회귀")

st.markdown(
    '<p class="muted">전체 데이터의 80%를 학습, 20%를 평가에 사용했습니다. '
    '학습 데이터 안에서 5겹 교차검증도 수행합니다.</p>',
    unsafe_allow_html=True
)


# ============================================================
# 5. 탭 구성
# ============================================================
tab_data, tab_stats, tab_model, tab_importance, tab_predict = st.tabs([
    "① 데이터 탐색",
    "② 통계 분석",
    "③ 회귀 모델 평가",
    "④ 변수 중요도",
    "⑤ 점수 예측"
])


# ============================================================
# 탭 1: 데이터 탐색
# ============================================================
with tab_data:

    st.subheader("데이터 구성")

    left, right = st.columns([1.1, 0.9])

    with left:

        variable_table = pd.DataFrame({
            "변수": [
                "studytime",
                "absences",
                "failures",
                "schoolsup",
                "famsup",
                "G3"
            ],
            "설명": [
                "주간 학습 시간 수준(1~4)",
                "결석 횟수",
                "과거 낙제 횟수",
                "학교의 추가 교육 지원 여부",
                "가족의 교육 지원 여부",
                "최종 수학 성적(0~20점)"
            ],
            "역할": ["예측 변수"] * 5 + ["예측 대상"]
        })

        st.dataframe(
            variable_table,
            hide_index=True,
            use_container_width=True
        )

    with right:

        stage_counts = (
            data["성취도 단계"]
            .value_counts()
            .reindex(STAGE_LABELS, fill_value=0)
            .rename_axis("성취도 단계")
            .reset_index(name="학생 수")
        )

        stage_fig = px.pie(
            stage_counts,
            names="성취도 단계",
            values="학생 수",
            hole=0.58,
            title="실제 G3 기준 성취도 단계 분포",
            color="성취도 단계",
            color_discrete_map={
                "낮음": "#E6A04B",
                "보통": "#E8C45B",
                "높음": "#2878B5"
            }
        )

        st.plotly_chart(
            stage_fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("학습 습관과 최종 성적의 관계")

    l, r = st.columns(2)

    with l:

        plot_data = data.copy()

        plot_data["학습 시간"] = plot_data["studytime"].map({
            1: "2시간 미만",
            2: "2~5시간",
            3: "5~10시간",
            4: "10시간 초과"
        })

        fig = px.box(
            plot_data,
            x="학습 시간",
            y="G3",
            color="학습 시간",
            title="주간 학습 시간별 최종 수학 성적 분포",
            category_orders={
                "학습 시간": [
                    "2시간 미만",
                    "2~5시간",
                    "5~10시간",
                    "10시간 초과"
                ]
            }
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="주간 학습 시간",
            yaxis_title="최종 성적 G3 (20점 만점)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with r:

        fig = px.scatter(
            data,
            x="absences",
            y="G3",
            color="성취도 단계",
            title="결석 횟수와 최종 수학 성적",
            color_discrete_map={
                "낮음": "#E6A04B",
                "보통": "#E8C45B",
                "높음": "#2878B5"
            },
            opacity=0.7
        )

        fig.update_layout(
            xaxis_title="결석 횟수",
            yaxis_title="최종 성적 G3 (20점 만점)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("성취도 단계별 평균 학습 습관")

    summary = (
        data.groupby("성취도 단계", observed=False)[
            numeric_features + ["G3"]
        ]
        .mean()
        .round(2)
        .rename(columns={
            "studytime": "평균 학습 시간 수준",
            "absences": "평균 결석 횟수",
            "failures": "평균 과거 낙제 횟수",
            "G3": "평균 G3 점수"
        })
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.caption(
        f"데이터 출처: {source_label}. "
        f"원본 {original_rows}행 중 G3 결측 또는 범위 밖인 "
        f"{original_rows-len(data)}행은 제외했습니다."
    )

    with st.expander("원본 데이터 일부 확인"):
        st.dataframe(
            raw.head(20),
            use_container_width=True
        )


# ============================================================
# 탭 2: 통계 분석
# ============================================================
with tab_stats:

    st.subheader("학습 습관과 성취도 단계의 관계")

    st.write(
        "세 단계의 수치형 변수 차이는 크루스칼-왈리스 검정으로, "
        "범주형 변수의 분포 차이는 카이제곱 독립성 검정으로 살펴봅니다."
    )

    rows = []

    for col in numeric_features:

        groups = [
            data.loc[data["성취도 단계"] == stage, col].dropna()
            for stage in STAGE_LABELS
        ]

        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:

            stat, p = kruskal(*groups)

            rows.append({
                "변수": col,
                "검정 방법": "크루스칼-왈리스",
                "검정 통계량": stat,
                "p값": p
            })

    for col in categorical_features:

        table = pd.crosstab(
            data[col],
            data["성취도 단계"]
        )

        if table.shape[0] >= 2 and table.shape[1] >= 2:

            chi2, p, _, expected = chi2_contingency(table)

            if table.shape == (2, 2) and (expected < 5).any():

                _, p = fisher_exact(table.to_numpy())
                method = "Fisher 정확 검정"

            else:
                method = "카이제곱 독립성"

            rows.append({
                "변수": col,
                "검정 방법": method,
                "검정 통계량": chi2,
                "p값": p
            })

    stats_df = pd.DataFrame(rows)

    if not stats_df.empty:

        stats_df["검정 결과"] = np.where(
            stats_df["p값"] < 0.05,
            "0.05 미만",
            "0.05 이상"
        )

        display = stats_df.copy()

        display["검정 통계량"] = display[
            "검정 통계량"
        ].map(lambda v: f"{v:.3f}")

        display["p값"] = display[
            "p값"
        ].map(lambda v: f"{v:.4f}")

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True
        )

    st.subheader("수치형 학습 습관과 G3의 상관관계")

    corr_rows = []

    for col in numeric_features:

        valid = data[[col, "G3"]].dropna()

        rho, p = (
            spearmanr(valid[col], valid["G3"])
            if len(valid) >= 3 and valid[col].nunique() > 1
            else (np.nan, np.nan)
        )

        corr_rows.append({
            "변수": col,
            "Spearman 상관계수": rho,
            "p값": p
        })

    corr = pd.DataFrame(corr_rows)

    show = corr.copy()

    show["Spearman 상관계수"] = show[
        "Spearman 상관계수"
    ].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")

    show["p값"] = show[
        "p값"
    ].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")

    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True
    )

    fig = px.bar(
        corr,
        x="변수",
        y="Spearman 상관계수",
        title="학습 습관 변수와 G3의 Spearman 상관계수",
        range_y=[-1, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "통계적 관계는 인과관계를 증명하지 않습니다. "
        "표본 수와 포르투갈 학교의 데이터 수집 환경도 고려해야 합니다."
    )


# ============================================================
# 탭 3: 회귀 모델 평가
# ============================================================
with tab_model:

    st.subheader("선형 회귀 모델 성능")

    st.write(
        "선형 회귀는 입력한 학습 습관과 최종 성적 G3 사이의 "
        "수치적 관계를 학습해 0~20점 범위의 점수를 예측합니다."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{metrics['MAE']:.2f}점",
        help="실제 점수와 예측 점수 차이의 절댓값 평균. 낮을수록 좋습니다."
    )

    m2.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}점",
        help="큰 예측 오차에 더 큰 영향을 받는 오차 지표. 낮을수록 좋습니다."
    )

    m3.metric(
        "R²",
        f"{metrics['R²']:.3f}",
        help="모델이 점수 변동을 설명하는 정도를 나타냅니다. 음수일 수도 있습니다."
    )

    st.caption(
        f"학습 데이터 5겹 교차검증 MAE 평균: "
        f"{cv_mae.mean():.2f}점 "
        f"(표준편차 {cv_mae.std():.2f})"
    )

    compare = pd.DataFrame({
        "평가 지표": ["MAE", "RMSE"],
        "선형 회귀": [
            metrics["MAE"],
            metrics["RMSE"]
        ],
        "평균값 기준 모델": [
            baseline_metrics["MAE"],
            baseline_metrics["RMSE"]
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="선형 회귀",
        x=compare["평가 지표"],
        y=compare["선형 회귀"]
    ))

    fig.add_trace(go.Bar(
        name="평균값 기준 모델",
        x=compare["평가 지표"],
        y=compare["평균값 기준 모델"]
    ))

    fig.update_layout(
        title="선형 회귀와 단순 기준 모델의 오차 비교 (낮을수록 좋음)",
        barmode="group",
        yaxis_title="오차 (점)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    actual_pred = pd.DataFrame({
        "실제 G3": y_test.to_numpy(),
        "예측 G3": pred_test
    })

    fig = px.scatter(
        actual_pred,
        x="실제 G3",
        y="예측 G3",
        title="실제 점수와 예측 점수 비교",
        opacity=0.75
    )

    fig.add_trace(go.Scatter(
        x=[0, 20],
        y=[0, 20],
        mode="lines",
        name="완벽한 예측 기준선",
        line=dict(dash="dash", color="gray")
    ))

    fig.update_xaxes(range=[0, 20])
    fig.update_yaxes(range=[0, 20])

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "평가 데이터는 모델 학습에 사용하지 않은 20% 자료입니다. "
        "MAE와 RMSE는 낮을수록 좋고, R²는 데이터에 따라 음수가 될 수도 있습니다."
    )

    export = pd.DataFrame({
        "실제 G3": y_test.to_numpy(),
        "예측 G3": pred_test,
        "오차": y_test.to_numpy() - pred_test
    })

    st.download_button(
        "실제·예측 점수 결과 다운로드",
        export.to_csv(index=False, encoding="utf-8-sig"),
        "regression_predictions.csv",
        "text/csv"
    )


# ============================================================
# 탭 4: 변수 중요도
# ============================================================
with tab_importance:

    st.subheader("예측에 중요한 변수는 무엇일까?")

    st.write(
        "순열 중요도는 평가 데이터에서 변수 하나의 값을 섞었을 때 "
        "MAE가 얼마나 증가하는지 측정합니다. "
        "증가량이 클수록 해당 변수가 예측에 더 유용했음을 뜻합니다."
    )

    if st.button("변수 중요도 계산하기"):

        with st.spinner("변수 중요도를 계산하고 있습니다..."):

            imp = permutation_importance(
                regression_model,
                X_test,
                y_test,
                n_repeats=15,
                random_state=42,
                scoring="neg_mean_absolute_error"
            )

        names = {
            "studytime": "주간 학습 시간",
            "absences": "결석 횟수",
            "failures": "과거 낙제 횟수",
            "schoolsup": "학교 교육 지원",
            "famsup": "가족 교육 지원"
        }

        imp_df = pd.DataFrame({
            "변수": [names[c] for c in features],
            "MAE 증가량 평균": imp.importances_mean,
            "표준편차": imp.importances_std
        }).sort_values("MAE 증가량 평균")

        st.session_state["reg_importance"] = imp_df

    if "reg_importance" in st.session_state:

        imp_df = st.session_state["reg_importance"]

        fig = px.bar(
            imp_df,
            x="MAE 증가량 평균",
            y="변수",
            orientation="h",
            error_x="표준편차",
            title="선형 회귀 모델의 순열 중요도"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            imp_df.round(4),
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info("'변수 중요도 계산하기' 버튼을 누르면 결과가 표시됩니다.")

    st.warning(
        "변수 중요도는 이 데이터와 모델에서의 상대적 예측 기여도이며, "
        "해당 변수가 성적을 직접 변화시킨다는 인과관계는 아닙니다."
    )


# ============================================================
# 탭 5: 점수 예측 및 3단계 해석
# ============================================================
with tab_predict:

    st.subheader("학습 습관을 입력해 수학 점수 예측하기")

    st.write(
        "선형 회귀 모델이 최종 수학 성적 G3를 0~20점으로 예측합니다. "
        "예측 점수에 따라 프로젝트에서 정한 성취도 단계를 함께 표시합니다."
    )

    with st.form("score_prediction_form"):

        l, r = st.columns(2)

        with l:

            studytime = st.select_slider(
                "주간 학습 시간",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda v: {
                    1: "2시간 미만",
                    2: "2~5시간",
                    3: "5~10시간",
                    4: "10시간 초과"
                }[v],
                help=(
                    "UCI 코드: 1은 2시간 미만, "
                    "2는 2~5시간, 3은 5~10시간, "
                    "4는 10시간 초과입니다."
                )
            )

            absences = st.number_input(
                "결석 횟수",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )

            failures = st.number_input(
                "과거 낙제 횟수",
                min_value=0,
                max_value=4,
                value=0,
                step=1
            )

        with r:

            schoolsup = st.selectbox(
                "학교의 추가 교육 지원 여부",
                ["yes", "no"],
                format_func=lambda v: "예" if v == "yes" else "아니요"
            )

            famsup = st.selectbox(
                "가족의 교육 지원 여부",
                ["yes", "no"],
                format_func=lambda v: "예" if v == "yes" else "아니요"
            )

        submitted = st.form_submit_button(
            "수학 점수 예측하기",
            use_container_width=True
        )

    if submitted:

        new = pd.DataFrame([{
            "studytime": studytime,
            "absences": absences,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup
        }])

        score = float(
            np.clip(regression_model.predict(new)[0], 0, 20)
        )

        converted = score * 5

        if score < 10:
            stage = "낮음"
            explanation = "프로젝트 기준으로 G3 10점 미만 구간입니다."

        elif score < 15:
            stage = "보통"
            explanation = "프로젝트 기준으로 G3 10점 이상 15점 미만 구간입니다."

        else:
            stage = "높음"
            explanation = "프로젝트 기준으로 G3 15점 이상 구간입니다."

        st.divider()
        st.subheader("예측 결과")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "예측 G3 점수",
            f"{score:.1f} / 20점"
        )

        c2.metric(
            "100점 만점 단순 환산",
            f"{converted:.1f} / 100점"
        )

        c3.metric(
            "성취도 단계",
            stage
        )

        st.info(explanation)

        st.caption(
            "성취도 단계는 회귀 모델이 예측한 점수를 "
            "낮음(0~9.9), 보통(10~14.9), 높음(15~20)으로 "
            "해석한 보조 표시입니다. 공식 등급이 아닙니다."
        )

        st.warning(
            "이 값은 포르투갈 학교 데이터에서 학습한 모델의 추정치입니다. "
            "한국 학교의 시험 난이도·평가 방식과 다르므로 "
            "실제 한국 학교 성적을 예측하거나 보장하지 않습니다. "
            "100점 환산은 G3에 5를 곱한 단순 비례 환산입니다."
        )


# ============================================================
# 6. 분석 기준 및 한계
# ============================================================
st.divider()

with st.expander("프로젝트의 분석 기준과 한계"):

    st.markdown("""
**분석 기준**

- 예측 대상은 최종 수학 성적 G3(0~20점)이며, 선형 회귀를 사용합니다.
- 예측 변수는 studytime, absences, failures, schoolsup, famsup의 다섯 가지입니다.
- G1·G2는 최종 성적과 밀접한 중간 성적이므로 예측 변수에서 제외했습니다.
- 데이터는 학습용 80%, 평가용 20%로 나누고 학습 데이터에서 5겹 교차검증을 수행했습니다.
- 보조 성취도 기준은 낮음(0~9점), 보통(10~14점), 높음(15~20점)으로 설정했습니다. 이 구간은 공식 등급이 아니라 프로젝트에서 점수 해석을 위해 정한 기준입니다.

**주의할 점과 한계**

- UCI Student Performance는 포르투갈 학교 학생 자료이므로 한국 학생 전체에 그대로 일반화할 수 없습니다.
- 학교마다 시험 난이도와 평가 방식이 다르므로 예측값을 한국 학교의 실제 성적으로 해석하면 안 됩니다.
- 100점 환산값은 G3에 5를 곱한 단순 비례 환산이며 한국식 내신 점수가 아닙니다.
- 학습 습관과 성적의 통계적 관계가 인과관계를 뜻하지 않습니다.
- 데이터에 포함되지 않은 가정·학교 환경과 개인차도 결과에 영향을 줄 수 있습니다.
- 예측값은 학습 데이터의 패턴에 따른 추정치이며 실제 점수와 다를 수 있습니다.
""")

st.caption(
    "MATH INSIGHT | UCI Student Performance 기반 데이터 과학 프로젝트"
)
