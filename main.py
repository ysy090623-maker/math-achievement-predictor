import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 페이지 설정 및 디자인
# ============================================================
st.set_page_config(
    page_title="MATH INSIGHT | 수학 성취도 예측",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F4F7FB; }
    .block-container { max-width: 1450px; padding-top: 1.7rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E1E8F0;
        border-radius: 13px; padding: 15px 17px;
    }
    div[data-testid="stTabs"] button { font-weight: 650; }
    .hero {
        background: linear-gradient(115deg, #132B4A 0%, #20577D 62%, #347D9D 100%);
        padding: 28px 30px; border-radius: 17px; color: white; margin-bottom: 16px;
    }
    .hero h2 { color: white; margin: 0 0 8px 0; font-size: 1.7rem; }
    .hero p { color: #E7F0F8; margin: 0; font-size: 1rem; }
    .muted { color: #64748B; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📘 MATH INSIGHT")
st.markdown(
    """
    <div class="hero">
      <h2>학생의 학습 습관과 교육 환경을 활용한 수학 성취도 예측</h2>
      <p>데이터 탐색 · 통계적 분석 · 회귀 모델 비교 · 변수 중요도 · 개인별 성적 예측</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "UCI Student Performance 데이터셋 · 수학 과목(student-mat.csv) · "
    "학습 습관과 교육 환경으로 최종 성적 G3 예측"
)


# ============================================================
# 2. 데이터 불러오기 및 변수 설정
# ============================================================
# 기존 5개 변수에 UCI 데이터에 실제로 존재하는 환경 변수를 추가했습니다.
feature_columns = [
    "studytime", "absences", "failures",
    "schoolsup", "famsup", "paid",
    "Medu", "Fedu", "internet", "traveltime", "higher",
]
numeric_features = [
    "studytime", "absences", "failures",
    "Medu", "Fedu", "traveltime",
]
categorical_features = [
    "schoolsup", "famsup", "paid", "internet", "higher",
]
target_column = "G3"

grade_order = ["하위", "중위", "상위"]
grade_colors = {"하위": "#E6A04B", "중위": "#7C91A8", "상위": "#2878B5"}

variable_names = {
    "studytime": "주간 학습 시간",
    "absences": "결석 횟수",
    "failures": "과거 낙제 횟수",
    "schoolsup": "학교의 추가 교육 지원",
    "famsup": "가족의 교육 지원",
    "paid": "수학 과목 추가 유료 수업",
    "Medu": "어머니의 교육 수준",
    "Fedu": "아버지의 교육 수준",
    "internet": "가정 내 인터넷 이용",
    "traveltime": "통학 시간",
    "higher": "고등교육 진학 희망",
}


def score_to_level(score):
    """예측 G3 점수를 프로젝트의 세 성취도 수준으로 변환합니다."""
    if score < 10:
        return "하위"
    if score < 15:
        return "중위"
    return "상위"


def benjamini_hochberg(p_values):
    """여러 통계 검정의 p값을 Benjamini-Hochberg 방식으로 보정합니다."""
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    if count == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranked = p_values[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def cramers_v(contingency_table):
    """카이제곱 독립성 검정의 효과 크기인 Cramer's V를 계산합니다."""
    chi2_value, _, _, _ = chi2_contingency(contingency_table)
    total = contingency_table.to_numpy().sum()
    rows, columns = contingency_table.shape
    denominator = total * min(rows - 1, columns - 1)
    if denominator <= 0:
        return np.nan
    return float(np.sqrt(chi2_value / denominator))


@st.cache_data
def load_data():
    """GitHub 저장소의 student-mat.csv를 읽습니다."""
    semicolon_data = pd.read_csv("student-mat.csv", sep=";")
    if len(semicolon_data.columns) == 1:
        semicolon_data = pd.read_csv("student-mat.csv")
    return semicolon_data


try:
    raw = load_data()
except Exception as exc:
    st.error(
        "student-mat.csv를 불러오지 못했습니다. "
        "GitHub 저장소에서 main.py와 student-mat.csv가 같은 최상위 위치에 있는지 확인해 주세요."
    )
    st.exception(exc)
    st.stop()

required_columns = feature_columns + [target_column]
missing_columns = [column for column in required_columns if column not in raw.columns]
if missing_columns:
    st.error(
        f"CSV에 필요한 열이 없습니다: {missing_columns}. "
        "UCI Student Performance의 수학 과목 student-mat.csv 파일인지 확인해 주세요."
    )
    st.stop()

data = raw[required_columns].copy()

for column in numeric_features + [target_column]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

for column in categorical_features:
    # pandas의 nullable string 형식 대신 일반 object 형식으로 정리해
    # scikit-learn의 SimpleImputer / OneHotEncoder와 호환되게 합니다.
    data[column] = data[column].map(
        lambda value: str(value).strip().lower() if pd.notna(value) else np.nan
    ).astype(object)

original_row_count = len(data)
data = data.dropna(subset=[target_column]).copy()
data = data[data[target_column].between(0, 20)].copy()
data["성취도"] = data[target_column].apply(score_to_level)

if len(data) < 50:
    st.error("모델을 학습하기에 유효한 데이터가 너무 적습니다. CSV 파일을 확인해 주세요.")
    st.stop()

X = data[feature_columns].copy()
y = data[target_column].astype(float).copy()


# ============================================================
# 3. 회귀 모델 학습 및 평가
# ============================================================
def build_pipeline(estimator):
    """결측값 처리와 범주형 인코딩을 모델과 하나의 파이프라인으로 묶습니다."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("regressor", estimator),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_and_evaluate(frame):
    """고정된 데이터에서 모델 비교와 평가를 수행하고 결과를 캐시합니다."""
    X_all = frame[feature_columns].copy()
    y_all = frame[target_column].astype(float).copy()

    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42
    )

    model_definitions = {
        "릿지 회귀": Ridge(alpha=5.0),
        "랜덤 포레스트 회귀": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "엑스트라 트리 회귀": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    trained = {}
    rows = []

    dummy_model = build_pipeline(DummyRegressor(strategy="mean"))
    dummy_cv = cross_validate(
        dummy_model,
        X_train_local,
        y_train_local,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        n_jobs=None,
    )
    dummy_model.fit(X_train_local, y_train_local)
    dummy_predictions = np.clip(dummy_model.predict(X_test_local), 0, 20)

    rows.append(
        {
            "모델": "기준 모델(Dummy)",
            "종류": "비교 기준",
            "교차검증 MAE": -dummy_cv["test_mae"].mean(),
            "교차검증 RMSE": -dummy_cv["test_rmse"].mean(),
            "평가 MAE": mean_absolute_error(y_test_local, dummy_predictions),
            "평가 RMSE": np.sqrt(mean_squared_error(y_test_local, dummy_predictions)),
            "평가 R²": r2_score(y_test_local, dummy_predictions),
        }
    )

    for model_name, estimator in model_definitions.items():
        pipeline = build_pipeline(estimator)
        cv_result = cross_validate(
            pipeline,
            X_train_local,
            y_train_local,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            n_jobs=None,
        )
        pipeline.fit(X_train_local, y_train_local)
        predictions = np.clip(pipeline.predict(X_test_local), 0, 20)

        trained[model_name] = pipeline
        rows.append(
            {
                "모델": model_name,
                "종류": "머신러닝 회귀",
                "교차검증 MAE": -cv_result["test_mae"].mean(),
                "교차검증 RMSE": -cv_result["test_rmse"].mean(),
                "평가 MAE": mean_absolute_error(y_test_local, predictions),
                "평가 RMSE": np.sqrt(mean_squared_error(y_test_local, predictions)),
                "평가 R²": r2_score(y_test_local, predictions),
            }
        )

    results_local = pd.DataFrame(rows)
    ml_results_local = (
        results_local[results_local["종류"] == "머신러닝 회귀"]
        .sort_values("교차검증 MAE", ascending=True)
        .reset_index(drop=True)
    )
    best_name_local = ml_results_local.loc[0, "모델"]
    best_model_local = trained[best_name_local]

    # 학습 데이터의 5겹 교차검증 예측으로 일반적인 오차 크기를 확인합니다.
    oof_predictions = cross_val_predict(
        best_model_local,
        X_train_local,
        y_train_local,
        cv=cv,
        n_jobs=None,
    )
    absolute_errors = np.abs(y_train_local.to_numpy() - oof_predictions)
    error_80 = float(np.quantile(absolute_errors, 0.80))

    best_test_predictions_local = np.clip(
        best_model_local.predict(X_test_local), 0, 20
    )
    actual_test_levels = np.array(
        [score_to_level(value) for value in y_test_local.to_numpy()]
    )
    predicted_test_levels = np.array(
        [score_to_level(value) for value in best_test_predictions_local]
    )

    return {
        "models": trained,
        "results": results_local,
        "best_name": best_name_local,
        "best_model": best_model_local,
        "X_test": X_test_local,
        "y_test": y_test_local,
        "best_test_predictions": best_test_predictions_local,
        "actual_test_levels": actual_test_levels,
        "predicted_test_levels": predicted_test_levels,
        "error_80": error_80,
        "dummy_model": dummy_model,
    }


with st.spinner("여러 회귀 모델을 학습하고 5겹 교차검증으로 비교하고 있습니다..."):
    training = train_and_evaluate(data)

results = training["results"]
best_model_name = training["best_name"]
best_model = training["best_model"]
X_test = training["X_test"]
y_test = training["y_test"]
best_test_predictions = training["best_test_predictions"]
actual_test_levels = training["actual_test_levels"]
predicted_test_levels = training["predicted_test_levels"]

actual_counts = (
    data["성취도"]
    .value_counts()
    .reindex(grade_order, fill_value=0)
)
predicted_counts = (
    pd.Series(predicted_test_levels)
    .value_counts()
    .reindex(grade_order, fill_value=0)
)


# ============================================================
# 4. 사이드바 안내
# ============================================================
with st.sidebar:
    st.header("분석 설정")
    st.markdown("**성취도 분류 기준**")
    st.write("• 하위: G3 0~9점")
    st.write("• 중위: G3 10~14점")
    st.write("• 상위: G3 15~20점")
    st.caption("세 수준은 이 프로젝트에서 설정한 기준이며 공식 등급은 아닙니다.")
    st.divider()
    st.markdown("**예측에 사용하는 정보**")
    st.write("• 학습 시간 · 결석 · 과거 낙제")
    st.write("• 학교·가족 지원 · 수학 추가 유료 수업")
    st.write("• 부모 교육 수준 · 인터넷 · 통학 시간 · 진학 희망")
    st.divider()
    st.caption("G1·G2는 최종 성적과 밀접한 중간 성적이므로 예측 변수에서 제외합니다.")


# ============================================================
# 5. 상단 요약 대시보드
# ============================================================
st.subheader("프로젝트 분석 현황")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("분석 학생 수", f"{len(data):,}명")
metric2.metric("하위 성취 그룹", f"{int(actual_counts['하위']):,}명")
metric3.metric("중위 성취 그룹", f"{int(actual_counts['중위']):,}명")
metric4.metric("상위 성취 그룹", f"{int(actual_counts['상위']):,}명")

best_cv_mae = float(
    results.loc[results["모델"] == best_model_name, "교차검증 MAE"].iloc[0]
)
best_test_mae = float(
    results.loc[results["모델"] == best_model_name, "평가 MAE"].iloc[0]
)
st.markdown(
    f'<p class="muted">선택 모델: {best_model_name} · '
    f'학습 데이터 80%, 평가 데이터 20% · '
    f'교차검증 평균 MAE {best_cv_mae:.2f}점</p>',
    unsafe_allow_html=True,
)


# ============================================================
# 6. 탭 구성
# ============================================================
tab_data, tab_stats, tab_models, tab_importance, tab_predict = st.tabs(
    [
        "① 데이터 탐색",
        "② 통계 분석",
        "③ 모델 성능 비교",
        "④ 변수 중요도",
        "⑤ 수학 성취도 예측",
    ]
)


# ============================================================
# 탭 1. 데이터 탐색
# ============================================================
with tab_data:
    st.subheader("데이터 구성")
    left, right = st.columns([1.15, 0.85])

    with left:
        variable_table = pd.DataFrame(
            {
                "변수": [
                    "studytime", "absences", "failures",
                    "schoolsup", "famsup", "paid",
                    "Medu / Fedu", "internet", "traveltime", "higher", "G3",
                ],
                "설명": [
                    "주간 학습 시간(1: 2시간 미만, 4: 10시간 초과)",
                    "결석 횟수",
                    "과거 낙제 횟수",
                    "학교의 추가 교육 지원 여부",
                    "가족의 교육 지원 여부",
                    "수학 과목의 추가 유료 수업 여부",
                    "어머니 / 아버지의 교육 수준(0~4)",
                    "가정 내 인터넷 이용 여부",
                    "통학 시간 수준(1~4)",
                    "고등교육 진학 희망 여부",
                    "최종 수학 성적(0~20점), 예측 대상",
                ],
                "역할": ["예측 변수"] * 10 + ["예측 대상"],
            }
        )
        st.dataframe(variable_table, hide_index=True, use_container_width=True)

    with right:
        group_chart = px.pie(
            data,
            names="성취도",
            hole=0.58,
            title="실제 성취도 그룹 구성 비율",
            color="성취도",
            category_orders={"성취도": grade_order},
            color_discrete_map=grade_colors,
        )
        group_chart.update_layout(margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(group_chart, use_container_width=True)

    st.divider()
    st.subheader("학습 습관과 최종 성적의 관계")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        study_plot_data = data.copy()
        study_plot_data["학습 시간"] = study_plot_data["studytime"].map(
            {
                1: "주 2시간 미만",
                2: "주 2~5시간",
                3: "주 5~10시간",
                4: "주 10시간 초과",
            }
        )
        study_chart = px.box(
            study_plot_data,
            x="학습 시간",
            y="G3",
            color="학습 시간",
            category_orders={
                "학습 시간": [
                    "주 2시간 미만", "주 2~5시간",
                    "주 5~10시간", "주 10시간 초과",
                ]
            },
            title="주간 학습 시간별 최종 수학 성적 분포",
        )
        study_chart.update_layout(
            xaxis_title="주간 학습 시간",
            yaxis_title="최종 성적 G3",
            showlegend=False,
        )
        st.plotly_chart(study_chart, use_container_width=True)

    with chart_right:
        absence_chart = px.scatter(
            data,
            x="absences",
            y="G3",
            color="성취도",
            opacity=0.7,
            title="결석 횟수와 최종 수학 성적",
            color_discrete_map=grade_colors,
            category_orders={"성취도": grade_order},
        )
        absence_chart.update_layout(
            xaxis_title="결석 횟수",
            yaxis_title="최종 성적 G3",
        )
        st.plotly_chart(absence_chart, use_container_width=True)

    st.subheader("성취도 그룹별 평균 학습·환경 특성")
    group_summary = (
        data.groupby("성취도", observed=False)[numeric_features]
        .mean()
        .reindex(grade_order)
        .round(2)
        .rename(columns=variable_names)
    )
    st.dataframe(group_summary, use_container_width=True)

    st.caption(
        f"원본 {original_row_count}행 중 G3가 없거나 0~20 범위를 벗어난 "
        f"{original_row_count - len(data)}행은 제외했습니다. "
        "예측 변수의 결측값은 학습 데이터의 중앙값 또는 최빈값으로 대체합니다."
    )

    with st.expander("원본 데이터 일부 확인"):
        st.dataframe(raw.head(20), use_container_width=True)


# ============================================================
# 탭 2. 통계 분석
# ============================================================
with tab_stats:
    st.subheader("세 성취도 그룹의 학습 습관·환경 차이")
    st.write(
        "수치형 변수는 크루스칼-왈리스 검정, 범주형 변수는 "
        "카이제곱 독립성 검정으로 세 그룹 사이의 차이를 살펴봅니다."
    )

    statistical_rows = []

    for feature in numeric_features:
        samples = [
            data.loc[data["성취도"] == group, feature].dropna().to_numpy()
            for group in grade_order
        ]
        nonempty_samples = [sample for sample in samples if len(sample) > 0]
        if len(nonempty_samples) >= 2:
            test_result = kruskal(*nonempty_samples)
            total_n = sum(len(sample) for sample in nonempty_samples)
            group_n = len(nonempty_samples)
            epsilon_squared = max(
                0.0, (test_result.statistic - group_n + 1) / max(total_n - group_n, 1)
            )
            statistical_rows.append(
                {
                    "변수": variable_names[feature],
                    "검정 방법": "크루스칼-왈리스",
                    "효과 크기": epsilon_squared,
                    "p값": test_result.pvalue,
                }
            )

    for feature in categorical_features:
        contingency = pd.crosstab(data[feature], data["성취도"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2_result = chi2_contingency(contingency)
            statistical_rows.append(
                {
                    "변수": variable_names[feature],
                    "검정 방법": "카이제곱 독립성",
                    "효과 크기": cramers_v(contingency),
                    "p값": chi2_result[1],
                }
            )

    stats_df = pd.DataFrame(statistical_rows)
    if not stats_df.empty:
        stats_df["BH 보정 p값"] = benjamini_hochberg(stats_df["p값"].values)
        stats_df["검정 결과"] = np.where(
            stats_df["BH 보정 p값"] < 0.05,
            "보정 후 0.05 미만",
            "보정 후 0.05 이상",
        )
        stats_display = stats_df.copy()
        for column in ["효과 크기", "p값", "BH 보정 p값"]:
            stats_display[column] = stats_display[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else "-"
            )
        st.dataframe(stats_display, hide_index=True, use_container_width=True)
    else:
        st.info("검정에 필요한 그룹별 데이터가 충분하지 않습니다.")

    st.caption(
        "크루스칼-왈리스 검정은 세 그룹 이상의 분포 차이를 살펴보는 비모수 검정입니다. "
        "효과 크기는 수치형 변수의 epsilon-squared 또는 범주형 변수의 Cramer's V입니다. "
        "여러 검정의 p값은 Benjamini-Hochberg 방식으로 보정했습니다."
    )

    st.subheader("학습·환경 변수와 최종 성적의 상관관계")
    correlation_rows = []
    for feature in numeric_features:
        valid = data[[feature, "G3"]].dropna()
        if len(valid) >= 3 and valid[feature].nunique() > 1:
            rho, p_value = spearmanr(valid[feature], valid["G3"])
        else:
            rho, p_value = np.nan, np.nan
        correlation_rows.append(
            {
                "변수": variable_names[feature],
                "Spearman 상관계수": rho,
                "p값": p_value,
            }
        )

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_display = correlation_df.copy()
    correlation_display["Spearman 상관계수"] = correlation_display[
        "Spearman 상관계수"
    ].map(lambda value: f"{value:.3f}" if pd.notna(value) else "-")
    correlation_display["p값"] = correlation_display["p값"].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else "-"
    )
    st.dataframe(correlation_display, hide_index=True, use_container_width=True)

    correlation_chart = px.bar(
        correlation_df,
        x="변수",
        y="Spearman 상관계수",
        color="Spearman 상관계수",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        title="수치형 변수와 최종 성적 G3의 Spearman 상관계수",
    )
    correlation_chart.update_layout(yaxis_title="Spearman 상관계수 (rho)")
    st.plotly_chart(correlation_chart, use_container_width=True)

    st.warning(
        "통계적 관계나 상관관계는 인과관계를 증명하지 않습니다. "
        "예를 들어 추가 교육 지원을 받는다는 사실만으로 성적이 낮거나 높다고 단정할 수 없습니다."
    )


# ============================================================
# 탭 3. 회귀 모델 성능 비교
# ============================================================
with tab_models:
    st.subheader("수학 성적을 숫자로 예측하는 회귀 모델 비교")
    st.write(
        "모델이 최종 성적 G3(0~20점)를 먼저 예측합니다. "
        "예측 점수에 따라 하위·중위·상위 세 수준을 표시합니다."
    )
    st.markdown(
        """
        - **MAE**: 실제 성적과 예측 성적의 평균적인 차이입니다. 낮을수록 좋습니다.
        - **RMSE**: 큰 예측 오차에 더 큰 불이익을 주는 지표입니다. 낮을수록 좋습니다.
        - **R²**: 모델이 성적 차이를 얼마나 설명하는지 나타냅니다. 1에 가까울수록 좋고, 0보다 작을 수도 있습니다.
        """
    )

    display_results = results.copy()
    for column in [
        "교차검증 MAE", "교차검증 RMSE", "평가 MAE", "평가 RMSE", "평가 R²"
    ]:
        display_results[column] = display_results[column].map(
            lambda value: f"{value:.3f}"
        )
    st.dataframe(display_results, hide_index=True, use_container_width=True)

    compare_long = results.melt(
        id_vars=["모델"],
        value_vars=["교차검증 MAE", "평가 MAE"],
        var_name="평가 구분",
        value_name="MAE (점)",
    )
    comparison_chart = px.bar(
        compare_long,
        x="모델",
        y="MAE (점)",
        color="평가 구분",
        barmode="group",
        title="모델별 평균 절대 오차 비교 (낮을수록 좋음)",
    )
    st.plotly_chart(comparison_chart, use_container_width=True)

    st.success(
        f"머신러닝 회귀 모델 중 5겹 교차검증 MAE가 가장 낮은 모델은 "
        f"'{best_model_name}'입니다. 이 모델을 개인별 예측에 사용합니다."
    )

    dummy_cv_mae = float(
        results.loc[
            results["모델"] == "기준 모델(Dummy)", "교차검증 MAE"
        ].iloc[0]
    )
    if best_cv_mae < dummy_cv_mae:
        st.write(
            f"선택된 모델의 교차검증 MAE는 단순 평균만 예측하는 기준 모델보다 "
            f"{dummy_cv_mae - best_cv_mae:.2f}점 낮았습니다."
        )
    else:
        st.warning(
            "선택된 회귀 모델의 교차검증 오차가 기준 모델보다 낮지 않습니다. "
            "현재 입력 변수만으로 성적을 충분히 예측하지 못할 수 있으므로 결과를 신중하게 해석해야 합니다."
        )

    st.divider()
    st.subheader("예측 점수를 성취도 세 수준으로 구분한 결과")
    true_group_order = ["하위", "중위", "상위"]
    matrix = confusion_matrix(
        actual_test_levels,
        predicted_test_levels,
        labels=true_group_order,
    )
    matrix_chart = px.imshow(
        matrix,
        text_auto=True,
        x=[f"예측: {group}" for group in true_group_order],
        y=[f"실제: {group}" for group in true_group_order],
        color_continuous_scale="Blues",
        title="평가 데이터의 3×3 성취도 혼동 행렬",
    )
    matrix_chart.update_layout(
        xaxis_title="예측한 성취도 수준",
        yaxis_title="실제 성취도 수준",
    )
    st.plotly_chart(matrix_chart, use_container_width=True)

    grouped_accuracy = accuracy_score(actual_test_levels, predicted_test_levels)
    grouped_macro_f1 = f1_score(
        actual_test_levels,
        predicted_test_levels,
        labels=true_group_order,
        average="macro",
        zero_division=0,
    )
    group_metric1, group_metric2 = st.columns(2)
    group_metric1.metric("세 수준 분류 정확도", f"{grouped_accuracy:.1%}")
    group_metric2.metric("세 수준 Macro F1", f"{grouped_macro_f1:.3f}")

    predicted_group_counts = (
        pd.Series(predicted_test_levels)
        .value_counts()
        .reindex(grade_order, fill_value=0)
    )
    group_count_chart = px.bar(
        x=predicted_group_counts.index,
        y=predicted_group_counts.values,
        color=predicted_group_counts.index,
        color_discrete_map=grade_colors,
        title="평가 데이터에서 모델이 예측한 성취도 그룹 인원",
        labels={"x": "예측 성취도", "y": "학생 수"},
    )
    st.plotly_chart(group_count_chart, use_container_width=True)

    if int(predicted_group_counts["상위"]) == 0:
        st.warning(
            "현재 평가 데이터에서는 모델이 상위 성취 그룹으로 예측한 학생이 없습니다. "
            "상위 결과가 나오도록 임의로 점수를 올리지 않고, 모델이 상위 학생의 패턴을 "
            "충분히 구별하는지 추가 점검해야 합니다."
        )
    else:
        st.info(
            f"평가 데이터에서 상위 성취 그룹으로 예측된 학생은 "
            f"{int(predicted_group_counts['상위'])}명입니다. "
            "이는 현재 평가 데이터에 대한 결과이며 모든 입력에서 상위 예측이 가능하다는 뜻은 아닙니다."
        )

    st.divider()
    st.subheader("상위 성취도 예측 가능성 점검")
    st.write(
        "입력 변수 중 학습 시간·결석·과거 낙제·부모 교육 수준·통학 시간 등을 "
        "상위 성취에 유리한 방향으로 설정한 가상 프로필을 시험합니다. "
        "지원 여부와 추가 유료 수업은 예/아니요 조합을 모두 확인합니다."
    )

    extreme_profiles = []
    for school_support in ["yes", "no"]:
        for family_support in ["yes", "no"]:
            for paid_class in ["yes", "no"]:
                extreme_profiles.append(
                    {
                        "studytime": 4,
                        "absences": 0,
                        "failures": 0,
                        "schoolsup": school_support,
                        "famsup": family_support,
                        "paid": paid_class,
                        "Medu": 4,
                        "Fedu": 4,
                        "internet": "yes",
                        "traveltime": 1,
                        "higher": "yes",
                    }
                )
    extreme_frame = pd.DataFrame(extreme_profiles, columns=feature_columns)
    extreme_scores = np.clip(best_model.predict(extreme_frame), 0, 20)
    extreme_result = extreme_frame[
        ["schoolsup", "famsup", "paid"]
    ].copy()
    extreme_result["예측 G3"] = extreme_scores.round(2)
    extreme_result["예측 성취도"] = [score_to_level(value) for value in extreme_scores]
    extreme_result["100점 환산 참고값"] = (extreme_scores * 5).round(1)
    st.dataframe(extreme_result, hide_index=True, use_container_width=True)

    if any(score_to_level(value) == "상위" for value in extreme_scores):
        st.success(
            "가상 프로필 조합 중 상위 성취도로 예측된 조합이 있습니다. "
            "다만 이 조합은 모델 점검용 가상 사례이며 실제 성적을 보장하지 않습니다."
        )
    else:
        st.warning(
            "상위에 유리한 가상 입력 조합에서도 상위 성취도 예측이 나오지 않았습니다. "
            "이는 현재 변수와 모델이 상위 성적을 충분히 구별하지 못할 수 있다는 신호입니다. "
            "사용자에게 상위 결과를 강제로 보여 주지 않고, 이 한계를 그대로 확인할 수 있도록 했습니다."
        )

    st.caption(
        "평가 지표는 학습에 사용하지 않은 20% 평가 데이터에서 계산했습니다. "
        "모델 선택은 학습 데이터 내부의 5겹 교차검증 MAE를 기준으로 합니다."
    )

    result_csv = results.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "회귀 모델 성능 비교 결과 다운로드",
        data=result_csv,
        file_name="regression_model_comparison.csv",
        mime="text/csv",
    )


# ============================================================
# 탭 4. 변수 중요도
# ============================================================
with tab_importance:
    st.subheader("모델의 예측에 중요한 변수는 무엇일까?")
    st.write(
        "순열 중요도는 평가 데이터에서 변수 하나의 값을 섞었을 때 "
        "MAE가 얼마나 나빠지는지 측정합니다. 값이 클수록 해당 변수에 대한 모델의 의존도가 큽니다."
    )

    if st.button("변수 중요도 계산하기"):
        with st.spinner("평가 데이터를 이용해 변수 중요도를 계산하고 있습니다..."):
            importance_result = permutation_importance(
                best_model,
                X_test,
                y_test,
                n_repeats=12,
                random_state=42,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
        importance_df = pd.DataFrame(
            {
                "변수": [variable_names[column] for column in feature_columns],
                "평균 중요도": importance_result.importances_mean,
                "표준편차": importance_result.importances_std,
            }
        ).sort_values("평균 중요도", ascending=True)
        st.session_state["importance_df"] = importance_df
        st.session_state["importance_model"] = best_model_name

    if (
        "importance_df" in st.session_state
        and st.session_state.get("importance_model") == best_model_name
    ):
        importance_df = st.session_state["importance_df"]
        importance_chart = px.bar(
            importance_df,
            x="평균 중요도",
            y="변수",
            orientation="h",
            error_x="표준편차",
            title=f"{best_model_name}의 변수별 순열 중요도",
        )
        importance_chart.update_layout(
            xaxis_title="MAE 증가량 (클수록 중요)",
            yaxis_title="예측 변수",
        )
        st.plotly_chart(importance_chart, use_container_width=True)
        st.dataframe(
            importance_df.round(4), hide_index=True, use_container_width=True
        )
    else:
        st.info("'변수 중요도 계산하기' 버튼을 누르면 분석 결과가 표시됩니다.")

    st.warning(
        "변수 중요도는 이 데이터와 선택된 모델의 예측에 대한 상대적 기여도입니다. "
        "해당 변수가 성적을 직접 변화시킨다는 인과관계를 의미하지 않습니다."
    )


# ============================================================
# 탭 5. 개인별 수학 성취도 예측
# ============================================================
with tab_predict:
    st.subheader("학습 습관과 교육 환경으로 수학 성적 예측하기")
    st.write(
        f"현재 가장 낮은 교차검증 MAE를 기록한 **{best_model_name}** 모델을 사용합니다. "
        "입력한 정보를 바탕으로 최종 성적 G3를 먼저 예측하고, 예측 점수에 따라 세 수준으로 분류합니다."
    )

    studytime_labels = {
        1: "1단계 · 주 2시간 미만",
        2: "2단계 · 주 2~5시간",
        3: "3단계 · 주 5~10시간",
        4: "4단계 · 주 10시간 초과",
    }
  education_labels = {
    0: "0 · 정규 교육을 받지 않음",
    1: "1 · 기초 초등교육",
    2: "2 · 초등·중등교육 일부",
    3: "3 · 중등교육",
    4: "4 · 고등교육",
}
    travel_labels = {
        1: "1 · 15분 미만",
        2: "2 · 15~30분",
        3: "3 · 30분~1시간",
        4: "4 · 1시간 초과",
    }

    default_absences = int(round(data["absences"].median()))
    max_absences = max(30, int(data["absences"].max()))
    default_failures = int(round(data["failures"].median()))
    max_failures = max(4, int(data["failures"].max()))

    with st.form("student_prediction_form"):
        st.markdown("**① 기본 학습 습관**")
        first_left, first_right = st.columns(2)

        with first_left:
            studytime = st.select_slider(
                "주간 학습 시간",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda value: studytime_labels[value],
                help="UCI 데이터 기준 주간 학습 시간 구간입니다.",
            )
            absences = st.number_input(
                "결석 횟수",
                min_value=0,
                max_value=max_absences,
                value=min(default_absences, max_absences),
                step=1,
            )
            failures = st.number_input(
                "과거 낙제 횟수(데이터 기록값)",
                min_value=0,
                max_value=max_failures,
                value=min(default_failures, max_failures),
                step=1,
            )

        with first_right:
            traveltime = st.selectbox(
                "집에서 학교까지의 통학 시간",
                options=[1, 2, 3, 4],
                format_func=lambda value: travel_labels[value],
                index=0,
            )
            higher = st.selectbox(
                "고등교육(대학 등) 진학 희망 여부",
                ["yes", "no"],
                format_func=lambda value: "예" if value == "yes" else "아니요",
            )
            internet = st.selectbox(
                "가정 내 인터넷 이용 가능 여부",
                ["yes", "no"],
                format_func=lambda value: "예" if value == "yes" else "아니요",
            )

        st.divider()
        st.markdown("**② 교육 지원 환경**")
        second_left, second_right = st.columns(2)

        with second_left:
            schoolsup = st.selectbox(
                "학교의 추가 교육 지원 여부",
                ["yes", "no"],
                format_func=lambda value: "예" if value == "yes" else "아니요",
            )
            famsup = st.selectbox(
                "가족의 교육 지원 여부",
                ["yes", "no"],
                format_func=lambda value: "예" if value == "yes" else "아니요",
            )
            paid = st.selectbox(
                "수학 과목의 추가 유료 수업 여부",
                ["yes", "no"],
                format_func=lambda value: "예" if value == "yes" else "아니요",
                help="학원·과외의 수준이나 지역을 뜻하는 변수가 아니라, 해당 과목의 추가 유료 수업 여부입니다.",
            )

        with second_right:
            Medu = st.selectbox(
                "어머니의 교육 수준",
                options=[0, 1, 2, 3, 4],
                format_func=lambda value: education_labels[value],
                index=2,
            )
            Fedu = st.selectbox(
                "아버지의 교육 수준",
                options=[0, 1, 2, 3, 4],
                format_func=lambda value: education_labels[value],
                index=2,
            )
          st.caption(
    "이 항목은 포르투갈 UCI 원자료의 부모 교육 수준을 나타냅니다. "
    "한국의 초등학교·중학교·고등학교 학년과 정확히 대응하는 값이 아니므로 "
    "특정 한국 학년으로 해석하지 않고 교육 단계로 구분했습니다."
)

        submitted = st.form_submit_button(
            "내 수학 성취도 예측하기", use_container_width=True
        )

    if submitted:
        new_student = pd.DataFrame(
            [{
                "studytime": int(studytime),
                "absences": int(absences),
                "failures": int(failures),
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "Medu": int(Medu),
                "Fedu": int(Fedu),
                "internet": internet,
                "traveltime": int(traveltime),
                "higher": higher,
            }],
            columns=feature_columns,
        )

        predicted_score = float(np.clip(best_model.predict(new_student)[0], 0, 20))
        predicted_level = score_to_level(predicted_score)
        converted_score = predicted_score * 5

        st.session_state["last_prediction_input"] = new_student
        st.session_state["last_prediction_score"] = predicted_score
        st.session_state["last_prediction_level"] = predicted_level

    if "last_prediction_input" in st.session_state:
        current_input = st.session_state["last_prediction_input"]
        predicted_score = float(st.session_state["last_prediction_score"])
        predicted_level = st.session_state["last_prediction_level"]
        converted_score = predicted_score * 5

        st.divider()
        st.subheader("예측 결과")
        result_left, result_right = st.columns([1.1, 0.9])

        with result_left:
            st.metric("예상 최종 수학 성적(G3)", f"{predicted_score:.1f} / 20점")
            st.metric("100점 만점 환산 참고값", f"{converted_score:.1f} / 100점")
            if predicted_level == "상위":
                st.success("예측 성취도: 상위 그룹 (15~20점)")
            elif predicted_level == "중위":
                st.info("예측 성취도: 중위 그룹 (10~14점)")
            else:
                st.warning("예측 성취도: 하위 그룹 (0~9점)")

        with result_right:
            st.markdown("**예측 점수가 속한 구간**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "구분": ["하위", "중위", "상위"],
                        "G3 기준": ["0~9점", "10~14점", "15~20점"],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"현재 예측값은 **{predicted_level}** 수준의 범위에 있습니다.")

        st.info(
            f"선택 모델의 20% 평가 데이터 MAE는 {best_test_mae:.2f}점입니다. "
            "이는 평가 데이터에서 실제 점수와 예측 점수 사이의 평균적인 차이이며, "
            "개별 학생의 오차가 정확히 이 값이라는 뜻은 아닙니다."
        )
        st.caption(
            f"학습 데이터 5겹 교차검증에서 절대 오차의 80백분위 값은 약 "
            f"±{training['error_80']:.1f}점이었습니다. 이는 과거 교차검증 예측의 오차 분포를 "
            "요약한 참고값이며, 개인별 성적의 확정 구간이나 보장 범위는 아닙니다."
        )

        st.divider()
        st.subheader("입력 조건을 바꾸면 예측 점수는 어떻게 달라질까?")
        st.write(
            "아래 표는 현재 입력에서 한 가지 항목만 바꾸고 나머지는 그대로 둔 가상 비교입니다. "
            "모델이 학습한 패턴을 확인하는 기능이며, 해당 조건이 성적을 직접 올리거나 내린다는 뜻은 아닙니다."
        )

        scenario_rows = []

        def add_scenario(feature, label, value, display_value):
            scenario = current_input.copy()
            scenario.loc[0, feature] = value
            scenario_score = float(np.clip(best_model.predict(scenario)[0], 0, 20))
            scenario_rows.append(
                {
                    "비교 항목": label,
                    "입력값": display_value,
                    "예상 G3": round(scenario_score, 2),
                    "현재 예측과 차이": round(scenario_score - predicted_score, 2),
                    "성취도": score_to_level(scenario_score),
                }
            )

        for value in [1, 2, 3, 4]:
            add_scenario(
                "studytime",
                "주간 학습 시간",
                value,
                studytime_labels[value],
            )

        absence_candidates = list(
            dict.fromkeys(
                [
                    0,
                    int(current_input.loc[0, "absences"]),
                    int(round(data["absences"].median())),
                ]
            )
        )
        for value in absence_candidates:
            add_scenario("absences", "결석 횟수", value, f"{value}회")

        failure_candidates = list(
            dict.fromkeys([0, int(current_input.loc[0, "failures"]), max_failures])
        )
        for value in failure_candidates:
            add_scenario("failures", "과거 낙제 횟수", value, f"{value}회")

        for feature, label in [
            ("paid", "수학 과목 추가 유료 수업"),
            ("schoolsup", "학교 추가 교육 지원"),
            ("famsup", "가족 교육 지원"),
        ]:
            for value in ["yes", "no"]:
                add_scenario(feature, label, value, "예" if value == "yes" else "아니요")

        scenario_df = pd.DataFrame(scenario_rows).drop_duplicates(
            subset=["비교 항목", "입력값"]
        )
        st.dataframe(scenario_df, hide_index=True, use_container_width=True)

        st.caption(
            "예상 점수 차이는 모델의 반응을 보여주는 비교값입니다. "
            "실제 환경에서 그 조건만 바꾸면 성적이 같은 폭으로 달라진다는 인과적 의미는 없습니다."
        )


# ============================================================
# 7. 프로젝트의 분석 기준과 한계
# ============================================================
st.divider()
with st.expander("프로젝트의 분석 기준과 한계"):
    st.markdown(
        """
        **분석 기준**
        - 예측 대상은 최종 수학 성적 G3(0~20점)입니다.
        - G3 예측 점수가 0~9점이면 하위, 10~14점이면 중위, 15~20점이면 상위로 분류합니다.
        - 세 수준은 본 프로젝트에서 설정한 기준이며 공식 성취도 등급은 아닙니다.
        - G1과 G2는 최종 성적과 밀접한 중간 성적이므로 예측 변수에서 제외했습니다.
        - 회귀 모델은 릿지 회귀, 랜덤 포레스트 회귀, 엑스트라 트리 회귀를 비교합니다.
        - 기준 모델은 학습 데이터의 평균 성적만 예측하는 Dummy 회귀 모델입니다.
        - 학습 데이터와 평가 데이터는 80:20으로 분리하고, 학습 데이터에서 5겹 교차검증으로 모델을 선택합니다.

        **데이터와 해석의 한계**
        - UCI 자료는 포르투갈의 특정 학교 학생 자료이며 한국 학생 전체를 대표하지 않습니다.
        - 데이터에는 강남·대치동 여부, 학원 수준, 사교육 시간, 학교별 시험 난이도 같은 정보가 없습니다.
        - 추가 유료 수업 여부(paid)는 수학 과목의 유료 수업 여부만 뜻하며 학원 수준이나 지역을 나타내지 않습니다.
        - 부모의 교육 수준, 인터넷 이용, 통학 시간 등은 데이터에 기록된 배경 정보이며 성적의 원인이라고 단정할 수 없습니다.
        - 모델 예측은 학습 데이터에서 발견한 패턴을 바탕으로 한 추정값이며 실제 개인의 성적을 보장하지 않습니다.
        - 100점 환산값은 G3에 5를 곱한 비례 환산 참고값이며 한국 학교의 실제 내신 점수와 같지 않습니다.
        """
    )

st.caption("MATH INSIGHT | UCI Student Performance 기반 수학 성취도 예측 프로젝트")
