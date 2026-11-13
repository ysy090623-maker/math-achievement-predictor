import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance


# ============================================================
# 1. 페이지 설정 및 디자인
# ============================================================
st.set_page_config(
    page_title="MATH INSIGHT | 수학 성취도 분석",
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
      <h2>학생의 학습 습관과 수학 성취도 분석</h2>
      <p>데이터 탐색 · 통계적 검정 · 머신러닝 모델 비교 · 변수 중요도 · 성취도 예측</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "UCI Student Performance 데이터셋 · 수학 과목(student-mat.csv) · "
    "프로젝트에서 설정한 성취도 분류 기준"
)


# ============================================================
# 2. 사이드바 설정 및 데이터 불러오기
# ============================================================
with st.sidebar:
    st.header("분석 설정")
    threshold = st.slider(
        "높은 성취도 분류 기준 (G3)",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
        help="최종 성적 G3가 기준 점수 이상이면 '높음', 미만이면 '낮음'으로 분류합니다.",
    )
    st.caption("10점은 이 프로젝트에서 선택한 기준이며 공식 등급은 아닙니다.")
    st.divider()
    st.markdown("**예측에 사용하는 변수**")
    st.write("• 주간 학습 시간(studytime)")
    st.write("• 결석 횟수(absences)")
    st.write("• 과거 낙제 횟수(failures)")
    st.write("• 학교 교육 지원(schoolsup)")
    st.write("• 가족 교육 지원(famsup)")
    st.divider()
    st.caption("G1·G2는 최종 성적과 밀접하므로 예측 변수에서 제외합니다.")


@st.cache_data
def load_data():
    """GitHub 저장소에 main.py와 함께 올린 CSV 파일을 읽습니다."""
    return pd.read_csv("student-mat.csv", sep=";")


try:
    raw = load_data()
except Exception as exc:
    st.error(
        "student-mat.csv를 불러오지 못했습니다. "
        "GitHub 저장소에서 main.py와 student-mat.csv가 같은 최상위 위치에 있는지 확인해 주세요."
    )
    st.exception(exc)
    st.stop()

required_columns = ["studytime", "absences", "failures", "schoolsup", "famsup", "G3"]
missing_columns = [column for column in required_columns if column not in raw.columns]
if missing_columns:
    st.error(f"CSV에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

data = raw[required_columns].copy()
for column in ["studytime", "absences", "failures", "G3"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

original_row_count = len(data)
data = data.dropna(subset=["G3"]).copy()
data["성취도"] = np.where(data["G3"] >= threshold, "높음", "낮음")

feature_columns = ["studytime", "absences", "failures", "schoolsup", "famsup"]
numeric_features = ["studytime", "absences", "failures"]
categorical_features = ["schoolsup", "famsup"]
X = data[feature_columns]
y = data["성취도"]

if len(data) < 40 or y.nunique() != 2 or y.value_counts().min() < 10:
    st.error(
        "현재 분류 기준에서는 두 그룹의 데이터가 모델 학습에 충분하지 않습니다. "
        "왼쪽 메뉴에서 기준 점수를 조정해 주세요."
    )
    st.stop()


# ============================================================
# 3. 통계 분석 및 모델 구성 함수
# ============================================================
def benjamini_hochberg(p_values):
    """여러 검정의 p값을 Benjamini-Hochberg 방식으로 보정합니다."""
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
    """카이제곱 검정의 효과 크기인 Cramer's V를 계산합니다."""
    chi2_value, _, _, _ = chi2_contingency(contingency_table)
    total = contingency_table.to_numpy().sum()
    rows, columns = contingency_table.shape
    denominator = total * min(rows - 1, columns - 1)
    if denominator <= 0:
        return np.nan
    return float(np.sqrt(chi2_value / denominator))


def build_model(estimator, preprocessor):
    """전처리와 분류기를 하나의 파이프라인으로 묶습니다."""
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", estimator),
        ]
    )


# ============================================================
# 4. 학습/평가 데이터 분리 및 전처리
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

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
    ]
)


# ============================================================
# 5. 기준 모델 및 머신러닝 모델 비교
# ============================================================
model_definitions = {
    "로지스틱 회귀": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "의사결정나무": DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=42
    ),
    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trained_models = {}
comparison_rows = []

with st.spinner("기준 모델과 머신러닝 모델 3개를 학습하고 검증하고 있습니다..."):
    # 기준 모델: 학습 데이터에서 가장 많은 그룹만 예측합니다.
    dummy_pipeline = build_model(
        DummyClassifier(strategy="most_frequent"), preprocessor
    )
    dummy_cv_scores = cross_val_score(
        dummy_pipeline, X_train, y_train,
        cv=cross_validator, scoring="f1_macro"
    )
    dummy_pipeline.fit(X_train, y_train)
    dummy_prediction = dummy_pipeline.predict(X_test)
    dummy_probabilities = dummy_pipeline.predict_proba(X_test)
    dummy_classes = list(dummy_pipeline.classes_)
    dummy_high_probability = dummy_probabilities[:, dummy_classes.index("높음")]

    comparison_rows.append(
        {
            "모델": "기준 모델(Dummy)",
            "종류": "비교 기준",
            "정확도": accuracy_score(y_test, dummy_prediction),
            "균형 정확도": balanced_accuracy_score(y_test, dummy_prediction),
            "정밀도(높음)": precision_score(
                y_test, dummy_prediction, pos_label="높음", zero_division=0
            ),
            "재현율(높음)": recall_score(
                y_test, dummy_prediction, pos_label="높음", zero_division=0
            ),
            "F1(높음)": f1_score(
                y_test, dummy_prediction, pos_label="높음", zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                (y_test == "높음").astype(int), dummy_high_probability
            ),
            "교차검증 F1(macro)": dummy_cv_scores.mean(),
            "교차검증 표준편차": dummy_cv_scores.std(),
        }
    )

    for model_name, estimator in model_definitions.items():
        pipeline = build_model(estimator, preprocessor)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cross_validator, scoring="f1_macro"
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        classes = list(pipeline.classes_)
        high_probability = probabilities[:, classes.index("높음")]

        trained_models[model_name] = pipeline
        comparison_rows.append(
            {
                "모델": model_name,
                "종류": "머신러닝",
                "정확도": accuracy_score(y_test, predictions),
                "균형 정확도": balanced_accuracy_score(y_test, predictions),
                "정밀도(높음)": precision_score(
                    y_test, predictions, pos_label="높음", zero_division=0
                ),
                "재현율(높음)": recall_score(
                    y_test, predictions, pos_label="높음", zero_division=0
                ),
                "F1(높음)": f1_score(
                    y_test, predictions, pos_label="높음", zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    (y_test == "높음").astype(int), high_probability
                ),
                "교차검증 F1(macro)": cv_scores.mean(),
                "교차검증 표준편차": cv_scores.std(),
            }
        )

results = pd.DataFrame(comparison_rows)
ml_results = (
    results[results["종류"] == "머신러닝"]
    .sort_values("교차검증 F1(macro)", ascending=False)
    .reset_index(drop=True)
)
best_model_name = ml_results.loc[0, "모델"]
best_model = trained_models[best_model_name]
best_test_predictions = best_model.predict(X_test)
best_test_probabilities = best_model.predict_proba(X_test)
best_classes = list(best_model.classes_)

# 요약 정보
counts = data["성취도"].value_counts()
high_count = int(counts.get("높음", 0))
low_count = int(counts.get("낮음", 0))


# ============================================================
# 6. 상단 요약 대시보드
# ============================================================
st.subheader("프로젝트 분석 현황")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("분석 학생 수", f"{len(data):,}명")
metric2.metric("높은 성취도 그룹", f"{high_count:,}명")
metric3.metric("낮은 성취도 그룹", f"{low_count:,}명")
metric4.metric("교차검증 기준 우수 모델", best_model_name)

st.markdown(
    '<p class="muted">학습용 80%, 평가용 20%로 분리했습니다. '
    '모델 선택은 학습 데이터의 5겹 교차검증 F1(macro) 평균을 기준으로 합니다.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# 7. 탭 구성
# ============================================================
tab_data, tab_stats, tab_models, tab_importance, tab_predict = st.tabs(
    [
        "① 데이터 탐색",
        "② 통계 분석",
        "③ 모델 성능 비교",
        "④ 변수 중요도",
        "⑤ 성취도 예측",
    ]
)


# ============================================================
# 탭 1. 데이터 탐색
# ============================================================
with tab_data:
    st.subheader("데이터 구성")
    left, right = st.columns([1.1, 0.9])

    with left:
        variable_table = pd.DataFrame(
            {
                "변수": [
                    "studytime", "absences", "failures",
                    "schoolsup", "famsup", "G3"
                ],
                "설명": [
                    "주간 학습 시간 수준(1~4)",
                    "결석 횟수",
                    "과거 낙제 횟수",
                    "학교의 추가 교육 지원 여부",
                    "가족의 교육 지원 여부",
                    "최종 수학 성적(0~20점)",
                ],
                "역할": ["예측 변수"] * 5 + ["성취도 분류 기준"],
            }
        )
        st.dataframe(variable_table, hide_index=True, use_container_width=True)

    with right:
        group_chart = px.pie(
            data,
            names="성취도",
            hole=0.58,
            title="성취도 그룹 구성 비율",
            color="성취도",
            color_discrete_map={"높음": "#2878B5", "낮음": "#E6A04B"},
        )
        group_chart.update_layout(margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(group_chart, use_container_width=True)

    st.divider()
    st.subheader("학습 습관과 최종 성적의 관계")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        study_plot_data = data.copy()
        study_plot_data["학습 시간 수준"] = study_plot_data["studytime"].map(
            {1: "1단계", 2: "2단계", 3: "3단계", 4: "4단계"}
        )
        study_chart = px.box(
            study_plot_data,
            x="학습 시간 수준",
            y="G3",
            color="학습 시간 수준",
            category_orders={
                "학습 시간 수준": ["1단계", "2단계", "3단계", "4단계"]
            },
            title="학습 시간 수준별 최종 수학 성적 분포",
        )
        study_chart.update_layout(
            xaxis_title="주간 학습 시간 수준",
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
            color_discrete_map={"높음": "#2878B5", "낮음": "#E6A04B"},
        )
        absence_chart.update_layout(
            xaxis_title="결석 횟수",
            yaxis_title="최종 성적 G3",
        )
        st.plotly_chart(absence_chart, use_container_width=True)

    st.subheader("성취도 그룹별 평균 학습 습관")
    group_summary = (
        data.groupby("성취도")[numeric_features]
        .mean()
        .round(2)
        .rename(
            columns={
                "studytime": "평균 학습 시간 수준",
                "absences": "평균 결석 횟수",
                "failures": "평균 과거 낙제 횟수",
            }
        )
    )
    st.dataframe(group_summary, use_container_width=True)

    st.caption(
        f"원본 {original_row_count}행 중 G3 결측 {original_row_count - len(data)}행은 제외했습니다. "
        "예측 변수 결측값은 모델 파이프라인에서 대체합니다."
    )

    with st.expander("원본 데이터 일부 확인"):
        st.dataframe(raw.head(20), use_container_width=True)


# ============================================================
# 탭 2. 통계 분석
# ============================================================
with tab_stats:
    st.subheader("학습 습관과 성취도 그룹의 통계적 관계")
    st.write(
        "수치형·순서형 변수는 맨-휘트니 U 검정, 범주형 변수는 "
        "카이제곱 독립성 검정으로 두 그룹의 차이를 살펴봅니다."
    )

    statistical_rows = []

    for feature in numeric_features:
        high_values = data.loc[data["성취도"] == "높음", feature].dropna()
        low_values = data.loc[data["성취도"] == "낮음", feature].dropna()

        if len(high_values) and len(low_values):
            test_result = mannwhitneyu(
                high_values, low_values, alternative="two-sided"
            )
            pair_count = len(high_values) * len(low_values)
            rank_biserial = (
                abs(1 - (2 * test_result.statistic / pair_count))
                if pair_count else np.nan
            )
            statistical_rows.append(
                {
                    "변수": feature,
                    "검정 방법": "맨-휘트니 U",
                    "높음 그룹 중앙값": high_values.median(),
                    "낮음 그룹 중앙값": low_values.median(),
                    "효과 크기": rank_biserial,
                    "p값": test_result.pvalue,
                }
            )

    for feature in categorical_features:
        contingency = pd.crosstab(data[feature], data["성취도"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2_result = chi2_contingency(contingency)
            expected = chi2_result[3]
            if contingency.shape == (2, 2) and (expected < 5).any():
                _, p_value = fisher_exact(contingency.to_numpy())
                test_name = "Fisher 정확 검정"
            else:
                p_value = chi2_result[1]
                test_name = "카이제곱 독립성"

            statistical_rows.append(
                {
                    "변수": feature,
                    "검정 방법": test_name,
                    "높음 그룹 중앙값": np.nan,
                    "낮음 그룹 중앙값": np.nan,
                    "효과 크기": cramers_v(contingency),
                    "p값": p_value,
                }
            )

    stats_df = pd.DataFrame(statistical_rows)
    stats_df["BH 보정 p값"] = benjamini_hochberg(stats_df["p값"].values)
    stats_df["검정 결과"] = np.where(
        stats_df["BH 보정 p값"] < 0.05,
        "보정 후 0.05 미만",
        "보정 후 0.05 이상",
    )

    stats_display = stats_df.copy()
    stats_display["효과 크기"] = stats_display["효과 크기"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "-"
    )
    stats_display["p값"] = stats_display["p값"].map(lambda value: f"{value:.4f}")
    stats_display["BH 보정 p값"] = stats_display["BH 보정 p값"].map(
        lambda value: f"{value:.4f}"
    )
    st.dataframe(stats_display, hide_index=True, use_container_width=True)

    st.caption(
        "효과 크기는 맨-휘트니 검정의 순위-이연 상관계수 절댓값 또는 "
        "카이제곱 검정의 Cramer's V입니다. 여러 검정의 p값은 Benjamini-Hochberg 방식으로 보정했습니다."
    )

    st.subheader("학습 습관 변수와 최종 성적의 상관관계")
    correlation_rows = []
    for feature in numeric_features:
        valid = data[[feature, "G3"]].dropna()
        if len(valid) >= 3 and valid[feature].nunique() > 1:
            rho, p_value = spearmanr(valid[feature], valid["G3"])
        else:
            rho, p_value = np.nan, np.nan

        correlation_rows.append(
            {"변수": feature, "Spearman 상관계수": rho, "p값": p_value}
        )

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_display = correlation_df.copy()
    correlation_display["Spearman 상관계수"] = (
        correlation_display["Spearman 상관계수"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else "-"
        )
    )
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
        title="학습 습관 변수와 G3의 Spearman 상관계수",
    )
    correlation_chart.update_layout(yaxis_title="Spearman 상관계수 (rho)")
    st.plotly_chart(correlation_chart, use_container_width=True)

    st.warning(
        "통계적 유의성은 변수 사이의 관계에 대한 근거이지 인과관계의 증거가 아닙니다. "
        "표본 수와 데이터 수집 환경의 한계도 고려해야 합니다."
    )


# ============================================================
# 탭 3. 모델 성능 비교
# ============================================================
with tab_models:
    st.subheader("기준 모델과 머신러닝 모델 비교")
    st.write(
        "기준 모델은 항상 학습 데이터에서 가장 많은 그룹만 예측합니다. "
        "머신러닝 모델의 성능이 이 기준보다 나은지 확인할 수 있습니다."
    )

    result_columns = [
        "모델", "종류", "정확도", "균형 정확도", "정밀도(높음)",
        "재현율(높음)", "F1(높음)", "ROC-AUC", "교차검증 F1(macro)",
        "교차검증 표준편차"
    ]
    formatted_results = results[result_columns].copy()
    for column in result_columns[2:]:
        formatted_results[column] = formatted_results[column].map(
            lambda value: f"{value:.1%}"
        )
    st.dataframe(formatted_results, hide_index=True, use_container_width=True)

    comparison_chart = go.Figure()
    for metric, label in [
        ("정확도", "정확도"),
        ("균형 정확도", "균형 정확도"),
        ("F1(높음)", "F1(높음)"),
        ("교차검증 F1(macro)", "교차검증 F1(macro)"),
    ]:
        comparison_chart.add_trace(
            go.Bar(name=label, x=results["모델"], y=results[metric])
        )
    comparison_chart.update_layout(
        title="모델별 성능 지표 비교",
        xaxis_title="모델",
        yaxis_title="점수",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        legend_title="평가 지표",
    )
    st.plotly_chart(comparison_chart, use_container_width=True)

    st.success(
        f"머신러닝 모델 중 교차검증 F1(macro) 평균이 가장 높은 모델은 "
        f"'{best_model_name}'입니다."
    )

    dummy_cv = float(
        results.loc[
            results["모델"] == "기준 모델(Dummy)", "교차검증 F1(macro)"
        ].iloc[0]
    )
    selected_cv = float(
        results.loc[
            results["모델"] == best_model_name, "교차검증 F1(macro)"
        ].iloc[0]
    )
    difference = selected_cv - dummy_cv

    if difference > 0:
        st.write(
            f"선택된 모델의 교차검증 F1(macro)은 기준 모델보다 "
            f"{difference:.3f} 높았습니다."
        )
    else:
        st.warning(
            "현재 설정에서는 선택된 모델의 교차검증 F1(macro)이 기준 모델보다 높지 않습니다. "
            "예측력이 충분하다고 단정하지 마세요."
        )

    st.subheader(f"{best_model_name}의 평가 데이터 혼동 행렬")
    matrix = confusion_matrix(
        y_test, best_test_predictions, labels=["낮음", "높음"]
    )
    matrix_chart = px.imshow(
        matrix,
        text_auto=True,
        x=["예측: 낮음", "예측: 높음"],
        y=["실제: 낮음", "실제: 높음"],
        color_continuous_scale="Blues",
        title="실제 성취도와 모델 예측 비교",
    )
    matrix_chart.update_layout(xaxis_title="모델 예측", yaxis_title="실제 정답")
    st.plotly_chart(matrix_chart, use_container_width=True)

    roc_chart = go.Figure()
    y_test_binary = (y_test == "높음").astype(int)
    for model_name, model in trained_models.items():
        model_probabilities = model.predict_proba(X_test)
        model_classes = list(model.classes_)
        positive_probabilities = model_probabilities[:, model_classes.index("높음")]
        fpr, tpr, _ = roc_curve(y_test_binary, positive_probabilities)
        auc_value = roc_auc_score(y_test_binary, positive_probabilities)
        roc_chart.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc_value:.3f})",
            )
        )
    roc_chart.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), name="무작위 기준"
        )
    )
    roc_chart.update_layout(
        title="모델별 ROC 곡선",
        xaxis_title="위양성률(False Positive Rate)",
        yaxis_title="재현율(True Positive Rate)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(roc_chart, use_container_width=True)

    result_csv = results.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "모델 성능 비교 결과 다운로드",
        data=result_csv,
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    st.caption(
        "최종 성능은 학습에 사용하지 않은 20% 평가 데이터에서 측정했습니다. "
        "모델 선택은 학습 데이터 내부의 5겹 교차검증 평균을 기준으로 합니다."
    )


# ============================================================
# 탭 4. 변수 중요도
# ============================================================
with tab_importance:
    st.subheader("모델의 예측에 중요한 변수는 무엇일까?")
    st.write(
        "순열 중요도(Permutation Importance)는 평가 데이터에서 변수 하나의 값을 섞었을 때 "
        "모델의 F1(macro) 점수가 얼마나 달라지는지 측정합니다."
    )

    if st.button("변수 중요도 계산하기"):
        with st.spinner("평가 데이터를 이용해 변수 중요도를 계산하고 있습니다..."):
            importance_result = permutation_importance(
                best_model,
                X_test,
                y_test,
                n_repeats=15,
                random_state=42,
                scoring="f1_macro",
            )
        variable_names = {
            "studytime": "주간 학습 시간",
            "absences": "결석 횟수",
            "failures": "과거 낙제 횟수",
            "schoolsup": "학교 교육 지원",
            "famsup": "가족 교육 지원",
        }
        importance_df = pd.DataFrame(
            {
                "변수": [variable_names[column] for column in feature_columns],
                "평균 중요도": importance_result.importances_mean,
                "표준편차": importance_result.importances_std,
            }
        ).sort_values("평균 중요도", ascending=True)
        st.session_state["importance_df"] = importance_df
        st.session_state["importance_model"] = best_model_name
        st.session_state["importance_threshold"] = threshold

    if (
        "importance_df" in st.session_state
        and st.session_state.get("importance_model") == best_model_name
        and st.session_state.get("importance_threshold") == threshold
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
            xaxis_title="F1(macro) 점수 변화량",
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
# 탭 5. 개인별 성취도 예측
# ============================================================
with tab_predict:
    st.subheader("학습 습관을 입력해 성취도 그룹 예측하기")
    st.write(
        f"교차검증 기준으로 선택된 **{best_model_name}** 모델을 사용합니다. "
        "아래 항목을 입력한 뒤 예측 버튼을 눌러 주세요."
    )

    with st.form("student_prediction_form"):
        input_left, input_right = st.columns(2)
        with input_left:
            studytime = st.select_slider(
                "주간 학습 시간 수준",
                options=[1, 2, 3, 4],
                value=2,
                help="1: 2시간 미만 / 2: 2~5시간 / 3: 5~10시간 / 4: 10시간 초과",
            )
            absences = st.number_input(
                "결석 횟수", min_value=0, max_value=100, value=2, step=1
            )
            failures = st.number_input(
                "과거 낙제 횟수", min_value=0, max_value=4, value=0, step=1
            )
        with input_right:
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
        submitted = st.form_submit_button(
            "수학 성취도 예측하기", use_container_width=True
        )

    if submitted:
        new_student = pd.DataFrame(
            [{
                "studytime": studytime,
                "absences": absences,
                "failures": failures,
                "schoolsup": schoolsup,
                "famsup": famsup,
            }]
        )
        predicted_group = best_model.predict(new_student)[0]
        probabilities = best_model.predict_proba(new_student)[0]
        class_names = list(best_model.classes_)
        low_probability = probabilities[class_names.index("낮음")]
        high_probability = probabilities[class_names.index("높음")]

        st.divider()
        st.subheader("예측 결과")
        if predicted_group == "높음":
            st.success("예측 결과: 높은 성취도 그룹")
        else:
            st.warning("예측 결과: 낮은 성취도 그룹")

        probability_left, probability_right = st.columns(2)
        probability_left.metric("높음 그룹 예측 확률", f"{high_probability:.1%}")
        probability_right.metric("낮음 그룹 예측 확률", f"{low_probability:.1%}")

        probability_data = pd.DataFrame(
            {
                "성취도 그룹": ["높음", "낮음"],
                "예측 확률": [high_probability, low_probability],
            }
        )
        probability_chart = px.bar(
            probability_data,
            x="성취도 그룹",
            y="예측 확률",
            color="성취도 그룹",
            text=probability_data["예측 확률"].map(lambda value: f"{value:.1%}"),
            color_discrete_map={"높음": "#2878B5", "낮음": "#E6A04B"},
            title="성취도 그룹별 모델 예측 확률",
        )
        probability_chart.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(probability_chart, use_container_width=True)
        st.caption(
            "확률은 모델이 학습한 패턴을 바탕으로 계산한 추정값이며 실제 성적을 보장하지 않습니다. "
            "입력하지 않은 환경적 요인과 개인차도 성취도에 영향을 줄 수 있습니다."
        )


# ============================================================
# 8. 프로젝트의 분석 기준과 한계
# ============================================================
st.divider()
with st.expander("프로젝트의 분석 기준과 한계"):
    st.markdown(
        f"""
        **분석 기준**
        - 최종 수학 성적 G3가 **{threshold}점 이상**이면 '높음', 미만이면 '낮음'으로 분류했습니다.
        - 이 기준은 본 프로젝트에서 정한 이진 분류 기준이며 공식 성취도 등급은 아닙니다.
        - 예측 변수는 studytime, absences, failures, schoolsup, famsup의 다섯 가지입니다.
        - G1과 G2는 최종 성적과 밀접한 중간 성적이므로 예측 변수에서 제외했습니다.
        - 데이터 분할은 학습용 80%, 평가용 20%이며 random_state는 42입니다.

        **분석의 한계**
        - 데이터는 특정 학교 표본을 바탕으로 하므로 모든 학생에게 일반화할 수 없습니다.
        - 모델 성능은 표본 분할과 평가 지표에 따라 달라질 수 있습니다.
        - 통계적 관계와 변수 중요도는 인과관계를 증명하지 않습니다.
        - 모델의 예측 확률은 실제 성적을 보장하거나 확정하는 값이 아닙니다.
        """
    )

st.caption("MATH INSIGHT | UCI Student Performance 기반 데이터 과학 프로젝트")
