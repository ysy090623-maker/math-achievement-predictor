import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import kruskal, chi2_contingency, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
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

st.markdown("""
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
""", unsafe_allow_html=True)

st.title("📘 MATH INSIGHT")
st.markdown("""
<div class="hero">
  <h2>학생의 학습 습관과 수학 성취도 분석</h2>
  <p>데이터 탐색 · 통계적 검정 · 머신러닝 모델 비교 · 변수 중요도 · 성취도 예측</p>
</div>
""", unsafe_allow_html=True)
st.caption(
    "UCI Student Performance 데이터셋 · 수학 과목(student-mat.csv) · "
    "프로젝트에서 설정한 3단계 성취도 기준"
)


# ============================================================
# 2. 사이드바 및 데이터
# ============================================================
with st.sidebar:
    st.header("분석 설정")
    st.markdown("**성취도 분류 기준 (G3)**")
    st.info("낮음: 0~9점  |  중간: 10~14점  |  높음: 15~20점")
    st.caption("위 기준은 이 프로젝트에서 정한 분석 기준이며 공식 등급은 아닙니다.")
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
missing_columns = [c for c in required_columns if c not in raw.columns]
if missing_columns:
    st.error(f"CSV에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

data = raw[required_columns].copy()
for c in ["studytime", "absences", "failures", "G3"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

original_row_count = len(data)
data = data.dropna(subset=["G3"]).copy()
data = data[data["G3"].between(0, 20)].copy()

# 3단계 성취도: 프로젝트에서 정한 기준
grade_order = ["낮음", "중간", "높음"]
data["성취도"] = pd.cut(
    data["G3"],
    bins=[-0.01, 9.99, 14.99, 20],
    labels=grade_order,
    include_lowest=True
).astype(str)

feature_columns = ["studytime", "absences", "failures", "schoolsup", "famsup"]
numeric_features = ["studytime", "absences", "failures"]
categorical_features = ["schoolsup", "famsup"]
X = data[feature_columns]
y = data["성취도"]

if len(data) < 40 or y.nunique() != 3 or y.value_counts().min() < 5:
    st.error(
        "세 성취도 그룹 중 데이터가 너무 적은 그룹이 있습니다. "
        "현재 데이터로는 안정적인 3단계 모델 학습이 어렵습니다."
    )
    st.stop()


# ============================================================
# 3. 분석 및 모델 함수
# ============================================================
def cramers_v(table):
    chi2_value, _, _, _ = chi2_contingency(table)
    total = table.to_numpy().sum()
    rows, cols = table.shape
    denom = total * min(rows - 1, cols - 1)
    return float(np.sqrt(chi2_value / denom)) if denom > 0 else np.nan


def build_model(estimator, preprocessor):
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", estimator),
    ])


# ============================================================
# 4. 학습/평가 데이터 분리 및 전처리
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])


# ============================================================
# 5. 기준 모델 및 머신러닝 모델 비교 (3단계 분류)
# ============================================================
model_definitions = {
    "로지스틱 회귀": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "의사결정나무": DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=42
    ),
    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=250, max_depth=5, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trained_models = {}
comparison_rows = []

with st.spinner("기준 모델과 머신러닝 모델을 학습하고 검증하고 있습니다..."):
    dummy_pipeline = build_model(DummyClassifier(strategy="most_frequent"), preprocessor)
    dummy_cv = cross_val_score(
        dummy_pipeline, X_train, y_train, cv=cross_validator, scoring="f1_macro"
    )
    dummy_pipeline.fit(X_train, y_train)
    dummy_pred = dummy_pipeline.predict(X_test)

    comparison_rows.append({
        "모델": "기준 모델(Dummy)",
        "종류": "비교 기준",
        "정확도": accuracy_score(y_test, dummy_pred),
        "균형 정확도": balanced_accuracy_score(y_test, dummy_pred),
        "정밀도(macro)": precision_score(y_test, dummy_pred, labels=grade_order,
                                           average="macro", zero_division=0),
        "재현율(macro)": recall_score(y_test, dummy_pred, labels=grade_order,
                                       average="macro", zero_division=0),
        "F1(macro)": f1_score(y_test, dummy_pred, labels=grade_order,
                               average="macro", zero_division=0),
        "교차검증 F1(macro)": dummy_cv.mean(),
        "교차검증 표준편차": dummy_cv.std(),
    })

    for name, estimator in model_definitions.items():
        pipeline = build_model(estimator, preprocessor)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cross_validator, scoring="f1_macro"
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        trained_models[name] = pipeline
        comparison_rows.append({
            "모델": name,
            "종류": "머신러닝",
            "정확도": accuracy_score(y_test, pred),
            "균형 정확도": balanced_accuracy_score(y_test, pred),
            "정밀도(macro)": precision_score(y_test, pred, labels=grade_order,
                                              average="macro", zero_division=0),
            "재현율(macro)": recall_score(y_test, pred, labels=grade_order,
                                          average="macro", zero_division=0),
            "F1(macro)": f1_score(y_test, pred, labels=grade_order,
                                   average="macro", zero_division=0),
            "교차검증 F1(macro)": cv_scores.mean(),
            "교차검증 표준편차": cv_scores.std(),
        })

results = pd.DataFrame(comparison_rows)
ml_results = results[results["종류"] == "머신러닝"].sort_values(
    "교차검증 F1(macro)", ascending=False
).reset_index(drop=True)
best_model_name = ml_results.loc[0, "모델"]
best_model = trained_models[best_model_name]
best_test_predictions = best_model.predict(X_test)

# 예측 확률을 점수로 환산할 때 사용할 학습 데이터의 그룹별 평균 G3
group_score_means = (
    data.loc[X_train.index]
    .groupby("성취도", observed=False)["G3"]
    .mean()
    .reindex(grade_order)
)
# 각 그룹 평균을 확률로 가중한 참고용 점수 추정치
counts = data["성취도"].value_counts().reindex(grade_order, fill_value=0)
low_count, middle_count, high_count = [int(counts[g]) for g in grade_order]


# ============================================================
# 6. 상단 요약 대시보드
# ============================================================
st.subheader("프로젝트 분석 현황")
m1, m2, m3, m4 = st.columns(4)
m1.metric("분석 학생 수", f"{len(data):,}명")
m2.metric("낮은 성취도", f"{low_count:,}명")
m3.metric("중간 성취도", f"{middle_count:,}명")
m4.metric("높은 성취도", f"{high_count:,}명")
st.markdown(
    '<p class="muted">학습용 80%, 평가용 20%로 분리했습니다. '
    '모델 선택은 학습 데이터의 5겹 교차검증 F1(macro) 평균을 기준으로 합니다.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# 7. 탭 구성
# ============================================================
tab_data, tab_stats, tab_models, tab_importance, tab_predict = st.tabs([
    "① 데이터 탐색", "② 통계 분석", "③ 모델 성능 비교",
    "④ 변수 중요도", "⑤ 성취도 예측"
])


# ============================================================
# 탭 1. 데이터 탐색
# ============================================================
with tab_data:
    st.subheader("데이터 구성")
    left, right = st.columns([1.1, 0.9])
    with left:
        variable_table = pd.DataFrame({
            "변수": ["studytime", "absences", "failures", "schoolsup", "famsup", "G3"],
            "설명": [
                "주간 학습 시간 수준(1~4)", "결석 횟수", "과거 낙제 횟수",
                "학교의 추가 교육 지원 여부", "가족의 교육 지원 여부",
                "최종 수학 성적(0~20점)"
            ],
            "역할": ["예측 변수"] * 5 + ["성취도 구분 및 환산 기준"]
        })
        st.dataframe(variable_table, hide_index=True, use_container_width=True)

    with right:
        pie = px.pie(
            data, names="성취도", hole=0.58, title="성취도 그룹 구성 비율",
            category_orders={"성취도": grade_order},
            color="성취도",
            color_discrete_map={"낮음": "#E6A04B", "중간": "#7B91A8", "높음": "#2878B5"}
        )
        pie.update_layout(margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(pie, use_container_width=True)

    st.divider()
    st.subheader("학습 습관과 최종 성적의 관계")
    c1, c2 = st.columns(2)
    with c1:
        study_data = data.copy()
        study_data["학습 시간 수준"] = study_data["studytime"].map({
            1: "1단계", 2: "2단계", 3: "3단계", 4: "4단계"
        })
        fig = px.box(
            study_data, x="학습 시간 수준", y="G3", color="성취도",
            category_orders={"성취도": grade_order},
            title="학습 시간 수준별 최종 수학 성적 분포",
            color_discrete_map={"낮음": "#E6A04B", "중간": "#7B91A8", "높음": "#2878B5"}
        )
        fig.update_layout(xaxis_title="주간 학습 시간 수준", yaxis_title="최종 성적 G3")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            data, x="absences", y="G3", color="성취도", opacity=0.7,
            title="결석 횟수와 최종 수학 성적",
            color_discrete_map={"낮음": "#E6A04B", "중간": "#7B91A8", "높음": "#2878B5"}
        )
        fig.update_layout(xaxis_title="결석 횟수", yaxis_title="최종 성적 G3")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("성취도 그룹별 평균 학습 습관")
    summary = data.groupby("성취도", observed=False)[numeric_features].mean().round(2)
    summary = summary.rename(columns={
        "studytime": "평균 학습 시간 수준",
        "absences": "평균 결석 횟수",
        "failures": "평균 과거 낙제 횟수"
    })
    st.dataframe(summary, use_container_width=True)

    st.subheader("G3 점수의 100점 만점 환산")
    st.write("원점수 G3(20점 만점)에 5를 곱해 100점 만점으로 비례 환산합니다.")
    st.metric("환산 공식", "G3 × 5")
    st.caption("이는 점수의 비례 환산이며 한국 학교의 실제 내신 점수나 등급을 의미하지 않습니다.")

    st.caption(
        f"원본 {original_row_count}행 중 G3 결측 또는 범위 밖 데이터 "
        f"{original_row_count - len(data)}행은 제외했습니다. 예측 변수 결측값은 모델에서 대체합니다."
    )
    with st.expander("원본 데이터 일부 확인"):
        st.dataframe(raw.head(20), use_container_width=True)


# ============================================================
# 탭 2. 통계 분석 (3개 그룹)
# ============================================================
with tab_stats:
    st.subheader("학습 습관과 성취도 그룹의 통계적 관계")
    st.write(
        "수치형·순서형 변수는 크루스칼-왈리스 검정으로 세 그룹의 차이를 살펴보고, "
        "범주형 변수는 카이제곱 독립성 검정을 사용합니다."
    )
    rows = []
    for feature in numeric_features:
        groups = [
            data.loc[data["성취도"] == g, feature].dropna()
            for g in grade_order
        ]
        if all(len(v) > 0 for v in groups):
            test = kruskal(*groups)
            rows.append({
                "변수": feature,
                "검정 방법": "크루스칼-왈리스",
                "낮음 그룹 중앙값": groups[0].median(),
                "중간 그룹 중앙값": groups[1].median(),
                "높음 그룹 중앙값": groups[2].median(),
                "검정 통계량": test.statistic,
                "p값": test.pvalue
            })

    for feature in categorical_features:
        table = pd.crosstab(data[feature], data["성취도"])
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            test = chi2_contingency(table)
            rows.append({
                "변수": feature,
                "검정 방법": "카이제곱 독립성",
                "낮음 그룹 중앙값": np.nan,
                "중간 그룹 중앙값": np.nan,
                "높음 그룹 중앙값": np.nan,
                "검정 통계량": test.statistic,
                "p값": test.pvalue
            })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        # BH 다중검정 보정
        p = stats_df["p값"].to_numpy()
        order = np.argsort(p)
        ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(len(p))
        adjusted[order] = np.clip(ranked, 0, 1)
        stats_df["BH 보정 p값"] = adjusted
        stats_df["검정 결과"] = np.where(
            stats_df["BH 보정 p값"] < 0.05, "보정 후 0.05 미만", "보정 후 0.05 이상"
        )
        st.dataframe(stats_df.round(4), hide_index=True, use_container_width=True)
    else:
        st.info("통계 검정을 수행할 수 있는 데이터가 부족합니다.")

    st.subheader("학습 습관 변수와 최종 성적의 상관관계")
    corr_rows = []
    for feature in numeric_features:
        valid = data[[feature, "G3"]].dropna()
        if len(valid) >= 3 and valid[feature].nunique() > 1:
            rho, pvalue = spearmanr(valid[feature], valid["G3"])
        else:
            rho, pvalue = np.nan, np.nan
        corr_rows.append({"변수": feature, "Spearman 상관계수": rho, "p값": pvalue})
    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(corr_df.round(4), hide_index=True, use_container_width=True)

    fig = px.bar(
        corr_df, x="변수", y="Spearman 상관계수",
        color="Spearman 상관계수", color_continuous_scale="RdBu",
        range_color=[-1, 1], title="학습 습관 변수와 G3의 Spearman 상관계수"
    )
    fig.update_layout(yaxis_title="Spearman 상관계수 (rho)")
    st.plotly_chart(fig, use_container_width=True)
    st.warning(
        "통계적 유의성은 변수 사이의 관계에 대한 근거이지 인과관계의 증거가 아닙니다. "
        "표본 수와 데이터 수집 환경의 한계도 고려해야 합니다."
    )


# ============================================================
# 탭 3. 모델 성능 비교 (다중분류)
# ============================================================
with tab_models:
    st.subheader("기준 모델과 머신러닝 모델 비교")
    st.write(
        "세 성취도 그룹(낮음·중간·높음)을 분류합니다. "
        "F1(macro)은 세 그룹의 F1 점수를 동일한 비중으로 평균한 지표입니다."
    )
    display_cols = [
        "모델", "종류", "정확도", "균형 정확도", "정밀도(macro)",
        "재현율(macro)", "F1(macro)", "교차검증 F1(macro)", "교차검증 표준편차"
    ]
    shown = results[display_cols].copy()
    for col in display_cols[2:]:
        shown[col] = shown[col].map(lambda v: f"{v:.1%}")
    st.dataframe(shown, hide_index=True, use_container_width=True)

    fig = go.Figure()
    for metric, label in [
        ("정확도", "정확도"),
        ("균형 정확도", "균형 정확도"),
        ("F1(macro)", "F1(macro)"),
        ("교차검증 F1(macro)", "교차검증 F1(macro)")
    ]:
        fig.add_trace(go.Bar(name=label, x=results["모델"], y=results[metric]))
    fig.update_layout(
        title="모델별 성능 지표 비교", xaxis_title="모델",
        yaxis_title="점수", yaxis=dict(range=[0, 1]), barmode="group",
        legend_title="평가 지표"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"머신러닝 모델 중 교차검증 F1(macro) 평균이 가장 높은 모델은 '{best_model_name}'입니다.")
    dummy_score = float(results.loc[
        results["모델"] == "기준 모델(Dummy)", "교차검증 F1(macro)"
    ].iloc[0])
    best_score = float(results.loc[
        results["모델"] == best_model_name, "교차검증 F1(macro)"
    ].iloc[0])
    if best_score > dummy_score:
        st.write(f"선택된 모델은 기준 모델보다 교차검증 F1(macro)이 {best_score-dummy_score:.3f} 높았습니다.")
    else:
        st.warning("선택된 모델의 교차검증 F1(macro)이 기준 모델보다 높지 않습니다. 예측력이 충분하다고 단정하지 마세요.")

    st.subheader(f"{best_model_name}의 평가 데이터 혼동 행렬")
    matrix = confusion_matrix(y_test, best_test_predictions, labels=grade_order)
    fig = px.imshow(
        matrix, text_auto=True,
        x=[f"예측: {g}" for g in grade_order],
        y=[f"실제: {g}" for g in grade_order],
        color_continuous_scale="Blues", title="실제 성취도와 모델 예측 비교"
    )
    fig.update_layout(xaxis_title="모델 예측", yaxis_title="실제 정답")
    st.plotly_chart(fig, use_container_width=True)

    csv = results.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "모델 성능 비교 결과 다운로드", data=csv,
        file_name="model_comparison_results.csv", mime="text/csv"
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
        "순열 중요도는 평가 데이터에서 변수 하나의 값을 섞었을 때 "
        "모델의 F1(macro) 점수가 얼마나 달라지는지 측정합니다."
    )
    if st.button("변수 중요도 계산하기"):
        with st.spinner("평가 데이터를 이용해 변수 중요도를 계산하고 있습니다..."):
            importance = permutation_importance(
                best_model, X_test, y_test, n_repeats=15,
                random_state=42, scoring="f1_macro"
            )
        names = {
            "studytime": "주간 학습 시간",
            "absences": "결석 횟수",
            "failures": "과거 낙제 횟수",
            "schoolsup": "학교 교육 지원",
            "famsup": "가족 교육 지원"
        }
        importance_df = pd.DataFrame({
            "변수": [names[c] for c in feature_columns],
            "평균 중요도": importance.importances_mean,
            "표준편차": importance.importances_std
        }).sort_values("평균 중요도", ascending=True)
        st.session_state["importance_df"] = importance_df
        st.session_state["importance_model"] = best_model_name

    if st.session_state.get("importance_model") == best_model_name and "importance_df" in st.session_state:
        importance_df = st.session_state["importance_df"]
        fig = px.bar(
            importance_df, x="평균 중요도", y="변수", orientation="h",
            error_x="표준편차", title=f"{best_model_name}의 변수별 순열 중요도"
        )
        fig.update_layout(xaxis_title="F1(macro) 점수 변화량", yaxis_title="예측 변수")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df.round(4), hide_index=True, use_container_width=True)
    else:
        st.info("'변수 중요도 계산하기' 버튼을 누르면 분석 결과가 표시됩니다.")

    st.warning(
        "변수 중요도는 이 데이터와 선택된 모델의 예측에 대한 상대적 기여도입니다. "
        "해당 변수가 성적을 직접 변화시킨다는 인과관계를 의미하지 않습니다."
    )


# ============================================================
# 탭 5. 개인별 성취도 예측 + 참고용 100점 환산
# ============================================================
with tab_predict:
    st.subheader("학습 습관을 입력해 성취도 예측하기")
    st.write(
        f"교차검증 기준으로 선택된 **{best_model_name}** 모델을 사용합니다. "
        "아래 항목을 입력한 뒤 예측 버튼을 눌러 주세요."
    )

    with st.form("student_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            studytime = st.select_slider(
                "주간 학습 시간 수준", options=[1, 2, 3, 4], value=2,
                format_func=lambda v: {
                    1: "1단계 (주 2시간 미만)",
                    2: "2단계 (주 2~5시간)",
                    3: "3단계 (주 5~10시간)",
                    4: "4단계 (주 10시간 초과)"
                }[v]
            )
            absences = st.number_input("결석 횟수", min_value=0, max_value=100, value=2, step=1)
            failures = st.number_input("과거 낙제 횟수", min_value=0, max_value=4, value=0, step=1)
        with col2:
            schoolsup = st.selectbox(
                "학교의 추가 교육 지원 여부", ["yes", "no"],
                format_func=lambda v: "예" if v == "yes" else "아니요"
            )
            famsup = st.selectbox(
                "가족의 교육 지원 여부", ["yes", "no"],
                format_func=lambda v: "예" if v == "yes" else "아니요"
            )
        submitted = st.form_submit_button("수학 성취도 예측하기", use_container_width=True)

    if submitted:
        new_student = pd.DataFrame([{
            "studytime": studytime,
            "absences": absences,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup
        }])
        predicted_group = best_model.predict(new_student)[0]
        probabilities = best_model.predict_proba(new_student)[0]
        classes = list(best_model.classes_)
        prob_by_group = {g: float(probabilities[classes.index(g)]) for g in grade_order}

        # 세 그룹의 학습 데이터 평균 G3에 각 그룹 예측 확률을 곱한 참고용 기대 점수
        estimated_g3 = sum(
            prob_by_group[g] * float(group_score_means[g]) for g in grade_order
        )
        estimated_100 = estimated_g3 * 5

        st.divider()
        st.subheader("예측 결과")
        if predicted_group == "높음":
            st.success(f"가장 가능성이 높은 성취도: {predicted_group}")
        elif predicted_group == "중간":
            st.info(f"가장 가능성이 높은 성취도: {predicted_group}")
        else:
            st.warning(f"가장 가능성이 높은 성취도: {predicted_group}")

        a, b = st.columns(2)
        a.metric("참고용 G3 점수 추정", f"{estimated_g3:.1f} / 20점")
        b.metric("100점 만점 환산 추정", f"{estimated_100:.1f} / 100점")

        st.caption(
            "※ 환산 점수는 모델이 예측한 세 그룹의 확률에 각 그룹의 학습 데이터 평균 G3를 가중해 계산한 참고용 추정치입니다. "
            "개별 학생의 정확한 점수를 직접 예측한 값이 아니며, 실제 성적이나 한국 학교의 내신 점수를 보장하지 않습니다."
        )

        prob_df = pd.DataFrame({
            "성취도 그룹": grade_order,
            "예측 확률": [prob_by_group[g] for g in grade_order]
        })
        fig = px.bar(
            prob_df, x="성취도 그룹", y="예측 확률",
            color="성취도 그룹", text=prob_df["예측 확률"].map(lambda v: f"{v:.1%}"),
            category_orders={"성취도 그룹": grade_order},
            color_discrete_map={"낮음": "#E6A04B", "중간": "#7B91A8", "높음": "#2878B5"},
            title="성취도 그룹별 모델 예측 확률"
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        prob_table = prob_df.copy()
        prob_table["예측 확률"] = prob_table["예측 확률"].map(lambda v: f"{v:.1%}")
        st.dataframe(prob_table, hide_index=True, use_container_width=True)

        st.caption(
            "예측 확률은 모델이 학습한 패턴을 바탕으로 계산한 추정값입니다. "
            "입력하지 않은 환경적 요인과 개인차도 성취도에 영향을 줄 수 있습니다."
        )


# ============================================================
# 8. 프로젝트의 분석 기준과 한계
# ============================================================
st.divider()
with st.expander("프로젝트의 분석 기준과 한계"):
    st.markdown("""
**분석 기준**
- 최종 수학 성적 G3를 낮음(0~9점), 중간(10~14점), 높음(15~20점)의 세 단계로 구분했습니다.
- 이 기준은 본 프로젝트에서 설정한 기준이며 공식 성취도 등급은 아닙니다.
- 예측 변수는 studytime, absences, failures, schoolsup, famsup의 다섯 가지입니다.
- G1과 G2는 최종 성적과 밀접한 중간 성적이므로 예측 변수에서 제외했습니다.
- 데이터는 학습용 80%, 평가용 20%로 분리하고, 학습 데이터 안에서 5겹 교차검증을 실시했습니다.
- 100점 환산은 G3에 5를 곱하는 비례 환산입니다. 개인별 결과의 환산 점수는 성취도별 예측 확률과 학습 데이터의 그룹 평균을 이용한 참고용 추정치입니다.

**분석의 한계**
- 데이터는 포르투갈 학교 학생들의 자료이므로 한국 학생에게 그대로 일반화하기 어렵습니다.
- 학교마다 시험 난이도와 평가 기준이 다를 수 있습니다.
- 성적에는 수업 참여도, 학습 방법, 개인별 이해도 등 분석에 포함하지 않은 요인도 영향을 줄 수 있습니다.
- 통계적 관계와 변수 중요도는 인과관계를 증명하지 않습니다.
- 모델의 예측 확률과 환산 점수는 실제 성적을 보장하거나 확정하는 값이 아닙니다.
""")

st.caption("MATH INSIGHT | UCI Student Performance 기반 데이터 과학 프로젝트")
