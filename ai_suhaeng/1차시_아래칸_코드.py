# ===== 분류: 센서 값으로 기계 고장 미리 알아채기 (예지보전) =====
# 위 칸에서 불러온 df(기계 고장 데이터)를 그대로 써요. 파일을 다시 읽지 않아요.
# [1] ~ [5] 제목 줄(주석과 print)은 지우지 마세요.

# [1] 전처리
print("[1] 전처리")
import numpy as np
from sklearn.model_selection import train_test_split

df.columns = df.columns.str.strip()
target = "Machine failure"                                   # 예측할 열 (0 정상, 1 고장)
leak = ["TWF", "HDF", "PWF", "OSF", "RNF"]                  # 고장 종류: 고장 뒤에야 아는 값 → 정답이 새므로 학습에서 뺌
data = df.drop(columns=["UDI", "Product ID"]).dropna().drop_duplicates()   # 번호·이름표 열은 고장과 관계없음
print(f"빈칸·중복 행을 빼서 {len(df)}행 → {len(data)}행")
fail_type = data[leak].copy()                               # 결과 해석에만 따로 보관
data = data.drop(columns=leak)
data["Type"] = data["Type"].map({"L": 0, "M": 1, "H": 2})   # 글자로 된 제품 등급을 숫자로
print("정답이 새는 열 뺌:", leak, "/ Type 을 L·M·H → 0·1·2 로 바꿈")
print(f"고장 {int(data[target].sum())}건 / {len(data)}건 → 고장 비율 {data[target].mean()*100:.1f}% (불균형 데이터)")

base_features = ["Type", "Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = data[base_features]
y = data[target]
X_train, X_test, y_train, y_test, ft_train, ft_test = train_test_split(
    X, y, fail_type, test_size=0.2, random_state=0, stratify=y)   # 훈련 80%, 테스트 20%, 고장 비율 똑같이
print(f"훈련 {len(X_train)}행(고장 {int(y_train.sum())}), 테스트 {len(X_test)}행(고장 {int(y_test.sum())})")

# [2] 모델 만들기
print("\n[2] 모델 만들기")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                               class_weight="balanced",     # 수가 적은 '고장'을 틀리면 벌점을 더 크게
                               random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
print("랜덤 포레스트(결정 트리 300개의 다수결, class_weight='balanced')를 훈련 데이터로 학습했습니다.")

# [3] 성능 평가
print("\n[3] 성능 평가")
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)

def report(name, y_true, p):
    print(f"{name}: 정확도 {accuracy_score(y_true, p):.3f} | 정밀도 {precision_score(y_true, p, zero_division=0):.3f}"
          f" | 재현율 {recall_score(y_true, p):.3f} | F1 {f1_score(y_true, p):.3f}")
    return f1_score(y_true, p)

report("비교용: 무조건 '정상'이라고 답하기", y_test, np.zeros(len(y_test), dtype=int))
pred = model.predict(X_test)
before = report("랜덤 포레스트(원래 특징 6개)", y_test, pred)
print("→ 고장이 3.4%뿐이라 정확도는 늘 높게 나와요. 실제 고장을 얼마나 잡았는지(재현율)와 F1로 판단해요.")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["정상", "고장"])
plt.title("혼동 행렬 (세로: 실제, 가로: 예측)")
plt.show()

# [4] 성능 개선 — 한 가지를 바꿔 바꾸기 전과 견주기
print("\n[4] 성능 개선")
print("바꾼 것: 고장 원리를 담은 새 특징 3개 추가 (온도차, 동력, 마모×토크)")

def add_features(t):
    t = t.copy()
    t["온도차 [K]"] = t["Process temperature [K]"] - t["Air temperature [K]"]          # 열이 안 빠지면 과열 고장
    t["동력 [W]"] = t["Torque [Nm]"] * t["Rotational speed [rpm]"] * 2 * np.pi / 60   # 너무 작거나 크면 전력 고장
    t["마모x토크"] = t["Tool wear [min]"] * t["Torque [Nm]"]                           # 닳은 공구에 큰 힘 → 과부하 고장
    return t

X_train2, X_test2 = add_features(X_train), add_features(X_test)
features = list(X_train2.columns)
model2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                random_state=0, n_jobs=-1)
model2.fit(X_train2, y_train)
pred2 = model2.predict(X_test2)
print("바꾸기 전 ↓")
report("  원래 특징 6개", y_test, pred)
print("바꾼 뒤 ↓")
after = report("  + 새 특징 3개", y_test, pred2)
print(f"F1: 바꾸기 전 {before:.3f} → 바꾼 뒤 {after:.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, pred2, display_labels=["정상", "고장"])
plt.title("개선 후 혼동 행렬 (세로: 실제, 가로: 예측)")
plt.show()

names = {"TWF": "공구 마모", "HDF": "열 방출", "PWF": "전력", "OSF": "과부하", "RNF": "무작위"}
print("고장 종류별로 잡은 비율 (고장 종류 열은 학습에 안 쓰고 해석에만 씀):")
for k in leak:
    m = (ft_test[k] == 1).values
    if m.sum():
        print(f"  {names[k]} 고장: {int(pred2[m].sum())}/{int(m.sum())}건")
imp = pd.Series(model2.feature_importances_, index=features).sort_values()
imp.plot(kind="barh", color="seagreen")
plt.title("특징 중요도 (개선 후 모델)")
plt.tight_layout()
plt.show()

# [5] 새 데이터로 예측
print("\n[5] 새 데이터로 예측")
new = pd.DataFrame([[0, 298.1, 308.6, 1551, 42.8, 30],     # 평범하게 돌아가는 기계
                    [0, 302.5, 311.0, 1300, 65.0, 210]],   # 토크가 크고 공구가 많이 닳은 기계
                   columns=base_features)
new = add_features(new)
print(new.round(1).to_string(index=False))
for i, (p, pr) in enumerate(zip(model2.predict(new), model2.predict_proba(new)[:, 1]), 1):
    print(f"기계 {i}: 고장 확률 {pr*100:.1f}% → 예측 {'고장 (점검 필요)' if p == 1 else '정상'}")
