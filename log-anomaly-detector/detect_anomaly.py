import pandas as pd
from parse_logs import load_logs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 로그 읽기
df = load_logs("access.log")

# 기본 특징 만들기
grouped = df.groupby('ip')

feature_df = grouped.size().reset_index(name='request_count')  # 요청 수
feature_df['unique_urls'] = grouped['url'].nunique().values    # 요청한 URL 종류 수
feature_df['error_rate'] = grouped.apply(lambda x: (x['status'] != '200').mean()).values  # 오류 비율
feature_df['avg_size'] = grouped['size'].apply(lambda x: x.astype(int).mean()).values  # 평균 응답 크기

# AI 모델
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(feature_df[['request_count', 'unique_urls', 'error_rate', 'avg_size']])
feature_df['anomaly'] = model.predict(feature_df[['request_count', 'unique_urls', 'error_rate', 'avg_size']])

# Step 2: 성능 평가용 라벨만들기
# 예를 들어 192.168.0.1은 공격자(이상행위)라고 가정
feature_df['true_label'] = feature_df['ip'].apply(lambda x: -1 if x == '192.168.0.1' else 1)

# 평가 지표 출력
print("\n📌 Classification Report:")
print(classification_report(feature_df['true_label'], feature_df['anomaly'], target_names=['정상', '이상']))

# 탐지 결과 출력
print("\n📊 전체 특징 + 이상 여부:")
print(feature_df)

print("\n🚨 이상행위로 탐지된 IP:")
print(feature_df[feature_df['anomaly'] == -1])

# 시각화
plt.figure(figsize=(8, 4))
plt.bar(feature_df['ip'], feature_df['request_count'], color=(feature_df['anomaly'] == -1).map({True: 'red', False: 'blue'}))
plt.title("IP별 요청 수 (빨간색 = 이상행위)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()