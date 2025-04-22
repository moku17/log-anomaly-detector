import streamlit as st
import pandas as pd
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from io import StringIO

# 로그 파싱 함수
def parse_log_line(line):
    pattern = r'(?P<ip>\S+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+)'
    match = re.match(pattern, line)
    if match:
        data = match.groupdict()
        data['datetime'] = datetime.strptime(data['datetime'], '%d/%b/%Y:%H:%M:%S %z')
        return data
    return None

def load_logs_from_text(text):
    lines = text.split('\n')
    parsed = [parse_log_line(line) for line in lines if parse_log_line(line)]
    return pd.DataFrame(parsed)

# 페이지 설정
st.set_page_config(page_title="이상행위 탐지기", layout="wide")
st.title("🕵️‍♂️ 이상행위 탐지 시스템 (Isolation Forest)")

# 파일 업로드
uploaded_file = st.file_uploader("🔽 로그 파일을 업로드하세요 (.log 형식)", type=["log", "txt"])

if uploaded_file is not None:
    # 로그 읽기
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = load_logs_from_text(stringio.read())

    if df.empty:
        st.warning("⚠️ 로그 형식이 올바르지 않습니다.")
    else:
        st.success("✅ 로그 파일이 성공적으로 업로드되었습니다.")
        grouped = df.groupby('ip')

        # 특징 추출
        feature_df = grouped.size().reset_index(name='request_count')
        feature_df['unique_urls'] = grouped['url'].nunique().values
        feature_df['error_rate'] = grouped.apply(lambda x: (x['status'] != '200').mean()).values
        feature_df['avg_size'] = grouped['size'].apply(lambda x: x.astype(int).mean()).values

        # 모델 학습 및 예측
        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(feature_df[['request_count', 'unique_urls', 'error_rate', 'avg_size']])
        feature_df['anomaly'] = model.predict(feature_df[['request_count', 'unique_urls', 'error_rate', 'avg_size']])

        # 결과 출력
        st.subheader("📊 IP별 요청 분석 결과")
        st.dataframe(feature_df)

        # 이상행위만 표시
        st.subheader("🚨 이상행위로 탐지된 IP")
        st.dataframe(feature_df[feature_df['anomaly'] == -1])

        # 그래프
        st.subheader("📈 이상행위 시각화")
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_color = feature_df['anomaly'].map({-1: 'red', 1: 'blue'})
        ax.bar(feature_df['ip'], feature_df['request_count'], color=bar_color)
        ax.set_title("IP별 요청 수 (빨간색 = 이상행위)")
        ax.set_ylabel("요청 수")
        ax.set_xticklabels(feature_df['ip'], rotation=45)
        st.pyplot(fig)