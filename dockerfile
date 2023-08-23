# 베이스 이미지로 Python 3.10.12를 사용합니다.
FROM python:3.10.12

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 라이브러리 설치
RUN pip install flask pillow torch torchvision

# Flask 애플리케이션 파일 복사
COPY app.py /app

# 모델 파일 및 가중치 파일 복사
COPY conjunctivitis.pt /app

EXPOSE 5000
# Flask 애플리케이션 실행
CMD ["python", "app.py"]