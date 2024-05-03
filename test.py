import streamlit as st
from ultralytics import YOLO
import cv2
from tkinter.messagebox import *
import numpy as np

# 모델 로드
model = YOLO("best.pt")

def main():
    st.title("주차 검사")

    uploaded_file = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 사진 표시
        st.image(img, caption="업로드된 사진", use_column_width=True)

        # 결과 예측
        results = model.predict(
            img,
            save=False,
            imgsz=640,
            conf=0.2,
            device="cpu",  # 또는 디바이스를 사용할 수 없는 경우 "cpu"로 설정
        )

        # 검사 결과 분석
        car_present = False
        scooter_present = False

        for r in results:
            cls = r.boxes.cls  # 현재 이미지의 bbox의 class들
            cls_dict = r.names  # 예시: {0: 'car', 1: 'scooter'}

            for cls_number in cls:
                cls_number_int = int(cls_number.item())
                cls_name = cls_dict[cls_number_int]

                if cls_name == "Car":
                    car_present = True
                elif cls_name == "Scooter":
                    scooter_present = True

        # 검사 결과 메시지 표시
        if car_present and scooter_present:
            st.error("주차 불가")
        elif not scooter_present:
            st.warning("스쿠터가 없습니다")
        elif scooter_present and not car_present:
            st.success("주차 가능")

if __name__ == "__main__":
    main()
