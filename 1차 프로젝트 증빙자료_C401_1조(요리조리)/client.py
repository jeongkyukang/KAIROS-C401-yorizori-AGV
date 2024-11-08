import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QTransform
import os
import socket
import threading
import time
import sqlite3

# UI file road
form_class = uic.loadUiType('design.ui')[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.SERVER_IP = '172.30.1.71'
        self.SERVER_PORT = 5000
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.SERVER_IP, self.SERVER_PORT))
        print("Connect server.")
        
        self.frame = None
        
        # DB에서 QR 방향 데이터 읽어오기
        conn = sqlite3.connect('agv_control.db')
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM commands')
        rows = cursor.fetchall()

        self.qr_directions_idx = 0
        self.qr_directions = []
        for row in rows:
            self.qr_directions.append(row[1])

        conn.close()
        
        # Qt_video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.Qt_video)
        self.timer.start(30)
        
        self.processing_thread = None 
        self.auto_mode = False #auto mode on off 
        self.rotation_angle = 0 #handle image angle
        self.handle_image = None #handle image 
        
        # 프레임 수신 스레드 시작
        self.running = True
        threading.Thread(target=self.receive_video, daemon=True).start()
        self.last_direction = "stop"

        #Qt Ui button command
        self.go.clicked.connect(self.go_command)
        self.back.clicked.connect(self.back_command)
        self.left.clicked.connect(self.left_command)
        self.right.clicked.connect(self.right_command)
        self.automode.clicked.connect(self.auto_command)
        self.stick.clicked.connect(self.stick_command)
        self.load_steering_wheel() #Qt handle image road
        self.label.setText("Manual mode...")
        
        # 회전 관련 변수 추가
        self.last_rotation = 0  # 마지막 회전 각도
        self.rotation_update_time = time.time()  # 마지막 회전 업데이트 시간
        self.MIN_ROTATION_INTERVAL = 0.5  # 최소 회전 업데이트 간격 (초)
        
        # qr
        self.last_qr_action_time = 0
        self.qr_action_completed = False
    def load_steering_wheel(self):
        try:
            steering_image_path = 'steering_wheel.png'
            self.handle_image = QPixmap(steering_image_path)
            if self.handle_image.isNull():
                print("Fail Handle image.")
                return
                
            # 초기 이미지 표시
            self.rotation_handle_image()
            
            # 이미지 2 로드 및 표시
            background_image_path = "background.png"
            if not os.path.exists(background_image_path):
                print(f"Error: {background_image_path} 파일을 찾을 수 없습니다.")
                return
                
            pixmap2 = QPixmap(background_image_path)
            if pixmap2.isNull():
                print(f"Error: {background_image_path} 파일을 불러올 수 없습니다.")
                return
                
            # 이미지2 표시
            self.background_image.setPixmap(pixmap2.scaled(
                self.background_image.width(),
                self.background_image.height(),
                Qt.KeepAspectRatio

            ))
            
            print("이미지 로드 성공!")
            
        except Exception as e:
            print(f"이미지 로드 중 오류 발생: {str(e)}")


    def receive_video(self):
        while self.running:
            try:
                # 프레임 길이 수신 (16바이트 고정 길이)
                frame_size_data = self.client_socket.recv(16)
                if not frame_size_data:
                    continue
                
                # 프레임 길이 파싱
                frame_size = int(frame_size_data.decode().strip())
                
                # 프레임 데이터 수신
                frame_data = b''
                remaining = frame_size
                
                while remaining > 0:
                    chunk = self.client_socket.recv(min(remaining, 4096))
                    if not chunk:
                        break
                    frame_data += chunk
                    remaining -= len(chunk)

                # 프레임 디코딩
                if len(frame_data) == frame_size:
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame = frame
            except Exception as e:
                print(f"Error receiving video: {str(e)}")
                time.sleep(0.1)

    def Qt_video(self):
        if self.frame is not None:
            try:
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
                target_width = self.camera.width()
                target_height = self.camera.height()
                
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    target_width, 
                    target_height,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                self.camera.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"Error updating frame: {str(e)}")

    def load_images(self):
        try:
            # 이미지 1 로드 및 표시
            image1_path = "steering_wheel.png"
            if not os.path.exists(image1_path):
                print(f"Error: {image1_path} 파일을 찾을 수 없습니다.")
                return
                
            self.handle_image = QPixmap(image1_path)
            if self.handle_image.isNull():
                print(f"Error: {image1_path} 파일을 불러올 수 없습니다.")
                return
                
            # 초기 이미지1 표시
            self.rotation_handle_image()

            # 이미지 2 로드 및 표시
            background_image_path = "background.png"
            if not os.path.exists(background_image_path):
                print(f"Error: {background_image_path} 파일을 찾을 수 없습니다.")
                return
                
            pixmap2 = QPixmap(background_image_path)
            if pixmap2.isNull():
                print(f"Error: {background_image_path} 파일을 불러올 수 없습니다.")
                return
                
            # 이미지2 표시
            self.background_image.setPixmap(pixmap2.scaled(
                self.background_image.width(),
                self.background_image.height(),
                Qt.KeepAspectRatio
            ))
            
            print("이미지 로드 성공!")
            
        except Exception as e:
            print(f"이미지 로드 중 오류 발생: {str(e)}")

    def rotation_handle_image(self):
        if self.handle_image is None or self.handle_image.isNull():
            print("Error: 회전할 이미지가 없습니다.")
            return
            
        try:
            # QTransform을 사용하여 이미지 회전
            transform = QTransform().rotate(self.rotation_angle)
            rotated_pixmap = self.handle_image.transformed(transform, Qt.SmoothTransformation)
            
            # 회전된 이미지를 QLabel 크기에 맞게 스케일링
            self.steering_wheel.setPixmap(rotated_pixmap.scaled(
                self.steering_wheel.width(),
                self.steering_wheel.height(),
                Qt.KeepAspectRatio
            ))
            print(f"이미지 회전 완료: {self.rotation_angle}도")
            
        except Exception as e:
            print(f"이미지 회전 중 오류 발생: {str(e)}")

    def go_command(self):
        print("go")
        self.rotation_angle = 0
        self.rotation_handle_image()
        if not self.auto_mode:
            self.send_command("go,30,1")
        
    def back_command(self):
        print("back")
        self.rotation_angle = 0
        self.rotation_handle_image()
        if not self.auto_mode:
            self.send_command("retreat,30,1")
        
    def left_command(self):
        print("left")
        self.rotation_angle = 270
        self.rotation_handle_image()
        if not self.auto_mode:
            self.send_command("left,10,1")
        
    def right_command(self):
        print("right")
        self.rotation_angle = 90
        self.rotation_handle_image()
        if not self.auto_mode:
            self.send_command("right,10,1")
        
    def auto_command(self):
        print("automode")
        self.auto_mode = True
        self.send_command("automode")
        
        # label에 텍스트 설정
        self.label.setText("자율 주행 중...")
        
        # 라인 트레이싱 스레드 시작
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.video_processing, daemon=True)
            self.processing_thread.start()

    def stick_command(self):
        print("stick")
        self.auto_mode = False  # 라인 트레이싱 중지
        self.send_command("stick")
        
        # label에 텍스트 설정
        self.label.setText("수동 제어 중...")
        
    def rotate_steering_img(self, new_angle):
        current_time = time.time()
        
        # 각도가 변경되었고 최소 업데이트 간격이 지났을 때만 회전
        if (new_angle != self.last_rotation and 
            current_time - self.rotation_update_time >= self.MIN_ROTATION_INTERVAL):
            self.rotation_angle = new_angle
            self.rotation_handle_image()
            self.last_rotation = new_angle
            self.rotation_update_time = current_time
            
    def detect_traffic_light(self):
        lower_red1 = np.array([0, 160, 180], dtype=np.uint8)
        upper_red1 = np.array([5, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([170, 160, 180], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # 빨간색 마스크 (두 개의 범위를 합쳐서 빨간색 탐지)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # 빨간색 물체가 있는지 확인
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in red_contours:
            if cv2.contourArea(contour) > 20:
                print("red stop")
                return True
        return False
    
    def qr_detect(self):
        detector = cv2.QRCodeDetector()
        return_value = False
        if self.frame is None:
            return return_value
        try:
            qr_data, bbox, _ = detector.detectAndDecode(self.frame)
            if qr_data and bbox is not None and len(bbox) > 0:
                area = cv2.contourArea(bbox)
                # QR 코드의 크기가 일정 이상일 때만 인식
                if area > 350:
                    print(f"Detected QR Code Data: {qr_data}")
                    current_time = time.time()
                    if current_time - self.last_qr_action_time >= 3:
                        return_value = qr_data
                        self.last_qr_action_time = time.time()
        except:
            pass
        return return_value
        
    def send_command(self, command):
        try:
            # 명령어 길이를 16바이트로 패딩
            data_length = str(len(command)).ljust(16).encode('utf-8')
            # 길이와 명령어 전송
            self.client_socket.sendall(data_length + command.encode('utf-8'))
            print(f"Sent command: {command}")
            time.sleep(0.6)
        except Exception as e:
            print(f"Error sending command: {str(e)}")

    def video_processing(self):
        #Qr code
        self.qr_mode = False
        self.qr_action_complete = False
        self.searching_line = False
        self.search_start_time = 0
        qr_data = self.qr_detect()
        while self.auto_mode and self.running:
            if self.frame is None:
                continue
            
            try:
                frame = self.frame.copy()
                height = frame.shape[0]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # detect traffic light 
                red_detected_flag = self.detect_traffic_light()
                if red_detected_flag:
                    self.send_command("stop")
                    continue

                # 노란색 라인 검출을 위한 마스크
                mask = cv2.inRange(hsv[height // 2:, :], np.array([20, 100, 100]), np.array([40, 255, 255]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result = np.zeros_like(frame)

                top_mask = cv2.inRange(hsv[height // 2:height // 4 * 3, :], np.array([20, 100, 100]), np.array([40, 255, 255]))
                top_contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if top_contours:
                    top_largest_contour = max(contours, key=cv2.contourArea)
                    top_M = cv2.moments(top_largest_contour)
                    if top_M['m00'] != 0:
                        top_cx = int(top_M['m10'] / top_M['m00'])
                        print("top cx: ", top_cx)

                        if top_cx <= 160:
                            print("last direction: left")
                            self.last_direction = "left,2,0.5"
                        else:
                            print("last direction: right")
                            self.last_direction = "right,2,0.5"

                # QR 코드 처리
                qr_data = self.qr_detect()
                if qr_data == "wooseok":
                    # db 값 활용
                    if self.qr_directions_idx  >= len(self.qr_directions):
                        self.qr_directions_idx = 0
                    if self.qr_directions[self.qr_directions_idx] == 1: # 갈림길 오른쪽
                        print("QR 코드 'wooseok' 감지됨 - Proceed right at the fork")
                        self.send_command("qr_right")
                    else: # 갈림길 왼쪽
                        print("QR 코드 'wooseok' 감지됨 - Proceed left at the fork")
                        self.send_command("qr_left")
                    time.sleep(2)
                    self.qr_directions_idx += 1
                    
                elif qr_data == "soomi":
                    print("QR 코드 'soomi' 감지됨 - 회피 동작 수행")
                    self.send_command("soomi")
                    time.sleep(6)
                    self.qr_action_completed = True
                    continue
                                
                # QR 동작이 완료되면 라인트레이싱으로 복귀
                if self.qr_action_completed:
                    self.qr_mode = False
                    self.qr_action_completed = False
                    print("라인트레이싱 모드로 복귀")

                # 라인트레이싱 로직
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result = np.zeros_like(frame)


                if contours:
                    if self.searching_line:
                        self.searching_line = False
                        print("라인 발견 - 트레이싱 재개")
                        self.search_start_time = 0

                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(result, [largest_contour], -1, (0, 255, 255), -1)

                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        # 방향 결정을 위한 임계값 조정
                        if cx < 80:
                            self.send_command("left")
                            self.rotate_steering_img(270)  # 좌회전
                        elif cx > 220:
                            self.send_command("right")
                            self.rotate_steering_img(90)   # 우회전
                        else:
                            self.send_command("go,15,0.5")
                            self.rotate_steering_img(0)    # 직진
                else:
                    # 라인을 잃어버렸을 때
                    if not self.searching_line:
                        print("라인 검색 시작 - 제자리 좌회전")
                        self.searching_line = True
                        self.search_start_time = time.time()
                    
                    current_time = time.time()
                    search_duration = current_time - self.search_start_time
                    
                    if search_duration > 60:
                        print("라인 검색 시간 초과 - 정지")
                        self.send_command("stop")
                        self.searching_line = False
                    else:
                        print(f"라인 검색 중... (경과 시간: {search_duration:.1f}초)")
                        self.send_command(self.last_direction)
                        if self.last_direction == "left":
                            self.rotate_steering_img(270)  # 좌회전 검색
                        elif self.last_direction == "right":
                            self.rotate_steering_img(90)
                        time.sleep(0.3)
 

                # 처리된 영상을 Qt 화면에 표시
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.camera.width(),
                    self.camera.height(),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                self.camera.setPixmap(scaled_pixmap)

                time.sleep(0.1)  # 전체 처리 주기 조절
                    
            except Exception as e:
                print(f"Error in video processing: {str(e)}")
                time.sleep(0.1)

    def closeEvent(self, event):
        self.running = False
        self.client_socket.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()