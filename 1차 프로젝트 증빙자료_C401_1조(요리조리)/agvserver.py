import cv2
import socket
import threading
from pymycobot.myagv import MyAgv
import time

agv = MyAgv('/dev/ttyAMA2', 115200)

# AGV 이동 명령
def agv_move(direction, speed=5, timeout=0.5):
    if direction == 'left':
        agv.counterclockwise_rotation(int(speed), timeout=float(timeout))
    elif direction == 'right':
        agv.clockwise_rotation(int(speed), timeout=float(timeout))
    elif direction == 'go':
        agv.go_ahead(int(speed), timeout=float(timeout))
    elif direction == "retreat":
        agv.retreat(int(speed), timeout=float(timeout))
    elif direction == 'stop':
        agv.stop()
    elif direction == "qr_left":
        agv.go_ahead(4, timeout=1.5)
        agv.counterclockwise_rotation(3, timeout=3)
    elif direction == "qr_right":
        agv.go_ahead(4, timeout=1.5)
        agv.clockwise_rotation(3, timeout=2.5)
    elif direction == "obstacle":
        agv.pan_left(10, timeout=2)
        agv.go_ahead(30, timeout=3)
        agv.pan_right(10, timeout=2)

# 서버 설정
HOST = '0.0.0.0'  # 모든 인터페이스에서 수신
PORT = 5000        # 포트 번호

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)  # 최대 1개의 대기 연결 허용
server_socket.settimeout(1.0)  # 1초 타임아웃 설정
print("starting server.. waiting client..")

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

frame = None
stop_thread_flag = False
stop_server_flag = False

# 영상 전송 함수
def send_video(client_socket):
    global stop_thread_flag, frame
    while not stop_thread_flag:
        ret, temp_frame = cap.read()
        if not ret:
            continue

        frame = temp_frame  # 프레임 업데이트

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        frame_length = len(frame_data)

        # 프레임 길이를 문자열로 변환하여 전송
        frame_length_bytes = str(frame_length).ljust(16).encode('utf-8')  # 16바이트로 패딩
        try:
            client_socket.sendall(frame_length_bytes)  # 프레임 길이 전송
            client_socket.sendall(frame_data)  # 프레임 데이터 전송
            time.sleep(0.01)
        except:
            print("The connection to the client has been lost. Video transmission stopped")
            break  # 연결 문제 발생 시 전송 중단

# 명령 수신 함수
def receive_commands(client_socket):
    global stop_thread_flag
    while not stop_thread_flag:
        try:
            length_data = client_socket.recv(16).decode()  # 16바이트 길이 수신
            if not length_data:
                continue
            data_length = int(length_data.strip())
            # 다음으로 해당 길이만큼의 데이터를 수신합니다.
            command = client_socket.recv(data_length).decode()
            if command:  # 빈 문자열이 아닐 때만 처리
                command = command.replace(" ", "").split(',')
                if len(command) == 1:
                    print(f"receive data: direction({command[0]})")
                    agv_move(command[0])
                else:
                    print(f"receive data: direction({command[0]}), speed({command[1]}), timeout({command[2]})")
                    agv_move(command[0], command[1], command[2])
        except:
            print("An error occurred in the connection with the client. Command reception stopped")
            break  # 연결 문제 발생 시 수신 중단

# 클라이언트 연결 처리 함수
def handle_client(client_socket, addr):
    global stop_thread_flag
    print(f"client {addr} connect")
    stop_thread_flag = False

    # 영상 전송 및 명령 수신 스레드 시작
    video_thread = threading.Thread(target=send_video, args=(client_socket,), daemon=True)
    command_thread = threading.Thread(target=receive_commands, args=(client_socket,), daemon=True)
    video_thread.start()
    command_thread.start()

    # 스레드가 종료될 때까지 대기
    video_thread.join()
    command_thread.join()

    client_socket.close()
    print(f"client {addr} connection end")

# 메인 서버 루프
try:
    while not stop_server_flag:
        # 클라이언트 연결 수락
        try:
            client_socket, addr = server_socket.accept()
            # 새로운 스레드에서 클라이언트 처리
            client_handler_thread = threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True)
            client_handler_thread.start()
        except socket.timeout:
            # 타임아웃 발생 시, 계속해서 서버 상태 확인
            pass

        # `q` 눌러서 서버 종료하기
        if frame is not None:
            cv2.imshow("AGV View", frame)  # 메인 스레드에서 화면 출력
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_server_flag = True
            stop_thread_flag = True
            print("Stopping server...")

except KeyboardInterrupt:
    print("server end")
finally:
    cap.release()
    server_socket.close()
    cv2.destroyAllWindows()
