
import sqlite3

# 데이터베이스 파일 연결
conn = sqlite3.connect('agv_control.db')
cursor = conn.cursor()

# 테이블 생성 (명령어 기록 테이블)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS commands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        direction INTEGER
    )
''')

# AGV 명령어 데이터 삽입 (1: 갈림길 오른쪽, 2: 갈림길 왼쪽)
direction_list = [2, 1]
for direction in direction_list:
    cursor.execute('''
        INSERT INTO commands (direction)
        VALUES (?)
    ''', (direction,))  # 튜플로 전달하기 위해 콤마 추가

# 변경 사항 저장
conn.commit()

# DB 확인
cursor.execute('SELECT * FROM commands')
rows = cursor.fetchall()

# 데이터 출력
for row in rows:
    print(f"ID: {row[0]}, Direction: {row[1]}")

conn.close()