# 프로세스 간 통신을 위한 파이프 사용 예제

from multiprocessing import Process, Pipe

# 메시지를 보내는 함수
def send_messages(conn, messages):
    for message in messages:
        print(f"Sending: {message}")
        conn.send(message)
    conn.close()

# 메시지를 받는 함수
def receive_messages(conn, num_messages):
    print("Receiving...")
    for _ in range(num_messages):
        message = conn.recv()
        print(f"Received: {message}")

if __name__ == '__main__':
    # 파이프 생성
    parent_conn, child_conn = Pipe()

    # 메시지 생성
    messages = ["Hello", "World", "!"]

    # 메시지 전송 프로세스 생성
    p1 = Process(target=send_messages, args=(child_conn, messages))
    p1.start()

    # 메시지 수신 프로세스 생성
    p2 = Process(target=receive_messages, args=(parent_conn, len(messages)))
    p2.start()

    # 프로세스가 종료될 때까지 기다림
    p1.join()
    p2.join()
