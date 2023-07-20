# DaemonThread(데몬스레드)
# 백그라운드에서 실행되는 스레드로 메인 스레드가 종료되면 즉시 종료되는 스레드
import threading

# 스레드 실행 함수
def thread_func(name, d):
    for i in d:
        print(i)

# 메인 영역
if __name__ == "__main__":
    x = threading.Thread(target=thread_func, args=('First', range(20000)), daemon=True)
    y = threading.Thread(target=thread_func, args=('Second', range(10000)), daemon=True)

    # 서브 스레드 시작
    x.start()
    y.start()

    # DaemonThread 확인
    print(x.isDaemon())
   