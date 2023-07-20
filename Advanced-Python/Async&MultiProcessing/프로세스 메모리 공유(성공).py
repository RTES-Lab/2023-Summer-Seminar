# 프로세스 메모리 공유 예제(공유 성공!)

from multiprocessing import Process, current_process, Value
import os

# 실행 함수
def generate_update_number(v : int):
    for i in range(50):
        v.value += 1
    print(current_process().name, "data", v.value)

def main():
    # 부모 프로세스 아이디
    parent_process_id = os.getpid()
    # 출력
    print(f"Parent process ID {parent_process_id}")

    # 프로세스 리스트  선언
    processes = list()

    # 프로세스 메모리 공유 변수 선언
    share_value = Value('i', 0)  # i는 정수형을 의미, 0은 초기값
    
    for _ in range(1,10):
        # 생성
        p = Process(target=generate_update_number, args=(share_value,))
        # 배열에 담기
        processes.append(p)
        # 실행
        p.start()
        
    # Join
    for p in processes:
        p.join()

    # 최종 프로세스 부모 변수 확인
    print("Final Data(share_value) in parent process",  share_value.value)

if __name__ == '__main__':
    main()