# 스레드 동기화(Thread Synchronization) 실패 예제

import logging
from concurrent.futures import ThreadPoolExecutor
import time

class DataStore:
    # 공유 변수(value)
    def __init__(self):
        self.value = 0

    # 변수 업데이트 함수
    def update(self, n):
        logging.info("Thread %s: starting update", n)

        # 뮤텍스 & Lock 등 동기화(Thread synchronization) 없음
        local_copy = self.value
        local_copy += 1
        time.sleep(0.1)
        self.value = local_copy

        logging.info("Thread %s: finishing update", n)


if __name__ == "__main__":
    # Logging format 설정
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # 클래스 인스턴스화
    store = DataStore()

    logging.info("Testing update. Starting value is %d.", store.value)

    # With Context 시작
    with ThreadPoolExecutor(max_workers=2) as executor:
        for n in ['First', 'Second', 'Third']:
            executor.submit(store.update, n)

    logging.info("Testing update. Ending value is %d.", store.value)