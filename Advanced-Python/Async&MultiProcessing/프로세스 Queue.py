# 프로세스 간 통신을 위한 큐

from multiprocessing import Process, Queue

def func(q, message):
    q.put(message)

if __name__ == '__main__':
    # 큐 생성
    q = Queue()

    # 프로세스 생성
    a = Process(target=func, args=(q,[42, None, 'hello']))
    b = Process(target=func, args=(q,[111, True, 'world']))
    c = Process(target=func, args=(q,"wow"))

    a.start()
    b.start()

    print(q.get())    # prints "[42, None, 'hello']"
    print(q.get())    # prints "[111, True, 'world']"

    c.start()

    a.join()
    b.join()

    print(q.get())    # prints "wow"
    c.join()