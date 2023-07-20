# Asyncio 모듈을 이용한 비동기 프로그래밍
import time
import asyncio

# 비동기 함수 
async def exe_calculate_async(name, n):
    for i in range(1, n + 1):
        print(f'{name} -> {i} of {n} is calculating..')
        await asyncio.sleep(1)
    print(f'## {name} - {n} working done!')
    
# 비동기 프로세스
async def process_async():
    start = time.time()
    await asyncio.wait([
        exe_calculate_async('One', 3),
        exe_calculate_async('Two', 2),
        exe_calculate_async('Three', 1),
    ])
    end = time.time()
    print(f'>>> total seconds : {end - start}')
    
# 비교를 위한 동기 함수
def exe_calculate_sync(name, n):
    for i in range(1, n + 1):
        print(f'{name} -> {i} of {n} is calculating..')
        time.sleep(1)
    print(f'## {name} - {n} working done!')
    
# 비교를 위한 동기 프로세스
def process_sync():
    start = time.time()
    
    exe_calculate_sync('One', 3)
    exe_calculate_sync('Two', 2)
    exe_calculate_sync('Three', 1)
    
    end = time.time()
    print(f'>>> total seconds : {end - start}')

if __name__ == '__main__':
    # Sync 실행
    process_sync()
    
    # Async 실행
    #asyncio.run(process_async())