from .api_model import RequestInfo, ResultInfo
from .func import query_housing_data_async, filter_housing_async
import asyncio
import time

async def fetch_recommend_properties_async(params: RequestInfo) -> list[ResultInfo]:
    '''异步算法接口，根据请求参数返回初步过滤结果及信息'''
    housings = await query_housing_data_async(params)
    print(f'第一步得到{len(housings)}条符合条件的房源，开始处理...')

    start_time = time.time()

    results = await filter_housing_async(housings, params)

    execution_time = time.time() - start_time
    print(f'第二步 filter_housing_async 执行时间: {execution_time:.2f} 秒')
    
    return results

def fetchRecommendProperties(params: RequestInfo) -> list[ResultInfo]:
    '''
    同步包装版本，自动检测当前是否存在事件循环；
    如果在异步上下文中，则通过 loop.run_until_complete
    '''
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已经在异步环境中
        coro = fetch_recommend_properties_async(params)
        return asyncio.ensure_future(coro)  # 返回一个 Future
    else:
        # 在同步环境中
        return asyncio.run(fetch_recommend_properties_async(params))