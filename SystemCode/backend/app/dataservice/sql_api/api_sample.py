from api_model import RequestInfo, ResultInfo
from api import fetchRecommendProperties, fetch_recommend_properties_async
import asyncio

test_request_params = [
    RequestInfo(
        min_monthly_rent=1000,
        max_monthly_rent=3000,
        school_id=3,
        max_school_limit=60,
        flat_type_preference=["HDB", "Condo", "Apartment"],
        max_mrt_distance=1000,
        importance_rent=5,
        importance_location=4,
        importance_facility=3
    ),
    RequestInfo(
        min_monthly_rent=900,
        max_monthly_rent=2000,
        school_id=1,
        target_district_id=3,
        max_school_limit=50,
        flat_type_preference=["HDB"],
        max_mrt_distance=1500,
        importance_rent=3,
        importance_location=1,
        importance_facility=3
    ),
    RequestInfo(
        min_monthly_rent=500,
        max_monthly_rent=3000,
        school_id=4,
        target_district_id=7,
        max_school_limit=30,
        flat_type_preference=["Apartment", "Executive Condo"],
    )
]


if __name__ == "__main__":
    # 示例请求参数
    request_params = RequestInfo(
        min_monthly_rent=1000,
        max_monthly_rent=3000,
        school_id=3,
        # target_district_id=3,
        max_school_limit=60,
        flat_type_preference=["HDB", "Condo", "Apartment"],
        max_mrt_distance=1000,
        importance_rent=5,
        importance_location=4,
        importance_facility=3
    )

    # -------- 异步调用示例 --------
    # 单请求
    async def single_request(request = request_params):
        filtered_properties = await fetch_recommend_properties_async(request)
        print(f'找到{len(filtered_properties)}个房源。')

        # 返回结果
        for property in filtered_properties:
            print(property.model_dump_json())

    asyncio.run(single_request())

    # 多请求批量测试
    async def multi_request():
        await asyncio.gather(*(single_request(param) for param in test_request_params))
    # asyncio.run(multi_request())

    # -------- 同步调用示例 --------
    # filtered_properties = fetchRecommendProperties(request_params)
    # print(f'找到{len(filtered_properties)}个房源。')

    # 返回结果
    # for property in filtered_properties:
    #     print(property.model_dump_json())