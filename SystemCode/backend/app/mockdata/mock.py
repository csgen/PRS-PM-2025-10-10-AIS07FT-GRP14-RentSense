import random
import math
import time
import csv
from pydantic import ValidationError
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from app.models.enquiry import EnquiryForm
from app.models.behavior import UserBehavior, UserBehaviorComplete
from app.dataservice.sql_api.api_model import RequestInfo, ResultInfo
from app.dataservice.sql_api.api import fetch_recommend_properties_async

# 模拟用户输入
property_types = ["HDB", "Condo", "Landed", "Apartment", "Executive Condo"]
def mock_property_types():
    if random.random() < 0.2:
        preference = None
    else:
        # 随机选择 1 到 5 个类型
        num_selected = random.randint(1, len(property_types))
        preference = random.sample(property_types, num_selected)
    return preference

def simulate_enquiry_forms(num_users: int = 3, 
                           output_path = 'enquiries.csv') -> list[EnquiryForm]:
    forms = []
    for i in range(num_users):
        min_monthly_rent=random.randint(8, 15) * 100
        max_monthly_rent=min_monthly_rent+random.randint(8, 15) * 100
        forms.append(
            EnquiryForm(
                device_id=f"device_{i+1}",
                min_monthly_rent=min_monthly_rent,
                max_monthly_rent=max_monthly_rent,
                school_id=random.randint(1, 6),
                target_district_id=random.choice([random.randint(1, 36),None]),
                max_school_limit=random.choice([random.choice([20, 30, 40, 50, 60]),None]),
                flat_type_preference=mock_property_types(),
                max_mrt_distance=random.choice([random.choice([500, 600, 800, 1000, 1500, 2000]),None]),
                importance_rent=random.randint(1, 5),
                importance_location=random.randint(1, 5),
                importance_facility=random.randint(1, 5),
            )
        )
    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=forms[0].model_dump().keys())
            writer.writeheader()
            for form in forms:
                writer.writerow(form.model_dump())
        print(f"✅ 已保存 {len(forms)} 条用户查询到 {output_path}")

    return forms

# 模拟返回推荐
async def simulate_fetch_recommend_properties(params: EnquiryForm) -> list[ResultInfo]:
    try:
        req = RequestInfo.model_validate(params.model_dump(), strict=False)
    except ValidationError as e:
        print(f"fail to convert EnquiryForm into reqinfo: {e}")
        return []

    filtered_properties = await fetch_recommend_properties_async(req)
    
    return filtered_properties

async def simulate_user_behaviors(num_users: int = 3, output_path = 'behaviors.csv'):
    all_behaviors = []
    forms = simulate_enquiry_forms(num_users)

    for form in forms:
        print(f"\n模拟用户 {form.device_id} 的推荐与行为")

        recommended = await simulate_fetch_recommend_properties(form)

        if not recommended:
            print(f"!!!!! 用户 {form.device_id} 没有拿到推荐结果，跳过")
            continue

        # 模拟每个用户浏览5-10条
        selected = random.sample(recommended, k=min(len(recommended), random.randint(5, 10)))

        favorite_count = 0
        selected_count = len(selected)
        # 记录上一次时间
        last_time = datetime.now(timezone.utc)
        for i in range(selected_count):
            r = selected[i]
            total_score = (
                form.importance_rent * r.costScore +
                form.importance_location * r.commuteScore +
                form.importance_facility * r.neighborhoodScore
            )

            normalized_score = total_score/15

            dwell_time = random.uniform(5, 60) * (1 + normalized_score)
            # favorite_prob = 1 / (1 + math.exp(-0.5 * (total_score - 5)))  # S形增长
            favorite_prob = 0.1 + 0.3 * normalized_score
            favorite = random.random() < favorite_prob
            # 为了方便训练，保证每个用户至少一个收藏
            if favorite:
                favorite_count+=1
            if i==selected_count-1 and favorite_count==0:
                favorite = True
                favorite_count+=1

            last_time+=timedelta(seconds=(dwell_time + random.uniform(15, 40)))

            behavior = UserBehaviorComplete(
                **r.model_dump(),
                device_id=form.device_id,
                dwell_time=round(dwell_time, 2),
                favorite=favorite,
                update_time=last_time,
            )
            all_behaviors.append(behavior)
        print(f"✅ 用户 {form.device_id} 产生了 {len(selected)} 条浏览记录，有{favorite_count}个收藏")

    if all_behaviors and output_path:
        EXCLUDE_FIELDS = {"img_src"}
        fieldnames = [k for k in all_behaviors[0].model_dump().keys() if k not in EXCLUDE_FIELDS]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for b in all_behaviors:
                data = b.model_dump()
                data.pop("img_src", None)  # 不存 img_src
                writer.writerow(data)
        print(f"✅ 已保存 {len(all_behaviors)} 条用户行为到 {output_path}")

    return all_behaviors

async def run_mock_user():
        start_time = time.time()
        behaviors = await simulate_user_behaviors(num_users=500)
        favorite_count=0
        for b in behaviors:
            # print(b.model_dump_json())
            if(b.favorite):
                favorite_count+=1

        execution_time = time.time() - start_time
        print(f'完成用户行为数据模拟{len(behaviors)},执行时间: {execution_time:.2f} 秒')
        print(f'{favorite_count}/{len(behaviors)}个收藏')