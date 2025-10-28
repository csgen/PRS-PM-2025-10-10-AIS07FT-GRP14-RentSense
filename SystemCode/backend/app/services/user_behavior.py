from ast import Dict, Tuple
from typing import List
from SystemCode.backend.app.models.property import Property
from dataclasses import dataclass

#用来测试的class，实际调用的时候可以改
@dataclass
class UserPreferenceScore:
    property_id: int
    final_score: float


def calculate_preference_score(
    propertyList: List[Property],
    omega_weights: Tuple[float, float, float] # 定义完omega的格式之后替换成omega的真正格式
) -> UserPreferenceScore:
    omega_cost, omega_commute, omega_neighbor = omega_weights
    total_omega = omega_cost + omega_commute + omega_neighbor
    w_cost = omega_cost / total_omega
    w_commute = omega_commute / total_omega  
    w_neighbor = omega_neighbor / total_omega
    final_score = (
        w_cost * propertyList.costScore +
        w_commute * propertyList.commuteScore + 
        w_neighbor * propertyList.neighborhoodScore
    )
    return UserPreferenceScore(
        property_id=propertyList.property_id,
        final_score=final_score,
    )
