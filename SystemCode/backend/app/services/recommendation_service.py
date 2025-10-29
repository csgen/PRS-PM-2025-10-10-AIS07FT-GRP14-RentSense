from typing import List, Optional, Tuple

from pydantic import ValidationError
from app.models.preference import UserPreference
from app.services.user_modeling.user_behavior import calculate_preference_score
from app.models import EnquiryForm, Property
from sqlmodel.ext.asyncio.session import AsyncSession

from app.dataservice.sql_api.api_model import RequestInfo as reqinfo, ResultInfo as resinfo
from app.dataservice.sql_api.api import fetch_recommend_properties_async
from app.database.crud import preference_crud


# Get recommended property list (unsorted)
async def fetch_recommend_properties(params: EnquiryForm) -> List[Property]:
    try:
        req = reqinfo.model_validate(params.model_dump(), strict=False)
    except ValidationError as e:
        print(f"fail to convert EnquiryForm into reqinfo: {e}")
        return []

    filtered_properties = await fetch_recommend_properties_async(req)

    try:
        results = [Property.model_validate(p.model_dump(), strict=False) for p in filtered_properties]
    except ValidationError as e:
        print(f"fail to convert resinfo into Property: {e}")
        return []
    
    return results


# Sort recommended property list
async def multi_objective_optimization_ranking(
        *,
        db: AsyncSession,
        enquiry: EnquiryForm,
        propertyList: List[Property]
) -> List[Property]:

    if not propertyList:
        return []

    valid_properties = _validate_and_filter(propertyList)
    if not valid_properties:
        return []

    normalized_properties = _normalize_scores(valid_properties)

    pareto_layers = _pareto_front_layering(normalized_properties)

    properties_with_crowding = _calculate_crowding_distance(pareto_layers)

    omega_weights: Optional[UserPreference] = \
        await preference_crud.get_user_preference_by_device_id(db=db,device_id=enquiry.device_id)

    ranked_properties = _final_ranking(
        properties_with_crowding=properties_with_crowding, 
        enquiry=enquiry,
        omega_weights=(
            omega_weights.costScore, 
            omega_weights.commuteScore, 
            omega_weights.neighborhoodScore
        ) if omega_weights else None
    )

    return ranked_properties


def _validate_and_filter(propertyList: List[Property]) -> List[Property]:
    valid_properties = []

    for prop in propertyList:
        if (hasattr(prop, 'costScore') and hasattr(prop, 'commuteScore') and hasattr(prop, 'neighborhoodScore')):
            if (0 < prop.costScore <= 1 and 0 < prop.commuteScore <= 1 and 0 < prop.neighborhoodScore <= 1):
                valid_properties.append(prop)

    return valid_properties


def _normalize_scores(properties: List[Property]) -> List[Property]:
    if len(properties) == 1:
        return properties

    cost_scores = [p.costScore for p in properties]
    commute_scores = [p.commuteScore for p in properties]
    neighborhood_scores = [p.neighborhoodScore for p in properties]

    cost_min, cost_max = min(cost_scores), max(cost_scores)
    commute_min, commute_max = min(commute_scores), max(commute_scores)
    neighborhood_min, neighborhood_max = min(neighborhood_scores), max(neighborhood_scores)

    for prop in properties:
        prop.costScore = _safe_normalize(prop.costScore, cost_min, cost_max)
        prop.commuteScore = _safe_normalize(prop.commuteScore, commute_min, commute_max)
        prop.neighborhoodScore = _safe_normalize(prop.neighborhoodScore, neighborhood_min, neighborhood_max)

    return properties


def _safe_normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val < 1e-6:
        return 1.0
    return (value - min_val) / (max_val - min_val)


def _pareto_front_layering(properties: List[Property]) -> List[List[Property]]:
    layers = []
    remaining = properties.copy()

    while remaining:
        current_layer = []
        dominated = []

        for prop in remaining:
            is_dominated = False

            for layer_prop in current_layer:
                if _dominates(layer_prop, prop):
                    is_dominated = True
                    break

            if not is_dominated:
                new_layer = []
                for layer_prop in current_layer:
                    if not _dominates(prop, layer_prop):
                        new_layer.append(layer_prop)
                    else:
                        dominated.append(layer_prop)

                new_layer.append(prop)
                current_layer = new_layer
            else:
                dominated.append(prop)

        layers.append(current_layer)
        remaining = dominated

    return layers


def _dominates(prop_a: Property, prop_b: Property) -> bool:
    not_worse = (
        prop_a.costScore >= prop_b.costScore and
        prop_a.commuteScore >= prop_b.commuteScore and
        prop_a.neighborhoodScore >= prop_b.neighborhoodScore
    )

    strictly_better = (
        prop_a.costScore > prop_b.costScore or
        prop_a.commuteScore > prop_b.commuteScore or
        prop_a.neighborhoodScore > prop_b.neighborhoodScore
    )

    return not_worse and strictly_better


def _calculate_crowding_distance(layers: List[List[Property]]) -> List[tuple]:
    properties_with_crowding = []

    for layer_idx, layer in enumerate(layers):
        if len(layer) <= 2:
            for prop in layer:
                properties_with_crowding.append((prop, layer_idx, float('inf')))
            continue

        crowding_distances = {id(prop): 0.0 for prop in layer}

        for objective in ['costScore', 'commuteScore', 'neighborhoodScore']:
            sorted_layer = sorted(layer, key=lambda p: getattr(p, objective), reverse=True)

            crowding_distances[id(sorted_layer[0])] = float('inf')
            crowding_distances[id(sorted_layer[-1])] = float('inf')

            obj_range = (getattr(sorted_layer[0], objective) - getattr(sorted_layer[-1], objective))

            if obj_range < 1e-6:
                continue

            for i in range(1, len(sorted_layer) - 1):
                if crowding_distances[id(sorted_layer[i])] != float('inf'):
                    distance = (getattr(sorted_layer[i - 1], objective) - getattr(sorted_layer[i + 1], objective)) / obj_range
                    crowding_distances[id(sorted_layer[i])] += distance

        for prop in layer:
            properties_with_crowding.append((prop, layer_idx, crowding_distances[id(prop)]))

    return properties_with_crowding


def _final_ranking(
    *,
    properties_with_crowding: List[tuple], 
    enquiry: EnquiryForm,
    omega_weights: Tuple[float, float, float] = None,
    alpha: float = 0.5 
) -> List[Property]:
    
    ranked_data = []
    
    for prop, layer_idx, crowding_dist in properties_with_crowding:
        explicit_score = (
            enquiry.importance_rent * prop.costScore +
            enquiry.importance_location * prop.commuteScore +
            enquiry.importance_facility * prop.neighborhoodScore
        )
        
        if omega_weights is not None:
            preference_result = calculate_preference_score(prop, omega_weights)
            implicit_score = preference_result.final_score
        else:
            implicit_score = 0
        
        if omega_weights is not None:
            final_score = alpha * implicit_score + (1 - alpha) * explicit_score
        else:
            final_score = explicit_score
        
        ranked_data.append({
            'property': prop,
            'layer': layer_idx,
            'crowding': crowding_dist,
            'weighted_score': final_score  
        })
    
    ranked_data.sort(
        key=lambda x: (
            x['layer'],
            -x['crowding'] if x['crowding'] != float('inf') else float('-inf'),
            -x['weighted_score']
        )
    )
    
    return [item['property'] for item in ranked_data]
