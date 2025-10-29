"use client"

import { useCallback } from "react"
import { getDeviceId } from "@/lib/device"
import { behaviorApi } from "@/lib/api"
/**
 * Custom hook for tracking user behaviors
 * Provides methods to track clicks, views, and favorites
 */
export function useBehaviorTracking() {
  /**
   * Track property click behavior
   * @param property_id - The ID of the property that was clicked
   */
  const trackClick = useCallback(async (property_id: number) => {
    const device_id = getDeviceId()
    await behaviorApi.trackClick(device_id, property_id)
  }, [])

  /**
   * Track property view behavior
   * @param property_id - The ID of the property that was viewed
   * @param dwell_time - Time spent viewing the property in seconds
   */
  const trackView = useCallback(async (property_id: number, dwell_time: number) => {
    const device_id = getDeviceId()
    await behaviorApi.trackView(device_id, property_id, dwell_time)
  }, [])

  /**
   * Track property favorite behavior
   * @param property_id - The ID of the property that was favorited/unfavorited
   * @param favorite - True if favorited, false if unfavorited
   */
  const trackFavorite = useCallback(async (property_id: number, favorite: boolean) => {
    const device_id = getDeviceId()
    await behaviorApi.trackFavorite(device_id, property_id, favorite)
  }, [])

  return {
    trackClick,
    trackView,
    trackFavorite,
  }
}
