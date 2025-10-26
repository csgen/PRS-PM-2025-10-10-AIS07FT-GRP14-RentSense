// Device ID generation and management utilities

/**
 * Generates a unique device ID using UUID v4
 */
function generateDeviceId(): string {
  // Generate UUID v4
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Gets or creates a device ID from localStorage
 * @returns The device ID string
 */
export function getDeviceId(): string {
  if (typeof window === "undefined") {
    // Server-side: return a temporary ID
    return "server-side-temp-id"
  }

  const STORAGE_KEY = "housefinder_device_id"

  try {
    // Try to get existing device ID from localStorage
    let deviceId = localStorage.getItem(STORAGE_KEY)

    if (!deviceId) {
      // Generate new device ID if not exists
      deviceId = generateDeviceId()
      localStorage.setItem(STORAGE_KEY, deviceId)
    }

    return deviceId
  } catch (error) {
    // Fallback if localStorage is not available
    console.error("[v0] Error accessing localStorage:", error)
    return generateDeviceId()
  }
}

/**
 * Clears the stored device ID (useful for testing)
 */
export function clearDeviceId(): void {
  if (typeof window !== "undefined") {
    try {
      localStorage.removeItem("housefinder_device_id")
    } catch (error) {
      console.error("[v0] Error clearing device ID:", error)
    }
  }
}
