/**
 * Utility functions for managing property favorites in localStorage
 */

const FAVORITES_KEY = "property_favorites"

export interface FavoriteProperty {
  property_id: number
  timestamp: number
}

/**
 * Get all favorited property IDs
 */
export function getFavorites(): number[] {
  if (typeof window === "undefined") return []

  try {
    const stored = localStorage.getItem(FAVORITES_KEY)
    if (!stored) return []

    const favorites: FavoriteProperty[] = JSON.parse(stored)
    return favorites.map((f) => f.property_id)
  } catch (error) {
    console.error("Error reading favorites:", error)
    return []
  }
}

/**
 * Check if a property is favorited
 */
export function isFavorite(propertyId: number): boolean {
  const favorites = getFavorites()
  return favorites.includes(propertyId)
}

/**
 * Add a property to favorites
 */
export function addFavorite(propertyId: number): void {
  if (typeof window === "undefined") return

  try {
    const favorites = getFavorites()
    if (!favorites.includes(propertyId)) {
      const stored = localStorage.getItem(FAVORITES_KEY)
      const favoritesData: FavoriteProperty[] = stored ? JSON.parse(stored) : []

      favoritesData.push({
        property_id: propertyId,
        timestamp: Date.now(),
      })

      localStorage.setItem(FAVORITES_KEY, JSON.stringify(favoritesData))
    }
  } catch (error) {
    console.error("Error adding favorite:", error)
  }
}

/**
 * Remove a property from favorites
 */
export function removeFavorite(propertyId: number): void {
  if (typeof window === "undefined") return

  try {
    const stored = localStorage.getItem(FAVORITES_KEY)
    if (!stored) return

    const favoritesData: FavoriteProperty[] = JSON.parse(stored)
    const filtered = favoritesData.filter((f) => f.property_id !== propertyId)

    localStorage.setItem(FAVORITES_KEY, JSON.stringify(filtered))
  } catch (error) {
    console.error("Error removing favorite:", error)
  }
}

/**
 * Toggle favorite status for a property
 */
export function toggleFavorite(propertyId: number): boolean {
  const isCurrentlyFavorite = isFavorite(propertyId)

  if (isCurrentlyFavorite) {
    removeFavorite(propertyId)
    return false
  } else {
    addFavorite(propertyId)
    return true
  }
}
