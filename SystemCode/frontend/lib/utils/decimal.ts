import Decimal from "decimal.js"

/**
 * Safely converts a value to a number
 * Handles Decimal objects, strings, and numbers
 */
export function toNumber(value: Decimal | string | number | undefined | null): number {
  if (value === undefined || value === null) {
    return 0
  }
  
  // If it's already a number, return it
  if (typeof value === 'number') {
    return value
  }
  
  // If it's a string, parse it
  if (typeof value === 'string') {
    const parsed = parseFloat(value)
    return isNaN(parsed) ? 0 : parsed
  }
  
  // If it's a Decimal object with toNumber method
  if (value && typeof value === 'object' && 'toNumber' in value) {
    return value.toNumber()
  }
  
  // Fallback
  return 0
}

/**
 * Safely converts a value to a Decimal object
 */
export function toDecimal(value: Decimal | string | number | undefined | null): Decimal {
  if (value === undefined || value === null) {
    return new Decimal(0)
  }
  
  // If it's already a Decimal, return it
  if (value instanceof Decimal) {
    return value
  }
  
  // Convert string or number to Decimal
  try {
    return new Decimal(value)
  } catch (e) {
    console.error('Error converting to Decimal:', value, e)
    return new Decimal(0)
  }
}

/**
 * Converts properties with coordinate fields from strings to Decimal objects
 */
export function convertPropertiesCoordinates<T extends { latitude?: any; longitude?: any }>(
  properties: T[]
): T[] {
  return properties.map((property) => ({
    ...property,
    latitude: toDecimal(property.latitude),
    longitude: toDecimal(property.longitude),
  }))
}