import axios, { type AxiosInstance, type AxiosRequestConfig } from "axios"
import type Decimal from "decimal.js"
import { convertPropertiesCoordinates } from "./utils/decimal"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/"

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
})

interface CustomAxiosRequestConfig extends AxiosRequestConfig {
  isMapRequest?: boolean
}

export interface ApiResponse<T = any> {
  code: number // 200 = success, others = error codes
  message: string
  data: T
}

// Request interceptor for adding auth tokens or logging
apiClient.interceptors.request.use(
  (config) => {
    console.log("[v0] API Request:", {
      method: config.method?.toUpperCase(),
      url: config.url,
      baseURL: config.baseURL,
      fullURL: `${config.baseURL}${config.url}`,
      data: config.data,
      params: config.params,
    })


    if (config.url && config.url.includes("/api/v1/properties/map")) {

      (config as CustomAxiosRequestConfig).isMapRequest = true

      config.responseType = 'text'

      config.headers.Accept = 'text/html'
    } else {

      config.headers.Accept = 'application/json, text/plain, */*'
    }

    // Add auth token if available
    const token = typeof window !== "undefined" ? sessionStorage.getItem("auth_token") : null
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    console.error("[v0] Request setup error:", error)
    return Promise.reject(error)
  },
)

apiClient.interceptors.response.use(
  (response) => {
    console.log("[v0] API Response:", {
      status: response.status,
      statusText: response.statusText,
      data: response.data,
      dataType: typeof response.data,
    })

    if ((response.config as CustomAxiosRequestConfig).isMapRequest) {
      console.log("[v0] Received HTML response (Map Request), returning raw data.")
      return response
    }

    // Original JSON processing logic (for standard API responses)
    if (response.data && typeof response.data === "object" && "code" in response.data) {
      const apiResponse = response.data as ApiResponse
      console.log("[v0] API Response Code:", apiResponse.code, "Message:", apiResponse.message)
      if (apiResponse.code === 200) {
        response.data = apiResponse.data
      } else {
        return Promise.reject({
          code: apiResponse.code,
          message: apiResponse.message,
          response: response,
        })
      }
    } else if (typeof response.data === "string" && response.data.trim().startsWith("<!DOCTYPE")) {
      console.log("[v0] Received HTML response (likely map HTML/error page) without map tag.")
      return response
    }

    return response
  },
  (error) => {
    if (error.response) {
      console.error("[v0] API Error Response:", {
        status: error.response.status,
        statusText: error.response.statusText,
        data: error.response.data,
        headers: error.response.headers,
      })

      if (error.response.status === 422 && error.response.data) {
        const errorData = error.response.data


        if (errorData.error_code) {
          return Promise.reject({
            code: errorData.error_code,
            message: errorData.message || "Validation error",
            missing_fields: errorData.missing_fields || [],
            response: error.response,
          })
        }
      }

      if (error.response.data && typeof error.response.data === "object") {
        return Promise.reject({
          code: error.response.data.code || error.response.data.error_code || error.response.status,
          message: error.response.data.message || error.response.statusText,
          response: error.response,
        })
      }
    } else if (error.request) {
      console.error("[v0] Network Error - No response received:", {
        message: error.message,
        url: error.config?.url,
        baseURL: error.config?.baseURL,
      })

      const corsError = new Error(
        "Unable to connect to the API server. Please check your network connection and ensure the backend server is running.",
      )
      corsError.name = "NetworkError"
      return Promise.reject(corsError)
    } else {
      console.error("[v0] Request Setup Error:", error.message)
    }
    return Promise.reject(error)
  },
)

export interface Property {
  property_id: number
  img_src: string
  name: string
  district: string
  price: string
  beds: number
  baths: number
  area: number
  build_time: string
  location: string
  time_to_school: number
  distance_to_mrt: number
  latitude: Decimal
  longitude: Decimal
  public_facilities: Array<{
    [key: string]: string
  }>
  facility_type: string
  recommand_reason: string
}

// Recommendations response interface
export interface RecommendationsResponse {
  properties: Property[]
  total_count: number
}

export interface FormData {
  min_monthly_rent: number
  max_monthly_rent: number
  school_id: number
  target_district_id?: number
  max_school_limit?: number
  flat_type_preference?: string[]
  max_mrt_distance?: number
  importance_rent?: number
  importance_location?: number
  importance_facility?: number
  device_id?: string
}

export interface MapData {
  html: string
}

export const api = {
  // Navigation data
  getNavigation: () => apiClient.get("/api/nav"),

  submitForm: async (data: FormData) => {
    console.log("[v0] Submitting form data:", data)
    const response = await apiClient.post<RecommendationsResponse>("/api/v1/properties/submit-form", data)
    console.log("[v0] Form submission response:", response.data)
    if (response.data && response.data.properties) {
      response.data.properties = convertPropertiesCoordinates(response.data.properties)
    }
    return response
  },

  submitDescription: async (device_id: string, requirement_description: string) => {
    console.log("[v0] Submitting description:", requirement_description)
    const response = await apiClient.post<RecommendationsResponse>("/api/v1/properties/submit-description", {
      device_id,
      requirement_description,
    })
    console.log("[v0] Description submission response:", response.data)
    if (response.data && response.data.properties) {
      response.data.properties = convertPropertiesCoordinates(response.data.properties)
    }
    return response
  },

  getRecommendationsNoSubmit: async () => {
    console.log("[v0] Fetching recommendations without submit (default)")
    const response = await apiClient.get<RecommendationsResponse>("/api/v1/properties/recommendation-no-submit")
    console.log("[v0] Default recommendations response:", response.data)
    if (response.data && response.data.properties) {

      response.data.properties = convertPropertiesCoordinates(response.data.properties)
    }
    return response
  },

  getPropertyDetail: async (propertyId: number) => {
    console.log("[v0] Fetching property detail:", propertyId)
    const response = await apiClient.get<Property>(`/api/v1/properties/${propertyId}`)
    console.log("[v0] Property detail response:", response.data)
    if (response.data) {
      const properties = convertPropertiesCoordinates([response.data])
      return { ...response, data: properties[0] }
    }
    return response
  },


  getPropertyMap: async (propertyId: number, latitude: number | Decimal | string, longitude: number | Decimal | string) => {

    const lat = typeof latitude === 'string' ? parseFloat(latitude) : Number(latitude)
    const lng = typeof longitude === 'string' ? parseFloat(longitude) : Number(longitude)

    console.log("[v0] Fetching property map:", { propertyId, latitude: lat, longitude: lng })


    const response = await apiClient.post<string>("/api/v1/properties/map", {
      property_id: propertyId,
      latitude: lat,
      longitude: lng
    })

    return {
      ...response,
      data: {
        html: response.data
      }
    }
  },
}

export default apiClient