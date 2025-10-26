"use client"

import { useEffect, useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { api } from "@/lib/api"
import { FALLBACK_MAP_HTML } from "@/lib/constants/fallback-map"
import { Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"

export default function MapPage() {
  const params = useParams()
  const searchParams = useSearchParams()

  const propertyId = params.property_id as string
  const latitude = searchParams.get("lat")
  const longitude = searchParams.get("lng")

  const [mapHtml, setMapHtml] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [usingFallback, setUsingFallback] = useState(false)

  useEffect(() => {
    const fetchMapData = async () => {
      if (!propertyId || !latitude || !longitude) {
        setError("Missing required map parameters (property_id, lat, lng)")
        setLoading(false)
        setMapHtml(FALLBACK_MAP_HTML)
        setUsingFallback(true)
        return
      }

      try {
        setLoading(true)
        setError(null)
        
        console.log("[v0] Fetching map with params:", { 
          propertyId, 
          latitude, 
          longitude 
        })
        

        const response = await api.getPropertyMap(
          Number(propertyId), 
          latitude,
          longitude
        )
        
        if (response.data && response.data.html) {
          console.log("[v0] Map HTML received successfully")
          setMapHtml(response.data.html)
          setUsingFallback(false)
        } else {
          console.warn("[v0] No map HTML in response, using fallback")
          setMapHtml(FALLBACK_MAP_HTML)
          setUsingFallback(true)
          setError("Map data not available from server")
        }
      } catch (err: any) {
        console.error("[v0] Error fetching map data:", {
          message: err.message,
          status: err.response?.status,
          statusText: err.response?.statusText,
          data: err.response?.data
        })
        
        setMapHtml(FALLBACK_MAP_HTML)
        setUsingFallback(true)
        

        if (err.response?.status === 404) {
          setError("Map endpoint not found (404). The API may not support this property or the endpoint is incorrect.")
        } else if (err.name === "NetworkError") {
          setError("Network error: Unable to connect to map service")
        } else {
          setError(`Unable to load property map: ${err.message || "Unknown error"}`)
        }
      } finally {
        setLoading(false)
      }
    }

    fetchMapData()
  }, [propertyId, latitude, longitude])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
          <p className="text-muted-foreground">Loading map...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <div className="bg-card border-b border-border p-4 shadow-sm">
        <div className="container mx-auto flex items-center justify-between">
          <div className="space-y-1">
            <h1 className="text-xl font-semibold text-foreground">
              Property Map {propertyId && `#${propertyId}`}
            </h1>
            {usingFallback && (
              <p className="text-sm text-amber-600">
                ⚠️ Using default Singapore location
              </p>
            )}
            {latitude && longitude && (
              <p className="text-xs text-muted-foreground">
                Coordinates: {latitude}, {longitude}
              </p>
            )}
          </div>
          <Button variant="outline" onClick={() => window.close()}>
            Close
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="container mx-auto px-4 pt-4">
          <Alert variant={usingFallback ? "default" : "destructive"}>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* Map Container */}
      <div className="flex-1 overflow-hidden">
        <iframe
          srcDoc={mapHtml}
          className="w-full h-full border-0"
          title="Property Map"
          sandbox="allow-scripts allow-same-origin"
        />
      </div>
    </div>
  )
}