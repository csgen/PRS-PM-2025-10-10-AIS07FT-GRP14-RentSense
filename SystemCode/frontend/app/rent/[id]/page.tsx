"use client"

import { useEffect, useState, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { api, type Property } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  ArrowLeft,
  MapPin,
  Home,
  Bed,
  Bath,
  Maximize,
  Calendar,
  School,
  Train,
  Star,
  Loader2,
  AlertCircle,
  ArrowUpRight,
  Heart,
} from "lucide-react"
import Image from "next/image"
import { FALLBACK_MAP_HTML } from "@/lib/constants/fallback-map"
import { useBehaviorTracking } from "@/hooks/use-behavior-tracking"
import { isFavorite, toggleFavorite } from "@/lib/favorites"

export default function RentDetailPage() {
  const params = useParams()
  const router = useRouter()
  const id = params.id as string

  const [property, setProperty] = useState<Property | null>(null)
  const [mapHtml, setMapHtml] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isFavorited, setIsFavorited] = useState(false)

  const { trackView, trackFavorite } = useBehaviorTracking()
  const entryTimeRef = useRef<number>(Date.now())
  const hasTrackedRef = useRef<boolean>(false)

  useEffect(() => {
    if (property) {
      setIsFavorited(isFavorite(property.property_id))
    }
  }, [property])

  useEffect(() => {
    // Reset entry time when component mounts
    entryTimeRef.current = Date.now()
    hasTrackedRef.current = false

    const sendViewTracking = () => {
      if (!property || hasTrackedRef.current) return

      const dwellTime = (Date.now() - entryTimeRef.current) / 1000
      const dwellTimeRounded = Math.round(dwellTime * 10) / 10 // Round to 1 decimal place

      console.log("[v0] Tracking view:", {
        property_id: property.property_id,
        dwell_time: dwellTimeRounded,
      })

      trackView(property.property_id, dwellTimeRounded)
      hasTrackedRef.current = true
    }

    // Track on beforeunload (page close, refresh, navigation)
    const handleBeforeUnload = () => {
      sendViewTracking()
    }

    window.addEventListener("beforeunload", handleBeforeUnload)

    // Cleanup: track on component unmount
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
      sendViewTracking()
    }
  }, [property, trackView])

  useEffect(() => {
    const fetchPropertyData = async () => {
      try {
        setLoading(true)
        setError(null)

        // 1. 首先尝试从 sessionStorage 获取（最优先，来自列表页直接点击）
        const sessionData = sessionStorage.getItem(`property_${id}`)
        if (sessionData) {
          try {
            const propertyData = JSON.parse(sessionData)
            console.log("[v0] Property data loaded from sessionStorage:", propertyData)
            setProperty(propertyData)
            await fetchMap(propertyData)
            setLoading(false)
            return
          } catch (parseErr) {
            console.error("[v0] Error parsing sessionStorage data:", parseErr)
          }
        }

        // 2. 尝试从 localStorage 的推荐列表中查找（备选方案）
        const savedData = localStorage.getItem("recommendations_data")
        if (savedData) {
          try {
            const data = JSON.parse(savedData)
            const foundProperty = data.properties?.find((p: Property) => p.property_id === Number(id))

            if (foundProperty) {
              console.log("[v0] Property data loaded from localStorage:", foundProperty)
              setProperty(foundProperty)
              // 同时保存到 sessionStorage 以便后续访问
              sessionStorage.setItem(`property_${id}`, JSON.stringify(foundProperty))
              await fetchMap(foundProperty)
              setLoading(false)
              return
            }
          } catch (parseErr) {
            console.error("[v0] Error parsing localStorage data:", parseErr)
          }
        }

        // 3. 如果都没找到，显示错误（不再调用后端 API）
        throw new Error("Property data not found. Please access from the recommendations page.")

      } catch (err: any) {
        console.error("[v0] Error loading property:", err)
        setError(err.message || "Failed to load property details. Please return to recommendations page and try again.")
      } finally {
        setLoading(false)
      }
    }

    const fetchMap = async (propertyData: Property) => {
      try {
        const lat = String(propertyData.latitude)
        const lng = String(propertyData.longitude)

        console.log("[v0] Fetching map with coordinates:", {
          propertyId: propertyData.property_id,
          lat,
          lng,
          latType: typeof lat,
          lngType: typeof lng,
        })

        const mapResponse = await api.getPropertyMap(propertyData.property_id, lat, lng)

        if (mapResponse.data && mapResponse.data.html) {
          console.log("[v0] Map HTML received successfully")
          setMapHtml(mapResponse.data.html)
        } else {
          console.log("[v0] No map HTML returned, using fallback")
          setMapHtml(FALLBACK_MAP_HTML)
        }
      } catch (mapErr: any) {
        console.error("[v0] Error fetching map HTML:", {
          message: mapErr.message,
          status: mapErr.response?.status,
          data: mapErr.response?.data,
        })
        setMapHtml(FALLBACK_MAP_HTML)
      }
    }

    if (id) {
      fetchPropertyData()
    }
  }, [id])

  const handleFavoriteClick = () => {
    if (!property) return
    const newFavoriteStatus = toggleFavorite(property.property_id)
    setIsFavorited(newFavoriteStatus)
    trackFavorite(property.property_id, newFavoriteStatus)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-50/50 to-background">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
          <p className="text-muted-foreground">Loading property details...</p>
        </div>
      </div>
    )
  }

  if (error || !property) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-b from-blue-50/50 to-background">
        <div className="max-w-md w-full space-y-4">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error || "Property not found. Please access from the recommendations page."}
            </AlertDescription>
          </Alert>
          <div className="flex gap-2">
            <Button onClick={() => router.push("/recomm")} className="flex-1">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Recommendations
            </Button>
            <Button onClick={() => router.push("/")} variant="outline" className="flex-1">
              Go Home
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/50 to-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <Button variant="ghost" onClick={() => router.push("/recomm")} className="mb-6">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Recommendations
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <Card className="overflow-hidden">
              <div className="relative h-96 w-full bg-muted">
                <Image
                  src={property.img_src || "/placeholder.svg"}
                  alt={property.name}
                  fill
                  className="object-cover"
                  priority
                />
                {property.facility_type && (
                  <Badge className="absolute top-4 right-4 bg-primary text-primary-foreground text-base px-4 py-2">
                    {property.facility_type}
                  </Badge>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-4 left-4 bg-white/90 hover:bg-white backdrop-blur-sm h-12 w-12"
                  onClick={handleFavoriteClick}
                >
                  <Heart
                    className={`h-6 w-6 transition-colors ${
                      isFavorited ? "fill-red-500 text-red-500" : "text-gray-600"
                    }`}
                  />
                </Button>
              </div>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <CardTitle className="text-3xl">{property.name}</CardTitle>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <MapPin className="h-4 w-4" />
                      <span>{property.location}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-primary">${property.price}</div>
                    <div className="text-sm text-muted-foreground">Per Month</div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    <Bed className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm text-muted-foreground">Bedrooms</div>
                      <div className="font-semibold">{property.beds}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    <Bath className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm text-muted-foreground">Bathrooms</div>
                      <div className="font-semibold">{property.baths}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    <Maximize className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm text-muted-foreground">Area</div>
                      <div className="font-semibold">{property.area} m²</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    <Calendar className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm text-muted-foreground">Built Year</div>
                      <div className="font-semibold">{property.build_time}</div>
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <MapPin className="h-5 w-5 text-primary" />
                    Location Information
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">District</div>
                      <div className="font-medium">{property.district}</div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">Detailed Address</div>
                      <div className="font-medium">{property.location}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <School className="h-4 w-4 text-primary" />
                      <div>
                        <span className="text-sm text-muted-foreground">To School:</span>
                        <span className="font-medium ml-2">{property.time_to_school}min</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Train className="h-4 w-4 text-primary" />
                      <div>
                        <span className="text-sm text-muted-foreground">To MRT Station:</span>
                        <span className="font-medium ml-2">{property.distance_to_mrt}m</span>
                      </div>
                    </div>
                  </div>
                </div>

                {property.public_facilities && property.public_facilities.length > 0 && (
                  <>
                    <Separator />
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Home className="h-5 w-5 text-primary" />
                        Nearby Facilities
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {property.public_facilities.map((facility, index) => {
                          const [name, distance] = Object.entries(facility)[0]
                          return (
                            <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                              <span className="font-medium text-sm">{name}</span>
                              <Badge variant="secondary">{distance}m</Badge>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {mapHtml && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MapPin className="h-5 w-5 text-primary" />
                    Map Location
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-96 rounded-lg overflow-hidden">
                    <iframe
                      srcDoc={mapHtml}
                      className="w-full h-full border-0"
                      title="Property Map"
                      sandbox="allow-scripts allow-same-origin"
                    />
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="space-y-6">
            <Card className="sticky top-24">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="h-5 w-5 text-primary" />
                  Recommendation Reason
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground leading-relaxed">
                  {property.recommand_reason || "This property matches your requirements."}
                </p>
                <Separator />
                <div className="space-y-3">
                  <Button
                    variant={isFavorited ? "default" : "outline"}
                    className="w-full"
                    size="lg"
                    onClick={handleFavoriteClick}
                  >
                    <Heart className={`h-4 w-4 mr-2 ${isFavorited ? "fill-current" : ""}`} />
                    {isFavorited ? "Favorited" : "Add to Favorites"}
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full bg-transparent"
                    size="lg"
                    onClick={() => window.open("https://www.propertyguru.com.sg/property-for-rent", "_blank")}
                  >
                    <ArrowUpRight className="h-4 w-4 mr-2" />
                    View on PropertyGuru
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full bg-transparent"
                    size="lg"
                    onClick={() => {
                      const lat = String(property.latitude)
                      const lng = String(property.longitude)
                      window.open(`/map/${property.property_id}?lat=${lat}&lng=${lng}`, "_blank")
                    }}
                  >
                    <MapPin className="h-4 w-4 mr-2" />
                    View Map in New Window
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}