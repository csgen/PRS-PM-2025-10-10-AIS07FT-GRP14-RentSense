"use client"

import type React from "react"

import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MapPin, School, Train, Star, ExternalLink } from "lucide-react"
import { useRouter } from "next/navigation"
import Image from "next/image"
import type { Property } from "@/lib/api"
import { toNumber } from "@/lib/utils/decimal"

interface RentBlockProps {
  property: Property
}

export default function RentBlock({ property }: RentBlockProps) {
  const router = useRouter()

  const handleCardClick = () => {
    router.push(`/rent/${property.property_id}`, { state: { property } } as any)
  }

  const handleViewMap = (e: React.MouseEvent) => {
    e.stopPropagation()
    const lat = toNumber(property.latitude)
    const lng = toNumber(property.longitude)
    window.open(`/map/${property.property_id}?lat=${lat}&lng=${lng}`, "_blank")
  }

  const renderStars = (count: number) => {
    return (
      <div className="flex gap-0.5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Star
            key={i}
            className={`h-4 w-4 ${i < count ? "fill-yellow-400 text-yellow-400" : "fill-gray-200 text-gray-200"}`}
          />
        ))}
      </div>
    )
  }

  return (
    <Card
      className="group cursor-pointer overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1"
      onClick={handleCardClick}
    >
      <CardHeader className="p-0">
        <div className="relative h-48 w-full overflow-hidden bg-muted">
          <Image
            src={property.img_src || "/placeholder.svg"}
            alt={property.name}
            fill
            className="object-cover transition-transform duration-300 group-hover:scale-110"
          />
          <Badge className="absolute top-3 right-3 bg-primary text-primary-foreground">{property.facility_type}</Badge>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-3">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold text-foreground line-clamp-1">{property.name}</h3>
          <div className="flex items-center gap-2 text-muted-foreground">
            <MapPin className="h-4 w-4" />
            <span className="text-sm">{property.district}</span>
          </div>
        </div>

        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-primary">{property.price}</span>
          <span className="text-sm text-muted-foreground">/month</span>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2 border-t">
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <School className="h-3.5 w-3.5" />
              <span>To School</span>
            </div>
            <p className="text-sm font-medium">{property.time_to_school}min</p>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Train className="h-3.5 w-3.5" />
              <span>To MRT</span>
            </div>
            <p className="text-sm font-medium">{property.distance_to_mrt}m</p>
          </div>
        </div>

        {property.public_facilities && property.public_facilities.length > 0 && (
          <div className="pt-2 border-t">
            <p className="text-xs text-muted-foreground mb-2">Nearby Facilities</p>
            <div className="flex flex-wrap gap-1">
              {property.public_facilities.slice(0, 3).map((facility, index) => {
                const [name, distance] = Object.entries(facility)[0]

                return (
                  <Badge key={index} variant="outline" className="text-xs">
                    {name} ({distance}m)
                  </Badge>
                )
              })}
              {property.public_facilities.length > 3 && (
                <Badge variant="outline" className="text-xs">
                  +{property.public_facilities.length - 3}
                </Badge>
              )}
            </div>
          </div>
        )}

        <div className="pt-2 border-t">
          <p className="text-sm text-muted-foreground italic line-clamp-2">{property.recommand_reason}</p>
        </div>
      </CardContent>

      <CardFooter className="p-4 pt-0 flex gap-2">
        <Button variant="outline" size="sm" className="flex-1 bg-transparent" onClick={handleViewMap}>
          <MapPin className="h-4 w-4 mr-1.5" />
          View Map
        </Button>
        <Button size="sm" className="flex-1" onClick={handleCardClick}>
          View Details
          <ExternalLink className="h-4 w-4 ml-1.5" />
        </Button>
      </CardFooter>
    </Card>
  )
}
