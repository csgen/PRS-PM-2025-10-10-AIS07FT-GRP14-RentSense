"use client"

import type React from "react"

import { Star } from "lucide-react"
import { cn } from "@/lib/utils"
import { useState } from "react"

interface StarRatingProps {
  value: number
  onChange: (value: number) => void
  max?: number
  label?: string
  className?: string
  allowHalf?: boolean // Added support for half-star ratings
}

export function StarRating({ value, onChange, max = 5, label, className, allowHalf = false }: StarRatingProps) {
  const [hoverValue, setHoverValue] = useState<number | null>(null)

  const displayValue = hoverValue !== null ? hoverValue : value

  const handleClick = (star: number, isHalf: boolean) => {
    if (allowHalf && isHalf) {
      onChange(star - 0.5)
    } else {
      onChange(star)
    }
  }

  const handleMouseMove = (star: number, e: React.MouseEvent<HTMLButtonElement>) => {
    if (!allowHalf) {
      setHoverValue(star)
      return
    }

    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const isLeftHalf = x < rect.width / 2

    setHoverValue(isLeftHalf ? star - 0.5 : star)
  }

  const handleMouseLeave = () => {
    setHoverValue(null)
  }

  const getStarFill = (star: number) => {
    const diff = displayValue - (star - 1)
    if (diff >= 1) return "full"
    if (diff > 0 && diff < 1) return "half"
    return "empty"
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {label && <label className="text-sm font-medium text-foreground">{label}</label>}
      <div className="flex gap-1" onMouseLeave={handleMouseLeave}>
        {Array.from({ length: max }, (_, i) => i + 1).map((star) => {
          const fillState = getStarFill(star)

          return (
            <button
              key={star}
              type="button"
              onClick={(e) => {
                if (!allowHalf) {
                  onChange(star)
                  return
                }
                const rect = e.currentTarget.getBoundingClientRect()
                const x = e.clientX - rect.left
                const isLeftHalf = x < rect.width / 2
                handleClick(star, isLeftHalf)
              }}
              onMouseMove={(e) => handleMouseMove(star, e)}
              className="relative transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
            >
              {fillState === "half" ? (
                <div className="relative">
                  <Star className="h-6 w-6 fill-none text-muted-foreground" />
                  <Star
                    className="h-6 w-6 fill-amber-400 text-amber-400 absolute top-0 left-0"
                    style={{ clipPath: "inset(0 50% 0 0)" }}
                  />
                </div>
              ) : (
                <Star
                  className={cn(
                    "h-6 w-6 transition-colors",
                    fillState === "full"
                      ? "fill-amber-400 text-amber-400"
                      : "fill-none text-muted-foreground hover:text-amber-400",
                  )}
                />
              )}
            </button>
          )
        })}
        <span className="ml-2 text-sm text-muted-foreground">
          ({displayValue.toFixed(allowHalf ? 1 : 0)}/{max})
        </span>
      </div>
    </div>
  )
}
