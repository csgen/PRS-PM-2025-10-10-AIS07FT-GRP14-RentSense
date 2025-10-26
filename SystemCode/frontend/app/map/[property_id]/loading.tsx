import { Loader2 } from "lucide-react"

export default function Loading() {
  return (
    <div className="h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-4">
        <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
        <p className="text-muted-foreground">Loading map...</p>
      </div>
    </div>
  )
}
