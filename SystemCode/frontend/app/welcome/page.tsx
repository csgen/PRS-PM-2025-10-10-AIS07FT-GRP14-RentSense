"use client"

import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { Home, MapPin, Shield, TrendingUp } from "lucide-react"

export default function WelcomePage() {
  const router = useRouter()

  const handleGetStarted = () => {
    router.push("/userin")
  }

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-background to-accent/10">
        <div className="container mx-auto px-4 py-20 md:py-32">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-16 items-center">
            {/* Left Content */}
            <div className="space-y-8 text-center lg:text-left">
              <div className="space-y-4">
                <h1 className="font-serif text-4xl md:text-5xl lg:text-6xl font-bold text-balance leading-tight">
                  Find Your
                  <span className="text-primary block mt-2">Ideal Home</span>
                </h1>
                <p className="text-lg md:text-xl text-muted-foreground text-pretty max-w-2xl mx-auto lg:mx-0">
                  Our intelligent recommendation system helps you quickly find the most suitable rental options, making
                  house hunting simple and efficient
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Button
                  size="lg"
                  onClick={handleGetStarted}
                  className="text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all"
                >
                  <Home className="mr-2 h-5 w-5" />
                  Start Searching
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => router.push("/recomm")}
                  className="text-lg px-8 py-6"
                >
                  Browse Recommendations
                </Button>
              </div>

              {/* Feature Stats */}
              <div className="grid grid-cols-3 gap-4 pt-8 border-t border-border">
                <div className="text-center lg:text-left">
                  <div className="text-2xl md:text-3xl font-bold text-primary">1000+</div>
                  <div className="text-sm text-muted-foreground">Quality Properties</div>
                </div>
                <div className="text-center lg:text-left">
                  <div className="text-2xl md:text-3xl font-bold text-primary">95%</div>
                  <div className="text-sm text-muted-foreground">Satisfaction</div>
                </div>
                <div className="text-center lg:text-left">
                  <div className="text-2xl md:text-3xl font-bold text-primary">24h</div>
                  <div className="text-sm text-muted-foreground">Quick Response</div>
                </div>
              </div>
            </div>

            {/* Right Image/Visual */}
            <div className="relative">
              <div className="relative aspect-[4/3] rounded-2xl overflow-hidden shadow-2xl">
                <img
                  src="/modern-apartment-building-exterior-with-blue-sky.jpg"
                  alt="Modern apartment building"
                  className="object-cover w-full h-full"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-primary/20 to-transparent" />
              </div>

              {/* Floating Cards */}
              <div className="absolute -bottom-6 -left-6 bg-card p-4 rounded-xl shadow-lg border border-border hidden md:block">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <Shield className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <div className="font-semibold">Safety Guarantee</div>
                    <div className="text-sm text-muted-foreground">Verified Identity</div>
                  </div>
                </div>
              </div>

              <div className="absolute -top-6 -right-6 bg-card p-4 rounded-xl shadow-lg border border-border hidden md:block">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <TrendingUp className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <div className="font-semibold">Smart Matching</div>
                    <div className="text-sm text-muted-foreground">AI Recommendations</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Decorative Elements */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl -z-10" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent/10 rounded-full blur-3xl -z-10" />
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">Why Choose HouseFinder</h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              We provide the most professional rental services to make your house hunting journey easier and more
              enjoyable
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-card p-8 rounded-xl shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="h-14 w-14 rounded-lg bg-primary/10 flex items-center justify-center mb-6">
                <MapPin className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold text-xl mb-3">Precise Location</h3>
              <p className="text-muted-foreground leading-relaxed">
                Based on your workplace, school location and other needs, we intelligently recommend the most suitable
                property locations
              </p>
            </div>

            <div className="bg-card p-8 rounded-xl shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="h-14 w-14 rounded-lg bg-primary/10 flex items-center justify-center mb-6">
                <Shield className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold text-xl mb-3">Safe & Reliable</h3>
              <p className="text-muted-foreground leading-relaxed">
                All properties are strictly verified, landlords are identity-verified, ensuring your rental safety
              </p>
            </div>

            <div className="bg-card p-8 rounded-xl shadow-sm border border-border hover:shadow-md transition-shadow">
              <div className="h-14 w-14 rounded-lg bg-primary/10 flex items-center justify-center mb-6">
                <TrendingUp className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold text-xl mb-3">Smart Recommendations</h3>
              <p className="text-muted-foreground leading-relaxed">
                AI algorithms analyze your needs and recommend the most matching properties, saving you time in house
                hunting
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
