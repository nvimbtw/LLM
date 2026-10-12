"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ClassicShowcase } from "./classic-showcase"
import { LiveShowcase } from "./live-showcase"

export function ThrobberShowcase() {
  return (
    <Tabs defaultValue="live">
      <TabsList variant="line" className="mb-6">
        <TabsTrigger value="live">Live art</TabsTrigger>
        <TabsTrigger value="classic">Classic showcase</TabsTrigger>
      </TabsList>
      <TabsContent value="live">
        <LiveShowcase />
      </TabsContent>
      <TabsContent value="classic">
        <ClassicShowcase />
      </TabsContent>
    </Tabs>
  )
}
