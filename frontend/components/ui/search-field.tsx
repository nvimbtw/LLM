import { Search } from "lucide-react"

import { Input, type InputProps } from "./input"
import { cn } from "@/lib/utils"

export interface SearchFieldProps extends InputProps {
  containerClassName?: string
}

/** Renders a labelled search input with one thin underline; input classes and refs retain the Input contract while container classes control its layout. */
export function SearchField({
  className,
  containerClassName,
  type = "search",
  ...props
}: SearchFieldProps) {
  return (
    <div className={cn("relative min-w-0", containerClassName)}>
      <Search
        aria-hidden="true"
        className="pointer-events-none absolute top-1/2 left-0 size-3.5 -translate-y-1/2 text-muted-foreground"
      />
      <Input
        {...props}
        type={type}
        variant="ruled"
        className={cn("h-9 pl-6 focus-visible:border-b", className)}
      />
    </div>
  )
}
