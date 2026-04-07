import * as React from "react"
import { cn } from "@/lib/utils"

interface SegmentedOption<T extends string> {
  value: T
  label: React.ReactNode
  title?: string
  disabled?: boolean
}

interface SegmentedProps<T extends string> {
  value: T
  onValueChange: (value: T) => void
  options: ReadonlyArray<SegmentedOption<T>>
  className?: string
}

/**
 * Segmented control: a row of equal-width buttons with an animated pill
 * sliding behind the active option. Same vibe as iOS / shadcn tabs.
 */
export function Segmented<T extends string>({
  value,
  onValueChange,
  options,
  className,
}: SegmentedProps<T>) {
  const activeIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value)
  )
  const n = options.length

  return (
    <div
      className={cn(
        "relative inline-flex h-9 w-full items-center rounded-md bg-muted p-1 text-muted-foreground",
        className
      )}
      role="tablist"
    >
      {/* Sliding active pill. Container inner width = 100% - 0.5rem
          (the p-1 padding); each slot is that divided by N. */}
      <div
        className="absolute top-1 bottom-1 rounded-sm bg-background shadow-sm transition-all duration-200 ease-out"
        style={{
          width: `calc((100% - 0.5rem) / ${n})`,
          left: `calc(0.25rem + (100% - 0.5rem) / ${n} * ${activeIndex})`,
        }}
      />
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={active}
            disabled={opt.disabled}
            title={opt.title}
            onClick={() => onValueChange(opt.value)}
            className={cn(
              "relative z-10 flex-1 rounded-sm px-3 py-1 text-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              "disabled:pointer-events-none disabled:opacity-50",
              active ? "text-foreground" : "hover:text-foreground"
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
