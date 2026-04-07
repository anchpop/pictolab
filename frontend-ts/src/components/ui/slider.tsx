import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"
import { cn } from "@/lib/utils"

type SliderProps = React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> & {
  /** Values that the thumb should magnetize toward when dragged near. */
  snapPoints?: number[]
  /** Tolerance in value units. Defaults to 1.5% of (max - min). */
  snapRadius?: number
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, value, defaultValue, snapPoints, snapRadius, onValueChange, min = 0, max = 100, ...props }, ref) => {
  const thumbCount =
    (Array.isArray(value) ? value.length : undefined) ??
    (Array.isArray(defaultValue) ? defaultValue.length : 1)

  const radius = snapRadius ?? (max - min) * 0.015
  const snap = React.useCallback(
    (v: number) => {
      if (!snapPoints || snapPoints.length === 0) return v
      let best = v
      let bestD = radius
      for (const p of snapPoints) {
        const d = Math.abs(v - p)
        if (d <= bestD) {
          best = p
          bestD = d
        }
      }
      return best
    },
    [snapPoints, radius]
  )

  const handleChange = React.useCallback(
    (vals: number[]) => {
      onValueChange?.(snapPoints ? vals.map(snap) : vals)
    },
    [onValueChange, snap, snapPoints]
  )

  return (
    <SliderPrimitive.Root
      ref={ref}
      value={value}
      defaultValue={defaultValue}
      min={min}
      max={max}
      onValueChange={handleChange}
      className={cn(
        "relative flex w-full touch-none select-none items-center",
        className
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-secondary">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      {Array.from({ length: thumbCount }).map((_, i) => (
        <SliderPrimitive.Thumb
          key={i}
          className="block h-4 w-4 rounded-full border-2 border-primary bg-background shadow ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50"
        />
      ))}
    </SliderPrimitive.Root>
  )
})
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }
