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
>(({ className, value, defaultValue, snapPoints, snapRadius, onValueChange, onValueCommit, min = 0, max = 100, step, disabled, ...props }, ref) => {
  const thumbCount =
    (Array.isArray(value) ? value.length : undefined) ??
    (Array.isArray(defaultValue) ? defaultValue.length : 1)
  const wrapRef = React.useRef<HTMLDivElement>(null)

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

  // Click-to-jump: pointer hits on the (taller) wrapper compute the value
  // from x position and move the nearest thumb. Radix supports this on the
  // track itself, but the track is only 6px tall and the thumb absorbs
  // vertical hits within its bounds — making the effective click target
  // tiny. Wrapping in a 36px-tall hit area fixes that.
  const handleWrapPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return
    // If the click landed on the thumb itself, let Radix handle it (drag).
    const target = e.target as HTMLElement
    if (target.closest('[role="slider"]')) return
    const w = wrapRef.current
    if (!w) return
    const rect = w.getBoundingClientRect()
    if (rect.width === 0) return
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const stepNum = (typeof step === 'number' && step > 0 ? step : 1) as number
    const raw = (min as number) + ratio * ((max as number) - (min as number))
    const stepped = Math.round(raw / stepNum) * stepNum
    const snapped = snap(stepped)
    if (Array.isArray(value) && value.length >= 1) {
      // Move the nearest thumb to the click position.
      let nearest = 0
      let bestD = Infinity
      for (let i = 0; i < value.length; i++) {
        const d = Math.abs(value[i] - snapped)
        if (d < bestD) {
          bestD = d
          nearest = i
        }
      }
      const next = [...value]
      next[nearest] = snapped
      onValueChange?.(next)
      onValueCommit?.(next)
    }
  }

  return (
    <div
      ref={wrapRef}
      className="relative flex h-9 w-full touch-none select-none items-center"
      onPointerDown={handleWrapPointerDown}
    >
    <SliderPrimitive.Root
      ref={ref}
      value={value}
      defaultValue={defaultValue}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={handleChange}
      onValueCommit={onValueCommit}
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
    </div>
  )
})
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }
