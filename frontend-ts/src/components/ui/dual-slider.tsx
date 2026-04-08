import * as React from "react"
import { cn } from "@/lib/utils"

interface DualSliderProps {
  min: number
  max: number
  step?: number
  value: [number, number]
  onValueChange: (value: [number, number]) => void
  /** Region [a, b] outside of which the track is shaded as a "danger" zone. */
  safeRange?: [number, number]
  /** Values that thumbs should magnetize toward when dragged near. */
  snapPoints?: number[]
  /** Tolerance in value units. Defaults to 1.5% of (max - min). */
  snapRadius?: number
  className?: string
}

/**
 * Two-thumb slider where each handle moves independently — the "low" thumb
 * may be dragged past the "high" thumb and vice versa. The filled range
 * always spans between them regardless of order.
 */
export function DualSlider({
  min,
  max,
  step = 1,
  value,
  onValueChange,
  safeRange,
  snapPoints,
  snapRadius,
  className,
}: DualSliderProps) {
  const radius = snapRadius ?? (max - min) * 0.015
  const snap = (v: number) => {
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
  }
  const trackRef = React.useRef<HTMLDivElement>(null)
  const activeRef = React.useRef<0 | 1 | null>(null)
  const [isHovered, setIsHovered] = React.useState(false)
  const [isFocused, setIsFocused] = React.useState(false)
  const [isDragging, setIsDragging] = React.useState(false)

  const pct = (v: number) => ((v - min) / (max - min)) * 100
  const lo = Math.min(value[0], value[1])
  const hi = Math.max(value[0], value[1])
  const showSafeRange = isHovered || isFocused || isDragging

  const valueFromClientX = (clientX: number) => {
    const track = trackRef.current
    if (!track) return min
    const rect = track.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const raw = min + ratio * (max - min)
    return snap(Math.round(raw / step) * step)
  }

  const handlePointerDown = (idx: 0 | 1) => (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    activeRef.current = idx
    setIsDragging(true)
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  const handlePointerMove = (e: React.PointerEvent) => {
    const idx = activeRef.current
    if (idx === null) return
    const v = valueFromClientX(e.clientX)
    const next: [number, number] = [...value] as [number, number]
    next[idx] = v
    onValueChange(next)
  }

  const handlePointerUp = (e: React.PointerEvent) => {
    if (activeRef.current === null) return
    activeRef.current = null
    setIsDragging(false)
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)
  }

  const handleTrackPointerDown = (e: React.PointerEvent) => {
    // Click-to-jump: move whichever thumb is closer.
    const v = valueFromClientX(e.clientX)
    const d0 = Math.abs(v - value[0])
    const d1 = Math.abs(v - value[1])
    const idx: 0 | 1 = d0 <= d1 ? 0 : 1
    const next: [number, number] = [...value] as [number, number]
    next[idx] = v
    onValueChange(next)
    activeRef.current = idx
    setIsDragging(true)
    ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
  }

  return (
    <div
      className={cn(
        "relative flex h-9 w-full touch-none select-none items-center",
        className
      )}
      onPointerEnter={() => setIsHovered(true)}
      onPointerLeave={() => setIsHovered(false)}
      onFocusCapture={() => setIsFocused(true)}
      onBlurCapture={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
          setIsFocused(false)
        }
      }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <div
        ref={trackRef}
        onPointerDown={handleTrackPointerDown}
        className="relative h-1.5 w-full grow cursor-pointer overflow-hidden rounded-full bg-secondary"
      >
        {safeRange && safeRange[0] > min && (
          <div
            className={cn(
              "absolute h-full bg-destructive/35 transition-opacity duration-150",
              showSafeRange ? "opacity-100" : "opacity-0"
            )}
            style={{ left: 0, width: `${pct(safeRange[0])}%` }}
          />
        )}
        {safeRange && safeRange[1] < max && (
          <div
            className={cn(
              "absolute h-full bg-destructive/35 transition-opacity duration-150",
              showSafeRange ? "opacity-100" : "opacity-0"
            )}
            style={{ left: `${pct(safeRange[1])}%`, width: `${100 - pct(safeRange[1])}%` }}
          />
        )}
        <div
          className="absolute h-full bg-primary"
          style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }}
        />
      </div>
      {[0, 1].map((i) => (
        <div
          key={i}
          role="slider"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value[i as 0 | 1]}
          tabIndex={0}
          onPointerDown={handlePointerDown(i as 0 | 1)}
          className={cn(
            "absolute h-4 w-4 -translate-x-1/2 cursor-grab rounded-full border-2 border-primary shadow ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 active:cursor-grabbing",
            // First handle filled, second handle hollow — gives a clear
            // visual cue to which endpoint is "low" vs "high" so an
            // inversion (swapped order) is obvious at a glance.
            i === 0 ? "bg-primary" : "bg-background"
          )}
          style={{ left: `${pct(value[i as 0 | 1])}%`, top: 'calc(50% - 8px)' }}
        />
      ))}
      {[0, 1].map((i) => (
        <div
          key={`label-${i}`}
          className="pointer-events-none absolute -translate-x-1/2 font-mono text-[10px] text-muted-foreground"
          style={{ left: `${pct(value[i as 0 | 1])}%`, top: 'calc(50% + 10px)' }}
        >
          {value[i as 0 | 1]}
        </div>
      ))}
    </div>
  )
}
