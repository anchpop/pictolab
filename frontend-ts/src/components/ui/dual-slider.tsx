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
  className,
}: DualSliderProps) {
  const trackRef = React.useRef<HTMLDivElement>(null)
  const activeRef = React.useRef<0 | 1 | null>(null)

  const pct = (v: number) => ((v - min) / (max - min)) * 100
  const lo = Math.min(value[0], value[1])
  const hi = Math.max(value[0], value[1])

  const valueFromClientX = (clientX: number) => {
    const track = trackRef.current
    if (!track) return min
    const rect = track.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const raw = min + ratio * (max - min)
    return Math.round(raw / step) * step
  }

  const handlePointerDown = (idx: 0 | 1) => (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    activeRef.current = idx
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
    ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
  }

  return (
    <div
      className={cn(
        "relative flex h-5 w-full touch-none select-none items-center",
        className
      )}
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
            className="absolute h-full bg-destructive/40"
            style={{ left: 0, width: `${pct(safeRange[0])}%` }}
          />
        )}
        {safeRange && safeRange[1] < max && (
          <div
            className="absolute h-full bg-destructive/40"
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
          className="absolute h-4 w-4 -translate-x-1/2 cursor-grab rounded-full border-2 border-primary bg-background shadow ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 active:cursor-grabbing"
          style={{ left: `${pct(value[i as 0 | 1])}%` }}
        />
      ))}
    </div>
  )
}
