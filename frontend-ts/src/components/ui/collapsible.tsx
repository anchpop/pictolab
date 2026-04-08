import * as React from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface CollapsibleProps {
  /** Header label rendered next to the chevron. */
  label: React.ReactNode
  /** Initial open state. */
  defaultOpen?: boolean
  /** Controlled open state. */
  open?: boolean
  /** Controlled state change callback. */
  onOpenChange?: (open: boolean) => void
  /** Optional right-side content shown in the header (e.g. a status pill). */
  meta?: React.ReactNode
  className?: string
  contentClassName?: string
  children: React.ReactNode
}

/**
 * Lightweight collapsible section: a header button that toggles a body
 * underneath. The chevron rotates when open. No animation library —
 * just a CSS rotate transition on the chevron and conditional render on
 * the body.
 */
export function Collapsible({
  label,
  defaultOpen = false,
  open: openProp,
  onOpenChange,
  meta,
  className,
  contentClassName,
  children,
}: CollapsibleProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = React.useState(defaultOpen)
  const open = openProp ?? uncontrolledOpen
  const setOpen = (next: boolean | ((prev: boolean) => boolean)) => {
    const resolved = typeof next === "function" ? next(open) : next
    if (openProp === undefined) setUncontrolledOpen(resolved)
    onOpenChange?.(resolved)
  }
  return (
    <div className={cn("space-y-2", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between rounded-md text-left text-xs font-medium text-muted-foreground hover:text-foreground"
        aria-expanded={open}
      >
        <span className="flex items-center gap-1">
          <ChevronDown
            className={cn(
              "h-3 w-3 transition-transform duration-150",
              open ? "rotate-0" : "-rotate-90"
            )}
          />
          {label}
        </span>
        {meta}
      </button>
      {open && <div className={cn("space-y-2", contentClassName)}>{children}</div>}
    </div>
  )
}
