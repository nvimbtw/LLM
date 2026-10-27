"use client"

import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { RuntimeProfile } from "@/lib/architecture/contracts.generated"

export interface RuntimeProfileFieldsProps {
  profile: RuntimeProfile
  devices: readonly string[]
  disabled?: boolean
  onChange: (profile: RuntimeProfile) => void
}

/** Edits prospective full-prefix shape/device/precision inputs; train/eval modes and strict policy remain explicit and these fields never represent measured evidence. */
export function RuntimeProfileFields({
  profile,
  devices,
  disabled,
  onChange,
}: RuntimeProfileFieldsProps) {
  const domain = profile.domains[0]
  return (
    <fieldset disabled={disabled} className="grid gap-4 sm:grid-cols-2">
      <legend className="mb-3 text-sm font-medium">
        Requested validation profile
      </legend>
      <div className="grid gap-2">
        <label className="text-sm" htmlFor="validation-device">
          Device
        </label>
        <Select
          value={profile.device}
          disabled={disabled}
          onValueChange={(device) => {
            if (device) onChange({ ...profile, device })
          }}
        >
          <SelectTrigger id="validation-device">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {devices.map((device) => (
              <SelectItem key={device} value={device}>
                {device}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-2">
        <label className="text-sm" htmlFor="validation-precision">
          Precision
        </label>
        <Select
          value={profile.precision}
          disabled={disabled}
          onValueChange={(precision) => {
            if (
              precision === "fp32" ||
              precision === "fp16" ||
              precision === "bf16"
            )
              onChange({ ...profile, precision })
          }}
        >
          <SelectTrigger id="validation-precision">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["fp32", "fp16", "bf16"].map((precision) => (
              <SelectItem key={precision} value={precision}>
                {precision}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {(
        ["batch_min", "batch_max", "sequence_min", "sequence_max"] as const
      ).map((field) => (
        <label key={field} className="grid gap-2 text-sm">
          {
            {
              batch_min: "Minimum batch",
              batch_max: "Maximum batch",
              sequence_min: "Minimum sequence",
              sequence_max: "Maximum sequence",
            }[field]
          }
          <Input
            type="number"
            min={1}
            step={1}
            value={Number.isFinite(domain[field]) ? domain[field] : ""}
            onChange={(event) =>
              onChange({
                ...profile,
                domains: [
                  {
                    ...domain,
                    [field]:
                      event.target.value === ""
                        ? Number.NaN
                        : Number(event.target.value),
                  },
                ],
              })
            }
          />
        </label>
      ))}
      <p className="text-xs text-muted-foreground sm:col-span-2">
        Train and evaluation · full prefix · {profile.torch_version}. The engine
        checks admission and measures the complete requested domain.
      </p>
    </fieldset>
  )
}
