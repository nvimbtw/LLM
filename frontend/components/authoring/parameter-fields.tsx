"use client"
import { useId } from "react"
import { FormError } from "@/components/ui/error-state"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import type { ParameterDeclaration } from "@/lib/architecture/contracts.generated"
import type { ParameterValue } from "@/lib/architecture/parameters"
export interface ParameterFieldsProps {
  declarations: Record<string, ParameterDeclaration>
  values: Record<string, ParameterValue | undefined>
  errors?: Record<string, string | undefined>
  disabled?: boolean
  onChange: (name: string, value: ParameterValue | undefined) => void
}
export interface ParameterFieldProps {
  name: string
  declaration: ParameterDeclaration
  value: ParameterValue | undefined
  error?: string
  disabled?: boolean
  onChange: (name: string, value: ParameterValue | undefined) => void
}
/** Renders accessible declared controls; changes remain local and never rewrite Python or submit requests. */
export function ParameterFields({
  declarations,
  values,
  errors,
  disabled,
  onChange,
}: ParameterFieldsProps) {
  return (
    <div className="grid gap-4">
      {Object.entries(declarations).map(([name, declaration]) => (
        <ParameterField
          key={name}
          name={name}
          declaration={declaration}
          value={values[name]}
          error={errors?.[name]}
          disabled={disabled}
          onChange={onChange}
        />
      ))}
    </div>
  )
}
/** Renders one typed declaration, preserving empty numeric drafts as unset and enum scalar types by index. */
export function ParameterField({
  name,
  declaration,
  value,
  error,
  disabled,
  onChange,
}: ParameterFieldProps) {
  const id = useId()
  const descriptionId = `${id}-description`
  const errorId = `${id}-error`
  const accessible = {
    id,
    disabled,
    "aria-invalid": Boolean(error),
    "aria-describedby":
      [declaration.description ? descriptionId : "", error ? errorId : ""]
        .filter(Boolean)
        .join(" ") || undefined,
  }
  const choices =
    declaration.type === "boolean" ? [true, false] : (declaration.choices ?? [])
  return (
    <div className="grid gap-2">
      <label htmlFor={id} className="text-sm font-medium">
        {declaration.title || name}
        {declaration.required ? " *" : ""}
      </label>
      {declaration.type === "boolean" && value !== undefined ? (
        <Switch
          {...accessible}
          checked={value === true}
          onCheckedChange={(checked) => onChange(name, checked)}
        />
      ) : declaration.type === "enum" || declaration.type === "boolean" ? (
        <Select
          value={
            value === undefined
              ? null
              : String(
                  choices.findIndex(
                    (choice) =>
                      typeof choice === typeof value && choice === value
                  )
                )
          }
          onValueChange={(index) =>
            onChange(name, index == null ? undefined : choices[Number(index)])
          }
          disabled={disabled}
        >
          <SelectTrigger {...accessible}>
            <SelectValue>
              {() => (value === undefined ? "Choose a value" : String(value))}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {choices.map((choice, index) => (
              <SelectItem key={index} value={String(index)}>
                {String(choice)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input
          {...accessible}
          type={declaration.type === "string" ? "text" : "number"}
          value={value === undefined ? "" : String(value)}
          min={declaration.minimum ?? undefined}
          max={declaration.maximum ?? undefined}
          step={declaration.type === "integer" ? 1 : "any"}
          maxLength={declaration.type === "string" ? 4096 : undefined}
          onChange={(event) =>
            onChange(
              name,
              declaration.type === "string"
                ? event.target.value
                : event.target.value === ""
                  ? undefined
                  : Number(event.target.value)
            )
          }
        />
      )}
      {declaration.description && (
        <p id={descriptionId} className="text-sm text-muted-foreground">
          {declaration.description}
        </p>
      )}
      {error && (
        <div id={errorId}>
          <FormError>{error}</FormError>
        </div>
      )}
    </div>
  )
}
