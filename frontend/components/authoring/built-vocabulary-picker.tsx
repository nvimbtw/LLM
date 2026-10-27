"use client"

import { useId, useState } from "react"
import type { TokenizerInfo } from "@/lib/tokenizers"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export interface BuiltVocabularyPickerProps {
  items: readonly TokenizerInfo[]
  value: string
  required: boolean
  disabled?: boolean
  onChange: (id: string) => void
}

/** Selects a built vocabulary from a searchable bounded list; displayed counts remain metadata and final bindings are read separately before submission. */
export function BuiltVocabularyPicker({
  items,
  value,
  required,
  disabled,
  onChange,
}: BuiltVocabularyPickerProps) {
  const id = useId()
  const [search, setSearch] = useState("")
  const matching = items.filter(
    (item) =>
      item.hasVocab &&
      (item.name.toLocaleLowerCase().includes(search.toLocaleLowerCase()) ||
        item.id === value)
  )
  const options = matching.slice(0, 200)
  const selected = items.find((item) => item.id === value && item.hasVocab)
  if (selected && !options.some((item) => item.id === value))
    options.push(selected)
  return (
    <div className="grid gap-2 text-sm">
      <label htmlFor={id}>
        Built vocabulary {required ? "(required)" : "(optional)"}
      </label>
      <Input
        aria-label="Search built vocabularies"
        value={search}
        maxLength={200}
        disabled={disabled}
        onChange={(event) => setSearch(event.target.value)}
      />
      <Select
        value={value || null}
        disabled={disabled}
        onValueChange={(next) => onChange(next ?? "")}
      >
        <SelectTrigger id={id}>
          <SelectValue>
            {() =>
              selected
                ? `${selected.name} · ${selected.vocabSize ?? "Unknown"} tokens`
                : "Choose a vocabulary"
            }
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map((item) => (
            <SelectItem key={item.id} value={item.id}>
              {item.name} · {item.vocabSize ?? "Unknown"} tokens
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {matching.length > 200 && (
        <p className="text-xs text-muted-foreground">
          Showing the first 200 matches. Narrow your search to find another
          vocabulary.
        </p>
      )}
      {value && (
        <Button
          className="w-fit"
          variant="ghost"
          disabled={disabled}
          onClick={() => onChange("")}
        >
          Clear vocabulary
        </Button>
      )}
    </div>
  )
}
