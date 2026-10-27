"use client"
import dynamic from "next/dynamic"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { decodeArchitecture } from "@/lib/architecture/decoder"
import { ParameterFields } from "./parameter-fields"
import type { ParameterValue } from "@/lib/architecture/parameters"
const PythonSourceEditor = dynamic(
  () =>
    import("../data/python-source-editor").then(
      (module) => module.PythonSourceEditor
    ),
  { ssr: false }
)
const SourceEditor = dynamic(
  () => import("./source-editor").then((module) => module.SourceEditor),
  { ssr: false }
)
/** Checks actual browser runtime schema decoding with valid and invalid parameter declarations. */
function decoderSmoke(): string {
  const value = decodeArchitecture("ParameterDeclaration", {
    type: "integer",
    default: 4,
    minimum: 1,
  })
  try {
    decodeArchitecture("ParameterDeclaration", {
      type: "integer",
      default: true,
    })
  } catch {
    return value.default === 4 ? "passed" : "failed"
  }
  return "failed"
}

/** Exercises real Python/YAML editor surfaces and declared controls using browser-local fixture drafts. */
export function AuthoringFixture() {
  const [decoder, setDecoder] = useState("pending")
  const [python, setPython] = useState("def forward(x):\n    return x\n")
  const [processor, setProcessor] = useState("processor_value = 1")
  const [yaml, setYaml] = useState("schema_version: 1\n")
  const [values, setValues] = useState<
    Record<string, ParameterValue | undefined>
  >({ width: 32, enabled: false, mode: "train" })
  return (
    <div className="grid gap-6 p-6">
      <h1>Authoring controls</h1>
      <Button onClick={() => setDecoder(decoderSmoke())}>
        Check runtime decoder
      </Button>
      <output aria-label="Runtime decoder result">{decoder}</output>
      <SourceEditor
        language="python"
        ariaLabel="Fixture Python source"
        value={python}
        onChange={setPython}
        readOnly={false}
        height="180px"
      />
      <SourceEditor
        language="yaml"
        ariaLabel="Fixture YAML source"
        value={yaml}
        onChange={setYaml}
        readOnly={false}
        height="180px"
      />
      <SourceEditor
        language="python"
        ariaLabel="Fixture read-only source"
        value="read_only = True"
        onChange={() => {}}
        readOnly
        height="100px"
      />
      <PythonSourceEditor
        value={processor}
        onChange={setProcessor}
        readOnly={false}
        height="100px"
      />
      <output aria-label="Processor draft">{processor}</output>
      <ParameterFields
        declarations={{
          width: { type: "integer", minimum: 1, maximum: 64 },
          enabled: { type: "boolean" },
          mode: { type: "enum", choices: ["train", "eval"] },
        }}
        values={values}
        onChange={(name, value) =>
          setValues((current) => ({ ...current, [name]: value }))
        }
      />
      <output aria-label="Python draft">{python}</output>
      <output aria-label="YAML draft">{yaml}</output>
      <output aria-label="Parameter draft">{JSON.stringify(values)}</output>
    </div>
  )
}
