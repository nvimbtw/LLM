import { test } from "node:test"
import assert from "node:assert/strict"
import { datasetComposition } from "./training-data"
import type { TrainConfig } from "./training"
import { configForMode, SFT_DEFAULT_LR } from "@/components/training/run-config"
import { DEFAULT_CONFIG } from "@/hooks/use-training"

const catalogue = [
  { id: "a", name: "Articles", documents: 100 },
  { id: "b", name: "Books", documents: 50 },
]

test("dataset shares resolve all/capped selections and count repeated curriculum prefixes once", () => {
  const cfg: TrainConfig = {
    ...DEFAULT_CONFIG,
    datasets: [
      { id: "a", count: 0 },
      { id: "b", count: 200 },
    ],
  }
  assert.deepEqual(
    datasetComposition(cfg, catalogue).datasets.map((row) => row.documents),
    [100, 50]
  )
  const staged: TrainConfig = {
    ...cfg,
    curriculum: [
      { epochs: 1, datasets: [{ id: "a", count: 20 }] },
      {
        epochs: 3,
        datasets: [
          { id: "a", count: 40 },
          { id: "b", count: 0 },
        ],
      },
    ],
  }
  assert.deepEqual(datasetComposition(staged, catalogue), {
    datasets: [
      { id: "a", name: "Articles", documents: 40, catalogueIndex: 0 },
      { id: "b", name: "Books", documents: 50, catalogueIndex: 1 },
    ],
    total: 90,
    complete: true,
  })
})

test("unknown sizes and invalid limits stay unavailable while measured empty datasets stay zero", () => {
  for (const count of [-1, NaN, Infinity, 0.5]) {
    assert.equal(
      datasetComposition(
        { ...DEFAULT_CONFIG, datasets: [{ id: "a", count }] },
        catalogue
      ).complete,
      false
    )
  }
  assert.equal(
    datasetComposition(
      { ...DEFAULT_CONFIG, datasets: [{ id: "missing", count: 0 }] },
      catalogue
    ).complete,
    false
  )
  assert.deepEqual(
    datasetComposition(
      { ...DEFAULT_CONFIG, datasets: [{ id: "a", count: 0 }] },
      [{ ...catalogue[0], documents: 0 }]
    ),
    {
      datasets: [
        { id: "a", name: "Articles", documents: 0, catalogueIndex: 0 },
      ],
      total: 0,
      complete: true,
    }
  )
})

test("switching dataset kinds clears hidden selections while preserving explicit learning rates", () => {
  const cfg = {
    ...DEFAULT_CONFIG,
    datasets: [{ id: "a", count: 0 }],
    curriculum: [{ datasets: [{ id: "a", count: 20 }], epochs: 2 }],
  }
  const fineTune = configForMode(cfg, "sft")
  assert.deepEqual(fineTune.datasets, [])
  assert.deepEqual(fineTune.curriculum, [])
  assert.equal(fineTune.lr, SFT_DEFAULT_LR)
  assert.equal(configForMode({ ...cfg, lr: 0.02 }, "sft").lr, 0.02)
  assert.equal(configForMode(cfg, "pretrain"), cfg)
  assert.deepEqual(
    configForMode(
      { ...fineTune, datasets: [{ id: "b", count: 0 }] },
      "pretrain"
    ).datasets,
    []
  )
})
