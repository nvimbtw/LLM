export const NODE_COUNT = 31
export const HEADER_FLOATS = 24
export const PARAMETER_FLOATS = HEADER_FLOATS + NODE_COUNT * 4

export function expression(
  seed: number,
  structure: number
): Float32Array<ArrayBuffer> {
  const random = generator(seed)
  const grammar = generator(structure)
  const nodes = new Float32Array(NODE_COUNT * 4)
  for (let node = 0; node < NODE_COUNT; node++) {
    const offset = node * 4
    nodes[offset] = Math.floor(grammar() * 4)
    nodes[offset + 1] = node >= 15 ? (random() - 0.5) * 5 : 0.7 + random() * 1.7
    nodes[offset + 2] = node >= 15 ? (random() - 0.5) * 5 : random()
    nodes[offset + 3] = (random() - 0.5) * Math.PI * 2
  }
  return nodes
}

function generator(seed: number): () => number {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let value = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}
