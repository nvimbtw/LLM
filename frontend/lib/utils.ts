import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Round a computed number to a fixed number of decimals, for values that get
 * rendered into an attribute on both sides of a hydration.
 *
 * `**` with a fractional exponent is implementation-defined in ECMAScript — the
 * spec lets an engine return any value within an implementation-approximated
 * result — so Node and the browser can disagree in the last bit. Nothing renders
 * differently, but React compares the *serialised* attribute, and
 * `0.09971291988180465` is not `0.09971291988180467`, so the whole tree reports a
 * hydration mismatch. The figures fade their marks with exponential easing and
 * hit this on every load.
 *
 * Four decimals is far past the precision an opacity can express and nowhere near
 * a rounding tie for these values, so both sides land on the same string.
 */
export function quantize(value: number, decimals = 4): number {
  const scale = 10 ** decimals
  return Math.round(value * scale) / scale
}
