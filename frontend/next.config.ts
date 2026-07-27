import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: {
    // Pin the workspace root to this dir so the dev server doesn't crawl
    // the parent repo (which contains the multi-GB pg19 dataset).
    root: __dirname,
  },
}

export default nextConfig
