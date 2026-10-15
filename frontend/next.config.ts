import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  devIndicators: false,
  /** Redirects legacy frontend composer/index paths; backend API paths and query values stay intact. */
  async redirects() {
    return [
      { source: "/benchmark", destination: "/bench", permanent: true },
      {
        source: "/jobs/new-train",
        destination: "/train/new-train",
        permanent: true,
      },
      {
        source: "/jobs/new/train",
        destination: "/train/new-train",
        permanent: true,
      },
      {
        source: "/jobs/new/benchmark",
        destination: "/bench/new-bench",
        permanent: true,
      },
      { source: "/jobs/new/rl", destination: "/gyms/new-gym", permanent: true },
    ]
  },
  turbopack: {
    // Pin the workspace root to this dir so the dev server doesn't crawl
    // the parent repo (which contains the multi-GB pg19 dataset).
    root: __dirname,
  },
}

export default nextConfig
