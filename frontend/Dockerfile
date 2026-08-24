# syntax=docker/dockerfile:1

FROM oven/bun:1 AS base
WORKDIR /app

FROM base AS deps
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM base AS dev
COPY --from=deps /app/node_modules ./node_modules
# Bind to every interface: the default is loopback *inside* the container, which
# the published port cannot reach.
CMD ["bun", "run", "dev", "--hostname", "0.0.0.0"]

FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN bun run build

FROM base AS runtime
COPY --from=build /app ./
CMD ["bun", "run", "start", "--hostname", "0.0.0.0"]
