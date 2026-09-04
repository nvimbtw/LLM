"use client"

import Link from "next/link"
import { Dumbbell, Gauge, MessageSquare, Plus, Target, Type } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

/**
 * Add job — a menu that routes to a composer, not a modal that contains one.
 *
 * Each kind's form is a full page worth of controls (the training rail alone is
 * a dozen hyperparameters), so they get their own URLs: deep-linkable, seedable
 * from a finished job's config, and survivable across a reload.
 *
 * Chat is on the list because a chat turn genuinely is a queued job now — but it
 * is composed by talking, so it points at /chat rather than a form.
 */
const OPTIONS = [
  { label: "Train a model", href: "/jobs/new/train", icon: Dumbbell },
  // A vocabulary build is a queued job like the rest, so it belongs on the one
  // menu every door to the queue goes through. Its composer lives beside the
  // list of what has been built, which is where you would be choosing anyway.
  { label: "Build a tokenizer", href: "/tokenizers", icon: Type },
  { label: "Benchmark a model", href: "/jobs/new/benchmark", icon: Gauge },
  // Practising is a training run and belongs on this menu, even though the gym
  // it practises against has a page of its own. Starting a run *there* evaluates
  // a model once; this is the loop that keeps going, and a loop is a job.
  { label: "Practise in a gym", href: "/jobs/new/rl", icon: Target },
  { label: "Chat", href: "/chat", icon: MessageSquare },
]

export function AddJobMenu() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button size="sm" className="gap-1.5">
            <Plus className="size-3.5" />
            Add job
          </Button>
        }
      />
      <DropdownMenuContent align="end">
        {OPTIONS.map((option) => (
          <DropdownMenuItem
            key={option.href}
            render={<Link href={option.href} />}
          >
            <option.icon />
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
