import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FluxThrobber, type ThrobberMode } from "@/components/chat/flux-throbber";

const MODES: { mode: ThrobberMode; description: string }[] = [
  {
    mode: "thinking",
    description: "Model reasoning — sun, facet, lattice. Fast, frequent switches.",
  },
  {
    mode: "generating",
    description: "Streaming the answer — wave, streak, pour. Distinct from thinking.",
  },
  {
    mode: "tool",
    description: "Reserved for tool calls — spark, facet. Snappy and mechanical.",
  },
  {
    mode: "loading",
    description: "Fetching chat history — a steadier, medium pace.",
  },
  {
    mode: "idle",
    description: "Ambient resting state — slow and calm.",
  },
  {
    mode: "training",
    description: "Reserved for a future training state — very slow.",
  },
];

export default function ThrobberPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Throbber
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Every mode of the generative loading panel.
        </p>
      </header>

      <div className="grid gap-6 sm:grid-cols-2">
        {MODES.map(({ mode, description }) => (
          <Card key={mode}>
            <CardHeader>
              <CardTitle className="capitalize">{mode}</CardTitle>
              <CardDescription>{description}</CardDescription>
            </CardHeader>
            <CardContent>
              <FluxThrobber mode={mode} className="h-40 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
