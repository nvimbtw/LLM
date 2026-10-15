import { TokenizerComposer } from "@/components/tokenizers/tokenizer-composer"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the page-native vocabulary build form. */
export default function NewTokenizerPage() {
  return (
    <PageShell width="reading">
      <TokenizerComposer />
    </PageShell>
  )
}
