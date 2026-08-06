//! The marker grammar, enforced while the model decodes.
//!
//! A reply is markup: `<|par|>`, `<|list|><|item|>`, `<|thinking|>`. The model at
//! this size cannot keep it well-formed on its own — the markup suite scores
//! `closer_matches_opener` at **52%**, against a 50% chance baseline, and
//! `nesting_order` at 63% — and served replies show exactly that:
//!
//! ```text
//! <|par|>Yes. The answer is what is what to be.<|par|>If you have the same
//! time?<|/user|><|assistant|><|par|>You can tell you the same thing.<|/par|>
//! ```
//!
//! A closer naming a block that is not open, a turn boundary invented mid-reply,
//! blocks left unclosed. None of that is a knowledge problem the model could study
//! its way out of — it is a *representable* state, and the fix is to stop
//! representing it. [`OpenBlocks`] tracks what is open and masks the markers that
//! would be illegal next, so the sampler can only draw from well-formed
//! continuations. Ordinary text is never masked, so there is always something to
//! draw.
//!
//! **This is a serving-time constraint and must stay out of `eval`.** The markup
//! suite exists to measure whether the model has *learned* the grammar; generating
//! its structure stage under the constraint would measure the constraint instead,
//! and score every model alike. `eval::generate::continue_tokens` therefore does
//! not use this, and the number to watch for real progress is still the suite's.
//!
//! Distinct from [`crate::eval::markup`], which scores marker use rather than
//! constraining it — the same relationship `chat_template` has with `eval::chat`.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use tokenize::pretokenize::{special_id, SPECIAL_TOKENS};

/// How deep blocks may nest before further openers are refused.
///
/// Not a property of the data — the deepest rendering in the SFT corpus is
/// `list > item > bold`, three — but a bound on a degenerate model that has learned
/// to open and never close. Without it a reply can spend its whole token budget
/// descending, and the drain at the end would then be longer than the reply.
const MAX_DEPTH: usize = 6;

/// Which marker blocks are open, and therefore which markers may come next.
///
/// One per generated reply. Fed every sampled token through [`Self::observe`], asked
/// about candidates through [`Self::allows`].
pub struct OpenBlocks {
    /// Opener ids, outermost first. The top is the block a closer must name.
    stack: Vec<u32>,
}

impl Default for OpenBlocks {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenBlocks {
    /// A fresh assistant turn: the `<|assistant|>` opener is part of the prompt and
    /// is not on the stack, so an empty stack means "the reply may end here".
    pub fn new() -> OpenBlocks {
        OpenBlocks { stack: Vec::new() }
    }

    /// Whether `id` is a legal next token.
    ///
    /// True for every ordinary (non-marker) token, unconditionally — which is what
    /// guarantees [`Self::mask`] can never empty the candidate set.
    pub fn allows(&self, id: u32) -> bool {
        let rules = rules();
        if !rules.is_special(id) {
            return true;
        }
        if rules.forbidden.contains(&id) {
            return false;
        }
        // A turn may only end from the outside of every block it opened. This is
        // also what stops a reply stopping mid-block: the model cannot reach the
        // end-of-turn marker without emitting the closers first.
        if id == rules.end_of_assistant || id == rules.end_of_text {
            return self.stack.is_empty();
        }
        if let Some(&opener) = rules.opener_for.get(&id) {
            return self.stack.last() == Some(&opener);
        }
        if rules.closer_for.contains_key(&id) {
            return self.may_open(id, rules);
        }
        // Standalone markers (`<|divider|>`) open nothing and close nothing.
        true
    }

    // An opener is legal when there is depth left for it, the enclosing block admits
    // it, and it does not itself demand a different parent.
    fn may_open(&self, id: u32, rules: &Rules) -> bool {
        if self.stack.len() >= MAX_DEPTH {
            return false;
        }
        let parent = self.stack.last();
        // `<|item|>` outside a list, or anything but `<|item|>` directly inside one.
        // Enforced from both sides so neither has to trust the other.
        if let Some(&required) = rules.requires_parent.get(&id) {
            if parent != Some(&required) {
                return false;
            }
        }
        match parent.and_then(|p| rules.only_child.get(p)) {
            Some(&only) => id == only,
            None => true,
        }
    }

    /// Records a token the sampler actually drew.
    ///
    /// A marker that [`Self::allows`] would have refused is ignored rather than
    /// applied — the stack must keep describing the text even when a caller samples
    /// without masking first, which is the only way the two could disagree.
    pub fn observe(&mut self, id: u32) {
        let rules = rules();
        if rules.closer_for.contains_key(&id) {
            if self.stack.len() < MAX_DEPTH {
                self.stack.push(id);
            }
            return;
        }
        if let Some(&opener) = rules.opener_for.get(&id) {
            if self.stack.last() == Some(&opener) {
                self.stack.pop();
            }
        }
    }

    /// Sets every illegal marker's logit to negative infinity, leaving the rest
    /// untouched.
    ///
    /// Only the reserved marker block is scanned — a few dozen ids out of the
    /// vocabulary — so this costs nothing against the forward pass it follows.
    pub fn mask(&self, logits: &mut [f32]) {
        for &id in rules().marker_ids.iter() {
            let Some(slot) = logits.get_mut(id as usize) else {
                continue;
            };
            if !self.allows(id) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }

    /// Whether the reply could legally stop here — nothing is left open.
    pub fn is_complete(&self) -> bool {
        self.stack.is_empty()
    }

    /// How many blocks are open, which is exactly how many closers a drain costs.
    ///
    /// The generation loop reserves this against its token cap, so the closers it may
    /// have to append are budgeted for rather than spent past the ceiling the caller
    /// set. Bounded by `MAX_DEPTH`.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// The closers that would drain everything still open, innermost first.
    ///
    /// For the token cap, which is the one stop the grammar cannot prevent: the loop
    /// appends these so a truncated reply is still well-formed markup rather than a
    /// half-open block the frontend has to guess at.
    pub fn closers(&self) -> Vec<u32> {
        let rules = rules();
        self.stack
            .iter()
            .rev()
            .filter_map(|opener| rules.closer_for.get(opener).copied())
            .collect()
    }
}

// The marker classification, derived once from `SPECIAL_TOKENS`.
struct Rules {
    closer_for: HashMap<u32, u32>,
    opener_for: HashMap<u32, u32>,
    // Markers the model must never emit: the other roles' scaffold, the budget cues
    // it only ever conditions on, and `<|reply|>`, whose content is a message id
    // rather than prose (see `synth::NOT_REPLY_MARKUP`).
    forbidden: HashSet<u32>,
    // An opener that is only legal directly inside a particular other block.
    requires_parent: HashMap<u32, u32>,
    // An open block that admits only one kind of opener directly inside it.
    only_child: HashMap<u32, u32>,
    marker_ids: Vec<u32>,
    end_of_assistant: u32,
    end_of_text: u32,
}

impl Rules {
    fn is_special(&self, id: u32) -> bool {
        // Contiguous by construction — the reserved block is `[256, 256 + NUM_SPECIAL)`
        // — but read off the collected ids rather than reconstructing that layout here,
        // which `pretokenize` owns.
        self.marker_ids.first().is_some_and(|&lo| id >= lo)
            && self.marker_ids.last().is_some_and(|&hi| id <= hi)
    }
}

// The pairs the model is allowed to use. Anything in `SPECIAL_TOKENS` that is not
// named here and not standalone is forbidden, so adding a marker fails closed:
// the model cannot emit it until this table says how it nests.
const PAIRS: &[(&str, &str)] = &[
    ("<|par|>", "<|/par|>"),
    ("<|code|>", "<|/code|>"),
    ("<|codeblock|>", "<|/codeblock|>"),
    ("<|title|>", "<|/title|>"),
    ("<|heading|>", "<|/heading|>"),
    ("<|subheading|>", "<|/subheading|>"),
    ("<|link|>", "<|/link|>"),
    ("<|list|>", "<|/list|>"),
    ("<|item|>", "<|/item|>"),
    ("<|quote|>", "<|/quote|>"),
    ("<|italic|>", "<|/italic|>"),
    ("<|bold|>", "<|/bold|>"),
    ("<|warn|>", "<|/warn|>"),
    ("<|thinking|>", "<|/thinking|>"),
];

// Markers that open and close nothing, and are legal wherever prose is.
const STANDALONE: &[&str] = &["<|divider|>"];

fn rules() -> &'static Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES.get_or_init(build_rules)
}

fn build_rules() -> Rules {
    let mut closer_for = HashMap::new();
    let mut opener_for = HashMap::new();
    for &(open, close) in PAIRS {
        // A marker missing from `SPECIAL_TOKENS` is a typo in `PAIRS`, and dropping
        // it silently would leave that block unconstrained. There is nothing to
        // recover to at this layer, so it fails the build of the table.
        let (open_id, close_id) = (
            marker_id(open).unwrap_or_else(|| panic!("unknown marker {open}")),
            marker_id(close).unwrap_or_else(|| panic!("unknown marker {close}")),
        );
        closer_for.insert(open_id, close_id);
        opener_for.insert(close_id, open_id);
    }

    let named: HashSet<u32> = closer_for
        .keys()
        .chain(closer_for.values())
        .chain(STANDALONE.iter().filter_map(|m| marker_id(m)).collect::<Vec<_>>().iter())
        .copied()
        .collect();

    let end_of_assistant = marker_id("<|/assistant|>").expect("<|/assistant|>");
    let end_of_text = marker_id("<|endoftext|>").expect("<|endoftext|>");

    let marker_ids: Vec<u32> = SPECIAL_TOKENS.iter().filter_map(|&(_, c)| special_id(c)).collect();
    // Everything the table does not account for. Fails closed by construction: a
    // marker added to `SPECIAL_TOKENS` and not to `PAIRS` lands here.
    let forbidden: HashSet<u32> = marker_ids
        .iter()
        .copied()
        .filter(|id| !named.contains(id) && *id != end_of_assistant && *id != end_of_text)
        .collect();

    let list = marker_id("<|list|>").expect("<|list|>");
    let item = marker_id("<|item|>").expect("<|item|>");

    Rules {
        closer_for,
        opener_for,
        forbidden,
        requires_parent: HashMap::from([(item, list)]),
        only_child: HashMap::from([(list, item)]),
        marker_ids,
        end_of_assistant,
        end_of_text,
    }
}

fn marker_id(marker: &str) -> Option<u32> {
    SPECIAL_TOKENS
        .iter()
        .find(|&&(name, _)| name == marker)
        .and_then(|&(_, ch)| special_id(ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(marker: &str) -> u32 {
        marker_id(marker).unwrap_or_else(|| panic!("no id for {marker}"))
    }

    // Walks a marker sequence through the automaton, asserting each step was legal.
    fn walk(markers: &[&str]) -> OpenBlocks {
        let mut blocks = OpenBlocks::new();
        for m in markers {
            assert!(blocks.allows(id(m)), "{m} was refused after {markers:?}");
            blocks.observe(id(m));
        }
        blocks
    }

    // The 52%-accuracy failure, made unrepresentable: `<|par|>` can only be closed by
    // `<|/par|>`, whatever the model would rather do.
    #[test]
    fn a_closer_must_name_the_block_it_closes() {
        let blocks = walk(&["<|par|>"]);
        assert!(blocks.allows(id("<|/par|>")));
        assert!(!blocks.allows(id("<|/item|>")));
        assert!(!blocks.allows(id("<|/quote|>")));
        assert!(!blocks.allows(id("<|/code|>")));
    }

    #[test]
    fn a_closer_with_nothing_open_is_refused() {
        let blocks = OpenBlocks::new();
        for closer in ["<|/par|>", "<|/item|>", "<|/list|>", "<|/thinking|>"] {
            assert!(!blocks.allows(id(closer)), "{closer} was allowed with nothing open");
        }
    }

    // `nesting_order`, the other paradigm the model scores near chance on.
    #[test]
    fn nested_blocks_close_innermost_first() {
        let blocks = walk(&["<|list|>", "<|item|>", "<|bold|>"]);
        assert!(blocks.allows(id("<|/bold|>")));
        assert!(!blocks.allows(id("<|/item|>")), "closed the outer block first");
        assert!(!blocks.allows(id("<|/list|>")), "closed the outermost block first");

        let blocks = walk(&["<|list|>", "<|item|>", "<|bold|>", "<|/bold|>", "<|/item|>"]);
        assert!(blocks.allows(id("<|/list|>")));
    }

    // The role leakage seen in served replies: the model generated `<|/user|>` and a
    // fresh `<|assistant|>` in the middle of its own turn and carried on.
    #[test]
    fn the_other_roles_scaffold_can_never_be_emitted() {
        let blocks = OpenBlocks::new();
        for marker in ["<|user|>", "<|/user|>", "<|system|>", "<|/system|>", "<|assistant|>"] {
            assert!(!blocks.allows(id(marker)), "{marker} was allowed inside an assistant turn");
        }
    }

    // Input-only by design: the model conditions on a budget marker and is never
    // trained to emit one, so only its embedding row is ever learned.
    #[test]
    fn the_budget_cues_can_never_be_emitted() {
        let blocks = OpenBlocks::new();
        for marker in ["<|think:low|>", "<|think:med|>", "<|think:high|>"] {
            assert!(!blocks.allows(id(marker)), "{marker} was allowed");
        }
    }

    // A turn that stops mid-block leaves markup the frontend has to guess at, so the
    // end-of-turn markers are gated on the stack being empty. This is also what makes
    // the model close its blocks: it cannot reach the stop token any other way.
    #[test]
    fn a_turn_can_only_end_from_outside_every_block() {
        let open = walk(&["<|par|>"]);
        assert!(!open.allows(id("<|/assistant|>")));
        assert!(!open.allows(id("<|endoftext|>")));
        assert!(!open.is_complete());

        let closed = walk(&["<|par|>", "<|/par|>"]);
        assert!(closed.allows(id("<|/assistant|>")));
        assert!(closed.allows(id("<|endoftext|>")));
        assert!(closed.is_complete());
    }

    #[test]
    fn an_empty_turn_may_end_immediately() {
        let blocks = OpenBlocks::new();
        assert!(blocks.is_complete());
        assert!(blocks.allows(id("<|/assistant|>")));
    }

    // `<|item|>` outside a list and prose blocks directly inside one are both the
    // same malformed shape, refused from both directions.
    #[test]
    fn items_live_in_lists_and_lists_hold_only_items() {
        let bare = OpenBlocks::new();
        assert!(!bare.allows(id("<|item|>")), "an item opened outside a list");

        let list = walk(&["<|list|>"]);
        assert!(list.allows(id("<|item|>")));
        assert!(!list.allows(id("<|par|>")), "a paragraph opened directly inside a list");
        assert!(!list.allows(id("<|quote|>")));

        // Inside the item, ordinary blocks are fine again.
        let item = walk(&["<|list|>", "<|item|>"]);
        assert!(item.allows(id("<|par|>")));
        assert!(item.allows(id("<|bold|>")));
    }

    // Ordinary text is legal in every state. This is the invariant that keeps `mask`
    // from ever emptying the candidate set, so it is asserted rather than assumed.
    #[test]
    fn ordinary_tokens_are_always_legal() {
        let states = [
            OpenBlocks::new(),
            walk(&["<|list|>"]),
            walk(&["<|list|>", "<|item|>"]),
            walk(&["<|par|>", "<|bold|>"]),
        ];
        for blocks in &states {
            for id in [0u32, 65, 255, 4000, 8191] {
                assert!(blocks.allows(id), "ordinary token {id} was refused");
            }
        }
    }

    #[test]
    fn masking_never_removes_every_candidate() {
        let mut logits = vec![0.0f32; 8192];
        walk(&["<|list|>"]).mask(&mut logits);
        assert!(logits.iter().any(|l| l.is_finite()), "nothing left to sample");
        // And it left the ordinary tokens exactly as they were.
        assert_eq!(logits[4000], 0.0);
        assert!(logits[id("<|par|>") as usize].is_infinite(), "an illegal marker survived");
    }

    // A model that opens and never closes would otherwise spend its whole budget
    // descending, and the drain would outrun the reply.
    #[test]
    fn nesting_is_bounded() {
        let mut blocks = OpenBlocks::new();
        for _ in 0..MAX_DEPTH {
            assert!(blocks.allows(id("<|bold|>")));
            blocks.observe(id("<|bold|>"));
        }
        assert!(!blocks.allows(id("<|bold|>")), "nesting passed MAX_DEPTH");
        assert!(blocks.allows(id("<|/bold|>")), "the way out must stay open");
    }

    // The token cap is the one stop the grammar cannot prevent, so what is still open
    // has to be closable in one pass, innermost first.
    #[test]
    fn the_drain_closes_everything_innermost_first() {
        let blocks = walk(&["<|list|>", "<|item|>", "<|bold|>"]);
        assert_eq!(
            blocks.closers(),
            vec![id("<|/bold|>"), id("<|/item|>"), id("<|/list|>")],
        );

        // And replaying that drain leaves nothing open.
        let mut drained = walk(&["<|list|>", "<|item|>", "<|bold|>"]);
        for closer in drained.closers() {
            assert!(drained.allows(closer));
            drained.observe(closer);
        }
        assert!(drained.is_complete());
    }

    #[test]
    fn a_complete_turn_needs_no_drain() {
        assert!(OpenBlocks::new().closers().is_empty());
        assert!(walk(&["<|par|>", "<|/par|>"]).closers().is_empty());
    }

    // `infer::stream_generation` reserves `depth()` tokens against its cap and then
    // spends `closers()` of them, so the two have to agree exactly — a `depth()` that
    // under-counted would let the drain run past the caller's ceiling.
    #[test]
    fn depth_is_exactly_what_the_drain_will_cost() {
        for markers in [
            vec![],
            vec!["<|par|>"],
            vec!["<|list|>", "<|item|>"],
            vec!["<|list|>", "<|item|>", "<|bold|>"],
            vec!["<|par|>", "<|bold|>", "<|/bold|>"],
        ] {
            let blocks = walk(&markers);
            assert_eq!(blocks.depth(), blocks.closers().len(), "after {markers:?}");
        }
    }

    // A reply may think again after it has started answering — `strip_thinking`
    // handles several blocks per reply for exactly this reason.
    #[test]
    fn a_reply_may_open_a_second_thinking_block() {
        let blocks = walk(&["<|thinking|>", "<|/thinking|>", "<|par|>", "<|/par|>"]);
        assert!(blocks.allows(id("<|thinking|>")));
    }

    // Fails closed: every marker is either in `PAIRS`, standalone, or refused. A
    // marker added to `SPECIAL_TOKENS` without a rule must not become emittable.
    #[test]
    fn every_marker_is_accounted_for() {
        let rules = rules();
        for &(name, ch) in SPECIAL_TOKENS {
            let id = special_id(ch).unwrap_or_else(|| panic!("no id for {name}"));
            let known = rules.closer_for.contains_key(&id)
                || rules.opener_for.contains_key(&id)
                || rules.forbidden.contains(&id)
                || id == rules.end_of_assistant
                || id == rules.end_of_text
                || STANDALONE.contains(&name);
            assert!(known, "{name} has no rule");
        }
    }

    // `observe` has to survive a caller that sampled without masking — otherwise the
    // stack stops describing the text and every later decision is made against a
    // fiction.
    #[test]
    fn an_illegal_closer_does_not_corrupt_the_stack() {
        let mut blocks = walk(&["<|par|>"]);
        blocks.observe(id("<|/item|>"));
        assert!(!blocks.is_complete(), "a mismatched closer popped the stack");
        assert!(blocks.allows(id("<|/par|>")));
    }
}
