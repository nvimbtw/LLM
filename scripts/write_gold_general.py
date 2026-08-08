#!/usr/bin/env python3
"""Writes data/raw/sft/gold-general.jsonl — the hand-written exemplar set.

This is the file every teach call is told to copy the VOICE of, and it exists
because the previous exemplar pool (synthetic.jsonl) is Claude Code session
transcripts: excellent data, but its register is "engineer auditing a codebase",
and a small teacher copies subject matter along with voice no matter how firmly
the prompt separates them. synthetic.jsonl stays in the corpus; it is simply no
longer what a teach run imitates.

Every conversation here is hand-written, general-domain, and factually
conservative — it lands in raw/sft/ like any other file, so it trains as well as
seeds. The voice is the one the corpus already teaches: lead with the answer,
dense, specific, no filler, no restating the question, markup sprinkled rather
than sprayed.

A third of them carry a <|thinking|> block, at deliberately different lengths —
short, middling and one genuinely long. The teach prompt *tells* the teacher that
the long tier is several paragraphs of real working thought; nothing ever showed
it one. Regenerate with: python3 scripts/write_gold_general.py (from the repo root)
"""

import json
import paths
import os

C = []


def conv(user, assistant, system=None):
    c = {"turns": [{"role": "user", "text": user},
                   {"role": "assistant", "text": assistant}]}
    if system:
        c = {"system": system, **c}
    C.append(c)


# ── cooking and food ────────────────────────────────────────────────────────

conv(
    "my bread didn't rise. i used a whole packet of yeast and left it 2 hours in a warm spot",
    "<|par|>Almost certainly the water. Yeast dies above roughly <|bold|>55°C<|/bold|>, and "
    "tap water that feels hot to the hand is already past that — a packet of dead yeast behaves "
    "exactly like no yeast at all.<|/par|>"
    "<|par|>Proof it before committing next time: stir the yeast into water at <|code|>35-40°C<|/code|> "
    "with a pinch of sugar and leave it 10 minutes. It should foam. No foam, dead packet.<|/par|>"
    "<|list|>"
    "<|item|>Water above 55°C — kills it outright.<|/item|>"
    "<|item|>Salt poured directly onto the yeast — slows it badly; mix salt into the flour first.<|/item|>"
    "<|item|>An expired packet stored somewhere warm — yeast is alive and has a shelf life.<|/item|>"
    "<|/list|>"
    "<|par|>A cold kitchen only makes rising <|italic|>slow<|/italic|>, never absent. Two hours "
    "with no movement at all is a dead-yeast signature, not a temperature one.<|/par|>",
)

conv(
    "recipe wants buttermilk and i have none. milk and lemon juice?",
    "<|thinking|>The substitution works because buttermilk does two things: it is acidic, which "
    "reacts with baking soda for lift, and it is thicker than milk. Lemon juice or vinegar in milk "
    "reproduces the acid faithfully. It does not reproduce the thickness, which matters for texture "
    "in pancakes and biscuits but not much in a cake batter that already has other liquid.<|/thinking|>"
    "<|par|>Yes. <|bold|>One tablespoon of lemon juice or white vinegar per cup of milk<|/bold|>, "
    "stirred and left 10 minutes until it clabbers.<|/par|>"
    "<|par|>It matches the acidity, which is the part that matters — the acid is what reacts with "
    "the baking soda for lift. What it does not match is thickness, so pancakes and scones come out "
    "slightly slacker. If that matters, plain yoghurt thinned with milk to a pourable consistency "
    "is the closer substitute.<|/par|>"
    "<|warn|>Do not use it in something with no baking soda or baking powder. Then the acid has "
    "nothing to react with and you have only made the batter sour.<|/warn|>",
)

conv(
    "how long is leftover rice actually safe in the fridge",
    "<|par|>Up to <|bold|>4 days<|/bold|> refrigerated — but the clock that matters starts before "
    "the fridge.<|/par|>"
    "<|par|>Uncooked rice carries <|italic|>Bacillus cereus<|/italic|> spores, which survive "
    "cooking. Left standing at room temperature they germinate and produce a toxin that reheating "
    "does <|bold|>not<|/bold|> destroy. So the real rule is about the cooling, not the storage.<|/par|>"
    "<|list|>"
    "<|item|>Cool it fast — spread it out rather than leaving the pot on the hob.<|/item|>"
    "<|item|>Into the fridge within an hour of cooking.<|/item|>"
    "<|item|>Reheat once, until steaming right through.<|/item|>"
    "<|/list|>"
    "<|par|>Rice that sat out overnight is not rescued by a hot pan. That is the one case where "
    "the usual \"it smells fine\" test tells you nothing, because the toxin has no smell.<|/par|>",
)

# ── money, work and admin ───────────────────────────────────────────────────

conv(
    "offer A is 62k with 25 days leave, offer B is 68k with 20 days. B is better right?",
    "<|thinking|>Compare them per working day rather than per year, since the leave difference is "
    "the whole question. A year has about 260 working days before leave. A works 235, B works 240. "
    "62000/235 is about 264. 68000/240 is about 283. B is ahead per day worked, and by more than "
    "the headline gap suggests, because B also has the higher salary. The five days are worth about "
    "1300 at A's own rate, well under the 6000 difference.<|/thinking|>"
    "<|par|>Yes, and by more than it looks. Per <|italic|>day actually worked<|/italic|>:<|/par|>"
    "<|list|>"
    "<|item|>A — 62,000 over ~235 working days = <|bold|>~264/day<|/bold|><|/item|>"
    "<|item|>B — 68,000 over ~240 working days = <|bold|>~283/day<|/bold|><|/item|>"
    "<|/list|>"
    "<|par|>The five extra days are worth roughly 1,300 at A's own daily rate, against a 6,000 gap "
    "in salary. B wins on the arithmetic with about 4,700 to spare.<|/par|>"
    "<|par|>What the arithmetic cannot price: pension contribution percentage, whether either "
    "carries leave over, and how firmly the leave is actually taken. Ask about the pension match "
    "before deciding — at typical rates it moves more money than the leave does.<|/par|>",
)

conv(
    "my savings account says 4.2% AER but the leaflet also says 4.12% gross. which is the real one",
    "<|par|>Both are real; they describe different things. <|bold|>Gross<|/bold|> is the simple "
    "annual rate before compounding. <|bold|>AER<|/bold|> is what you actually end up with once "
    "interest paid during the year earns interest itself.<|/par|>"
    "<|par|>4.12% paid monthly compounds to about 4.2% over a year, which is exactly the gap you "
    "are looking at. AER exists so accounts paying monthly, quarterly and annually can be compared "
    "on one number.<|/par|>"
    "<|par|>So: <|bold|>compare accounts on AER<|/bold|>. Use gross only if you intend to withdraw "
    "the interest as it is paid, in which case nothing compounds and gross is what you get.<|/par|>",
)

# ── health, fitness and sleep ───────────────────────────────────────────────

conv(
    "no soreness after my workout today. did i waste the session",
    "<|par|>No. Soreness measures <|italic|>novelty<|/italic|>, not effort — it comes largely from "
    "unfamiliar movements and the lengthening part of a lift, and it fades as you adapt to a "
    "programme that is working exactly as intended.<|/par|>"
    "<|par|>The things that do track progress:<|/par|>"
    "<|list|>"
    "<|item|>Load, reps or sets going up over weeks.<|/item|>"
    "<|item|>The same session feeling easier at the same weight.<|/item|>"
    "<|item|>Being able to train the movement again sooner.<|/item|>"
    "<|/list|>"
    "<|par|>Chasing soreness pushes you toward constant novelty, which is the opposite of the "
    "repetition that actually drives adaptation.<|/par|>",
)

conv(
    "why do i wake at 3am every time i drink in the evening, even just two glasses",
    "<|thinking|>Alcohol is sedating on the way in and stimulating on the way out. It shortens "
    "sleep latency and suppresses REM early in the night, then as the body clears it there is a "
    "rebound — lighter sleep, more REM, more awakenings, and a rise in heart rate and adrenaline. "
    "Two glasses clear in roughly two to three hours, which puts the rebound in the middle of the "
    "night. That timing is the thing being asked about, so it is what the answer should lead "
    "with.<|/thinking|>"
    "<|par|>Because your body has finished clearing it by then, and the rebound is the opposite of "
    "the sedation.<|/par|>"
    "<|par|>Alcohol gets you to sleep faster and suppresses REM for the first few hours. Two "
    "standard drinks clear in roughly <|bold|>2-3 hours<|/bold|>. What follows is a rebound: "
    "lighter sleep, REM catching up all at once, a higher heart rate and a small adrenaline rise. "
    "You surface — and at 3am there is nothing to distract you from having surfaced.<|/par|>"
    "<|par|>The practical lever is timing rather than quantity. The same two glasses finished "
    "<|italic|>four hours<|/italic|> before bed put the rebound before you sleep rather than in the "
    "middle of it.<|/par|>",
)

# ── home, repair and making ─────────────────────────────────────────────────

conv(
    "fridge is humming but everything inside is warm. freezer is fine",
    "<|par|>Freezer cold and fridge warm is a strong signal: it is almost always <|bold|>airflow, "
    "not cooling<|/bold|>. Most fridge-freezers make cold in one place and blow it into the fridge "
    "compartment through a duct.<|/par|>"
    "<|par|>Check in this order:<|/par|>"
    "<|list|>"
    "<|item|>The vent between freezer and fridge — often behind a panel at the back. Blocked by a "
    "box or iced over.<|/item|>"
    "<|item|>The evaporator fan — open the freezer and listen; it should stop when you press the "
    "door switch.<|/item|>"
    "<|item|>Frost build-up on the back wall of the freezer, which means the defrost cycle has "
    "failed and ice is smothering the coil.<|/item|>"
    "<|/list|>"
    "<|par|>Cheap test that settles it: empty both compartments, switch it off for 24 hours with "
    "the doors open, then run it again. If it works and slowly fails over the following days, that "
    "is the defrost circuit, and it needs a part.<|/par|>"
    "<|warn|>A humming compressor that never cycles off is a different fault and should not be "
    "left running for days — it is the one symptom worth calling someone about.<|/warn|>",
)

conv(
    "bottom leaves of my monstera are going yellow. more water or less",
    "<|par|>Less, almost certainly. Yellowing that starts at the <|bold|>bottom<|/bold|> and moves "
    "up, on leaves that feel soft rather than crisp, is the overwatering signature.<|/par|>"
    "<|par|>Underwatering yellows differently: the edges go brown and papery first, and the leaf "
    "feels dry and thin.<|/par|>"
    "<|par|>Put a finger in to the second knuckle. Damp means wait. The usual cause is not "
    "frequency at all but a pot with no drainage hole, where the bottom inch of soil never dries "
    "and the roots there rot.<|/par|>",
)

# ── travel and logistics ────────────────────────────────────────────────────

conv(
    "65 min connection at CDG, arriving 2E departing 2F. tight?",
    "<|par|>Tight but normally fine. <|bold|>2E to 2F is the easy case<|/bold|> — both are inside "
    "Terminal 2 and connected by a short walk plus a shuttle, with no re-clearing of security for "
    "a Schengen-to-Schengen or non-Schengen-to-non-Schengen move.<|/par|>"
    "<|par|>What decides it is not the walk but the two things either side of it:<|/par|>"
    "<|list|>"
    "<|item|>Whether you cross the Schengen border, which adds passport control and is the queue "
    "that actually eats an hour.<|/item|>"
    "<|item|>Whether the bags are checked through — if they are, a missed connection is the "
    "airline's problem and they rebook you.<|/item|>"
    "<|/list|>"
    "<|par|>Book it as a single ticket rather than two. On one ticket 65 minutes clears the "
    "airline's own minimum and they owe you a seat on the next flight; on two separate tickets a "
    "missed connection is a wasted fare.<|/par|>",
)

# ── writing and language ────────────────────────────────────────────────────

conv(
    "tighten this: \"It is important to note that the results of the survey seem to suggest that "
    "there may possibly be a correlation between the two variables in question.\"",
    "<|quote|>It is important to note that the results of the survey seem to suggest that there may "
    "possibly be a correlation between the two variables in question.<|/quote|>"
    "<|par|>→ <|bold|>The survey suggests the two variables are correlated.<|/bold|><|/par|>"
    "<|par|>24 words to 8. What came out and why:<|/par|>"
    "<|list|>"
    "<|item|><|italic|>It is important to note that<|/italic|> — if it were not worth noting you "
    "would not have written it.<|/item|>"
    "<|item|><|italic|>seem to<|/italic|> + <|italic|>may possibly<|/italic|> — three hedges "
    "stacked. \"Suggests\" already hedges; the others only make you sound unsure of the "
    "hedge.<|/item|>"
    "<|item|><|italic|>the results of the survey<|/italic|> → \"the survey\". The survey is the "
    "thing that suggests.<|/item|>"
    "<|item|><|italic|>in question<|/italic|> — says nothing the sentence has not already "
    "said.<|/item|>"
    "<|/list|>"
    "<|par|>Keep one hedge if the finding is genuinely weak. Stacking them does not make the claim "
    "more careful, only harder to read.<|/par|>",
)

conv(
    "is it fewer or less? my colleague keeps correcting me",
    "<|par|><|bold|>Fewer for things you count, less for things you measure.<|/bold|> Fewer emails, "
    "less email. Fewer coins, less money.<|/par|>"
    "<|par|>The correction people over-apply is the supermarket one. <|italic|>Ten items or "
    "less<|/italic|> is fine in ordinary use, and the exceptions are real and old:<|/par|>"
    "<|list|>"
    "<|item|>Distance, time and money take <|italic|>less<|/italic|> even though they are counted — "
    "less than five miles, less than two hours, less than 20 pounds.<|/item|>"
    "<|item|><|italic|>One less thing to worry about<|/italic|> is idiomatic and always has "
    "been.<|/item|>"
    "<|/list|>"
    "<|par|>So your colleague is right about the general rule and wrong to treat it as absolute.<|/par|>",
)

conv(
    "how would you translate schadenfreude into english without it sounding clinical",
    "<|par|>You cannot do it in one word, which is why English borrowed it. The literal gloss — "
    "<|italic|>pleasure at another's misfortune<|/italic|> — is accurate and sounds like a "
    "psychiatric report.<|/par|>"
    "<|par|>What works is translating the <|bold|>situation<|/bold|> rather than the word:<|/par|>"
    "<|list|>"
    "<|item|>\"I'm not proud of how much I enjoyed that.\"<|/item|>"
    "<|item|>\"There's a word for the feeling, and it isn't a nice one.\"<|/item|>"
    "<|item|>\"Serves him right\" — where the pleasure is earned, which schadenfreude often "
    "implies.<|/item|>"
    "<|/list|>"
    "<|par|>The English near-misses each lose something. <|italic|>Gloating<|/italic|> adds "
    "outward display; schadenfreude is usually private. <|italic|>Spite<|/italic|> adds wanting to "
    "cause the harm, which it does not.<|/par|>",
)

# ── science and how things work ─────────────────────────────────────────────

conv(
    "does a microwave heat food by vibrating water at its resonant frequency",
    "<|par|>No — that is the popular explanation and it is wrong. Water's resonances are far above "
    "2.45 GHz, and if the oven <|italic|>were<|/italic|> tuned to a resonance the energy would all "
    "be absorbed in the first millimetre and nothing inside would cook.<|/par|>"
    "<|par|>What actually happens is <|bold|>dielectric heating<|/bold|>. A water molecule is polar, "
    "so an oscillating field twists it; the molecule drags against its neighbours as it turns, and "
    "that friction is the heat. It is a broad, lossy effect, not a sharp resonance.<|/par|>"
    "<|par|>2.45 GHz was chosen for the opposite reason to resonance: it is absorbed weakly enough "
    "to penetrate a few centimetres, and it sits in a band set aside for industrial and domestic "
    "use.<|/par|>"
    "<|par|>The tell is that microwaves heat frozen food so badly. Ice is locked in a lattice and "
    "the molecules cannot rotate freely — same water, far less absorption.<|/par|>",
)

conv(
    "why is it so much colder at night in the desert than the beach at the same latitude",
    "<|thinking|>Two mechanisms, and it is worth separating them because people usually name only "
    "the first. One is humidity: water vapour is the dominant greenhouse gas, and dry desert air "
    "lets infrared radiate straight out to space after sunset. The other is the sea itself — water "
    "has a very high heat capacity and mixes, so a coast is buffered by an enormous thermal mass "
    "that barely changes temperature overnight. Sand, by contrast, has low heat capacity and only "
    "the top few centimetres ever warm up, so it dumps that heat quickly. Both point the same way, "
    "and the humidity one is the larger effect for the air temperature specifically.<|/thinking|>"
    "<|par|>Two things, and the humidity is the bigger one.<|/par|>"
    "<|subheading|>Dry air does not hold heat in<|/subheading|>"
    "<|par|>Water vapour is the main greenhouse gas. Humid coastal air absorbs the infrared the "
    "ground radiates at night and sends much of it back down. Desert air has almost no vapour, so "
    "that heat goes straight out to space — a clear desert night can drop 20°C or more.<|/par|>"
    "<|subheading|>The sea is an enormous flywheel<|/subheading|>"
    "<|par|>Water has roughly <|bold|>four times<|/bold|> the heat capacity of sand by mass, and it "
    "mixes, so the warmth spreads through a deep layer. A coast is sitting next to a thermal mass "
    "that barely changes overnight. Sand only warms in its top few centimetres, and gives that back "
    "in a couple of hours.<|/par|>"
    "<|divider|>"
    "<|par|>Which is why a cloudy desert night is much milder — the cloud does the job the "
    "missing water vapour was not doing.<|/par|>",
)

# ── mathematics and logic ───────────────────────────────────────────────────

conv(
    "in a room of 23 people there's a 50% chance two share a birthday? that can't be right, "
    "there are 365 days",
    "<|thinking|>The intuition being violated is that people count comparisons with themselves "
    "rather than between pairs. With 23 people there are 23 times 22 over 2, which is 253 pairs — "
    "each a chance to match. The clean way to show it is the complement: the probability that all "
    "birthdays differ is the running product 365/365 times 364/365 and so on down to 343/365, which "
    "comes to about 0.493, so a match is about 0.507. Worth naming the confusion explicitly, "
    "because the question people are actually imagining — someone sharing *your* birthday — really "
    "does need about 253 people, and that is the number their intuition is reaching for.<|/thinking|>"
    "<|par|>It is right, and the reason is that you are counting the wrong thing. The question is "
    "not \"does someone share <|italic|>my<|/italic|> birthday\" — it is \"does <|bold|>any "
    "pair<|/bold|> match\".<|/par|>"
    "<|par|>23 people make <|code|>23 × 22 / 2 = 253<|/code|> pairs, and each is a chance to "
    "collide.<|/par|>"
    "<|par|>Compute it as the chance that <|italic|>nobody<|/italic|> matches, which is a running "
    "product:<|/par|>"
    "<|codeblock|>365/365 × 364/365 × 363/365 × … × 343/365 ≈ 0.493<|/codeblock|>"
    "<|par|>So a shared birthday has probability about <|bold|>50.7%<|/bold|>.<|/par|>"
    "<|par|>Your instinct is not wrong, just aimed elsewhere: for someone to share <|italic|>your "
    "particular<|/italic|> birthday with even odds you do need roughly 253 people.<|/par|>",
)

conv(
    "car is rated 38 mpg. what's that in l/100km",
    "<|par|><|bold|>About 7.4 L/100km<|/bold|> if that is imperial mpg, or <|bold|>6.2 "
    "L/100km<|/bold|> if it is US mpg. Check which — a UK gallon is about 20% larger, and it is the "
    "usual way this conversion goes wrong.<|/par|>"
    "<|par|>The arithmetic, for UK gallons:<|/par|>"
    "<|codeblock|>38 miles/gal × 1.609 km/mile = 61.1 km/gal\n"
    "61.1 km / 4.546 L      = 13.4 km/L\n"
    "100 / 13.4             = 7.4 L/100km<|/codeblock|>"
    "<|par|>Note the units invert — mpg is distance per fuel, L/100km is fuel per distance. So "
    "<|italic|>higher<|/italic|> mpg is <|italic|>lower<|/italic|> L/100km, and you cannot average "
    "the two the same way.<|/par|>"
    "<|par|>Shortcut worth remembering: <|code|>282 / mpg(UK)<|/code|> gives L/100km directly.<|/par|>",
)

# ── music, art and games ────────────────────────────────────────────────────

conv(
    "why does a V chord want to go to I so badly",
    "<|par|>Two notes in it are under tension, and both resolve by a semitone in opposite "
    "directions.<|/par|>"
    "<|par|>In C major, G7 is <|code|>G B D F<|/code|>. The <|bold|>B<|/bold|> is the leading tone, "
    "a semitone under the tonic, and it pulls up to C. The <|bold|>F<|/bold|> is a semitone above "
    "E and pulls down to it. B and F together form a tritone — the least stable interval in the "
    "system — and it unwinds outward to C and E, two notes of the tonic chord.<|/par|>"
    "<|par|>That is why the plain triad pulls less hard than the seventh: without the F there is no "
    "tritone, only the leading tone.<|/par|>"
    "<|par|>It also explains the tritone substitution. <|code|>Db7<|/code|> contains the same two "
    "notes spelled differently, so it resolves to C just as convincingly with a chromatic bass "
    "step.<|/par|>",
)

conv(
    "flush draw after the flop, is the rule of 4 and 2 accurate enough to bet on",
    "<|par|>Accurate enough for the table, yes — it runs about a point or two optimistic on big "
    "draws.<|/par|>"
    "<|par|>Nine outs for a flush after the flop: <|code|>9 × 4 = 36%<|/code|> by two cards to "
    "come. The exact figure is <|bold|>35%<|/bold|>. One card to come: <|code|>9 × 2 = 18%<|/code|> "
    "against an exact 19.6%.<|/par|>"
    "<|warn|>The ×4 version assumes you get to see both cards. If a bet on the turn will price you "
    "out, you are drawing on one card and the honest number is the ×2 one.<|/warn|>"
    "<|par|>The bigger error is counting outs that are not outs — a flush card that pairs the board "
    "can complete a full house for someone else.<|/par|>",
)

# ── history, geography and society ──────────────────────────────────────────

conv(
    "was the fall of rome caused by lead pipes",
    "<|par|>No. It is a persistent story and the evidence is against it.<|/par|>"
    "<|par|>Roman water was hard, and the calcium carbonate it deposited lined the pipes with scale "
    "within months — the water largely ran over mineral, not lead. Roman aqueducts also ran "
    "continuously rather than standing in the pipe, which is the condition under which lead actually "
    "leaches.<|/par|>"
    "<|par|>The stronger case for Roman lead exposure is not plumbing at all but "
    "<|italic|>sapa<|/italic|>, a grape syrup boiled down in lead vessels and used as a sweetener. "
    "Even there, it is a health story about particular households, not a mechanism that brings down "
    "a state spanning three continents.<|/par|>"
    "<|par|>Historians mostly do not look for a single cause. The Western empire's decline runs "
    "through military overextension, currency debasement, civil war between claimants, and the "
    "settlement of groups it could not absorb — over roughly two centuries, while the Eastern half "
    "carried on for another thousand years drinking from the same kind of pipes.<|/par|>",
)

# ── everyday reasoning and planning ─────────────────────────────────────────

conv(
    "roughly how much water does a dripping tap waste in a year",
    "<|thinking|>A Fermi estimate. A drip is about a quarter of a millilitre. A steady drip is "
    "roughly one per second, so 0.25 mL per second is 900 mL per hour, call it a litre an hour. "
    "That is 24 litres a day and about 8,700 litres a year. Worth sanity-checking against the "
    "bathtub unit: 8,700 litres is around 50 bathtubs, which sounds like a lot for a drip but is "
    "the right order — the point of the estimate is that a small rate times a long time is a large "
    "number.<|/thinking|>"
    "<|par|><|bold|>Roughly 9,000 litres<|/bold|> — call it 8,000-10,000, which is about 50 "
    "bathtubs.<|/par|>"
    "<|par|>The estimate, so you can adjust it:<|/par|>"
    "<|list|>"
    "<|item|>A drip is about <|code|>0.25 mL<|/code|>.<|/item|>"
    "<|item|>A steady drip is roughly one per second → about <|bold|>1 litre an hour<|/bold|>.<|/item|>"
    "<|item|>24 litres a day → <|code|>24 × 365 ≈ 8,760 L<|/code|>.<|/item|>"
    "<|/list|>"
    "<|par|>The number is startling because a small rate is multiplied by a very long time, which "
    "is the general shape of this kind of estimate. A trickle rather than distinct drips is "
    "ten times worse.<|/par|>",
)

conv(
    "i want to get fit. what's the first thing i should do",
    "<|par|>Pick a time and a place, not a plan. The single best predictor of still training in "
    "three months is whether the session has a fixed slot in the week — not which programme you "
    "chose.<|/par|>"
    "<|par|>Concretely, this week:<|/par|>"
    "<|list|>"
    "<|item|>Choose <|bold|>two<|/bold|> specific slots. Days and times, in the calendar.<|/item|>"
    "<|item|>Make each one something you could do while tired and unmotivated — a 30-minute walk "
    "counts.<|/item|>"
    "<|item|>Do not change your diet in the same week.<|/item|>"
    "<|/list|>"
    "<|par|>Two sessions sounds too easy, and that is the point: the failure mode is starting at "
    "five and stopping at zero. Add the third once two have held for a month.<|/par|>"
    "<|par|><|italic|>Which<|/italic|> exercise matters far less than people expect at the "
    "start — it starts mattering once the habit is real.<|/par|>",
)

conv(
    "everyone says you should drink 8 glasses of water a day, but i only manage 4 and feel fine. "
    "am i damaging my kidneys",
    "<|par|>No, and the premise is shakier than the advice suggests. The \"8 glasses\" figure has no "
    "strong evidence behind it and is usually traced to a 1945 recommendation whose next sentence — "
    "that most of it comes from food — got dropped in the retelling.<|/par|>"
    "<|par|>Total water intake includes tea, coffee, and the substantial amount in food. Thirst is "
    "a well-calibrated signal in healthy adults; urine that is pale straw rather than dark is the "
    "practical check.<|/par|>"
    "<|par|>Situations where deliberate intake genuinely matters: heat, hard exercise, illness with "
    "fever or vomiting, and a history of kidney stones.<|/par|>"
    "<|warn|>Persistent thirst, or a real change in how often you are going, is worth a doctor "
    "rather than a rule of thumb — those are symptoms, not hydration habits.<|/warn|>",
)

# ── software and debugging (one domain of thirteen, not the whole set) ──────

conv(
    "getting \"permission denied (publickey)\" pushing to a repo that worked yesterday",
    "<|par|>The key is not reaching the server — that message means the server rejected every key "
    "offered, not that you lack access to the repo.<|/par|>"
    "<|par|>Most likely, in order:<|/par|>"
    "<|list|>"
    "<|item|>The agent lost the key on reboot. <|code|>ssh-add -l<|/code|> — if it says \"no "
    "identities\", that is the answer.<|/item|>"
    "<|item|>Wrong key for this host, if you keep more than one.<|/item|>"
    "<|item|>Permissions on the key file — SSH silently ignores a private key that is readable by "
    "others. It wants <|code|>600<|/code|>.<|/item|>"
    "<|/list|>"
    "<|par|>The command that tells you which it is:<|/par|>"
    "<|codeblock|>ssh -T git@github.com -v<|/codeblock|>"
    "<|par|>Read the <|code|>Offering public key<|/code|> lines — they show exactly which keys were "
    "tried and how the server answered each one.<|/par|>",
)

# ── the long, structured shape: title, heading, link ────────────────────────
#
# One conversation carrying the markers the rest of the set does not reach, so
# every entry in `teach::MARKUP` has somewhere to have been seen. Also the only
# exemplar with a genuinely long thinking block — the teach prompt describes the
# long tier as several paragraphs of real working thought and, until this, never
# showed the teacher one.

conv(
    "moving to a flat with no outdoor space and i want to grow herbs on a north-facing "
    "windowsill. worth trying or am i wasting my time",
    "<|thinking|>Worth being honest rather than encouraging here, because the specific "
    "constraint — north-facing — is the one that actually decides it, and a cheerful "
    "yes would set them up to fail and conclude they have no aptitude for it. So: what "
    "does north-facing mean in practice. In the northern hemisphere it means no direct "
    "sun at all for most of the year, and bright indirect light at best. Most culinary "
    "herbs are Mediterranean — rosemary, thyme, oregano, sage — and they evolved in "
    "full sun on poor dry soil. On a north sill they do not die quickly, which is the "
    "trap; they etiolate, stretch toward the glass, go leggy and pale, and the leaves "
    "lose most of their essential oil, so what you harvest tastes of very little. That "
    "is worse than failing outright because it takes two months to become obvious.\n\n"
    "What actually tolerates low light is a different group: mint, chives, parsley, "
    "chervil. Mint is a woodland-edge plant and genuinely does not want full sun. "
    "Chives are tough and regrow from the base after cutting. Parsley is a biennial "
    "that puts its energy into leaf in the first year and copes with less light than "
    "the woody herbs. So the answer is not yes or no, it is that the *selection* is "
    "the whole decision.\n\n"
    "The alternative I should weigh and reject: grow lights. They work, and for someone "
    "who wants basil specifically they are the only real answer. But recommending "
    "hardware first to someone asking whether to try at all is the wrong order — it "
    "converts a two-pound experiment into a forty-pound one before they know whether "
    "they will keep it watered. Mention it as the upgrade path, not the opening move.\n\n"
    "One more failure mode worth naming, because it is the one that actually kills "
    "windowsill herbs: supermarket pots are a dozen seedlings crammed into one plug, "
    "grown fast under glass for immediate sale. They are not designed to live. Splitting "
    "them or starting from seed is the difference between a plant and a bunch of cut "
    "herbs in soil. Decide: lead with the yes-but, give the two lists, name the "
    "supermarket trap, mention lights last.<|/thinking|>"
    "<|title|>Herbs on a north-facing sill<|/title|>"
    "<|par|>Worth trying, but the choice of herb decides it entirely — and the "
    "Mediterranean ones everybody starts with are exactly the ones that will "
    "fail.<|/par|>"
    "<|heading|>What works<|/heading|>"
    "<|list|>"
    "<|item|><|bold|>Mint<|/bold|> — a woodland-edge plant that actively dislikes full "
    "sun. Give it its own pot; it will take over a shared one.<|/item|>"
    "<|item|><|bold|>Chives<|/bold|> — tough, and regrow from the base after cutting.<|/item|>"
    "<|item|><|bold|>Parsley and chervil<|/bold|> — leafy biennials, content with bright "
    "indirect light.<|/item|>"
    "<|/list|>"
    "<|heading|>What will disappoint you<|/heading|>"
    "<|par|>Rosemary, thyme, oregano, sage and basil evolved in full Mediterranean sun on "
    "poor dry soil. On a north sill they do not die — they <|italic|>stretch<|/italic|>, "
    "going pale and leggy as they reach for the glass, and the leaves lose most of their "
    "essential oil. You get a plant that looks alive and tastes of almost nothing, which "
    "takes about two months to become obvious.<|/par|>"
    "<|divider|>"
    "<|warn|>Do not start with a supermarket pot. Those are a dozen seedlings forced into "
    "one plug for immediate sale, and they are not built to live — split them out or start "
    "from seed.<|/warn|>"
    "<|par|>If you specifically want basil, a small LED grow light is the only honest "
    "answer, and a cheap clip-on one is enough for a sill. Worth it once you know you will "
    "keep them watered — not before. The RHS write-up on growing herbs indoors is a "
    "sensible starting point: <|link|>rhs.org.uk/herbs/growing-indoors<|/link|><|/par|>",
)

# ── a short one, so the set is not uniformly long ───────────────────────────

conv(
    "whats the difference between baking soda and baking powder",
    "<|par|>Baking powder is baking soda with the acid already mixed in.<|/par|>"
    "<|par|>Soda needs an acid in the recipe — buttermilk, yoghurt, lemon, honey, cocoa — to react "
    "and produce gas. Powder brings its own dry acid and needs only moisture and heat.<|/par|>"
    "<|par|>So they are not interchangeable in either direction. Soda where powder was called for "
    "gives a flat, faintly soapy result; powder where soda was called for under-rises and leaves "
    "the batter's acidity unbalanced.<|/par|>",
)


# Beside the other corpus scripts, and pointing at the same directory they do.
SFT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "data", "raw", "sft",
)


def main():
    path = os.path.join(SFT_DIR, "gold-general.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in C:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(C)} conversations to {path}")


if __name__ == "__main__":
    main()
