#!/usr/bin/env python3
"""
Generate synthetic thinking-trace training data in an assistant style.

Each file is a short assistant interaction: an inner-monologue <|thinking|> block
where the assistant reasons about how to help, followed by a <|par|> reply. The
substance is deliberately plain natural-language prose — no math, equations, or
symbols — because the point is to teach the *thinking pattern* (open the block,
reason in words, close it, then answer), not to teach arithmetic. Math-heavy
content pollutes the BPE vocabulary and a small model just learns to emit broken
equations, so it's avoided entirely.

Usage:
    python3 generate_synthetic.py [--count 100] [--out /path/to/dir]
"""

import argparse
import os
import random
import re

SEED = 42
OUT_DIR = "/home/x_x/Programming/LLM/backend/data/raw/synthetic"


# Each topic is one short assistant turn (some have a follow-up). A step is a
# triple: (ask, think, reply).
#   ask   — the user's request, restated so the monologue has something to chew on
#   think — the assistant's actual reasoning, in plain prose
#   reply — the helpful answer that goes in the <|par|> block
TOPICS = [
    {
        "title": "A Book Recommendation",
        "steps": [
            (
                "They loved a fast, suspenseful mystery and want something similar to read next.",
                "What they enjoyed was the pace and the pull of not knowing what happens, so a slow literary novel would miss the mark. A tightly plotted thriller with short chapters is the safer match. I don't know which books they've already read, so rather than guess a single title I should describe the kind of book that fits and offer to narrow it down once they tell me a little more.",
                "Since you liked the pace and the suspense, you'd probably enjoy a tightly plotted thriller with short chapters and a strong hook early on. If you tell me whether you prefer a clever detective puzzle or more of a chase, I can point you to a specific author who fits.",
            ),
        ],
    },
    {
        "title": "Wording a Polite Decline",
        "steps": [
            (
                "They want help saying no to an invitation without sounding rude.",
                "The worry here is coming across as cold, so the reply should be warm and brief, thank them, decline clearly, and leave the door open. A long explanation usually makes it feel more awkward, not less, so shorter is kinder.",
                "A short, warm note works best. Something like: thank them, say you won't be able to make it this time, and add that you hope it goes well and would love to catch up soon. You don't owe a detailed reason, and keeping it brief actually feels more sincere.",
            ),
        ],
    },
    {
        "title": "Calming Pre-Presentation Nerves",
        "steps": [
            (
                "They're nervous before giving a presentation tomorrow and want advice.",
                "Nerves usually come from feeling unprepared or from imagining the worst, so the most useful advice is practical and reassuring rather than a lecture. A run-through out loud and a calm opening tend to help the most, and it's worth reminding them that a little nervous energy is normal and even useful.",
                "Try running through it out loud once or twice tonight, especially the first minute, since a steady opening settles the rest. A few slow breaths before you start help more than you'd expect. And remember, a bit of nervous energy is normal and usually makes you sharper, not worse.",
            ),
        ],
    },
    {
        "title": "Choosing Between Two Cafes",
        "steps": [
            (
                "They can't decide between a quiet cafe far away and a lively one nearby.",
                "There's no objectively right answer here; it depends on what they want from the visit. Instead of picking for them, the helpful move is to surface the trade-off clearly so they can decide, since one is better for focus and the other for atmosphere.",
                "It really comes down to what you're after today. If you want to read or get work done, the quiet one is worth the extra walk. If you'd rather soak up some atmosphere and don't mind the noise, the nearby one wins. What's the mood you're in?",
            ),
        ],
    },
    {
        "title": "How to Start Reading More",
        "steps": [
            (
                "They say they want to read more but never seem to find the time.",
                "Usually the problem isn't time so much as friction and unrealistic goals. So the advice should be small and concrete: a tiny daily amount, a book they actually enjoy, and a fixed moment to attach it to. Telling them to just read more isn't useful.",
                "Start smaller than feels worth it, even ten minutes a day, and attach it to something you already do, like right before bed. Pick a book you genuinely enjoy rather than one you think you should read, since finishing one builds the habit far better than struggling through something dull.",
            ),
        ],
    },
    {
        "title": "Comforting a Friend Who Did Poorly",
        "steps": [
            (
                "Their friend just failed an important test and they want to know what to say.",
                "The instinct to fix it or hand out study tips is the wrong first move; right now the friend mostly needs to feel heard. So the reply should lead with acknowledgment and support, and only gently point toward next steps if it fits.",
                "Lead with how they feel, not with advice. Something simple like, that sounds really disappointing, I'm sorry, and you're allowed to be upset about it. Once they feel heard, you can gently ask if they want to talk about what's next, but let that come after, not instead of, the support.",
            ),
        ],
    },
    {
        "title": "A Gift for a Gardener",
        "steps": [
            (
                "They need a gift idea for someone who loves gardening.",
                "Gardeners often already own the basic tools, so a generic trowel might miss. Something that's either a small upgrade on the everyday or a nice-to-have they wouldn't buy themselves tends to land better. I should give a couple of options at different price points and let them choose.",
                "A safe bet is something that upgrades an everyday task, like a really good pair of gloves or comfortable kneeling pad. If you want something they'd enjoy but wouldn't buy for themselves, a nice packet of unusual seeds or a small potted plant they don't have yet works well. Do you know roughly what they grow?",
            ),
        ],
    },
    {
        "title": "Explaining a Metaphor Simply",
        "steps": [
            (
                "They ask what a metaphor is, in plain terms.",
                "The trap is defining it with jargon like tenor and vehicle, which won't help. A plain definition plus one clear everyday example is far more useful, and contrasting it briefly with a simile makes it stick.",
                "A metaphor describes one thing as if it were another to make a point, without using like or as. Saying time is a thief is a metaphor, since time doesn't literally steal, but calling it a thief captures how it slips away. If you add like or as, as in time is like a thief, it becomes a simile instead.",
            ),
        ],
    },
    {
        "title": "Getting Better Sleep",
        "steps": [
            (
                "They've been sleeping badly and want a couple of practical tips.",
                "Sleep advice can balloon into a huge list, but that's overwhelming and rarely acted on. Better to give two or three high-impact, easy changes rather than everything at once. Screens and an inconsistent schedule are the usual culprits, so I'll focus there.",
                "Two changes tend to help the most. First, try to go to bed and wake up around the same time, even on weekends, since a steady rhythm matters more than the exact hours. Second, dim the lights and put your phone down half an hour before bed. Start with just those rather than overhauling everything.",
            ),
        ],
    },
    {
        "title": "How to Apologize Sincerely",
        "steps": [
            (
                "They hurt a friend's feelings and want to apologize the right way.",
                "A good apology isn't about clever wording, it's about taking responsibility without excuses. The common mistake is the sorry if you were upset phrasing, which shifts blame. So I should steer them toward naming what they did, acknowledging the effect, and not tacking on a but.",
                "The key is to own the specific thing you did, not to apologize for how they reacted. Say what you did, acknowledge that it hurt them, and stop there, without adding a but or an excuse. Something like, I'm sorry I snapped at you earlier, that wasn't fair and I can see it upset you. Then give them room to respond.",
            ),
        ],
    },
    {
        "title": "A Thank-You Note",
        "steps": [
            (
                "They want help writing a short thank-you note for a gift.",
                "Thank-you notes feel hard mostly because people overthink them. The useful structure is simple: name the gift, say something specific about why it's appreciated, and add a warm closing. Specificity is what makes it feel genuine rather than generic.",
                "Keep it short and specific. Name the gift, say one real thing about why you like it or how you'll use it, and close warmly. For example: thank you so much for the scarf, the color is perfect and I've already worn it twice. It was so thoughtful of you. That one specific detail is what makes it feel heartfelt.",
            ),
        ],
    },
    {
        "title": "Staying Focused While Studying",
        "steps": [
            (
                "They keep getting distracted while studying and want to fix it.",
                "Distraction usually has two sources: the phone within reach and sessions that are too long to sustain. So the practical fix is to remove the temptation and work in shorter focused blocks with breaks. Willpower advice alone rarely works.",
                "Try putting your phone in another room, not just face down, since being out of reach makes a real difference. Then work in shorter blocks, say twenty-five minutes, with a short break after each one. It feels less daunting than a long stretch, and you'll usually get more done.",
            ),
        ],
    },
    {
        "title": "First Day at a New Job",
        "steps": [
            (
                "They're anxious about their first day at a new job and want advice.",
                "First-day anxiety is mostly about the unknown, so reassurance plus a couple of small, doable actions helps more than big-picture career advice. Nobody expects them to know everything yet, and asking questions early is a good sign, not a bad one.",
                "Go in expecting to learn, not to perform. Nobody expects you to know how everything works on day one, so it's completely fine to ask questions and write things down. Learn a few names, be friendly, and give yourself permission to feel a little lost at first. That settles within a week or two.",
            ),
        ],
    },
    {
        "title": "Being a Better Listener",
        "steps": [
            (
                "They want to become a better listener in conversations.",
                "The most common listening mistake is planning your reply while the other person is still talking. So the core advice is to stay with what they're saying and hold back the urge to jump in or fix. A small concrete habit is more useful than a vague be present.",
                "The biggest shift is to stop planning your response while they're still talking, and just follow what they're actually saying. A simple trick is to pause for a beat before you reply, and sometimes to reflect back what you heard. Resist the urge to immediately fix or one-up, since often people just want to feel heard.",
            ),
        ],
    },
    {
        "title": "Congratulating Someone",
        "steps": [
            (
                "A friend got a promotion and they want to say something more than just congrats.",
                "A bare congrats is fine but forgettable. What makes it land is naming something specific, that they earned it or worked hard for it, so it feels personal rather than reflexive. I'll suggest a warm, specific line.",
                "Add something specific so it doesn't feel automatic. Instead of just congratulations, try something like: this is so well deserved, I know how hard you've worked for it, I'm really happy for you. Naming that you saw the effort is what makes it feel genuine.",
            ),
        ],
    },
    {
        "title": "Breaking Down a Big Task",
        "steps": [
            (
                "They feel stuck on a big project and don't know where to start.",
                "Feeling stuck usually means the task is too vague and too large to hold in your head. The fix is to make the first step small enough that it's almost silly not to do it. So I should encourage them to define one tiny concrete starting action rather than plan the whole thing.",
                "When something feels too big, the trick is to shrink the first step until it's almost too easy to skip, not plan the whole thing. Pick one small, concrete action you could do in a few minutes, like opening the document and writing a rough outline. Starting is the hard part, and one small step usually loosens the rest.",
            ),
        ],
    },
    {
        "title": "Choosing a Movie for a Group",
        "steps": [
            (
                "A group with very different tastes can't agree on a movie.",
                "Trying to find a film everyone loves usually fails; the realistic goal is one nobody objects to. Something broadly liked, a bit of comedy and a bit of adventure, tends to be the safe middle ground. I can also suggest a quick fair way to pick.",
                "Aim for something nobody dislikes rather than everyone's favorite, since that's more achievable. A film with a bit of both comedy and adventure usually pleases a mixed group. If you're still stuck, have everyone veto one option and pick from what survives, that's quick and feels fair.",
            ),
        ],
    },
    {
        "title": "Asking for Help Politely",
        "steps": [
            (
                "They need to ask a busy coworker for help and don't want to impose.",
                "The concern is imposing, so the request should be respectful of the coworker's time: acknowledge they're busy, be specific about what's needed, and make it easy to say not now. Vague or open-ended asks are what actually feel like an imposition.",
                "Make it easy to say yes or no. Acknowledge they're busy, be specific about exactly what you need and how long it'll take, and offer a couple of times rather than expecting them to drop everything. Something like: I know you're swamped, but when you get a spare ten minutes, could you take a look at this one thing? Being specific respects their time.",
            ),
        ],
    },
    {
        "title": "Encouraging a New Musician",
        "steps": [
            (
                "They just started learning an instrument and feel discouraged by slow progress.",
                "Early discouragement is almost always about comparing where you are to where you want to be. The helpful reframe is that progress at this stage is invisible day to day but real over weeks, and that a little consistent practice beats occasional long sessions.",
                "Progress on an instrument is sneaky, it barely shows day to day but adds up a lot over a few weeks, so it's normal to feel like you're not improving even when you are. A little practice most days beats a long session once in a while. Try recording yourself now and again, it's often the clearest proof of how far you've come.",
            ),
        ],
    },
    {
        "title": "Keeping a Conversation Going",
        "steps": [
            (
                "They run out of things to say in conversations and want help.",
                "Conversations stall when questions can be answered in one word. The practical fix is open questions and following up on what the other person just said rather than hunting for a brand-new topic. Genuine curiosity does most of the work.",
                "The trick is to ask questions that can't be answered in one word, and to follow up on what they just said instead of scrambling for a new topic. If someone mentions a trip, ask what the best part was rather than moving on. Being a little curious takes the pressure off you to keep talking.",
            ),
        ],
    },
    {
        "title": "Giving Feedback Kindly",
        "steps": [
            (
                "They need to give a friend honest feedback without hurting them.",
                "The tension is between honesty and kindness, and the resolution is that they're not opposites. Being specific, focusing on the work rather than the person, and framing it around what would make it stronger keeps it kind and useful. Vague softness actually helps less.",
                "Be honest but specific, and aim at the work, not the person. Instead of a vague this isn't great, point to one concrete thing and how it could be stronger. It also helps to lead with what genuinely worked before the suggestion. Kind and honest aren't opposites, specific and caring is usually both.",
            ),
        ],
    },
    {
        "title": "Unwinding After a Stressful Day",
        "steps": [
            (
                "They had a rough day and want a few ways to unwind that aren't just scrolling.",
                "They've already flagged that scrolling isn't what they want, so I should skip it and suggest small, low-effort things that actually reset you: a bit of movement, something with your hands, or stepping outside. A short list is easier to act on than a big one.",
                "A few things tend to reset you more than a screen does. A short walk, even around the block, changes your head surprisingly fast. Something with your hands, like cooking or tidying one small space, gives a sense of order. And even a few minutes outside or a warm shower can quietly take the edge off.",
            ),
        ],
    },
    {
        "title": "Affect Versus Effect",
        "steps": [
            (
                "They always mix up affect and effect and want an easy way to remember.",
                "A grammar rule with exceptions will overwhelm them; a simple memory trick covers the everyday cases. Affect is usually the verb and effect the noun, and a short mnemonic makes that stick without a lecture on the rare uses.",
                "For everyday use, affect is usually the action, the verb, and effect is usually the thing, the noun. The rain affects your mood; the rain has an effect on your mood. A quick trick: affect is action, both start with A. There are rarer uses, but that covers almost everything you'll write.",
            ),
        ],
    },
    {
        "title": "Setting a Small Daily Habit",
        "steps": [
            (
                "They want to build a new daily habit but keep giving up after a few days.",
                "Habits usually fail because they're too ambitious and not anchored to anything. The fix is to make the habit tiny and attach it to something they already do every day, so it doesn't rely on motivation. Missing a day is fine as long as they don't miss two.",
                "Make it almost embarrassingly small to start, and attach it to something you already do, like doing it right after your morning coffee. Tiny and anchored beats big and motivated, because you're not relying on willpower. And if you miss a day, just don't miss two in a row, that's what actually keeps a habit alive.",
            ),
        ],
    },
    {
        "title": "An Overwhelming To-Do List",
        "steps": [
            (
                "Their to-do list feels endless and they don't know what to do first.",
                "When everything feels urgent, the real problem is that nothing is prioritized. The useful move is to pick just one or two things that matter most today and let the rest wait, rather than trying to do it all. Naming the single most important task is the unlock.",
                "When the list feels endless, the answer usually isn't to work faster, it's to choose. Ask yourself which one thing, if you finished it today, would make the biggest difference, and start there. Let the rest sit. A long list is far less scary once you've picked the one thing that actually matters most right now.",
            ),
        ],
    },
    {
        "title": "Making a Good First Impression",
        "steps": [
            (
                "They're meeting someone important soon and want to make a good impression.",
                "People overthink this and try to impress, which usually backfires. The reassuring and practical truth is that warmth and genuine attention land better than a polished performance. A couple of small concrete behaviors help more than trying to seem impressive.",
                "The best first impression usually comes from being warm and genuinely attentive, not from trying to impress. Show up on time, remember their name, and actually listen rather than waiting to talk. People remember how you made them feel far more than anything clever you said, so relax and be present.",
            ),
        ],
    },
    {
        "title": "What to Pack for a Short Trip",
        "steps": [
            (
                "They're going away for a couple of nights and always overpack.",
                "Overpacking comes from packing for imagined scenarios rather than the actual trip. The fix is to plan around what they'll really do, pack versatile pieces, and accept that they can buy anything they forget. A rule of thumb is easier than a full list.",
                "Pack for what you'll actually do, not every possible what-if. For a couple of nights, a few versatile pieces you can mix and match beat a bag full of just-in-case items. Lay everything out, then take out about a third of it. And remember, almost anything you forget can be bought when you get there.",
            ),
        ],
    },
    {
        "title": "A Name for a New Pet",
        "steps": [
            (
                "They just got a puppy and can't settle on a name.",
                "Naming is personal, so instead of just handing over a random list, it's more helpful to give a way to choose: names that are easy to call out, that suit the dog's look or personality, and that they won't tire of. Then a few examples to spark ideas.",
                "A good dog name is easy to say out loud and call across a park, so two syllables often work well. Beyond that, it helps to wait a day or two and let their personality suggest something. If you want a starting point, tell me what they look like or how they act, and I'll toss out a few ideas that fit.",
            ),
        ],
    },
    {
        "title": "Explaining Why the Sky Is Blue",
        "steps": [
            (
                "A child asked them why the sky is blue and they want a simple way to explain it.",
                "The audience is a child, so the explanation has to be simple and picture-friendly, not a physics lecture. The core idea, that sunlight is made of colors and the blue light gets scattered around the sky the most, can be told in one or two friendly sentences.",
                "You can keep it simple: sunlight looks white but is really made of all the colors mixed together. As it comes through the air, the blue light bounces around and spreads across the whole sky more than the other colors, so that's the color we see everywhere we look up. For a child, that's plenty.",
            ),
        ],
    },
    {
        "title": "Planning a Relaxing Weekend",
        "steps": [
            (
                "They want to actually relax this weekend but usually end up busy or bored.",
                "The two failure modes they named are overbooking and doing nothing, so the sweet spot is a loose plan with one or two nice things and plenty of open space. The advice should protect rest without scheduling it to death.",
                "Aim for a loose plan rather than an empty calendar or a packed one. Pick one or two things you'd genuinely enjoy, maybe a walk somewhere new and a meal you like, and leave the rest open on purpose. A little structure keeps you from drifting into boredom, while the open space is what actually lets you rest.",
            ),
        ],
    },
    {
        "title": "Remembering People's Names",
        "steps": [
            (
                "They're bad at remembering names right after meeting someone.",
                "Names slip because we're not really paying attention in the moment of introduction, we're a bit self-conscious. So the fix is to slow down and use the name once right away, which both anchors it and shows warmth. A simple, doable trick beats a memory-palace lecture.",
                "Most of the time names don't stick because we're a little distracted the second we hear them. The simplest fix is to say the name back right away, like nice to meet you, Sarah, and then use it once more in the next minute or two. Repeating it early anchors it, and it comes across as warm too.",
            ),
        ],
    },
]


def expand_thinking(rng, ask: str, think: str, reply: str) -> list[tuple[str, str]]:
    """Turn one (ask, think, reply) turn into an inner-monologue block and a reply."""
    openers = [
        "Okay, let me make sure I understand what they're really asking.",
        "Let me think about what would genuinely help here.",
        "Right, let me work out the most useful way to answer this.",
        "Let me consider this for a moment before I reply.",
        "Let me think about what they actually need, not just what they literally said.",
    ]
    checks = [
        "Let me make sure this actually answers what they asked and doesn't wander into things they didn't.",
        "I want to keep this clear and genuinely useful, so let me check the answer is easy to follow.",
        "Before I reply, let me see whether there's a simpler, more direct way to put it.",
        "Let me make sure I'm not overcomplicating this; the clearest answer is usually the most helpful one.",
        "I should check that this stays balanced and doesn't push them one way without good reason.",
    ]
    leadins = [
        "So here's how I'd put it.",
        "That gives me a clear way to respond.",
        "Alright, I think I've got the right answer for them.",
        "Good, I'll keep the reply simple and direct.",
    ]

    parts = [f"{rng.choice(openers)} {ask}", think]
    if rng.random() < 0.5:
        parts.insert(2, rng.choice(checks))
    parts.append(rng.choice(leadins))

    return [(" ".join(parts), reply)]


def format_file(title: str, thought_pairs: list[tuple[str, str]], rng) -> str:
    """Format a file with alternating thinking and par blocks."""
    lines = [f"<|title|>{title}<|/title|>", ""]
    for thought_text, response_text in thought_pairs:
        lines.append(f"<|thinking|>{thought_text}<|/thinking|>")
        lines.append("")
        lines.append(f"<|par|>{response_text}<|/par|>")
        lines.append("")
    lines.append("<|endoftext|>")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic thinking-trace training data.")
    ap.add_argument("--count", type=int, default=100, help="Number of files to generate")
    ap.add_argument("--out", default=OUT_DIR, help="Output directory")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    existing = [f for f in os.listdir(args.out) if f.startswith("synthetic") and f.endswith(".txt")]
    if existing:
        nums = [int(re.match(r"synthetic(\d+)\.txt", f).group(1)) for f in existing if re.match(r"synthetic(\d+)\.txt", f)]
        start = max(nums) + 1 if nums else 1
    else:
        start = 1

    for i in range(args.count):
        topic = rng.choice(TOPICS)
        idx = start + i
        all_pairs = []
        for ask, think, reply in topic["steps"]:
            all_pairs.extend(expand_thinking(rng, ask, think, reply))
        content = format_file(topic["title"], all_pairs, rng)
        fname = f"synthetic{idx:04d}.txt"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {args.count} files -> {args.out}")


if __name__ == "__main__":
    main()
