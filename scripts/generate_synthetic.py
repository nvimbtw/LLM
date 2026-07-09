#!/usr/bin/env python3
"""
Generate synthetic thinking-trace training data in an assistant style.

Each file is a short assistant interaction: an inner-monologue <|thinking|> block
where the assistant reasons at length about how to help, followed by a <|par|>
reply. The substance is deliberately plain natural-language prose — no math,
equations, or symbols — because the point is to teach the *thinking pattern*
(open the block, reason step by step in words, weigh options, self-correct,
then answer), not to teach arithmetic. Math-heavy content pollutes the BPE
vocabulary and a small model just learns to emit broken equations, so it's
avoided entirely.

The reasoning for each topic is written as a list of short "beats" — one thought
per sentence — so the monologue reads like a real chain of thought rather than a
single lump. The assembler weaves those beats together with openers, a mid-way
reflection, a self-check, and a lead-in, producing long thinking chains with
plenty of variation between files.

Usage:
    python3 generate_synthetic.py [--count 100] [--out /path/to/dir]
"""

import argparse
import os
import random
import re

SEED = 42
OUT_DIR = "/home/x_x/Programming/LLM/backend/data/raw/synthetic"


# Each topic is one short assistant turn. A step is a triple:
#   ask   — the user's request, restated so the monologue has something to chew on
#   think — the assistant's reasoning, as a list of one-sentence beats (plain prose)
#   reply — the helpful answer that goes in the <|par|> block
TOPICS = [
    {
        "title": "A Book Recommendation",
        "steps": [
            (
                "They loved a fast, suspenseful mystery and want something similar to read next.",
                [
                    "What they enjoyed most was the pace and the pull of not knowing what happens next, so that pull is the thing to protect.",
                    "A slow, meandering literary novel would miss the mark completely, even if it's beautifully written.",
                    "A tightly plotted thriller with short chapters and an early hook is the safer match for that feeling.",
                    "The trouble is I don't know which books they've already read, so naming a single title risks suggesting one they've finished.",
                    "Rather than guess, it's better to describe the kind of book that fits and then invite one detail that lets me narrow it down.",
                    "That way the suggestion is useful now but can get sharper the moment they tell me a little more.",
                ],
                "Since you liked the pace and the suspense, you'd probably enjoy a tightly plotted thriller with short chapters and a strong hook early on. If you tell me whether you prefer a clever detective puzzle or more of a chase, I can point you to a specific author who fits.",
            ),
        ],
    },
    {
        "title": "Wording a Polite Decline",
        "steps": [
            (
                "They want help saying no to an invitation without sounding rude.",
                [
                    "The real worry underneath this is sounding cold or ungrateful, so warmth has to come first.",
                    "The reply should thank them, decline clearly, and leave the door open, all without dragging on.",
                    "A long, detailed explanation tends to make a decline feel more awkward, not less.",
                    "Over-explaining can even read as guilt, which quietly invites them to push back.",
                    "So the kinder shape is short: appreciation, a clear no, and a friendly note about next time.",
                ],
                "A short, warm note works best. Something like: thank them, say you won't be able to make it this time, and add that you hope it goes well and would love to catch up soon. You don't owe a detailed reason, and keeping it brief actually feels more sincere.",
            ),
        ],
    },
    {
        "title": "Calming Pre-Presentation Nerves",
        "steps": [
            (
                "They're nervous before giving a presentation tomorrow and want advice.",
                [
                    "Nerves like this usually come from feeling underprepared or from picturing the worst outcome.",
                    "That means a lecture on public speaking won't land; something practical and reassuring will.",
                    "A run-through out loud, especially of the opening, tends to steady the whole thing.",
                    "A few slow breaths right before starting genuinely helps settle the body.",
                    "It's also worth naming that a little nervous energy is normal and can actually sharpen a performance.",
                    "Keeping the advice to a couple of concrete actions makes it easier to follow tonight.",
                ],
                "Try running through it out loud once or twice tonight, especially the first minute, since a steady opening settles the rest. A few slow breaths before you start help more than you'd expect. And remember, a bit of nervous energy is normal and usually makes you sharper, not worse.",
            ),
        ],
    },
    {
        "title": "Choosing Between Two Cafes",
        "steps": [
            (
                "They can't decide between a quiet cafe far away and a lively one nearby.",
                [
                    "There's no objectively correct choice here; it hinges on what they want from the visit.",
                    "One cafe is better for focus, the other for atmosphere, and those pull in opposite directions.",
                    "Picking for them would be presumptuous and might not match their mood today.",
                    "The more helpful move is to lay the trade-off out plainly so the decision is easy.",
                    "Ending with a small question about their mood nudges them to choose without pressure.",
                ],
                "It really comes down to what you're after today. If you want to read or get work done, the quiet one is worth the extra walk. If you'd rather soak up some atmosphere and don't mind the noise, the nearby one wins. What's the mood you're in?",
            ),
        ],
    },
    {
        "title": "How to Start Reading More",
        "steps": [
            (
                "They say they want to read more but never seem to find the time.",
                [
                    "The stated problem is time, but usually the real obstacles are friction and goals that are too big.",
                    "Telling them to simply read more just restates the problem instead of solving it.",
                    "A tiny daily amount feels achievable and sidesteps the all-or-nothing trap.",
                    "Attaching it to something they already do gives it a reliable trigger.",
                    "Choosing a book they genuinely enjoy matters more than a worthy one they'll abandon.",
                    "Finishing one enjoyable book builds momentum far better than grinding through something dull.",
                ],
                "Start smaller than feels worth it, even ten minutes a day, and attach it to something you already do, like right before bed. Pick a book you genuinely enjoy rather than one you think you should read, since finishing one builds the habit far better than struggling through something dull.",
            ),
        ],
    },
    {
        "title": "Comforting a Friend Who Did Poorly",
        "steps": [
            (
                "Their friend just failed an important test and they want to know what to say.",
                [
                    "The instinct to jump straight to fixing it or offering study tips is the wrong first move.",
                    "Right now the friend mainly needs to feel heard, not managed.",
                    "Leading with advice can accidentally imply the feelings aren't valid.",
                    "So the reply should open with acknowledgment and permission to be upset.",
                    "Only after that, and only gently, does it make sense to point toward next steps.",
                    "The order matters: comfort first, and next steps only if the friend seems ready.",
                ],
                "Lead with how they feel, not with advice. Something simple like, that sounds really disappointing, I'm sorry, and you're allowed to be upset about it. Once they feel heard, you can gently ask if they want to talk about what's next, but let that come after, not instead of, the support.",
            ),
        ],
    },
    {
        "title": "A Gift for a Gardener",
        "steps": [
            (
                "They need a gift idea for someone who loves gardening.",
                [
                    "Keen gardeners usually already own the basic tools, so a generic trowel could easily miss.",
                    "The gifts that land are either a small upgrade on an everyday task or a nice-to-have they wouldn't buy themselves.",
                    "Guessing blindly is risky, so offering a couple of options at different price points is safer.",
                    "Letting them choose keeps the answer useful even though I don't know the person.",
                    "A light question about what the gardener grows could sharpen the suggestion further.",
                ],
                "A safe bet is something that upgrades an everyday task, like a really good pair of gloves or comfortable kneeling pad. If you want something they'd enjoy but wouldn't buy for themselves, a nice packet of unusual seeds or a small potted plant they don't have yet works well. Do you know roughly what they grow?",
            ),
        ],
    },
    {
        "title": "Explaining a Metaphor Simply",
        "steps": [
            (
                "They ask what a metaphor is, in plain terms.",
                [
                    "The trap is defining it with jargon like tenor and vehicle, which helps no one.",
                    "A plain definition plus one clear everyday example will do far more work.",
                    "Choosing a familiar example, like calling time a thief, makes the idea concrete.",
                    "Briefly contrasting it with a simile helps the distinction actually stick.",
                    "Keeping it to a couple of sentences avoids turning a simple question into a lesson.",
                ],
                "A metaphor describes one thing as if it were another to make a point, without using like or as. Saying time is a thief is a metaphor, since time doesn't literally steal, but calling it a thief captures how it slips away. If you add like or as, as in time is like a thief, it becomes a simile instead.",
            ),
        ],
    },
    {
        "title": "Getting Better Sleep",
        "steps": [
            (
                "They've been sleeping badly and want a couple of practical tips.",
                [
                    "Sleep advice can balloon into an enormous checklist, which is overwhelming and rarely acted on.",
                    "It's better to hand over two or three high-impact changes than to list everything at once.",
                    "The usual culprits are screens late at night and an inconsistent schedule.",
                    "A steady sleep and wake time tends to matter more than the exact number of hours.",
                    "Dimming the lights and putting the phone down beforehand is easy and effective.",
                    "Framing it as starting with just those two keeps it from feeling like a full overhaul.",
                ],
                "Two changes tend to help the most. First, try to go to bed and wake up around the same time, even on weekends, since a steady rhythm matters more than the exact hours. Second, dim the lights and put your phone down half an hour before bed. Start with just those rather than overhauling everything.",
            ),
        ],
    },
    {
        "title": "How to Apologize Sincerely",
        "steps": [
            (
                "They hurt a friend's feelings and want to apologize the right way.",
                [
                    "A good apology isn't about clever wording; it's about taking responsibility without excuses.",
                    "The classic mistake is the sorry if you were upset phrasing, which quietly shifts the blame.",
                    "So the guidance should steer them toward naming the specific thing they did.",
                    "Then acknowledging the effect it had, plainly, without softening it into nothing.",
                    "And crucially, no tacked-on but, since a but erases the apology that came before it.",
                    "Ending by giving the other person room to respond keeps it from feeling like a performance.",
                ],
                "The key is to own the specific thing you did, not to apologize for how they reacted. Say what you did, acknowledge that it hurt them, and stop there, without adding a but or an excuse. Something like, I'm sorry I snapped at you earlier, that wasn't fair and I can see it upset you. Then give them room to respond.",
            ),
        ],
    },
    {
        "title": "A Thank-You Note",
        "steps": [
            (
                "They want help writing a short thank-you note for a gift.",
                [
                    "These feel hard mainly because people overthink them and reach for grand phrasing.",
                    "A simple structure carries it: name the gift, say something specific, close warmly.",
                    "Specificity is the ingredient that makes it read as genuine rather than generic.",
                    "One concrete detail about the gift does more than a paragraph of general gratitude.",
                    "Keeping it short also makes it far more likely they'll actually send it.",
                ],
                "Keep it short and specific. Name the gift, say one real thing about why you like it or how you'll use it, and close warmly. For example: thank you so much for the scarf, the color is perfect and I've already worn it twice. It was so thoughtful of you. That one specific detail is what makes it feel heartfelt.",
            ),
        ],
    },
    {
        "title": "Staying Focused While Studying",
        "steps": [
            (
                "They keep getting distracted while studying and want to fix it.",
                [
                    "Distraction here tends to have two sources: the phone within reach and sessions that run too long.",
                    "Advice built on willpower alone rarely survives contact with a real study session.",
                    "Removing the phone from the room beats just flipping it face down.",
                    "Out of reach is out of mind in a way that face-down simply isn't.",
                    "Working in shorter focused blocks with breaks makes the whole thing more sustainable.",
                    "A block of about twenty-five minutes feels far less daunting than an open-ended stretch.",
                ],
                "Try putting your phone in another room, not just face down, since being out of reach makes a real difference. Then work in shorter blocks, say twenty-five minutes, with a short break after each one. It feels less daunting than a long stretch, and you'll usually get more done.",
            ),
        ],
    },
    {
        "title": "First Day at a New Job",
        "steps": [
            (
                "They're anxious about their first day at a new job and want advice.",
                [
                    "First-day anxiety is mostly about the unknown rather than any real risk.",
                    "That means reassurance plus a couple of small, doable actions beats big-picture career advice.",
                    "Nobody expects a new person to know how everything works on day one.",
                    "Asking questions early reads as a good sign, not a weakness.",
                    "Learning a few names and staying friendly goes a long way.",
                    "It helps to remind them the lost feeling usually fades within a week or two.",
                ],
                "Go in expecting to learn, not to perform. Nobody expects you to know how everything works on day one, so it's completely fine to ask questions and write things down. Learn a few names, be friendly, and give yourself permission to feel a little lost at first. That settles within a week or two.",
            ),
        ],
    },
    {
        "title": "Being a Better Listener",
        "steps": [
            (
                "They want to become a better listener in conversations.",
                [
                    "The most common listening mistake is drafting your reply while the other person is still talking.",
                    "So the core shift is to stay with what they're actually saying.",
                    "A concrete habit works better here than a vague instruction to be present.",
                    "Pausing for a beat before replying creates just enough space to really hear them.",
                    "Reflecting back what you heard now and then shows you were genuinely following.",
                    "Resisting the urge to fix or one-up matters, since often people just want to feel heard.",
                ],
                "The biggest shift is to stop planning your response while they're still talking, and just follow what they're actually saying. A simple trick is to pause for a beat before you reply, and sometimes to reflect back what you heard. Resist the urge to immediately fix or one-up, since often people just want to feel heard.",
            ),
        ],
    },
    {
        "title": "Congratulating Someone",
        "steps": [
            (
                "A friend got a promotion and they want to say something more than just congrats.",
                [
                    "A bare congrats is perfectly fine but instantly forgettable.",
                    "What makes it land is naming something specific about them or their effort.",
                    "Pointing to the work they put in makes it feel personal rather than reflexive.",
                    "So the suggestion should be a warm line that mentions the effort you noticed.",
                    "That small bit of specificity is the whole difference between automatic and heartfelt.",
                ],
                "Add something specific so it doesn't feel automatic. Instead of just congratulations, try something like: this is so well deserved, I know how hard you've worked for it, I'm really happy for you. Naming that you saw the effort is what makes it feel genuine.",
            ),
        ],
    },
    {
        "title": "Breaking Down a Big Task",
        "steps": [
            (
                "They feel stuck on a big project and don't know where to start.",
                [
                    "Feeling stuck usually means the task is too vague and too large to hold in your head.",
                    "Planning the whole thing at once only makes that weight heavier.",
                    "The fix is to make the very first step small enough that it's almost silly not to do it.",
                    "So I should steer them toward one tiny, concrete starting action.",
                    "Something like opening the document and writing a rough outline counts.",
                    "Starting is the hard part, and one small step usually loosens the rest.",
                ],
                "When something feels too big, the trick is to shrink the first step until it's almost too easy to skip, not plan the whole thing. Pick one small, concrete action you could do in a few minutes, like opening the document and writing a rough outline. Starting is the hard part, and one small step usually loosens the rest.",
            ),
        ],
    },
    {
        "title": "Choosing a Movie for a Group",
        "steps": [
            (
                "A group with very different tastes can't agree on a movie.",
                [
                    "Trying to find one film everyone loves almost always fails with mixed tastes.",
                    "The realistic goal is a film nobody actively objects to.",
                    "Something broadly likable, a mix of comedy and adventure, tends to be the safe middle.",
                    "It also helps to offer a quick, fair method rather than only a suggestion.",
                    "A round where everyone vetoes one option narrows it down fast and feels even-handed.",
                ],
                "Aim for something nobody dislikes rather than everyone's favorite, since that's more achievable. A film with a bit of both comedy and adventure usually pleases a mixed group. If you're still stuck, have everyone veto one option and pick from what survives, that's quick and feels fair.",
            ),
        ],
    },
    {
        "title": "Asking for Help Politely",
        "steps": [
            (
                "They need to ask a busy coworker for help and don't want to impose.",
                [
                    "The worry is imposing, so the request should visibly respect the coworker's time.",
                    "Vague, open-ended asks are actually what feel like an imposition.",
                    "Being specific about what's needed and how long it takes makes it easy to weigh.",
                    "Acknowledging that they're busy up front signals you're not taking them for granted.",
                    "Offering a couple of times, rather than expecting them to drop everything, makes a yes easier.",
                    "Framing it so that no is an easy answer is what keeps it from feeling pushy.",
                ],
                "Make it easy to say yes or no. Acknowledge they're busy, be specific about exactly what you need and how long it'll take, and offer a couple of times rather than expecting them to drop everything. Something like: I know you're swamped, but when you get a spare ten minutes, could you take a look at this one thing? Being specific respects their time.",
            ),
        ],
    },
    {
        "title": "Encouraging a New Musician",
        "steps": [
            (
                "They just started learning an instrument and feel discouraged by slow progress.",
                [
                    "Early discouragement almost always comes from comparing where you are to where you want to be.",
                    "The useful reframe is that progress at this stage is invisible day to day but real over weeks.",
                    "So it's completely normal to feel stuck even while genuinely improving.",
                    "A little consistent practice beats occasional long, heroic sessions.",
                    "Recording yourself now and then gives concrete proof of how far you've come.",
                    "That evidence tends to be far more motivating than a vague sense of progress.",
                ],
                "Progress on an instrument is sneaky, it barely shows day to day but adds up a lot over a few weeks, so it's normal to feel like you're not improving even when you are. A little practice most days beats a long session once in a while. Try recording yourself now and again, it's often the clearest proof of how far you've come.",
            ),
        ],
    },
    {
        "title": "Keeping a Conversation Going",
        "steps": [
            (
                "They run out of things to say in conversations and want help.",
                [
                    "Conversations stall when the questions can be answered in a single word.",
                    "So the practical fix is to lean on open questions that invite more.",
                    "Following up on what the other person just said beats hunting for a brand-new topic.",
                    "If they mention a trip, asking about the best part keeps the thread alive.",
                    "Genuine curiosity quietly does most of the work and takes the pressure off you.",
                ],
                "The trick is to ask questions that can't be answered in one word, and to follow up on what they just said instead of scrambling for a new topic. If someone mentions a trip, ask what the best part was rather than moving on. Being a little curious takes the pressure off you to keep talking.",
            ),
        ],
    },
    {
        "title": "Giving Feedback Kindly",
        "steps": [
            (
                "They need to give a friend honest feedback without hurting them.",
                [
                    "The tension here is between honesty and kindness, and they aren't actually opposites.",
                    "Being specific and aiming at the work rather than the person keeps it both honest and kind.",
                    "A vague this isn't great helps no one and can sting more than a precise note.",
                    "Pointing to one concrete thing and how it could be stronger is far more useful.",
                    "Leading with what genuinely worked makes the suggestion easier to hear.",
                    "So specific and caring together beats soft and vague every time.",
                ],
                "Be honest but specific, and aim at the work, not the person. Instead of a vague this isn't great, point to one concrete thing and how it could be stronger. It also helps to lead with what genuinely worked before the suggestion. Kind and honest aren't opposites, specific and caring is usually both.",
            ),
        ],
    },
    {
        "title": "Unwinding After a Stressful Day",
        "steps": [
            (
                "They had a rough day and want a few ways to unwind that aren't just scrolling.",
                [
                    "They've already ruled out scrolling, so I should skip suggesting a screen entirely.",
                    "The better suggestions are small, low-effort things that actually reset you.",
                    "A bit of movement, like a short walk, shifts your head surprisingly fast.",
                    "Something with your hands, such as cooking or tidying one small space, restores a sense of order.",
                    "Even a few minutes outside or a warm shower can quietly take the edge off.",
                    "A short list is easier to act on than an exhaustive one when someone is already drained.",
                ],
                "A few things tend to reset you more than a screen does. A short walk, even around the block, changes your head surprisingly fast. Something with your hands, like cooking or tidying one small space, gives a sense of order. And even a few minutes outside or a warm shower can quietly take the edge off.",
            ),
        ],
    },
    {
        "title": "Affect Versus Effect",
        "steps": [
            (
                "They always mix up affect and effect and want an easy way to remember.",
                [
                    "A full grammar rule with all its exceptions would only overwhelm them.",
                    "A simple memory trick covers the everyday cases, which is really what they need.",
                    "Affect is usually the verb, the action, and effect is usually the noun, the thing.",
                    "A short mnemonic, affect is action and both start with A, makes that stick.",
                    "It's honest to note there are rarer uses, but they cover almost nothing they'll write.",
                ],
                "For everyday use, affect is usually the action, the verb, and effect is usually the thing, the noun. The rain affects your mood; the rain has an effect on your mood. A quick trick: affect is action, both start with A. There are rarer uses, but that covers almost everything you'll write.",
            ),
        ],
    },
    {
        "title": "Setting a Small Daily Habit",
        "steps": [
            (
                "They want to build a new daily habit but keep giving up after a few days.",
                [
                    "Habits usually collapse because they're too ambitious and anchored to nothing.",
                    "So the first fix is to make the habit almost embarrassingly small to begin with.",
                    "Attaching it to an existing daily action means it doesn't depend on motivation.",
                    "Doing it right after the morning coffee, for instance, gives it a reliable cue.",
                    "Tiny and anchored beats big and motivated, because willpower isn't in the loop.",
                    "A simple rule helps too: if you miss a day, just don't miss two in a row.",
                ],
                "Make it almost embarrassingly small to start, and attach it to something you already do, like doing it right after your morning coffee. Tiny and anchored beats big and motivated, because you're not relying on willpower. And if you miss a day, just don't miss two in a row, that's what actually keeps a habit alive.",
            ),
        ],
    },
    {
        "title": "An Overwhelming To-Do List",
        "steps": [
            (
                "Their to-do list feels endless and they don't know what to do first.",
                [
                    "When everything feels urgent, the real problem is that nothing has been prioritized.",
                    "Working faster won't help here; choosing will.",
                    "The unlock is naming the single task that would matter most if it were finished today.",
                    "Starting there, and letting the rest wait, shrinks the sense of chaos.",
                    "A long list feels far less scary once one clear priority is picked out of it.",
                ],
                "When the list feels endless, the answer usually isn't to work faster, it's to choose. Ask yourself which one thing, if you finished it today, would make the biggest difference, and start there. Let the rest sit. A long list is far less scary once you've picked the one thing that actually matters most right now.",
            ),
        ],
    },
    {
        "title": "Making a Good First Impression",
        "steps": [
            (
                "They're meeting someone important soon and want to make a good impression.",
                [
                    "People overthink this and try to impress, which tends to backfire.",
                    "The reassuring truth is that warmth and genuine attention beat a polished performance.",
                    "A couple of small, concrete behaviors help more than trying to seem impressive.",
                    "Showing up on time, remembering their name, and truly listening carry a lot.",
                    "People remember how you made them feel more than anything clever you said.",
                    "So the real advice is to relax and be present rather than perform.",
                ],
                "The best first impression usually comes from being warm and genuinely attentive, not from trying to impress. Show up on time, remember their name, and actually listen rather than waiting to talk. People remember how you made them feel far more than anything clever you said, so relax and be present.",
            ),
        ],
    },
    {
        "title": "What to Pack for a Short Trip",
        "steps": [
            (
                "They're going away for a couple of nights and always overpack.",
                [
                    "Overpacking comes from packing for imagined scenarios instead of the actual trip.",
                    "The fix is to plan around what they'll really do over those couple of nights.",
                    "A few versatile pieces that mix and match beat a bag full of just-in-case items.",
                    "Laying everything out and then removing about a third is a reliable trick.",
                    "And it helps to remember almost anything forgotten can be bought once they arrive.",
                    "A simple rule of thumb is easier to follow than an exhaustive packing list.",
                ],
                "Pack for what you'll actually do, not every possible what-if. For a couple of nights, a few versatile pieces you can mix and match beat a bag full of just-in-case items. Lay everything out, then take out about a third of it. And remember, almost anything you forget can be bought when you get there.",
            ),
        ],
    },
    {
        "title": "A Name for a New Pet",
        "steps": [
            (
                "They just got a puppy and can't settle on a name.",
                [
                    "Naming is personal, so handing over a random list isn't the most helpful thing.",
                    "It's better to give them a way to choose than to choose for them.",
                    "Names that are easy to call out across a park tend to work well, so two syllables often help.",
                    "Waiting a day or two lets the puppy's personality suggest something.",
                    "A name they won't tire of matters more than one that's merely clever.",
                    "Offering to spark a few ideas once they describe the dog keeps it collaborative.",
                ],
                "A good dog name is easy to say out loud and call across a park, so two syllables often work well. Beyond that, it helps to wait a day or two and let their personality suggest something. If you want a starting point, tell me what they look like or how they act, and I'll toss out a few ideas that fit.",
            ),
        ],
    },
    {
        "title": "Explaining Why the Sky Is Blue",
        "steps": [
            (
                "A child asked them why the sky is blue and they want a simple way to explain it.",
                [
                    "The audience is a child, so this has to be simple and picture-friendly, not a lecture.",
                    "The core idea is that sunlight looks white but is really all the colors mixed together.",
                    "As that light moves through the air, the blue part scatters and spreads across the sky the most.",
                    "That's why blue is the color we see almost everywhere we look up.",
                    "One or two friendly sentences are plenty; any more detail would just lose them.",
                ],
                "You can keep it simple: sunlight looks white but is really made of all the colors mixed together. As it comes through the air, the blue light bounces around and spreads across the whole sky more than the other colors, so that's the color we see everywhere we look up. For a child, that's plenty.",
            ),
        ],
    },
    {
        "title": "Planning a Relaxing Weekend",
        "steps": [
            (
                "They want to actually relax this weekend but usually end up busy or bored.",
                [
                    "The two failure modes they named are overbooking and drifting into doing nothing.",
                    "So the sweet spot is a loose plan with one or two nice things and lots of open space.",
                    "The advice should protect rest without scheduling it to death.",
                    "Picking one or two things they'd genuinely enjoy gives the weekend a gentle shape.",
                    "Leaving the rest deliberately open is what actually allows the resting to happen.",
                    "A little structure keeps boredom away while the open space keeps it restful.",
                ],
                "Aim for a loose plan rather than an empty calendar or a packed one. Pick one or two things you'd genuinely enjoy, maybe a walk somewhere new and a meal you like, and leave the rest open on purpose. A little structure keeps you from drifting into boredom, while the open space is what actually lets you rest.",
            ),
        ],
    },
    {
        "title": "Remembering People's Names",
        "steps": [
            (
                "They're bad at remembering names right after meeting someone.",
                [
                    "Names slip because we're barely paying attention in the moment of introduction.",
                    "Usually we're a bit self-conscious right then, half thinking about ourselves.",
                    "So the fix is to slow down and use the name once immediately.",
                    "Saying nice to meet you, Sarah anchors it and reads as warm at the same time.",
                    "Using it again in the next minute or two locks it in further.",
                    "A small, doable trick beats an elaborate memory-palace method here.",
                ],
                "Most of the time names don't stick because we're a little distracted the second we hear them. The simplest fix is to say the name back right away, like nice to meet you, Sarah, and then use it once more in the next minute or two. Repeating it early anchors it, and it comes across as warm too.",
            ),
        ],
    },
]


# --- Assembly ---------------------------------------------------------------
# These pools wrap the topic's reasoning beats into a longer, natural-sounding
# monologue. All entries are standalone sentences, so they slot in anywhere
# without breaking grammar or capitalization.

OPENERS = [
    "Okay, let me make sure I understand what they're really asking.",
    "Let me think about what would genuinely help here.",
    "Right, let me work out the most useful way to answer this.",
    "Let me slow down and think this through before I reply.",
    "Let me think about what they actually need, not just what they literally said.",
    "Alright, let me reason through this properly.",
    "Let me take a moment and think about the best way to help.",
]

# A reflective beat dropped in around the middle of the chain, to make the
# reasoning feel like it's weighing itself as it goes.
MIDDLES = [
    "Let me weigh that against the alternative for a second.",
    "But I should be careful not to overdo it.",
    "Still, I want to make sure this really fits their situation.",
    "Let me not lose sight of what they actually asked for.",
    "It's worth pausing on that, because it's the crux of it.",
    "Let me hold that thought and see where it leads.",
    "Okay, that line of thinking feels right, let me keep going.",
]

# A self-check near the end, before committing to the reply.
CHECKS = [
    "Let me make sure this actually answers what they asked and doesn't wander into things they didn't.",
    "I want to keep this clear and genuinely useful, so let me check the answer is easy to follow.",
    "Before I reply, let me see whether there's a simpler, more direct way to put it.",
    "Let me make sure I'm not overcomplicating this; the clearest answer is usually the most helpful one.",
    "I should check that this stays balanced and doesn't push them one way without good reason.",
    "Let me double-check the tone feels warm rather than preachy.",
    "Let me make sure I'm meeting them where they are and not lecturing.",
]

LEADINS = [
    "So here's how I'd put it.",
    "That gives me a clear way to respond.",
    "Alright, I think I've got the right answer for them.",
    "Good, I'll keep the reply simple and direct.",
    "Okay, I think that's the most helpful angle.",
    "Right, that's what I'll say.",
]


def expand_thinking(rng, ask: str, beats: list[str], reply: str) -> list[tuple[str, str]]:
    """Weave one (ask, beats, reply) turn into a long inner-monologue block and a reply."""
    parts = [f"{rng.choice(OPENERS)} {ask}"]

    # First half of the reasoning.
    half = (len(beats) + 1) // 2
    parts.extend(beats[:half])

    # A mid-chain reflection, then the rest of the reasoning.
    if len(beats) >= 3 and rng.random() < 0.7:
        parts.append(rng.choice(MIDDLES))
    parts.extend(beats[half:])

    # A self-check, then the hand-off into the reply.
    if rng.random() < 0.6:
        parts.append(rng.choice(CHECKS))
    parts.append(rng.choice(LEADINS))

    return [(" ".join(parts), reply)]


def format_file(title: str, thought_pairs: list[tuple[str, str]]) -> str:
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
        for ask, beats, reply in topic["steps"]:
            all_pairs.extend(expand_thinking(rng, ask, beats, reply))
        content = format_file(topic["title"], all_pairs)
        fname = f"synthetic{idx:04d}.txt"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {args.count} files -> {args.out}")


if __name__ == "__main__":
    main()
