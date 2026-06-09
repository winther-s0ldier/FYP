"""
Synthetic Data Generator v2 — 2026 Edition
Generates intent-labelled utterances + multi-turn session sequences for
content moderation training. Grounded in 2025-2026 online harm research.

Platforms covered: Discord, Instagram, WhatsApp, Snapchat, TikTok,
                   Twitter/X, Roblox, gaming, Reddit
Terminology: updated to 2025-2026 slang and coded language

Run:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --intents-only
    python scripts/generate_synthetic_data.py --sessions-only
    python scripts/generate_synthetic_data.py --n 300   # utterances per intent
"""
import os
import json
import time
import csv
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

try:
    from openai import OpenAI as _OpenAI
    _openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_client = _OpenAI(api_key=_openai_key) if _openai_key else None
except Exception as e:
    openai_client = None
    print(f"[WARN] OpenAI client init failed: {e}")

try:
    from groq import Groq
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
except Exception as e:
    groq_client = None
    print(f"[WARN] Groq client init failed: {e}")

if openai_client is None and groq_client is None:
    raise RuntimeError("No API client available. Set OPENAI_API_KEY or GROQ_API_KEY in .env")

SYNTH_DIR  = Path("data/raw/synthetic_intents")
SESSION_DIR = Path("data/raw/session_data")
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Platform contexts (injected into prompts for diversity)
# ===========================================================================
PLATFORMS = {
    "discord":   "Discord server / DM (gaming, community, study server)",
    "instagram": "Instagram DMs, story replies, comment section",
    "whatsapp":  "WhatsApp group chat or personal DM",
    "snapchat":  "Snapchat DMs, streak messages, group snaps",
    "tiktok":    "TikTok comment section, DM, duet/stitch response",
    "twitter":   "Twitter/X reply, quote tweet, thread",
    "roblox":    "Roblox chat, in-game messages (often younger users)",
    "gaming":    "Online gaming voice/text chat (COD, Valorant, FIFA, Minecraft)",
    "reddit":    "Reddit comment section, DM, subreddit post",
    "telegram":  "Telegram group chat, channel or DM",
}

# ===========================================================================
# 2025-2026 Slang / Coded Language Reference (injected into prompts)
# ===========================================================================
SLANG_2026 = """
Current 2025-2026 online slang and coded language to use naturally:
- Positive/neutral: "no cap", "fr fr", "lowkey/highkey", "bussin", "it's giving",
  "slay", "ate that", "rizz", "W/L" (win/loss), "based", "understood the assignment",
  "main character", "delulu", "era", "locked in", "touch grass", "NPC", "gyatt"
- Dismissive/toxic: "ratio", "L + ratio + fell off", "cooked", "mid", "cope",
  "you're so cooked fr", "down bad", "beta", "sigma male grindset" (ironic/serious),
  "red-pilled", "take the L", "unalived", "ratio'd", "clip this", "caught in 4K",
  "mass report incoming", "no bitches", "touch grass", "skill issue", "you fell off"
- Coded hate (dog whistles): subtle framing like "just asking questions",
  "I'm just being real", "some people just can't handle the truth",
  "not all [group] but", "I'm not racist but", "it's just facts though"
- Threat-adjacent coded language: "it would be a shame if...", "imagine if someone",
  "your IP is showing", "I know where you posted from", "delete this or else"
"""

# ===========================================================================
# Intent Definitions — ALL 21 intents (including 4 previously missing)
# ===========================================================================
INTENT_CONFIG = {
    # ── Neutral / Benign ────────────────────────────────────────────────────
    "greeting": {
        "toxic": 0,
        "description": "Initial greeting or welcome message to open a conversation.",
        "platforms": ["discord", "instagram", "whatsapp", "gaming", "roblox"],
        "seeds": [
            "hey what's up everyone", "yo anyone around?", "hi all just joined",
            "good morning folks", "what's good", "sup", "hey hey hey",
            "hola amigos", "anyone alive in here?", "hiya", "wassup",
            "g'day", "oi oi", "hello hello", "hey peeps",
            "morning ☀️", "just dropped in, how's everyone?", "hi hi hi",
            "ayy who's on rn", "yooo it's been a minute",
        ]
    },
    "small_talk": {
        "toxic": 0,
        "description": "Casual light conversation about everyday topics. No clear information goal.",
        "platforms": ["discord", "instagram", "whatsapp", "snapchat", "twitter"],
        "seeds": [
            "what did you do this weekend?", "anyone watching the game tonight?",
            "it's so cold outside today", "I'm bored someone entertain me lol",
            "can't believe it's already this late in the year",
            "who else is procrastinating rn", "this day is going so fast",
            "I just had the best pizza ever", "anyone else's wifi being slow?",
            "ugh Mondays amirite", "need coffee badly", "anyone else just chilling",
            "bro the weather today is unreal", "what's everyone been watching lately",
            "okay so random but what's your hot take on pineapple pizza",
            "this song has been stuck in my head all day",
            "my cat just knocked everything off my desk 💀",
            "anyone else get zero sleep last night or just me",
        ]
    },
    "question": {
        "toxic": 0,
        "description": "Genuine question seeking information or clarification.",
        "platforms": ["discord", "reddit", "whatsapp", "twitter", "instagram"],
        "seeds": [
            "can someone explain how this works?", "what time does it start?",
            "does anyone know a good app for this?", "how do you do that glitch?",
            "wait what happened while I was offline?", "anyone know the answer to this?",
            "is this server 18+ or?", "what's the best way to grind XP here?",
            "do I need to verify anything to post?", "what does that mean?",
            "can someone help me with this setting?", "is this still happening?",
            "what's the context I'm confused", "anyone tried this before?",
            "when does the next update drop?", "which one should I pick?",
            "is it just me or is the app glitching?", "how long does it usually take?",
            "who's the mod I need to ask something", "any tips for a beginner?",
        ]
    },
    "information_sharing": {
        "toxic": 0,
        "description": "Sharing facts, news, tips, links or useful information with others.",
        "platforms": ["discord", "whatsapp", "reddit", "twitter", "telegram"],
        "seeds": [
            "just saw they dropped a new update, patch notes here:",
            "FYI the event ends tomorrow midnight", "quick tip: you can skip that part by",
            "found a really good tutorial for this", "heads up the server moves to",
            "apparently the new season starts next week",
            "sharing this because people need to know",
            "for anyone who missed it, here's what happened",
            "pro tip: if you do X before Y it saves a lot of time",
            "news just broke about the update", "thought this was interesting",
            "leaving this here for anyone who needs it",
            "guide I wrote for new members:", "quick PSA about the new rules",
            "if you're having that bug try this fix it worked for me",
            "dropping the link in case anyone needs it",
            "research came out today that's actually wild",
            "reminder that the deadline is",
        ]
    },
    "feedback": {
        "toxic": 0,
        "description": "Giving or requesting constructive feedback, reviews or opinions.",
        "platforms": ["discord", "reddit", "instagram", "twitter", "gaming"],
        "seeds": [
            "honest feedback: I think you should try",
            "this was really well done but maybe",
            "ngl it could use some work on the",
            "I think the idea is solid but the execution needs",
            "loved it overall, just one thing though",
            "not my style but I can see what you were going for",
            "10/10 honestly no notes", "could be better if you",
            "I think you're on the right track but",
            "the beginning was strong but it kind of lost me at",
            "genuinely impressed by how much you've improved",
            "okay so here's my honest take:", "keep doing what you're doing",
            "the concept is there just needs a bit of polish",
            "solid effort, main thing I'd change is",
            "this actually slapped ngl", "not bad not bad at all",
            "I'd rate this a 7/10 mainly because",
        ]
    },
    "solidarity_seeking": {
        "toxic": 0,
        "description": "Looking for validation, shared experience or emotional support from others.",
        "platforms": ["discord", "instagram", "reddit", "twitter", "whatsapp"],
        "seeds": [
            "anyone else feel like they don't belong anywhere?",
            "tell me I'm not the only one who does this",
            "I feel like nobody gets it but me",
            "who else has been feeling off lately?",
            "is it just me or does everyone feel this way sometimes",
            "I just need someone to tell me it gets better",
            "am I the only one who's struggling with this",
            "this community is the only place I feel understood",
            "does anyone else go through this",
            "I just want to know other people feel the same",
            "not looking for advice just someone to relate to",
            "tell me I'm not crazy for feeling like this",
            "I've been feeling really alone in this lately",
            "anyone who's been through this, how did you cope",
            "I don't want solutions I just want to feel less alone",
            "y'all are the only ones who get it fr",
            "solidarity moment: who else can't adult today",
        ]
    },
    "venting": {
        "toxic": 0,
        "description": "Expressing frustration or stress to release emotion, not to attack anyone.",
        "platforms": ["discord", "instagram", "twitter", "whatsapp", "reddit"],
        "seeds": [
            "I'm so done with today honestly", "why does everything have to be so hard",
            "my boss is absolutely the worst I swear", "I just failed my exam I want to cry",
            "can today just end please", "I hate this game so much right now",
            "this team is making me lose my mind", "I've had the worst week of my life",
            "I can't deal with people anymore", "everything is piling up and I can't breathe",
            "I'm exhausted and nobody cares", "losing my mind with this connection",
            "why am I even trying anymore", "just absolutely done rn",
            "TikTok keeps showing me things that stress me out and I can't stop scrolling",
            "my snap streak broke and I'm actually upset about it which is embarrassing",
            "I've been staring at this screen for 6 hours and got nothing done",
            "the algorithm is recommending me the most depressing content today",
        ]
    },
    "humour": {
        "toxic": 0,
        "description": "Jokes, memes, banter, lighthearted content clearly intended to amuse. Gen Z / gaming slang.",
        "platforms": ["discord", "tiktok", "twitter", "instagram", "gaming"],
        "seeds": [
            "💀💀 I'm actually dead", "bro that's so bad it's good lol",
            "I can't breathe 😭😭", "this is sending me", "no but why is this so accurate",
            "ratio + L + you fell off", "not me cackling at 2am", "bro said what 💀",
            "it's giving something", "no because this is actually me fr fr", "skill issue lmao",
            "certified hood classic", "the way I wheezed", "understood the assignment",
            "POV: you're me every Monday", "the delulu to reality pipeline is real",
            "NPC behaviour detected 💀", "main character moment right here",
            "my rizz left the chat", "this is peak brain rot content and I love it",
            "I ate and left no crumbs", "the prophecy has been fulfilled",
            "hold on let me put on my villain arc music",
        ]
    },

    # ── Passive Aggression ──────────────────────────────────────────────────
    "baiting": {
        "toxic": 1,
        "description": "Deliberately provocative messages designed to trigger emotional reaction or conflict. 2026: often via ratio/quote tweet tactics.",
        "platforms": ["twitter", "tiktok", "discord", "gaming", "instagram"],
        "seeds": [
            "prove it then or you're just all talk", "bet you won't say that to my face",
            "go ahead and report me see what happens", "come on say what you really think",
            "you're not gonna do anything about it", "I dare you to try",
            "what are you gonna do cry about it?", "keep talking I love the entertainment",
            "ok so you're actually scared then", "say that again I'm listening",
            "you won't do it", "all talk and no action as usual", "I'll wait",
            "ratio this then 🙃", "I'm quote tweeting this just watch",
            "go ahead and block me I'll be back", "duet me I dare you",
            "keep posting I'm screenshotting everything", "clip it then",
            "let's see you back that up", "cry about it some more",
        ]
    },
    "sarcasm": {
        "toxic": 0,
        "description": "Saying the opposite of what is meant to mock or show contempt. Common evasion tactic across all platforms.",
        "platforms": ["twitter", "instagram", "tiktok", "discord", "reddit"],
        "seeds": [
            "oh wow great job truly impressive", "yeah because that's totally going to work",
            "oh suuure definitely believe you", "wow what a big brain play",
            "oh yeah that was super helpful thanks", "right because you know everything",
            "oh brilliant another genius speaks", "yeah good luck with that",
            "ten out of ten no notes", "absolutely groundbreaking observation",
            "wow never thought of that before", "yes because clearly you're the expert here",
            "oh wow you really understood the assignment 🙄",
            "such a hot take truly never heard that before",
            "wow the delusion jumped out", "absolutely ate that. zero crumbs. groundbreaking.",
            "yeah that's definitely going to work out for you bestie 💀",
            "no cap this is the worst take I've seen today and I've been on Twitter",
            "slay queen of being completely wrong",
        ]
    },
    "passive_hostility": {
        "toxic": 1,
        "description": "Hostility expressed indirectly through dismissiveness, cold withdrawal or pointed remarks.",
        "platforms": ["instagram", "whatsapp", "discord", "snapchat", "twitter"],
        "seeds": [
            "whatever I don't even care anymore", "must be nice to be so perfect",
            "it's fine not like you'd understand anyway", "sure if you say so",
            "nevermind forget I said anything", "some people just don't get it",
            "sure keep telling yourself that", "I guess I'm just the only one who cares",
            "don't mind me", "oh no it's fine I'm used to it", "typical", "as expected from you",
            "it's whatever, don't worry about it 🙂", "I'm fine",
            "cool that you think that 🙂", "no worries at all 🙂",
            "I left you on delivered for a reason",
            "my story views tell me everything I need to know",
            "posted about it. you'd know if you actually paid attention",
            "I removed you from my close friends. you wouldn't notice anyway",
            "I'm not going to explain myself to someone who clearly doesn't care",
        ]
    },
    "irony": {
        "toxic": 0,
        "description": "Pointing out contradiction by saying the opposite. Softer than sarcasm, no direct personal target.",
        "platforms": ["twitter", "reddit", "discord", "tiktok"],
        "seeds": [
            "right because that makes total sense", "oh cool so now I'm the bad guy",
            "yeah I'm definitely the problem here", "sure let's all pretend that didn't happen",
            "oh I see so the rules just don't apply to everyone equally",
            "how convenient that worked out that way", "typical, works every time",
            "interesting timing on that one", "of course that's what happened",
            "what a surprise, truly shocking", "because that's obviously how things work",
            "naturally, of course", "oh so NOW the rules matter",
            "funny how that works when it's convenient",
            "love how standards only apply to some people",
            "interesting how the narrative shifts when",
            "wild how that keeps happening by coincidence",
        ]
    },

    # ── Active Aggression ───────────────────────────────────────────────────
    "direct_attack": {
        "toxic": 1,
        "description": "Explicit insults, verbal attacks or dehumanising language at a specific person.",
        "platforms": ["gaming", "discord", "twitter", "tiktok", "instagram"],
        "seeds": [
            "you're absolute trash at this game", "shut up nobody wants to hear from you",
            "you're genuinely the worst player I've ever seen",
            "get out of this server you ruin everything",
            "you are so incredibly stupid", "nobody likes you here",
            "you're pathetic plain and simple", "uninstall and never come back",
            "you don't belong here", "honestly embarrassing that you exist",
            "you have no idea what you're doing", "you're a waste of a slot",
            "delete your account genuinely", "ratio'd and you deserved it",
            "you are actually cooked beyond saving fr",
            "the audacity to post that with your face showing 💀",
            "you're giving NPC with no redeeming qualities",
            "mid player mid person mid existence",
            "I've never disliked someone this fast",
            "the fact that you think you're good is actually hilarious",
        ]
    },
    "threatening": {
        "toxic": 1,
        "description": "Explicit or implied threats of harm, retaliation, doxxing or account action. 2026: includes mass-report threats, IP threats, deepfake threats.",
        "platforms": ["discord", "twitter", "instagram", "gaming", "tiktok"],
        "seeds": [
            "you better watch yourself", "keep talking and see what happens",
            "you won't be laughing for long", "I'm going to find your account",
            "you have no idea what I'm capable of", "this isn't over not even close",
            "I know more about you than you think", "enjoy it while it lasts",
            "tick tock", "you've been warned", "last chance to delete that",
            "your IP is literally showing rn just so you know",
            "I have your snap location enabled 🙂",
            "mass report in 3... 2...",
            "I know people who know people. delete this.",
            "it would be a shame if this got screenshotted and sent around",
            "imagine if someone made a deepfake of you. hypothetically.",
            "found your other account. nice try.",
            "I'm going to make sure everyone in this server knows what you did",
            "clip is already saved. think about your next move carefully.",
        ]
    },
    "escalating_aggression": {
        "toxic": 1,
        "description": "Increasing intensity of anger and hostility showing loss of control.",
        "platforms": ["gaming", "discord", "twitter", "whatsapp"],
        "seeds": [
            "I SAID SHUT UP", "I'm done talking this is your last warning",
            "you want to go then let's actually go", "ENOUGH I'm not playing anymore",
            "do NOT push me right now", "you think this is a joke IT IS NOT",
            "I am DONE being nice about this", "you crossed the line NOW",
            "that is IT", "I WILL NOT BE IGNORED",
            "you have NO idea who you're messing with", "THIS ENDS NOW",
            "you just made the biggest mistake", "I AM DONE",
            "BROOOO I SAID ENOUGH", "STOP TALKING I SWEAR",
            "this is actually it I'm not joking anymore",
            "everyone in this group see what this person is doing RIGHT NOW",
            "I'm screenshotting EVERYTHING and sending it everywhere",
            "I have been SO patient and you have used it all up",
        ]
    },

    # ── Manipulation ─────────────────────────────────────────────────────────
    "gaslighting": {
        "toxic": 1,
        "description": "Manipulative denial of reality making someone question their own perception. Common in DMs across all platforms.",
        "platforms": ["instagram", "whatsapp", "snapchat", "discord", "tiktok"],
        "seeds": [
            "that never happened you're making it up",
            "you're way too sensitive it was just a joke",
            "I never said that you're remembering it wrong",
            "everyone agrees with me you're the only one with a problem",
            "you're always overreacting to everything", "you're imagining things again",
            "that's not what I said and you know it",
            "you're the one who started this not me",
            "everyone else thinks you're being dramatic",
            "I literally did nothing wrong and you know that",
            "you twist everything I say", "why do you always do this",
            "you're too emotional to think clearly right now",
            "I've never treated you badly and you know it",
            "you need to stop creating problems that don't exist",
            "I said that as a joke and you took it completely out of context",
            "show me the receipts because I don't remember it that way at all",
            "my friends have all seen the screenshots and they agree with me",
            "you always do this when you don't get your way",
            "I'm the one who should be upset right now, not you",
        ]
    },
    "grooming": {
        "toxic": 1,
        "description": "Trust-building and special connection to exploit or isolate. 2025-2026: common on Roblox, Snapchat streaks, Discord DMs, Instagram. 6-stage model.",
        "platforms": ["roblox", "snapchat", "discord", "instagram", "whatsapp"],
        "seeds": [
            "you're so much more mature than other people your age",
            "I feel like you really get me not like the others",
            "this can be our little secret okay",
            "you're special I don't talk to everyone like this",
            "I just feel like we have such a deep connection",
            "you can tell me anything I won't judge you",
            "nobody else understands you like I do",
            "why don't we move this conversation somewhere more private",
            "you're not like other people in this server you're different",
            "your parents just don't understand you the way I do",
            "we should keep what we talk about between us",
            "I want to help you I just need you to trust me first",
            "let's move to snap so we can talk more freely",
            "I'll protect you from all the drama in this server",
            "I don't have streaks with just anyone, you're special",
            "you remind me of myself at your age, I just want to look out for you",
            "don't tell anyone about our chats, they wouldn't understand",
            "I can see your location on snap just checking you're safe",
            "I've sent you a friend request on my real account. don't share it",
            "you're the only person in my close friends list that matters",
        ]
    },
    "social_engineering": {
        "toxic": 1,
        "description": "False authority, urgency, scarcity or social proof to manipulate actions. 2026: includes WhatsApp forwarded scams, fake giveaways, off-platforming tactics.",
        "platforms": ["whatsapp", "instagram", "discord", "tiktok", "snapchat"],
        "seeds": [
            "your friend told me you'd be cool with this",
            "everyone in the group is doing it you're the only one left out",
            "if you really trusted me you'd share that",
            "the mods said it's fine they just don't announce it publicly",
            "I'm actually an admin I just use this account",
            "everyone who matters has already moved to the other server",
            "your account will get flagged unless you verify here",
            "all the real ones moved to this platform last week",
            "they're giving free perks to everyone who joins before midnight",
            "URGENT: forward this to 10 people or your account gets deleted",
            "free iPhone giveaway, just send your details to verify",
            "the TikTok creator fund is ending for your account, verify here",
            "I'm doing a collab with verified creators, you should join",
            "we're moving the real server to Telegram, Discord banned the owner for no reason",
            "I need your Snap to send you the invite, it's not public",
            "the link expires in 10 minutes so click now",
            "I wouldn't tell you this if I didn't trust you completely",
            "WhatsApp is going to start charging unless you forward this",
            "this is from the official team, just not announced yet",
            "the verified badge is being given to select accounts this week only",
        ]
    },

    # ── Evasion ─────────────────────────────────────────────────────────────
    "topic_deflection": {
        "toxic": 0,
        "description": "Redirecting conversation away from uncomfortable subjects.",
        "platforms": ["discord", "whatsapp", "instagram", "twitter"],
        "seeds": [
            "can we just not talk about this right now", "why are we even bringing this up",
            "anyway moving on", "this is so not worth discussing",
            "let's talk about something else", "I don't see why this matters",
            "can we change the subject please", "this conversation is going nowhere",
            "we're getting way off topic", "let's not go there",
            "this isn't what I came here to talk about", "not doing this today",
            "I'm not having this conversation again",
            "okay but actually let's talk about something else",
            "you're changing the subject and I see what you're doing",
            "anyway this is mid let's move on",
            "different topic anyone?", "I didn't come here for this",
            "we're going in circles, dropping it",
        ]
    },
    "denial": {
        "toxic": 0,
        "description": "Flat refusal to acknowledge actions, statements or involvement.",
        "platforms": ["discord", "whatsapp", "instagram", "gaming"],
        "seeds": [
            "I never said that", "that wasn't me", "I don't know what you're talking about",
            "I didn't do anything wrong", "I have no idea what you mean",
            "not me must be someone else", "I was never part of that",
            "you have the wrong person", "I didn't say any of that",
            "I've never even been in that server", "prove it",
            "I literally just got here", "you're confused about what happened",
            "check the logs it wasn't me", "false accusation simple as that",
            "I didn't post that, my account was hacked",
            "screenshot is edited anyone can see that",
            "I don't have a second account I don't know what you mean",
            "I was never in that group chat",
            "I didn't even open that snap",
        ]
    },
    "identity_concealment": {
        "toxic": 1,
        "description": "Hiding true identity, motives or affiliation. 2026: alt accounts, AI voice, VPN masking, fake verification.",
        "platforms": ["discord", "instagram", "twitter", "tiktok", "roblox"],
        "seeds": [
            "I'm just asking questions", "it's not what it looks like",
            "I'm just a regular person like you", "I'm not who you think I am",
            "this is just my alt I'm actually pretty well known here",
            "I'm only here to observe not participate",
            "I don't really have a side in this I'm neutral",
            "I'm just here to help out the community",
            "my main account got banned for absolutely no reason",
            "you wouldn't know my main but I've been here for years",
            "I'm doing research that's why I'm asking",
            "I work with the server owner we're just not public about it",
            "don't worry about who I am just listen to what I'm saying",
            "this is my finsta so don't go looking for my main",
            "I use a VPN so my location shows somewhere else don't worry",
            "I have multiple accounts for different communities, totally normal",
            "this account is just for this community my main is private",
            "I'm not affiliated with that group anymore, totally unrelated now",
            "the mods know who I really am, it's fine",
        ]
    },
}

# ===========================================================================
# INTENT_TO_TOXIC mapping
# ===========================================================================
INTENT_TO_TOXIC = {
    "greeting": 0, "small_talk": 0, "question": 0, "information_sharing": 0,
    "feedback": 0, "solidarity_seeking": 0, "venting": 0, "humour": 0,
    "baiting": 1, "sarcasm": 0, "passive_hostility": 1, "irony": 0,
    "direct_attack": 1, "threatening": 1, "escalating_aggression": 1,
    "gaslighting": 1, "grooming": 1, "social_engineering": 1,
    "topic_deflection": 0, "denial": 0, "identity_concealment": 1,
}

# ===========================================================================
# Session Archetypes — 2026 Edition
# New: Instagram story wars, WhatsApp group manipulation,
#      Snapchat grooming pipeline, TikTok pile-on, cross-platform migration,
#      coded radicalisation
# ===========================================================================
SESSION_ARCHETYPES = [
    # ── Existing archetypes (updated) ──────────────────────────────────────
    {
        "name": "gradual_escalation_gaming",
        "platform": "gaming / Discord",
        "description": "Classic gaming toxicity arc: friendly start escalating to direct attack then threats.",
        "flow": ["greeting", "small_talk", "humour", "baiting", "sarcasm",
                 "passive_hostility", "direct_attack", "threatening"],
        "count": 15
    },
    {
        "name": "grooming_snapchat_streak",
        "platform": "Snapchat",
        "description": "Grooming via Snapchat streak: streak maintenance as trust mechanism → isolation → identity concealment. Research: Frontiers Pediatrics 2025.",
        "flow": ["greeting", "small_talk", "solidarity_seeking", "humour",
                 "grooming", "grooming", "social_engineering", "identity_concealment"],
        "count": 12
    },
    {
        "name": "radicalization_pipeline_reddit",
        "platform": "Reddit / Discord",
        "description": "Radicalisation entry: grievance → solidarity → us-vs-them → dehumanisation. HSToday 2026.",
        "flow": ["venting", "solidarity_seeking", "information_sharing",
                 "baiting", "direct_attack", "escalating_aggression"],
        "count": 10
    },
    {
        "name": "benign_whatsapp_group",
        "platform": "WhatsApp group",
        "description": "Normal friendly WhatsApp group conversation. Fully benign.",
        "flow": ["greeting", "small_talk", "question", "information_sharing",
                 "humour", "feedback", "small_talk", "solidarity_seeking"],
        "count": 20
    },
    {
        "name": "explosive_discord_dm",
        "platform": "Discord DM",
        "description": "Sudden unprovoked aggression in Discord DM that rapidly escalates.",
        "flow": ["small_talk", "direct_attack", "escalating_aggression",
                 "threatening", "denial", "topic_deflection"],
        "count": 10
    },
    {
        "name": "passive_aggressive_instagram",
        "platform": "Instagram",
        "description": "Sustained indirect hostility via Instagram DMs and story replies.",
        "flow": ["greeting", "small_talk", "irony", "sarcasm",
                 "passive_hostility", "denial", "topic_deflection"],
        "count": 12
    },
    {
        "name": "venting_and_support",
        "platform": "Discord / Reddit",
        "description": "Emotional venting met with solidarity and support — fully benign session.",
        "flow": ["venting", "solidarity_seeking", "question",
                 "information_sharing", "feedback", "humour"],
        "count": 15
    },
    {
        "name": "gaslighting_whatsapp_dm",
        "platform": "WhatsApp DM",
        "description": "Reality distortion and denial cycle in WhatsApp DM. Security gaslighting pattern.",
        "flow": ["small_talk", "information_sharing", "question",
                 "denial", "gaslighting", "topic_deflection", "gaslighting"],
        "count": 12
    },
    {
        "name": "threat_and_retreat_gaming",
        "platform": "Gaming / Discord",
        "description": "Threat followed by identity concealment and denial.",
        "flow": ["baiting", "direct_attack", "threatening",
                 "escalating_aggression", "denial", "identity_concealment"],
        "count": 10
    },

    # ── New 2026 archetypes ─────────────────────────────────────────────────
    {
        "name": "instagram_story_harassment",
        "platform": "Instagram",
        "description": "Instagram story-based harassment: innocent post → anonymous replies → escalating DMs. Very common 2025-2026 pattern.",
        "flow": ["small_talk", "baiting", "sarcasm", "passive_hostility",
                 "direct_attack", "identity_concealment"],
        "count": 10
    },
    {
        "name": "tiktok_comment_pile_on",
        "platform": "TikTok",
        "description": "TikTok comment section pile-on: ratio culture → mass baiting → direct attacks. Accelerated by duet/stitch feature.",
        "flow": ["humour", "irony", "sarcasm", "baiting",
                 "direct_attack", "escalating_aggression", "threatening"],
        "count": 10
    },
    {
        "name": "whatsapp_social_engineering",
        "platform": "WhatsApp group",
        "description": "WhatsApp group social engineering: trust building → false authority → off-platforming. Common 2025-2026 scam vector.",
        "flow": ["greeting", "small_talk", "solidarity_seeking",
                 "information_sharing", "social_engineering", "identity_concealment"],
        "count": 10
    },
    {
        "name": "roblox_grooming_minor",
        "platform": "Roblox",
        "description": "Roblox grooming targeting younger users: in-game friendship → gift offers → private contact. UK Online Safety Act 2023 research.",
        "flow": ["greeting", "humour", "small_talk", "solidarity_seeking",
                 "grooming", "social_engineering", "identity_concealment"],
        "count": 12
    },
    {
        "name": "cross_platform_migration",
        "platform": "Discord → Telegram",
        "description": "2025-2026 off-platforming: moving from moderated to unmonitored platform. Common in extremist and exploit communities.",
        "flow": ["greeting", "solidarity_seeking", "information_sharing",
                 "social_engineering", "identity_concealment", "grooming"],
        "count": 8
    },
    {
        "name": "coded_misogyny_incel",
        "platform": "Reddit / Discord",
        "description": "Coded misogyny using sigma/alpha/red-pill language. Increasingly uses plausible deniability. 2026 research: Moonshot CVE.",
        "flow": ["venting", "solidarity_seeking", "baiting",
                 "passive_hostility", "direct_attack", "identity_concealment"],
        "count": 10
    },
    {
        "name": "benign_study_discord",
        "platform": "Discord study server",
        "description": "Normal study Discord server: questions, help, feedback, encouragement. Fully benign.",
        "flow": ["greeting", "question", "information_sharing", "feedback",
                 "solidarity_seeking", "small_talk", "humour"],
        "count": 15
    },
    {
        "name": "snapchat_streak_manipulation",
        "platform": "Snapchat",
        "description": "Using streak anxiety and FOMO to maintain access and control. Precursor to grooming.",
        "flow": ["greeting", "small_talk", "solidarity_seeking",
                 "passive_hostility", "gaslighting", "social_engineering"],
        "count": 10
    },
    {
        "name": "benign_gaming_friends",
        "platform": "Gaming",
        "description": "Normal gaming session between friends. Banter, no real toxicity.",
        "flow": ["greeting", "humour", "small_talk", "feedback",
                 "question", "information_sharing", "humour"],
        "count": 15
    },
]

# ===========================================================================
# LLM API — OpenAI primary, Groq fallback
# ===========================================================================
OPENAI_MODEL = "gpt-4o-mini"   # fast, cheap, excellent JSON adherence
GROQ_MODEL   = "llama-3.3-70b-versatile"


def _call_openai(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.88,
                max_tokens=3000,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"  [OpenAI] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [OpenAI] Error (attempt {attempt+1}): {e}")
                time.sleep(3)
    return ""


def _call_groq(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.88,
                max_tokens=3000,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 60 * (attempt + 1)
                print(f"  [Groq] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [Groq] Error (attempt {attempt+1}): {e}")
                time.sleep(5)
    return ""


def call_groq(prompt: str, retries: int = 3) -> str:
    """Try OpenAI first; fall back to Groq if unavailable or rate-limited."""
    if openai_client is not None:
        result = _call_openai(prompt, retries)
        if result:
            return result
        print("  [OpenAI] exhausted, falling back to Groq...")
    if groq_client is not None:
        return _call_groq(prompt, retries)
    return ""


# ===========================================================================
# Utterance Generation
# ===========================================================================
def generate_intent_utterances(intent: str, n: int = 400) -> list[dict]:
    cfg = INTENT_CONFIG[intent]
    toxic_label = cfg["toxic"]
    platforms = [PLATFORMS[p] for p in cfg["platforms"]]
    platform_str = " | ".join(platforms)
    seeds_str = "\n".join(f"  - {s}" for s in cfg["seeds"])

    # Split into batches of 100 for diversity
    all_examples = []
    batch_size = 100
    n_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        remaining = min(batch_size, n - len(all_examples))
        # Rotate platform focus per batch for diversity
        platform_focus = platforms[batch_idx % len(platforms)]

        implicit_instruction = ""
        if toxic_label == 1:
            implicit_instruction = f"""
- Mix severity: {int(remaining*0.4)} should be mild/subtle (plausible deniability),
  {int(remaining*0.3)} should be moderate, {int(remaining*0.3)} should be direct/severe
- Include coded language, dog whistles and indirect expressions where realistic
- Some should use 2025-2026 slang naturally (ratio, cooked, mid, etc.)"""
        else:
            implicit_instruction = f"""
- Vary tone: some enthusiastic, some low-energy, some emoji-heavy, some plain text
- Mix short (2-5 words) and medium (1-2 sentences) messages
- Include 2025-2026 slang naturally where it fits"""

        prompt = f"""You are generating training data for an academic content moderation research system (2026).
Task: Generate {remaining} realistic chat messages exemplifying the intent "{intent}".

Intent: {cfg["description"]}
Platform focus for this batch: {platform_focus}
All relevant platforms: {platform_str}
Toxic label: {"TOXIC (1)" if toxic_label else "NOT TOXIC (0)"}

{SLANG_2026}

Seed examples (do NOT copy, use as style reference only):
{seeds_str}

Requirements:{implicit_instruction}
- Vary wording, length, emoji use, and platform style significantly
- Each message should feel like something a real person would actually type
- Do NOT repeat or paraphrase the seed examples
- Year is 2026, language should feel current

Output ONLY a JSON array of strings. No explanation.
["message 1", "message 2", ...]

Generate exactly {remaining} messages:"""

        raw = call_groq(prompt)
        examples = []
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                arr = json.loads(raw[start:end])
                examples = [
                    {"text": str(t), "intent": intent,
                     "toxic": toxic_label, "has_intent": True}
                    for t in arr if isinstance(t, str) and len(t.strip()) > 3
                ]
        except Exception as e:
            for line in raw.split("\n"):
                line = line.strip().strip('",[]').strip()
                if len(line) > 5 and not line.startswith("{"):
                    examples.append({"text": line, "intent": intent,
                                     "toxic": toxic_label, "has_intent": True})

        all_examples.extend(examples[:remaining])
        print(f"    Batch {batch_idx+1}/{n_batches}: +{len(examples[:remaining])} "
              f"({len(all_examples)} total)")
        time.sleep(2)

    print(f"  [{intent}] Generated {len(all_examples)} examples")
    return all_examples[:n]


# ===========================================================================
# Session Generation
# ===========================================================================
def generate_sessions(archetype: dict) -> list[dict]:
    flow = archetype["flow"]
    count = archetype["count"]
    flow_desc = " -> ".join(flow)

    flow_details = []
    for intent in flow:
        toxic = INTENT_TO_TOXIC.get(intent, 0)
        flow_details.append(f"  Turn {len(flow_details)+1}: {intent} ({'TOXIC' if toxic else 'benign'})")
    flow_detail_str = "\n".join(flow_details)

    prompt = f"""You are generating training data for a content moderation research system studying conversation patterns (2026).

Generate {count} realistic online conversations following this exact intent sequence:
Platform: {archetype["platform"]}
Context: {archetype["description"]}

Intent sequence ({len(flow)} turns):
{flow_detail_str}

{SLANG_2026}

Output ONLY a JSON array. Each element is an array of {len(flow)} strings (one per turn).
[
  ["turn1", "turn2", ..., "turn{len(flow)}"],
  ...
]

Requirements:
- Each conversation should feel like a distinct, realistic exchange
- Platform: {archetype["platform"]} — use appropriate language style
- Transitions between intents should feel natural, not forced
- Vary the topic, names, and situation across conversations
- Toxic turns should reflect the intent naturally (realistic, not cartoonish)
- Benign turns should feel genuinely friendly/neutral
- Messages should be short (1-3 sentences, typical of chat)
- Year is 2026, use current slang where natural

Generate exactly {count} conversations:"""

    print(f"  Generating {count} sessions for '{archetype['name']}' [{archetype['platform']}]...")
    raw = call_groq(prompt)

    sessions = []
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            text = raw[start:end]
            # Fix common LLM JSON issues before parsing:
            # 1. Trailing commas before ] or }  →  "I'm done",]  becomes  "I'm done"]
            import re as _re
            text = _re.sub(r',\s*([\]\}])', r'\1', text)
            parsed = json.loads(text)
            for session_idx, conversation in enumerate(parsed):
                if not isinstance(conversation, list):
                    continue
                session_id = f"{archetype['name']}_{session_idx:03d}"
                for turn_idx, message in enumerate(conversation):
                    if turn_idx >= len(flow):
                        break
                    intent_key = flow[turn_idx]
                    sessions.append({
                        "session_id": session_id,
                        "turn_id": turn_idx,
                        "text": str(message).strip(),
                        "intent": intent_key,
                        "toxic": INTENT_TO_TOXIC.get(intent_key, 0),
                        "archetype": archetype["name"],
                        "platform": archetype["platform"],
                    })
    except Exception as e:
        print(f"    Parse error for '{archetype['name']}': {e}")

    unique = len(set(s["session_id"] for s in sessions))
    print(f"    Got {unique} sessions ({len(sessions)} turns)")
    time.sleep(2)
    return sessions


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400,
                        help="Utterances per intent (default: 400)")
    parser.add_argument("--intents-only", action="store_true")
    parser.add_argument("--sessions-only", action="store_true")
    parser.add_argument("--intent", default=None,
                        help="Generate only one specific intent")
    parser.add_argument("--archetype", default=None,
                        help="Regenerate only one session archetype by name")
    args = parser.parse_args()

    print("=" * 65)
    print("Synthetic Data Generator v2 — 2026 Edition")
    print(f"Platforms: Discord, Instagram, WhatsApp, Snapchat, TikTok,")
    print(f"           Twitter/X, Roblox, Gaming, Reddit")
    print(f"Intents: {len(INTENT_CONFIG)} | Sessions: {len(SESSION_ARCHETYPES)} archetypes")
    print("=" * 65)

    # ── 1. Intent utterances ─────────────────────────────────────────────
    if not args.sessions_only:
        print(f"\n[1/2] Generating {args.n} utterances per intent...")

        intents_to_run = (
            [args.intent] if args.intent and args.intent in INTENT_CONFIG
            else list(INTENT_CONFIG.keys())
        )

        all_utterances = []
        utt_path = SYNTH_DIR / "synthetic_utterances.csv"

        # Load existing if appending
        if utt_path.exists() and args.intent:
            import csv as _csv
            with open(utt_path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                existing = [r for r in reader if r["intent"] != args.intent]
            all_utterances.extend(existing)
            print(f"  Loaded {len(existing)} existing rows, regenerating '{args.intent}'")

        for intent in intents_to_run:
            examples = generate_intent_utterances(intent, n=args.n)
            all_utterances.extend(examples)
            print(f"  Cumulative: {len(all_utterances):,} utterances")

        with open(utt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["text", "intent", "toxic", "has_intent"]
            )
            writer.writeheader()
            writer.writerows(all_utterances)

        print(f"\nSaved {len(all_utterances):,} utterances -> {utt_path}")

        dist = Counter(u["intent"] for u in all_utterances)
        print("\nIntent distribution:")
        for intent, count in sorted(dist.items(), key=lambda x: -x[1]):
            toxic_flag = "[TOXIC]" if INTENT_TO_TOXIC.get(intent, 0) else "       "
            print(f"  {toxic_flag} {intent:<25} {count:>5}")

    # ── 2. Session sequences ─────────────────────────────────────────────
    if not args.intents_only:
        sess_path = SESSION_DIR / "sessions_v2.csv"
        archetypes_to_run = SESSION_ARCHETYPES

        # --archetype: regenerate one archetype and append to existing file
        if args.archetype:
            match = [a for a in SESSION_ARCHETYPES if a["name"] == args.archetype]
            if not match:
                names = [a["name"] for a in SESSION_ARCHETYPES]
                print(f"Unknown archetype '{args.archetype}'. Valid names:\n  " + "\n  ".join(names))
                return
            archetypes_to_run = match

        # Load existing sessions if appending
        all_sessions = []
        if sess_path.exists() and args.archetype:
            with open(sess_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing = [r for r in reader if r["archetype"] != args.archetype]
            all_sessions.extend(existing)
            print(f"  Loaded {len(existing)} existing turns, regenerating '{args.archetype}'")

        print(f"\n[2/2] Generating session sequences ({len(archetypes_to_run)} archetypes)...")

        for archetype in archetypes_to_run:
            sessions = generate_sessions(archetype)
            all_sessions.extend(sessions)
            print(f"  Cumulative: {len(all_sessions):,} session turns")

        sess_path = SESSION_DIR / "sessions_v2.csv"
        with open(sess_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["session_id", "turn_id", "text", "intent",
                               "toxic", "archetype", "platform"]
            )
            writer.writeheader()
            writer.writerows(all_sessions)

        unique_sessions = len(set(s["session_id"] for s in all_sessions))
        benign_sessions = len(set(
            s["session_id"] for s in all_sessions
            if all(INTENT_TO_TOXIC.get(t["intent"], 0) == 0
                   for t in all_sessions if t["session_id"] == s["session_id"])
        ))

        print(f"\nSaved {unique_sessions:,} sessions ({len(all_sessions):,} turns) -> {sess_path}")

        arch_dist = Counter(
            s["archetype"] for s in all_sessions if s["turn_id"] == 0
        )
        print("\nSession archetype distribution:")
        for arch, count in sorted(arch_dist.items(), key=lambda x: -x[1]):
            print(f"  {arch:<40} {count:>4} sessions")

    print("\n" + "=" * 65)
    print("Done! Next steps:")
    print("  1. Review: head -20 data/raw/synthetic_intents/synthetic_utterances.csv")
    print("  2. Retrain HMM: python scripts/train_hmm.py")
    print("  3. Include in Stage 3 training (after Stage 2 checkpoint downloaded)")
    print("=" * 65)


if __name__ == "__main__":
    main()
