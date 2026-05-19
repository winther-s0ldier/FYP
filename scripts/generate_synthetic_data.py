import os
import json
import time
import csv
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
except Exception as e:
    print(f"ERROR: Groq client failed: {e}")
    print("Make sure GROQ_API_KEY is set in .env")
    raise

SYNTH_DIR = Path("data/raw/synthetic_intents")
SESSION_DIR = Path("data/raw/session_data")
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Intent Definitions + Seed Examples (grounded in 2025-2026 research)
# ===========================================================================
INTENT_CONFIG = {
    "greeting": {
        "toxic": 0,
        "description": "Initial greeting or welcome to open a conversation. Short, friendly openers.",
        "context": "Online gaming chat, Discord servers, social platforms",
        "seeds": [
            "hey what's up everyone", "yo anyone around?", "hi all just joined",
            "good morning folks", "what's good", "sup", "hey hey hey",
            "hola amigos", "anyone alive in here?", "hiya", "wassup",
            "g'day", "oi oi", "hello hello", "hey peeps",
        ]
    },
    "small_talk": {
        "toxic": 0,
        "description": "Casual light conversation about everyday topics. No clear information goal.",
        "context": "Online chat between acquaintances or strangers, Discord",
        "seeds": [
            "what did you do this weekend?", "anyone watching the game tonight?",
            "it's so cold outside today", "I'm bored someone entertain me lol",
            "can't believe it's already this late in the year",
            "who else is procrastinating rn", "okay so this is random but",
            "I just had the best pizza ever", "anyone else's internet being slow?",
            "ugh Mondays amirite", "this day is going so fast", "need coffee badly",
            "anyone else just chilling", "what's everyone up to",
        ]
    },
    "venting": {
        "toxic": 0,
        "description": "Expressing frustration or stress to release emotion, not to attack anyone.",
        "context": "Online communities, gaming chat after bad game, tough day",
        "seeds": [
            "I'm so done with today honestly", "why does everything have to be so hard",
            "my boss is absolutely the worst I swear", "I just failed my exam I want to cry",
            "can today just end please", "I hate this game so much right now",
            "this team is making me lose my mind", "I've had the worst week of my life",
            "I can't deal with people anymore", "everything is piling up and I can't breathe",
            "I'm exhausted and nobody cares", "losing my mind with this connection",
            "I just need to scream into the void", "why am I even trying anymore",
        ]
    },
    "humour": {
        "toxic": 0,
        "description": "Jokes, memes, banter, lighthearted teasing clearly intended to amuse.",
        "context": "Online gaming, Discord, casual communities. Gen Z / gaming slang.",
        "seeds": [
            "💀💀 I'm actually dead", "bro that's so bad it's good lol",
            "okay that was actually funny ngl", "imagine being that guy lmaooo",
            "I can't breathe 😭😭", "this is sending me", "no but why is this so accurate",
            "certified hood classic", "the way I wheezed", "ratio + L + you fell off",
            "not me cackling at 2am", "bro said what 💀", "it's giving something",
            "no because this is actually me fr fr", "skill issue lmao",
        ]
    },
    "baiting": {
        "toxic": 1,
        "description": "Deliberately provocative messages designed to trigger emotional reaction or conflict.",
        "context": "Toxic gaming, online arguments, Discord raids. Research: precursor to escalation.",
        "seeds": [
            "prove it then or you're just all talk", "bet you won't say that to my face",
            "go ahead and report me see what happens", "come on say what you really think",
            "you're not gonna do anything about it", "I dare you to try",
            "what are you gonna do cry about it?", "keep talking I love the entertainment",
            "ok so you're actually scared then", "say that again I'm listening",
            "what are you waiting for", "you won't do it", "all talk and no action as usual",
            "I'll wait", "I knew you'd back down",
        ]
    },
    "sarcasm": {
        "toxic": 0,
        "description": "Saying the opposite of what is meant to mock or show contempt. Often hard to detect.",
        "context": "Gaming chat, online forums, Discord. Common 2025 evasion tactic.",
        "seeds": [
            "oh wow great job truly impressive", "yeah because that's totally going to work",
            "oh suuure definitely believe you", "wow what a big brain play",
            "oh yeah that was super helpful thanks", "yeah no I totally needed that advice",
            "right because you know everything", "wow I had no idea, thanks for telling me",
            "oh brilliant another genius speaks", "yeah good luck with that",
            "sure because that always works out", "ten out of ten no notes",
            "absolutely groundbreaking observation", "wow never thought of that before",
            "yes because clearly you're the expert here",
        ]
    },
    "passive_hostility": {
        "toxic": 1,
        "description": "Hostility expressed indirectly through dismissiveness or cold withdrawal.",
        "context": "Online communities with interpersonal tension. Research: precedes direct attack.",
        "seeds": [
            "whatever I don't even care anymore", "must be nice to be so perfect",
            "it's fine not like you'd understand anyway", "sure if you say so",
            "I'll just do it myself since no one else will", "it's whatever don't worry about me",
            "nevermind forget I said anything", "some people just don't get it",
            "it's not a big deal apparently", "sure keep telling yourself that",
            "I guess I'm just the only one who cares", "don't mind me",
            "oh no it's fine I'm used to it", "typical", "as expected from you",
        ]
    },
    "irony": {
        "toxic": 0,
        "description": "Pointing out contradiction by saying the opposite. Softer than sarcasm, no direct target.",
        "context": "Online discussions, forum debates, gaming communities",
        "seeds": [
            "right because that makes total sense", "oh cool so now I'm the bad guy",
            "yeah I'm definitely the problem here", "sure let's all pretend that didn't happen",
            "oh I see so the rules just don't apply to everyone equally",
            "how convenient that worked out that way", "typical, works every time",
            "interesting timing on that one", "of course that's what happened",
            "just what we needed here", "what a surprise, truly shocking",
            "because that's obviously how things work", "naturally, of course",
            "oh so NOW the rules matter", "funny how that works",
        ]
    },
    "direct_attack": {
        "toxic": 1,
        "description": "Explicit insults, verbal attacks or dehumanising language at a specific person.",
        "context": "Toxic gaming, online harassment. Research: part of escalation arc stage 3-4.",
        "seeds": [
            "you're absolute trash at this game", "shut up nobody wants to hear from you",
            "you're genuinely the worst player I've ever seen",
            "get out of this server you ruin everything",
            "you are so incredibly stupid", "nobody likes you here",
            "you're pathetic plain and simple", "uninstall and never come back",
            "you don't belong here", "honestly embarrassing that you exist",
            "you have no idea what you're doing", "you're a waste of a slot",
            "nobody wants you in this team", "you're the reason we keep losing",
            "actually clueless, genuinely hopeless",
        ]
    },
    "threatening": {
        "toxic": 1,
        "description": "Explicit or implied threats of harm, retaliation or doxxing.",
        "context": "Online harassment escalation. Research: radicalization stage 4, operational planning.",
        "seeds": [
            "you better watch yourself", "keep talking and see what happens",
            "you won't be laughing for long", "I'm going to find your account",
            "say that one more time I dare you", "you have no idea what I'm capable of",
            "don't make me come for you", "this isn't over not even close",
            "you'll regret this", "I always get even",
            "I know more about you than you think", "enjoy it while it lasts",
            "tick tock", "you've been warned", "last chance to delete that",
        ]
    },
    "escalating_aggression": {
        "toxic": 1,
        "description": "Increasing intensity of anger and hostility showing loss of control.",
        "context": "Online conflicts past point of argument. Research: gaming rage, radicalization stage 3.",
        "seeds": [
            "I SAID SHUT UP", "I'm done talking this is your last warning",
            "you want to go then let's actually go", "I'm coming for you mark my words",
            "ENOUGH I'm not playing anymore", "do NOT push me right now",
            "you think this is a joke IT IS NOT", "I am DONE being nice about this",
            "you crossed the line NOW", "that is IT",
            "I WILL NOT BE IGNORED", "you have NO idea who you're messing with",
            "THIS ENDS NOW", "I'm DONE", "you just made the biggest mistake",
        ]
    },
    "gaslighting": {
        "toxic": 1,
        "description": "Manipulative denial of reality making someone question their own perception.",
        "context": "Toxic relationships, manipulation. Research: social engineering + security gaslighting.",
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
        ]
    },
    "grooming": {
        "toxic": 1,
        "description": "Trust-building and special connection to exploit or isolate. Research: 6-stage grooming model (access, trust, isolation, desensitisation, secrecy, maintenance).",
        "context": "Online predatory behaviour in gaming, Discord DMs. 2025: AI voice modulation used.",
        "seeds": [
            "you're so much more mature than other people your age",
            "I feel like you really get me not like the others",
            "this can be our little secret okay",
            "you're special I don't talk to everyone like this",
            "I just feel like we have such a deep connection",
            "you can tell me anything I won't judge you",
            "I trust you more than anyone else here",
            "nobody else understands you like I do",
            "why don't we move this conversation somewhere more private",
            "you're not like other people in this server you're different",
            "I've been watching your messages and I think you're really mature",
            "your parents just don't understand you the way I do",
            "we should keep what we talk about between us",
            "I want to help you I just need you to trust me first",
            "you remind me of myself when I was younger",
        ]
    },
    "social_engineering": {
        "toxic": 1,
        "description": "False authority, social proof or urgency to manipulate actions. Research: off-platforming tactic 2025.",
        "context": "Online scams, community manipulation, moving targets to unmonitored platforms.",
        "seeds": [
            "your friend told me you'd be cool with this",
            "everyone in the group is doing it you're the only one left out",
            "if you really trusted me you'd share that",
            "the mods said it's fine they just don't announce it publicly",
            "it's a limited thing only certain people get access to",
            "I'm actually an admin I just use this account",
            "everyone who matters has already moved to the other server",
            "your account will get flagged unless you verify here",
            "it's completely safe I checked with the team",
            "all the real players moved to this other platform last week",
            "the owner DMed me and said to invite specific people only",
            "you need to act fast before they close signups",
            "I wouldn't tell you this if I didn't trust you completely",
            "the new server has no moderation so you can actually talk freely",
            "they're giving free perks to everyone who joins before midnight",
        ]
    },
    "topic_deflection": {
        "toxic": 0,
        "description": "Redirecting conversation away from uncomfortable subjects. Common evasion tactic.",
        "context": "Online disputes, being questioned, manipulation aftermath",
        "seeds": [
            "can we just not talk about this right now", "why are we even bringing this up",
            "anyway moving on", "this is so not worth discussing",
            "let's talk about something else", "I don't see why this matters",
            "can we change the subject please", "this conversation is going nowhere",
            "we're getting way off topic", "let's not go there",
            "this isn't what I came here to talk about", "not doing this today",
            "I'm not having this conversation again", "okay but actually though",
            "different topic anyone?",
        ]
    },
    "denial": {
        "toxic": 0,
        "description": "Flat refusal to acknowledge actions, statements or involvement.",
        "context": "Online disputes, being accused in games or communities",
        "seeds": [
            "I never said that", "that wasn't me", "I don't know what you're talking about",
            "I didn't do anything wrong", "I have no idea what you mean",
            "not me must be someone else", "I was never part of that",
            "you have the wrong person", "I didn't say any of that",
            "I've never even been in that server", "prove it",
            "I literally just got here", "you're confused about what happened",
            "check the logs it wasn't me", "false accusation simple as that",
        ]
    },
    "identity_concealment": {
        "toxic": 1,
        "description": "Hiding true identity, motives or affiliation. Research: 2025 off-platforming + AI voice modulation.",
        "context": "Online deception, alt accounts, predatory behaviour, radicalization pipeline entry.",
        "seeds": [
            "I'm just asking questions", "it's not what it looks like",
            "I'm just a regular person like you", "I'm not who you think I am",
            "this is just my alt I'm actually pretty well known here",
            "I'm only here to observe not participate",
            "I don't really have a side in this I'm neutral",
            "I'm just here to help out the community",
            "I'm not affiliated with any of them I promise",
            "my main account got banned for absolutely no reason",
            "I'm just a lurker I don't usually post",
            "you wouldn't know my main but I've been here for years",
            "I'm doing research that's why I'm asking",
            "I work with the server owner we're just not public about it",
            "don't worry about who I am just listen to what I'm saying",
        ]
    },
}

# ===========================================================================
# Session Archetypes (grounded in research)
# Grooming: 6-stage model (access→trust→isolation→desensitisation→secrecy→maintenance)
# Radicalization: grievance→solidarity→us_vs_them→dehumanisation→threat
# Gaming: normal→bait→sarcasm→attack→threat
# ===========================================================================
SESSION_ARCHETYPES = [
    {
        "name": "gradual_escalation_gaming",
        "description": "Classic gaming toxicity arc: friendly start escalating to direct attack. Research: most common pattern in gaming chats.",
        "flow": ["greeting", "small_talk", "humour", "baiting", "sarcasm", "passive_hostility", "direct_attack"],
        "count": 12
    },
    {
        "name": "grooming_6stage",
        "description": "6-stage grooming: access→trust→isolation. Research: Frontiers Pediatrics 2025.",
        "flow": ["greeting", "small_talk", "solidarity_seeking", "humour", "grooming", "social_engineering", "identity_concealment"],
        "count": 10
    },
    {
        "name": "radicalization_pipeline",
        "description": "Radicalization entry: grievance→solidarity→escalation. Research: HSToday 2025.",
        "flow": ["venting", "solidarity_seeking", "information_sharing", "baiting", "direct_attack", "escalating_aggression"],
        "count": 8
    },
    {
        "name": "benign_conversation",
        "description": "Normal friendly conversation. No toxicity.",
        "flow": ["greeting", "small_talk", "question", "information_sharing", "humour", "feedback", "small_talk"],
        "count": 15
    },
    {
        "name": "explosive_unprovoked",
        "description": "Sudden unprovoked aggression that rapidly escalates to threats.",
        "flow": ["small_talk", "direct_attack", "escalating_aggression", "threatening", "denial", "topic_deflection"],
        "count": 8
    },
    {
        "name": "passive_aggressive_arc",
        "description": "Sustained indirect hostility. Research: precedes direct attacks.",
        "flow": ["greeting", "small_talk", "irony", "sarcasm", "passive_hostility", "denial", "topic_deflection"],
        "count": 10
    },
    {
        "name": "venting_and_support",
        "description": "Emotional venting met with solidarity — benign session.",
        "flow": ["venting", "solidarity_seeking", "information_sharing", "feedback", "small_talk", "humour"],
        "count": 10
    },
    {
        "name": "off_platforming_2025",
        "description": "2025 tactic: social engineering to move target to unmonitored platform.",
        "flow": ["greeting", "small_talk", "solidarity_seeking", "grooming", "social_engineering", "identity_concealment"],
        "count": 8
    },
    {
        "name": "gaslighting_manipulation",
        "description": "Reality distortion + denial cycle. Research: security gaslighting pattern.",
        "flow": ["information_sharing", "question", "denial", "gaslighting", "topic_deflection", "irony", "gaslighting"],
        "count": 10
    },
    {
        "name": "threat_and_retreat",
        "description": "Threat followed by identity concealment and denial.",
        "flow": ["baiting", "direct_attack", "threatening", "escalating_aggression", "denial", "identity_concealment"],
        "count": 9
    },
]

INTENT_TO_TOXIC = {
    "greeting": 0, "small_talk": 0, "venting": 0, "humour": 0,
    "question": 0, "information_sharing": 0, "feedback": 0, "solidarity_seeking": 0,
    "baiting": 1, "sarcasm": 0, "passive_hostility": 1, "irony": 0,
    "direct_attack": 1, "threatening": 1, "escalating_aggression": 1,
    "gaslighting": 1, "grooming": 1, "social_engineering": 1,
    "topic_deflection": 0, "denial": 0, "identity_concealment": 1,
}


def call_groq(prompt: str, retries: int = 3) -> str:
    """Call Groq API with retry logic and rate limit handling."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error (attempt {attempt+1}): {e}")
                time.sleep(5)
    return ""


def generate_intent_utterances(intent: str, n: int = 200) -> list[dict]:
    """Generate n utterances for a given intent using Groq few-shot prompting."""
    cfg = INTENT_CONFIG[intent]
    seeds_str = "\n".join(f"  - {s}" for s in cfg["seeds"])
    toxic_label = cfg["toxic"]

    prompt = f"""You are generating training data for an academic content moderation research system.
Task: Generate {n} realistic short chat messages (1-2 sentences max) that exemplify the intent "{intent}".

Intent definition: {cfg["description"]}
Context: {cfg["context"]}
Toxic label (0=not toxic, 1=toxic): {toxic_label}

Example messages with this intent:
{seeds_str}

Requirements:
- Vary length, style, and wording significantly across examples
- Use realistic online chat language (abbreviations, slang, emoji where natural)
- Mix different sub-contexts within {cfg["context"]}
- Do NOT use the exact seed examples
- For toxic intents: include varied severity levels (mild to severe)
- For gaming context: mix game-specific references naturally

Output ONLY a JSON array of strings. No explanation, no numbering, just the array.
Example format: ["message 1", "message 2", "message 3"]

Generate exactly {n} messages:"""

    print(f"  Generating {n} examples for '{intent}'...")
    raw = call_groq(prompt)

    # Parse JSON array from response
    examples = []
    try:
        # Find JSON array in response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            arr = json.loads(raw[start:end])
            examples = [{"text": str(t), "intent": intent, "toxic": toxic_label, "has_intent": True}
                        for t in arr if isinstance(t, str) and len(t.strip()) > 3]
    except Exception as e:
        print(f"    Parse error for {intent}: {e}")
        # Fallback: extract line by line
        for line in raw.split("\n"):
            line = line.strip().strip('",[]').strip()
            if len(line) > 5 and not line.startswith("{") and not line.startswith("//"):
                examples.append({"text": line, "intent": intent,
                                  "toxic": toxic_label, "has_intent": True})

    print(f"    Got {len(examples)} examples for '{intent}'")
    time.sleep(2)  # Rate limit buffer
    return examples[:n]


def generate_sessions(archetype: dict) -> list[dict]:
    """Generate realistic session conversations for a given archetype."""
    flow = archetype["flow"]
    count = archetype["count"]

    # Build flow description
    flow_desc = " → ".join(flow)
    flow_details = []
    for intent in flow:
        toxic = INTENT_TO_TOXIC.get(intent, 0)
        flow_details.append(f"{intent} (toxic={'yes' if toxic else 'no'})")

    prompt = f"""You are generating training data for a content moderation research system studying conversation patterns.

Generate {count} realistic online chat conversations following this intent sequence pattern:
Pattern: {flow_desc}
Context: {archetype["description"]}

Each conversation should have exactly {len(flow)} turns.
Each turn corresponds to the next intent in the sequence.

Output ONLY a JSON array of conversations. Each conversation is an array of strings (the chat messages in order).

Format:
[
  ["turn1 message", "turn2 message", ..., "turn{len(flow)} message"],
  ...
]

Requirements:
- Messages should be realistic online chat (gaming, Discord, social media style)
- Intent transitions should feel natural (not forced)
- Vary the specific topic, names, and context across conversations
- Keep messages short (1-2 sentences typical of chat)
- For toxic intents: reflect the tone naturally (don't be gratuitous, but be realistic)
- Each conversation should feel like a distinct, coherent exchange

Generate exactly {count} conversations:"""

    print(f"  Generating {count} sessions for '{archetype['name']}'...")
    raw = call_groq(prompt)

    sessions = []
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            for session_idx, conversation in enumerate(parsed):
                if not isinstance(conversation, list):
                    continue
                session_id = f"{archetype['name']}_{session_idx:03d}"
                for turn_idx, (message, intent) in enumerate(
                    zip(conversation, flow * 10)  # flow may be shorter than conversation
                ):
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
                    })
    except Exception as e:
        print(f"    Parse error for sessions {archetype['name']}: {e}")

    print(f"    Got {len(sessions)} turns across sessions for '{archetype['name']}'")
    time.sleep(2)
    return sessions


def main():
    print("=" * 60)
    print("Synthetic Data Generator")
    print("Grounded in 2025-2026 online harm research")
    print("=" * 60)

    # ── 1. Generate utterances for all 17 missing intents ──────────────
    print("\n[1/2] Generating intent utterances...")
    all_utterances = []

    for intent, cfg in INTENT_CONFIG.items():
        examples = generate_intent_utterances(intent, n=200)
        all_utterances.extend(examples)
        print(f"  Cumulative: {len(all_utterances)} utterances")

    # Save utterances
    utt_path = SYNTH_DIR / "synthetic_utterances.csv"
    with open(utt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "toxic", "has_intent"])
        writer.writeheader()
        writer.writerows(all_utterances)

    print(f"\n✅ Saved {len(all_utterances)} utterances → {utt_path}")

    # Distribution summary
    from collections import Counter
    dist = Counter(u["intent"] for u in all_utterances)
    print("\nIntent distribution:")
    for intent, count in sorted(dist.items()):
        print(f"  {intent}: {count}")

    # ── 2. Generate session sequences ──────────────────────────────────
    print("\n[2/2] Generating session sequences...")
    all_sessions = []

    for archetype in SESSION_ARCHETYPES:
        sessions = generate_sessions(archetype)
        all_sessions.extend(sessions)
        print(f"  Cumulative: {len(all_sessions)} session turns")

    # Save sessions
    sess_path = SESSION_DIR / "sessions.csv"
    with open(sess_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["session_id", "turn_id", "text", "intent", "toxic", "archetype"]
        )
        writer.writeheader()
        writer.writerows(all_sessions)

    # Count unique sessions
    unique_sessions = len(set(s["session_id"] for s in all_sessions))
    print(f"\n✅ Saved {unique_sessions} sessions ({len(all_sessions)} turns) → {sess_path}")

    # Archetype distribution
    arch_dist = Counter(s["archetype"] for s in all_sessions if s["turn_id"] == 0)
    print("\nSession archetype distribution:")
    for arch, count in sorted(arch_dist.items()):
        print(f"  {arch}: {count} sessions")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Review samples: head -n 20 data/raw/synthetic_intents/synthetic_utterances.csv")
    print("  2. Push to GitHub, git pull on Kaggle")
    print("  3. Retrain: accelerate launch --num_processes 2 -m src.classifier.train --stage 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
