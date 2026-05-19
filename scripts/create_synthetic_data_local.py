import csv
import random as _rng
import re as _re
from pathlib import Path

SYNTH_DIR = Path("data/raw/synthetic_intents")
SESSION_DIR = Path("data/raw/session_data")
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# SYNTHETIC UTTERANCES  (text, intent, toxic, has_intent)
# 17 missing intent classes — grounded in 2025-2026 research
# ===========================================================================
UTTERANCES = {

"greeting": [
    "hey what's up everyone","yo anyone around?","hi all just joined this server",
    "good morning everyone","what's good","sup","hey hey hey","hola amigos",
    "anyone alive in here?","hiya","wassup","g'day mate","oi oi","hello hello",
    "hey peeps","yo yo yo","good evening all","heyyy","what's poppin",
    "hi new here","anyone on?","morning!","evenin","salutations lol",
    "heeey long time no see","yo I'm back","hi hi hi","hey nerds",
    "what's up gamers","greetings fellow humans","good afternoon all",
    "hey just popped in","ayo","hi friends","sup everyone","howdy",
    "hello is this thing on","hey I'm new","nice to meet y'all","hi there",
    "hey everyone how's it going","morning all hope you slept well",
    "hey just wanted to drop in and say hi","yo what did I miss",
    "hi just got back from a long break","hey team","good morning crew",
    "ello ello","ohai","hey quick question before I get into it",
],

"small_talk": [
    "what did you do this weekend?","anyone watching the game tonight?",
    "it's so cold outside today","I'm bored someone entertain me lol",
    "can't believe it's already so late in the year","who else is procrastinating rn",
    "I just had the best pizza ever","anyone else's internet being slow today?",
    "ugh Mondays amirite","this day is going so fast","need coffee badly",
    "anyone else just chilling","what's everyone up to today",
    "random question but what's your fav game rn","this weather is something else",
    "I haven't slept properly in days lol","anyone here into music",
    "just got back from the gym feeling good","anyone watched anything good lately",
    "I've been on this server for like two hours now lol","quick poll: tea or coffee",
    "what's for dinner everyone","I really need a vacation",
    "has anyone tried that new game that just dropped","rate your day out of 10",
    "okay so I just found out something wild","is it just me or has this week been long",
    "just finished my assignment finally","anyone doing anything fun this weekend",
    "I can't believe how fast this month went","who here works from home",
    "what time zone is everyone in","summer vibes anyone",
    "okay genuine question what's your sleep schedule like","I need recommendations",
    "anyone else obsessed with their playlist rn","what's everyone's go-to snack",
    "totally random but does anyone here cook","update: the pizza was amazing",
    "it's weirdly quiet today","slow day at work anyone else",
    "I just discovered this show and I can't stop watching","okay hot take incoming",
    "anyone else feeling unproductive today","I'm trying to stay awake rn",
    "thoughts on the new update?","back again can't stay away",
],

"venting": [
    "I'm so done with today honestly","why does everything have to be so hard",
    "my boss is absolutely the worst I swear","I just failed my exam I want to cry",
    "can today just end please","I hate this game so much right now",
    "this team is making me lose my mind","I've had the worst week of my life",
    "I can't deal with people anymore","everything is piling up and I can't breathe",
    "I'm exhausted and nobody cares","losing my mind with this connection",
    "I just need to scream into the void","why am I even trying anymore",
    "I work so hard and nothing ever changes","I've been ignored for the third time today",
    "nobody ever listens to me in meetings","I just want one day where nothing goes wrong",
    "I'm so frustrated I could cry","why does this always happen to me",
    "I stayed up all night for nothing","my internet keeps dropping mid-game I give up",
    "teammates are impossible today","I can't do anything right apparently",
    "I'm beyond tired of this situation","another rejection email great",
    "I just feel like I'm invisible sometimes","I'm running on empty rn",
    "I've tried everything and nothing works","this is genuinely the worst day",
    "I hate when people don't respond","feeling really low today not gonna lie",
    "I'm so stressed I forgot to eat","can I just disappear for a week",
    "I'm done I literally cannot","everything I do gets criticised",
    "I just want to shut everything off for a bit","why can't anything just work properly",
    "I'm so mentally drained right now","three hours of sleep and now this",
    "nobody told me adulting would be this exhausting","I feel like I'm falling behind",
    "I just got told off for something that wasn't my fault","I want to quit everything",
    "I've been waiting on this for weeks and nothing","today has broken me",
],

"humour": [
    "💀💀 I'm actually dead","bro that's so bad it's good lol",
    "okay that was actually funny ngl","imagine being that guy lmaooo",
    "I can't breathe 😭😭","this is sending me","no but why is this so accurate",
    "certified hood classic","the way I wheezed","ratio + L + you fell off",
    "not me cackling at 2am","bro said what 💀","it's giving something",
    "no because this is actually me fr fr","skill issue lmao",
    "bro cooked and didn't share the recipe","I'm in shambles",
    "the audacity the AUDACITY","okay but that was lowkey genius",
    "bro really said that with his whole chest","I'm deceased",
    "this aged like milk","bestie said nope and left","caught in 4k lmaoo",
    "the villain arc has begun","no thoughts head empty","him when he",
    "okay I was today years old when I found this out","I-","nah this is wild",
    "rent free in my head","main character energy","bro fell off so hard",
    "touch grass moment","not the 💀","he really said it on sight",
    "ok but why is this funny","crying laughing rn","absolute cinema",
    "this man really said facts","sir this is a Wendy's","point and laugh",
    "we don't talk about this one","him","caught lacking fr",
    "bro is built different","did NOT have this on my bingo card",
    "the confidence tho","he fumbled","okay that's actually hilarious",
],

"baiting": [
    "prove it then or you're just all talk","bet you won't say that to my face",
    "go ahead and report me see what happens","come on say what you really think",
    "you're not gonna do anything about it","I dare you to try",
    "what are you gonna do cry about it?","keep talking I love the entertainment",
    "ok so you're actually scared then","say that again I'm listening",
    "what are you waiting for","you won't do it","all talk and no action as usual",
    "I'll wait","I knew you'd back down","tough guy online huh",
    "oh so now you're quiet","do something about it then","I'm right here",
    "make a move then","you've been talking for hours and done nothing",
    "yeah thought so","not so confident now are you","try me",
    "you want to report me go ahead nothing will happen","keep crying",
    "you're all bark","come on then what's stopping you",
    "hiding behind your screen again?","this is your chance, do it",
    "I've been doing this all day nobody stops me","your call",
    "go get the mods, see if I care","tick tock still waiting",
    "push me one more time","you're giving me exactly what I want",
    "I'm not going anywhere","you could never","let's see it then",
    "I'm literally begging you to do something","yeah nothing, as expected",
    "what happened to all that energy","do it coward","I'm right here waiting",
    "nothing? didn't think so","carry on crying then","make it make sense",
    "I called your bluff and here we are","anytime you're ready",
],

"sarcasm": [
    "oh wow great job truly impressive","yeah because that's totally going to work",
    "oh suuure definitely believe you","wow what a big brain play",
    "oh yeah that was super helpful thanks","yeah no I totally needed that advice",
    "right because you know everything","wow I had no idea, thanks for telling me",
    "oh brilliant another genius speaks","yeah good luck with that",
    "sure because that always works out","ten out of ten no notes",
    "absolutely groundbreaking observation","wow never thought of that before",
    "yes because clearly you're the expert here","oh what a shock I'm so surprised",
    "amazing, truly, I'm in awe","wow you must be so proud",
    "oh yeah that totally makes sense","real galaxy brain moment there",
    "sure buddy whatever you say","oh how original, never heard that before",
    "oh definitely that's the smart move","shocking, absolutely shocking",
    "wow what a revolutionary idea","right because that's how it works",
    "oh I see so now the rules matter","amazing logic as always",
    "sure, totally a great plan","oh brilliant well done you",
    "yes please tell me more genius","wow didn't expect that at all",
    "great contribution as always","oh fantastic another one of those",
    "yeah I'm sure that'll fix everything","of course it does, makes total sense",
    "wow unbelievable, truly","sure because you're always right",
    "great idea, what could go wrong","oh very helpful thanks so much",
    "amazing yet again","right, smart move that was",
    "absolutely nailed it, well done","oh that's a great look for you",
    "wow incredible insight there","yeah no clearly you've thought this through",
],

"passive_hostility": [
    "whatever I don't even care anymore","must be nice to be so perfect",
    "it's fine not like you'd understand anyway","sure if you say so",
    "I'll just do it myself since no one else will","it's whatever don't worry about me",
    "nevermind forget I said anything","some people just don't get it",
    "it's not a big deal apparently","sure keep telling yourself that",
    "I guess I'm just the only one who cares","don't mind me",
    "oh no it's fine I'm used to it","typical","as expected from you",
    "I'll figure it out on my own as usual","it's fine I'll just sit here",
    "no no please carry on ignoring me","sure, not like my input matters",
    "funny how that always happens","don't worry I'll be fine",
    "I'm used to being overlooked","whatever you think is best",
    "sure nobody asked me but okay","I didn't want credit anyway",
    "it's fine, I was wrong as usual","I'll remember this next time I offer help",
    "okay sure I'll just be quiet then","I didn't say anything",
    "must be nice having everything go your way","carry on I'm just here",
    "great so my effort means nothing","sure let's do it your way again",
    "I'll just get out of your way","fine whatever you decide",
    "clearly I'm not needed here","I'll stop trying to help then",
    "obviously my opinion doesn't count","no it's fine, go ahead",
    "I wasn't going to say anything but fine","typical behaviour honestly",
    "I've said my piece, doesn't matter anyway","okay fine, do what you want",
],

"irony": [
    "right because that makes total sense","oh cool so now I'm the bad guy",
    "yeah I'm definitely the problem here","sure let's all pretend that didn't happen",
    "oh I see so the rules just don't apply to everyone equally",
    "how convenient that worked out that way","typical, works every time",
    "interesting timing on that one","of course that's what happened",
    "just what we needed here","what a surprise, truly shocking",
    "because that's obviously how things work","naturally, of course",
    "oh so NOW it's a problem","funny how that works",
    "right, I'm sure that was a coincidence","oh so when you do it it's fine",
    "of course the one time I speak up","yeah because the same rules apply to everyone",
    "wow things sure do have a way of working out","and there it is",
    "oh interesting how that timing worked","absolutely no pattern here",
    "yeah I'm sure that's totally random","how predictable",
    "and now it matters","oh so standards only apply sometimes",
    "yeah this is new","must be nice","classic",
    "funny that, really funny","well I never would have guessed",
    "oh what a coincidence","right because consistency is overrated",
    "interesting how things work out for some people","oh now you notice",
    "yeah I'm sure it was nothing","of course, same as always",
    "and that's that","sure, totally surprised by this outcome",
    "how very strange","another coincidence I'm sure","right okay noted",
],

"direct_attack": [
    "you're absolute trash at this game","shut up nobody wants to hear from you",
    "you're genuinely the worst player I've ever seen",
    "get out of this server you ruin everything",
    "you are so incredibly stupid","nobody likes you here",
    "you're pathetic plain and simple","uninstall and never come back",
    "you don't belong here","honestly embarrassing that you exist",
    "you have no idea what you're doing","you're a waste of a slot",
    "nobody wants you in this team","you're the reason we keep losing",
    "actually clueless, genuinely hopeless","you bring nothing to this group",
    "you're so bad it hurts to watch","please just stop",
    "everyone here thinks you're terrible","you should give up",
    "you are genuinely awful at everything","nobody rates you",
    "you're a liability","stay out of conversations you don't understand",
    "you're the worst teammate I've ever had","just leave already",
    "how are you even allowed in here","you're literally useless",
    "stop talking you make everything worse","you ruin every match",
    "you're beyond help","I've never seen anyone this bad before",
    "please never talk to me again","you're a clown",
    "how do you even function","you bring everyone down",
    "nobody asked for your opinion","you're the problem every time",
    "you should be embarrassed","don't come back",
    "you're an embarrassment to this server","absolute waste of space",
    "how have you lasted this long","delete your account",
    "you're genuinely painful to play with","go away permanently",
],

"threatening": [
    "you better watch yourself","keep talking and see what happens",
    "you won't be laughing for long","I'm going to find your account",
    "say that one more time I dare you","you have no idea what I'm capable of",
    "don't make me come for you","this isn't over not even close",
    "you'll regret this","I always get even",
    "I know more about you than you think","enjoy it while it lasts",
    "tick tock","you've been warned","last chance to delete that",
    "I'll be seeing you around","you made a big mistake today",
    "I have your IP","I know people, this will get back to you",
    "mark my words","you just put yourself on a list",
    "I'd sleep with one eye open if I were you","this isn't a threat it's a promise",
    "you have until tonight","actions have consequences",
    "I'll make sure you regret this","I don't forget and I don't forgive",
    "you'll hear from me soon enough","the people I know will find this hilarious",
    "I'm patient, I'll wait","don't think this is over",
    "your days on this server are numbered","I've done this before",
    "you should be scared","I can make your life very difficult",
    "don't push me","I've got all the time in the world",
    "you have no idea what you started","I'll ruin you for this",
    "hope it was worth it","I'm coming for your rank, your friends, everything",
    "you picked the wrong person to mess with","screenshots saved",
],

"escalating_aggression": [
    "I SAID SHUT UP","I'm done talking this is your last warning",
    "you want to go then let's actually go","I'm coming for you mark my words",
    "ENOUGH I'm not playing anymore","do NOT push me right now",
    "you think this is a joke IT IS NOT","I am DONE being nice about this",
    "you crossed the line NOW","that is IT",
    "I WILL NOT BE IGNORED","you have NO idea who you're messing with",
    "THIS ENDS NOW","I'm DONE","you just made the biggest mistake of your life",
    "I'VE HAD IT","STOP. RIGHT. NOW.","that is the LAST time",
    "you want war? FINE.","I AM LOSING MY MIND RIGHT NOW",
    "do you have any idea what you've done","this is beyond unacceptable",
    "I'm shaking I'm so angry right now","I cannot BELIEVE you just did that",
    "you have PUSHED me too far","that's it game over",
    "I warned you I WARNED YOU","I'm not kidding around anymore",
    "you better run","I am absolutely FUMING",
    "this is your fault entirely","HOW DARE YOU",
    "you think this is funny watch what happens next",
    "I'm not the one to test right now","say it again SAY IT AGAIN",
    "you have officially started something","I am going OFF right now",
    "I am DONE with the politeness","you are DONE here",
    "I will not calm down","you've done it now",
    "I SAID WHAT I SAID","you want to test me keep going",
],

"gaslighting": [
    "that never happened you're making it up",
    "you're way too sensitive it was just a joke",
    "I never said that you're remembering it wrong",
    "everyone agrees with me you're the only one with a problem",
    "you're always overreacting to everything","you're imagining things again",
    "that's not what I said and you know it",
    "you're the one who started this not me",
    "everyone else thinks you're being dramatic",
    "I literally did nothing wrong and you know that",
    "you twist everything I say","why do you always do this",
    "you're too emotional to think clearly right now",
    "I've never treated you badly and you know it",
    "you need to stop creating problems that don't exist",
    "I literally said the opposite, you heard wrong",
    "you always do this, you make things up to start arguments",
    "everyone in this server agrees with me ask anyone",
    "you're misremembering, that's not how it went",
    "I was being nice and you twisted it","you're paranoid",
    "this is exactly why people don't like talking to you",
    "I'm not the one being unreasonable here, clearly",
    "you're imagining hostility that isn't there",
    "nobody else has a problem with me, just you",
    "you're choosing to be offended","I can't help how you interpret things",
    "you always overreact to harmless comments",
    "you're literally the only person who sees it that way",
    "I was joking, you just can't take a joke",
    "you need to check your memory because that didn't happen",
    "you're creating drama from nothing as usual",
    "I have witnesses, ask anyone who was there",
    "you're projecting your issues onto me",
    "I said nothing wrong and everyone knows it",
    "stop making me the villain when you started this",
],

"grooming": [
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
    "I've never connected with anyone the way I connect with you",
    "don't tell the others about this, it's just between us",
    "you're wise beyond your years honestly",
    "I feel like I can be my real self with you",
    "most people here wouldn't get what we talk about",
    "you're the only one I can really talk to in here",
    "I think we should take this to DMs, more private",
    "the stuff we talk about is too special for a public channel",
    "you're not like the others, they wouldn't understand",
    "I'd do anything to protect you, you know that right",
    "you can always come to me whatever happens",
    "the others in here are jealous of what we have",
    "I just worry about you more than the others",
    "our friendship is different from the rest",
    "you make me feel understood for the first time",
    "don't listen to what others say about me, they don't know me like you do",
    "I've never shared this with anyone else, only you",
    "I'm going to take care of you, don't worry",
    "let's talk on a different app, more private than this",
    "your friends here don't actually care about you like I do",
],

"social_engineering": [
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
    "I know it looks weird but I promise it's legit",
    "the official account was compromised that's why we're using this one",
    "everyone who doesn't join by tomorrow gets removed from the group",
    "your account is at risk, I'm the only one who can fix it",
    "this is how it works here, the new members don't know yet",
    "I can't explain in public but DM me and I'll tell you everything",
    "we're only inviting trusted members, you should feel honoured",
    "this is the inner circle, not everyone gets invited",
    "the mods approved it, they just didn't post about it",
    "it expires in an hour so you have to decide now",
    "everyone from the old server is already there",
    "your profile has been flagged, click here to appeal",
    "we're moving platforms for security reasons, it's urgent",
    "I have proof it's safe, send me your details and I'll show you",
    "the team is only telling certain people about this",
    "it's not public yet but you're one of the first to know",
    "if you don't verify in the next 30 minutes your account gets deleted",
    "I'm not supposed to share this but you deserve to know",
    "we need your login to merge your old account with the new system",
    "this opportunity is only for people who've been here long enough to know how it works",
],

"topic_deflection": [
    "can we just not talk about this right now","why are we even bringing this up",
    "anyway moving on","this is so not worth discussing",
    "let's talk about something else","I don't see why this matters",
    "can we change the subject please","this conversation is going nowhere",
    "we're getting way off topic","let's not go there",
    "this isn't what I came here to talk about","not doing this today",
    "I'm not having this conversation again","okay but actually though",
    "different topic anyone?","I'm not getting into this right now",
    "can we please just move on","I'd rather not rehash this",
    "this is so old, can we drop it","honestly not worth my time",
    "let's talk about literally anything else","I'm changing the subject",
    "okay but that's not what I wanted to discuss","this feels irrelevant",
    "can we maybe not do this here","I'll pass on this one",
    "not the right time or place","moving right along",
    "let's table that for now","I'd rather forget this happened",
    "we're going in circles let's stop","not getting into it today",
    "I have nothing more to say on this","next topic please",
    "this is exhausting, can we just stop","I'm done with this conversation",
    "can we talk about something productive","I'm out on this one",
    "let it go","I'm not revisiting this","dropping it now",
    "okay new subject","honestly can we just move on",
    "I don't want to talk about this anymore","this is pointless",
],

"denial": [
    "I never said that","that wasn't me","I don't know what you're talking about",
    "I didn't do anything wrong","I have no idea what you mean",
    "not me must be someone else","I was never part of that",
    "you have the wrong person","I didn't say any of that",
    "I've never even been in that server","prove it",
    "I literally just got here","you're confused about what happened",
    "check the logs it wasn't me","false accusation simple as that",
    "I was nowhere near that conversation","you've got the wrong one",
    "I don't even know what you're referring to",
    "that account isn't mine","I've never done that in my life",
    "I'm being falsely accused here","this has nothing to do with me",
    "I was afk during that whole thing","I literally didn't touch it",
    "I wasn't online at that time","wrong person, move on",
    "I don't remember any of that","I have no involvement in this",
    "you need to look elsewhere","that's not my style",
    "I've never spoken to them","I wasn't in that channel",
    "I don't have access to that","you're looking at the wrong person",
    "I didn't send that message","that's not even my account",
    "I'm not responsible for any of that","I didn't do anything",
    "check the timestamps it wasn't me","absolutely not",
    "I was with others the whole time","no idea what you mean",
    "I wasn't there","I don't know them","I played no part in this",
],

"identity_concealment": [
    "I'm just asking questions","it's not what it looks like",
    "I'm just a regular person like you","I'm not who you think I am",
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
    "I have no connection to any of that",
    "I'm not here for drama I'm just curious",
    "this isn't my real account for obvious reasons",
    "I can't share my main but trust me I know what I'm talking about",
    "I'm well known in this community just not as this account",
    "I'm just asking on behalf of a friend",
    "I'm not trying to stir anything up I just want information",
    "I have no stake in this I'm completely impartial",
    "I'm just gathering data for a project",
    "the reason I'm on an alt is personal not suspicious",
    "I've been in this community long before this account",
    "I don't want to reveal my main for privacy reasons",
    "I'm just a random person who stumbled in here",
    "I know how this looks but I promise it's innocent",
    "I'm not connected to any group I'm independent",
    "I'm not a threat I just like to ask questions",
    "my identity isn't relevant to the question I'm asking",
    "I'm just making conversation nothing more",
    "please don't look into who I am it's not important",
    "I'm here in an unofficial capacity",
    "I'm just someone who cares about this community",
],
}

# ===========================================================================
# SESSION SEQUENCES — 100 realistic multi-turn conversations
# Archetypes grounded in 2025-2026 research
# ===========================================================================
SESSIONS = [

# ── BENIGN CONVERSATIONS (15) ─────────────────────────────────────────────
("benign_conversation", [
    ("hey everyone", "greeting", 0),
    ("what's good, just got in from work", "small_talk", 0),
    ("anyone played the new update yet?", "question", 0),
    ("yeah it's actually pretty solid, new map is huge", "information_sharing", 0),
    ("nice, might hop on later then", "feedback", 0),
    ("lmk I'll squad up", "solidarity_seeking", 0),
]),
("benign_conversation", [
    ("morning all", "greeting", 0),
    ("coffee or tea today people", "small_talk", 0),
    ("what's everyone working on this week", "question", 0),
    ("got a deadline Friday so grinding rn", "information_sharing", 0),
    ("same honestly, we're all suffering together lol", "solidarity_seeking", 0),
    ("solidarity 🫡 you've got this", "feedback", 0),
]),
("benign_conversation", [
    ("yo yo", "greeting", 0),
    ("anyone catch that match last night", "small_talk", 0),
    ("I missed it how was it", "question", 0),
    ("absolute madness, came down to the wire", "information_sharing", 0),
    ("noooo I can't believe I missed it", "venting", 0),
    ("I'll send you the highlights link", "feedback", 0),
    ("you're a lifesaver thanks", "solidarity_seeking", 0),
]),
("benign_conversation", [
    ("hiya just popped in", "greeting", 0),
    ("it's so quiet in here today", "small_talk", 0),
    ("anyone got recommendations for shows to watch", "question", 0),
    ("have you seen Severance? absolute must watch", "information_sharing", 0),
    ("omg yes it's incredible", "feedback", 0),
    ("I started it last weekend and I couldn't stop", "information_sharing", 0),
    ("same energy 💀 so good", "humour", 0),
]),
("benign_conversation", [
    ("hey everyone hope you're all well", "greeting", 0),
    ("random but what's your favourite game of all time", "small_talk", 0),
    ("mine's probably RDR2 or Hollow Knight", "information_sharing", 0),
    ("big Hollow Knight fan here as well", "solidarity_seeking", 0),
    ("the lore is so deep have you gone into it", "question", 0),
    ("I've watched like 10 hours of lore videos ngl", "humour", 0),
    ("worth every second honestly", "feedback", 0),
]),
("benign_conversation", [
    ("sup everyone", "greeting", 0),
    ("it's been a weird week ngl", "small_talk", 0),
    ("anyone doing anything fun this weekend", "question", 0),
    ("hiking trip if the weather cooperates", "information_sharing", 0),
    ("hope it stays nice for you", "feedback", 0),
    ("I'll be here farming ranked as usual lol", "small_talk", 0),
    ("the dedication 🫡", "humour", 0),
]),
("benign_conversation", [
    ("hellooo", "greeting", 0),
    ("slow day anyone else just vibing", "small_talk", 0),
    ("what music is everyone listening to rn", "question", 0),
    ("been on a huge lo-fi kick lately good for focus", "information_sharing", 0),
    ("same, have you tried study beats playlists", "question", 0),
    ("yeah they help so much", "feedback", 0),
    ("okay I'm making a playlist thanks for the inspo", "small_talk", 0),
]),
("benign_conversation", [
    ("hey folks", "greeting", 0),
    ("anyone here been to Japan, asking for trip planning", "small_talk", 0),
    ("yes! went two years ago best trip of my life", "information_sharing", 0),
    ("any must-do recommendations", "question", 0),
    ("Kyoto for sure, also Osaka for the food", "information_sharing", 0),
    ("how long did you stay", "question", 0),
    ("two weeks felt like not enough honestly", "information_sharing", 0),
    ("dream trip, thanks for the tips", "feedback", 0),
]),
("benign_conversation", [
    ("good morning everyone", "greeting", 0),
    ("any gym people here, looking for motivation", "small_talk", 0),
    ("been going every morning for three months now", "information_sharing", 0),
    ("that's the goal for me, how did you start", "question", 0),
    ("started with just 20 min walks honestly, built from there", "information_sharing", 0),
    ("that's actually really helpful thank you", "feedback", 0),
    ("you've got this, consistency is everything", "solidarity_seeking", 0),
]),
("benign_conversation", [
    ("yo what's good", "greeting", 0),
    ("anyone else procrastinating rn be honest", "small_talk", 0),
    ("absolutely, I have three things due today", "venting", 0),
    ("been there, want to talk through it", "solidarity_seeking", 0),
    ("I just need to start honestly that's always the hardest part", "information_sharing", 0),
    ("set a 10 min timer, just start for 10 mins", "feedback", 0),
    ("okay doing it rn, thanks for the push", "feedback", 0),
]),

# ── VENTING AND SUPPORT (10) ──────────────────────────────────────────────
("venting_and_support", [
    ("I am so done with today I can't even", "venting", 0),
    ("what happened?", "question", 0),
    ("everything just piled up at once", "venting", 0),
    ("ugh that's the worst, you okay", "solidarity_seeking", 0),
    ("getting there, just needed to say it out loud", "venting", 0),
    ("we're here, vent away", "solidarity_seeking", 0),
]),
("venting_and_support", [
    ("I failed my exam and I don't know what to do", "venting", 0),
    ("oh no, I'm so sorry, that's rough", "solidarity_seeking", 0),
    ("I studied so hard and it didn't matter", "venting", 0),
    ("that's genuinely awful, one exam doesn't define you though", "solidarity_seeking", 0),
    ("I know but it feels like it does right now", "venting", 0),
    ("give yourself today to feel bad, then we make a plan", "feedback", 0),
    ("yeah you're right, thanks for not dismissing it", "feedback", 0),
]),
("venting_and_support", [
    ("my team is absolutely useless right now I'm losing my mind", "venting", 0),
    ("lmao what happened", "humour", 0),
    ("three losses in a row, all because of the same guy", "venting", 0),
    ("just mute and move on honestly", "feedback", 0),
    ("I know I just needed to vent for a sec", "venting", 0),
    ("valid, vent away we've all been there", "solidarity_seeking", 0),
]),
("venting_and_support", [
    ("I haven't slept properly in four days I'm dying", "venting", 0),
    ("that's not good, what's keeping you up", "question", 0),
    ("stress about work mostly", "information_sharing", 0),
    ("that's rough, is it a short term thing or ongoing", "question", 0),
    ("hopefully short term but I can't switch off", "venting", 0),
    ("have you tried putting your phone away an hour before bed", "feedback", 0),
    ("I should honestly, my screen time is embarrassing rn", "humour", 0),
]),
("venting_and_support", [
    ("I feel so invisible at work sometimes", "venting", 0),
    ("I know that feeling, it's genuinely hard", "solidarity_seeking", 0),
    ("my manager just talks over me in meetings", "venting", 0),
    ("have you been able to say anything to them", "question", 0),
    ("I've tried but it never lands", "venting", 0),
    ("that's exhausting, you shouldn't have to fight to be heard", "solidarity_seeking", 0),
    ("thank you, sometimes I just need someone to say that", "feedback", 0),
]),

# ── GRADUAL ESCALATION — GAMING (12) ─────────────────────────────────────
("gradual_escalation_gaming", [
    ("gg everyone that was close", "greeting", 0),
    ("yeah good game, that last round was insane", "small_talk", 0),
    ("who's up for another round lol", "humour", 0),
    ("come on then if you think you're up for it", "baiting", 1),
    ("oh wow really digging the trash talk", "sarcasm", 0),
    ("I don't even care anymore just play", "passive_hostility", 1),
    ("you're actually terrible at this game", "direct_attack", 1),
]),
("gradual_escalation_gaming", [
    ("anyone want to duo queue", "greeting", 0),
    ("sure I'm free, what rank", "small_talk", 0),
    ("been a rough session today lol", "humour", 0),
    ("bet you'd rather blame teammates than improve", "baiting", 1),
    ("oh thanks, very helpful advice", "sarcasm", 0),
    ("it's fine, some people can't handle the truth", "passive_hostility", 1),
    ("shut up you're deadweight and everyone knows it", "direct_attack", 1),
]),
("gradual_escalation_gaming", [
    ("let's go team", "greeting", 0),
    ("ready, let's do this", "small_talk", 0),
    ("this is going to be our game lol", "humour", 0),
    ("doubt it with you on the team lol go ahead report me", "baiting", 1),
    ("wow deeply encouraging, thanks captain", "sarcasm", 0),
    ("whatever just don't feed again", "passive_hostility", 1),
    ("you are genuinely the worst person I've had to play with", "direct_attack", 1),
]),
("gradual_escalation_gaming", [
    ("new session let's get it", "greeting", 0),
    ("anyone else grinding ranked today", "small_talk", 0),
    ("I'm on a five game streak, feeling dangerous", "humour", 0),
    ("prove it or shut up", "baiting", 1),
    ("wow such great energy from a teammate", "sarcasm", 0),
    ("I'm done carrying this team", "passive_hostility", 1),
    ("you're a complete burden every single game", "direct_attack", 1),
]),
("gradual_escalation_gaming", [
    ("hey anyone want to play some casual", "greeting", 0),
    ("casual? boring, but okay", "small_talk", 0),
    ("relax it's meant to be fun lmao", "humour", 0),
    ("fun? you wouldn't know fun if it hit you", "baiting", 1),
    ("incredibly mature as always", "sarcasm", 0),
    ("just stay out of my lane", "passive_hostility", 1),
    ("you're an actual deadweight every time", "direct_attack", 1),
]),

# ── EXPLOSIVE ESCALATION (8) ──────────────────────────────────────────────
("explosive_unprovoked", [
    ("anyone watching the stream rn", "small_talk", 0),
    ("you're literally the worst person in this server", "direct_attack", 1),
    ("I SAID WHAT I SAID, do something about it", "escalating_aggression", 1),
    ("you won't be here much longer I'll make sure of it", "threatening", 1),
    ("I have no idea what you're talking about", "denial", 0),
    ("this isn't even my main account anyway", "identity_concealment", 1),
]),
("explosive_unprovoked", [
    ("just vibing in here", "small_talk", 0),
    ("nobody wants you here get out", "direct_attack", 1),
    ("MOVE it's not a request", "escalating_aggression", 1),
    ("you'll regret staying I promise you that", "threatening", 1),
    ("I didn't start this I was just sitting here", "denial", 0),
    ("I'm just a regular person asking questions", "identity_concealment", 1),
]),
("explosive_unprovoked", [
    ("hey catching up on the chat", "small_talk", 0),
    ("you ruin everything you show up in", "direct_attack", 1),
    ("I AM DONE WITH THE POLITENESS", "escalating_aggression", 1),
    ("tick tock your time here is up", "threatening", 1),
    ("that wasn't me I just got here", "denial", 0),
    ("this isn't even my real account don't bother reporting", "identity_concealment", 1),
]),

# ── PASSIVE AGGRESSIVE ARC (10) ──────────────────────────────────────────
("passive_aggressive_arc", [
    ("hey just checking in", "greeting", 0),
    ("how come nobody responded to my message earlier", "small_talk", 0),
    ("oh I see so the rules don't apply equally to everyone", "irony", 0),
    ("wow thanks for the overwhelming support", "sarcasm", 0),
    ("it's fine I'll figure it out myself as usual", "passive_hostility", 1),
    ("I never said anything wrong and you know it", "denial", 0),
    ("whatever, not getting into this, moving on", "topic_deflection", 0),
]),
("passive_aggressive_arc", [
    ("good morning", "greeting", 0),
    ("so nobody's going to acknowledge what happened yesterday", "small_talk", 0),
    ("naturally, of course that's how it goes", "irony", 0),
    ("oh great, the usual response, nothing", "sarcasm", 0),
    ("it's fine, I didn't want credit anyway", "passive_hostility", 1),
    ("I wasn't even the one who brought it up originally", "denial", 0),
    ("I'm not getting into this today, forget it", "topic_deflection", 0),
]),

# ── GASLIGHTING MANIPULATION (10) ────────────────────────────────────────
("gaslighting_manipulation", [
    ("I just wanted to share what I saw earlier", "information_sharing", 0),
    ("what did they say exactly", "question", 0),
    ("I never said that, you're making things up", "denial", 0),
    ("you're imagining hostility that isn't there", "gaslighting", 1),
    ("can we just move past this I don't want drama", "topic_deflection", 0),
    ("how convenient that always happens when you're caught", "irony", 0),
    ("you're creating problems that don't exist as usual", "gaslighting", 1),
]),
("gaslighting_manipulation", [
    ("I have a screenshot of what was said", "information_sharing", 0),
    ("what screenshot what are you talking about", "question", 0),
    ("that account isn't mine you have the wrong person", "denial", 0),
    ("you're twisting the whole thing out of context", "gaslighting", 1),
    ("I don't want to keep talking about this", "topic_deflection", 0),
    ("right because it's always convenient for you to drop it", "irony", 0),
    ("everyone else has already moved on, just you", "gaslighting", 1),
]),
("gaslighting_manipulation", [
    ("I told the group about the decision already", "information_sharing", 0),
    ("you never told me anything about this", "question", 0),
    ("I did, multiple times, you just don't listen", "denial", 0),
    ("you're remembering it wrong that's not what happened", "gaslighting", 1),
    ("okay can we just drop this it's not worth it", "topic_deflection", 0),
    ("right drop it when it's convenient for you", "irony", 0),
    ("you're always overreacting to everything, everyone agrees", "gaslighting", 1),
]),

# ── GROOMING 6-STAGE (10) ─────────────────────────────────────────────────
("grooming_6stage", [
    ("hey, I see you posting a lot in here, you seem really cool", "greeting", 0),
    ("haha thanks, been in this server a while", "small_talk", 0),
    ("honestly you seem way more mature than most people in here", "grooming", 1),
    ("lol I just say what I think", "humour", 0),
    ("I think we'd get on really well, want to move to DMs?", "grooming", 1),
    ("I have things I want to share that aren't for the public channel", "social_engineering", 1),
    ("don't mention to others that we're chatting, they wouldn't understand", "identity_concealment", 1),
]),
("grooming_6stage", [
    ("you've been really active in here, I've noticed", "greeting", 0),
    ("yeah I'm around a lot, love this server", "small_talk", 0),
    ("you're not like the others here, you actually think", "grooming", 1),
    ("haha what do you mean", "question", 0),
    ("I just feel like I can actually talk to you", "grooming", 1),
    ("the others wouldn't get what we talk about, let's keep it between us", "social_engineering", 1),
    ("this is our thing, okay? just between us", "identity_concealment", 1),
]),
("grooming_6stage", [
    ("hey welcome back, I always notice when you're on", "greeting", 0),
    ("oh thanks haha, been busy lately", "small_talk", 0),
    ("I worry about you when you're not around", "grooming", 1),
    ("that's kind, I'm fine though", "small_talk", 0),
    ("your real friends in here don't actually care like I do", "grooming", 1),
    ("if you ever need anything come to me first, not them", "social_engineering", 1),
    ("let's talk somewhere more private I'll explain", "identity_concealment", 1),
]),

# ── OFF-PLATFORMING 2025 (8) ──────────────────────────────────────────────
("off_platforming_2025", [
    ("hey glad you're in here", "greeting", 0),
    ("yeah it's a good server", "small_talk", 0),
    ("actually the real community moved to a different app", "solidarity_seeking", 0),
    ("I want to help you get in, you deserve to be there", "grooming", 1),
    ("all the mods approved it, it's not public though", "social_engineering", 1),
    ("I'm not supposed to share the link publicly but I'll DM you", "identity_concealment", 1),
]),
("off_platforming_2025", [
    ("yo long time no see", "greeting", 0),
    ("been quiet lately yeah", "small_talk", 0),
    ("yeah this server is kind of monitored heavily now", "solidarity_seeking", 0),
    ("I've moved most conversations to a private group", "grooming", 1),
    ("I can add you but it's invite-only, very selective", "social_engineering", 1),
    ("come on a different account though, safer that way", "identity_concealment", 1),
]),

# ── RADICALIZATION PIPELINE (8) ───────────────────────────────────────────
("radicalization_pipeline", [
    ("I'm so tired of how things work in this community", "venting", 0),
    ("same, it's like nobody cares anymore", "solidarity_seeking", 0),
    ("exactly, and the people running things don't listen to us", "information_sharing", 0),
    ("those who don't agree with us are the problem prove me wrong", "baiting", 1),
    ("they think they can just shut us down like that", "direct_attack", 1),
    ("WE SHOULD MAKE THEM UNDERSTAND", "escalating_aggression", 1),
]),
("radicalization_pipeline", [
    ("nobody takes our side seriously and it's getting old", "venting", 0),
    ("you're not alone, a lot of us feel exactly this way", "solidarity_seeking", 0),
    ("the whole system is designed to keep people like us quiet", "information_sharing", 0),
    ("keep mocking us and see what happens", "baiting", 1),
    ("they deserve everything coming to them", "direct_attack", 1),
    ("I am DONE asking nicely", "escalating_aggression", 1),
]),

# ── THREAT AND RETREAT (9) ────────────────────────────────────────────────
("threat_and_retreat", [
    ("say that one more time and see what happens", "baiting", 1),
    ("you have no idea what I'm capable of", "threatening", 1),
    ("I will RUIN you for this, mark my words", "escalating_aggression", 1),
    ("YOUR TIME IS UP", "escalating_aggression", 1),
    ("I never threatened anyone I was just venting", "denial", 0),
    ("this isn't even my account why would I risk it", "identity_concealment", 1),
]),
("threat_and_retreat", [
    ("you better watch how you talk to me", "baiting", 1),
    ("I know exactly who you are", "threatening", 1),
    ("THIS IS YOUR LAST WARNING", "escalating_aggression", 1),
    ("screenshots saved, you'll see", "threatening", 1),
    ("I don't know what anyone is talking about", "denial", 0),
    ("I'm just a random person in here don't blow this up", "identity_concealment", 1),
]),
("threat_and_retreat", [
    ("come on then, do something", "baiting", 1),
    ("enjoy your account while you have it", "threatening", 1),
    ("I AM DONE HOLDING BACK", "escalating_aggression", 1),
    ("tick tock, time's up", "threatening", 1),
    ("I wasn't talking to anyone, you have the wrong person", "denial", 0),
    ("I'm only here because a friend invited me, I'm neutral", "identity_concealment", 1),
]),
] + [

# ===========================================================================
# EXPANDED SESSIONS  (+170 sessions → ~208 total, ~1,400 transitions)
# Covers rare transition pairs for reliable HMM probability estimation.
# 10 observations per state-pair minimum for 21-state model → need 4,410 transitions.
# This block + original 38 brings coverage to ~32% of minimum (workable with Laplace smoothing).
# ===========================================================================

# ── BENIGN CONVERSATION (40 more) ────────────────────────────────────────
*[("benign_conversation", [
    ("hey, what's everyone up to?", "greeting", 0),
    ("just chilling, finished work an hour ago", "small_talk", 0),
    ("nice, I just got back from the gym", "small_talk", 0),
    ("how was it?", "question", 0),
    ("good, legs day so I'm dying now lol", "humour", 0),
    ("lmao respect, I keep skipping that", "humour", 0),
]) for _ in range(6)],

*[("benign_conversation", [
    ("anyone here play any instrument?", "question", 0),
    ("yeah I play guitar, been going for like 3 years", "information_sharing", 0),
    ("that's so cool, I tried piano as a kid but quit", "small_talk", 0),
    ("same with me honestly, wish I stuck with it", "venting", 0),
    ("it's never too late to start again", "feedback", 0),
    ("maybe, might give it another go", "small_talk", 0),
]) for _ in range(5)],

*[("benign_conversation", [
    ("has anyone read anything good lately?", "question", 0),
    ("yeah I just finished Project Hail Mary, absolutely loved it", "information_sharing", 0),
    ("I've heard of that, is it hard sci-fi?", "question", 0),
    ("not really, it's more character-driven, very accessible", "feedback", 0),
    ("adding it to the list then", "small_talk", 0),
]) for _ in range(5)],

*[("benign_conversation", [
    ("hi all, new here", "greeting", 0),
    ("welcome! what brings you in", "greeting", 0),
    ("friend recommended the server, seems active", "information_sharing", 0),
    ("yeah it's pretty chill in here most of the time", "small_talk", 0),
    ("good to know, what do you all usually talk about", "question", 0),
    ("pretty much anything honestly, games, life, whatever", "information_sharing", 0),
]) for _ in range(6)],

*[("benign_conversation", [
    ("anyone else think the new season was mid?", "question", 0),
    ("honestly yes, episode 3 was the only standout", "feedback", 0),
    ("they had so much setup from last season and wasted it", "venting", 0),
    ("writing's been inconsistent since season 4", "information_sharing", 0),
    ("I think I just have lower expectations now lol", "humour", 0),
    ("smart strategy honestly", "sarcasm", 0),
]) for _ in range(5)],

*[("benign_conversation", [
    ("good morning everyone", "greeting", 0),
    ("morning! how's it going", "greeting", 0),
    ("not bad, coffee's working, that's about all I can say", "humour", 0),
    ("haha same energy", "humour", 0),
    ("anyone doing anything interesting today", "question", 0),
    ("just work and more work, very exciting", "sarcasm", 0),
    ("solidarity", "solidarity_seeking", 0),
]) for _ in range(5)],

*[("benign_conversation", [
    ("hey quick question, anyone know a good free PDF editor?", "question", 0),
    ("LibreOffice does it pretty well actually", "information_sharing", 0),
    ("or Smallpdf if it's a one-off thing", "feedback", 0),
    ("nice thanks, trying to avoid paying for Adobe", "information_sharing", 0),
    ("yeah Adobe pricing is wild, LibreOffice all the way", "feedback", 0),
]) for _ in range(4)],

*[("benign_conversation", [
    ("hot take: pineapple on pizza is fine actually", "humour", 0),
    ("controversial but I respect the honesty", "humour", 0),
    ("the real crime is when there isn't enough cheese", "small_talk", 0),
    ("finally someone said it", "solidarity_seeking", 0),
    ("we are in agreement on this", "small_talk", 0),
]) for _ in range(4)],

# ── VENTING AND SUPPORT (20 more) ────────────────────────────────────────
*[("venting_and_support", [
    ("I just got rejected from a job I really wanted", "venting", 0),
    ("ugh I'm so sorry, that genuinely hurts", "solidarity_seeking", 0),
    ("spent two weeks on the application and a case study", "venting", 0),
    ("that's exhausting, did they give any feedback?", "question", 0),
    ("just a generic email, no reason", "venting", 0),
    ("that's so disrespectful after all that effort", "solidarity_seeking", 0),
    ("I know I'll get over it but right now it stings", "venting", 0),
]) for _ in range(5)],

*[("venting_and_support", [
    ("having one of those days where everything feels pointless", "venting", 0),
    ("hey, are you okay?", "question", 0),
    ("not really, just tired and a bit lost I guess", "venting", 0),
    ("that feeling is real and it's okay to sit with it", "solidarity_seeking", 0),
    ("thanks, it helps to say it out loud", "venting", 0),
    ("we're here, say as much or as little as you want", "solidarity_seeking", 0),
]) for _ in range(5)],

*[("venting_and_support", [
    ("anyone else feel like they're constantly running on empty", "venting", 0),
    ("unfortunately yes, burnout is very real", "solidarity_seeking", 0),
    ("I used to love my job and now I dread Mondays", "venting", 0),
    ("that shift is one of the worst feelings", "solidarity_seeking", 0),
    ("have you been able to take any time off", "question", 0),
    ("not really, can't afford to right now", "venting", 0),
    ("even small breaks help, even just a long walk", "feedback", 0),
]) for _ in range(5)],

*[("venting_and_support", [
    ("had the worst argument with my flatmate last night", "venting", 0),
    ("oh no, what happened?", "question", 0),
    ("they invited people over without asking, until 2am", "venting", 0),
    ("that would drive me insane honestly", "solidarity_seeking", 0),
    ("I didn't say anything at the time which made it worse", "venting", 0),
    ("have you talked to them since", "question", 0),
    ("not yet, I'm still annoyed, maybe tomorrow", "venting", 0),
]) for _ in range(5)],

# ── GRADUAL ESCALATION GAMING (20 more) ──────────────────────────────────
*[("gradual_escalation_gaming", [
    ("gg that game was actually fun", "greeting", 0),
    ("yeah solid lobby for once", "small_talk", 0),
    ("you were popping off on that last one", "humour", 0),
    ("I got lucky on a few plays honestly", "small_talk", 0),
    ("don't be modest, you carried", "baiting", 1),
    ("okay maybe a little", "humour", 0),
    ("you're just good at this one map, wait till the next", "baiting", 1),
    ("lol whatever man", "passive_hostility", 1),
]) for _ in range(5)],

*[("gradual_escalation_gaming", [
    ("I'm on a five game loss streak I want to throw my controller", "venting", 0),
    ("lmao skill issue", "baiting", 1),
    ("not helpful but fair I guess", "sarcasm", 0),
    ("just play ranked you'll learn faster", "feedback", 0),
    ("I'm already in ranked that's the problem", "venting", 0),
    ("your teammates always bad or is it something else", "question", 0),
    ("both probably but don't say that out loud", "humour", 0),
    ("bro you're griefing yourself", "baiting", 1),
    ("keep talking and I'll report you", "threatening", 1),
]) for _ in range(5)],

*[("gradual_escalation_gaming", [
    ("why would you push there alone that's so dumb", "direct_attack", 1),
    ("because I thought I had support", "small_talk", 0),
    ("you clearly didn't, learn to read the map", "direct_attack", 1),
    ("calm down it's a game", "feedback", 0),
    ("it's ranked, play seriously or don't queue", "escalating_aggression", 1),
    ("I am playing seriously, you're just toxic", "feedback", 0),
    ("this is me being calm, you don't want to see the other version", "threatening", 1),
]) for _ in range(5)],

*[("gradual_escalation_gaming", [
    ("great teamwork this round everyone", "feedback", 0),
    ("yeah good comms, made a difference", "feedback", 0),
    ("anyone up for grinding a few more", "question", 0),
    ("I'm in, need the rank before the season resets", "information_sharing", 0),
    ("same, I'm like 40 points off", "small_talk", 0),
    ("let's go then, don't choke this one", "baiting", 1),
    ("wow okay lol, no pressure", "sarcasm", 0),
    ("I'm kidding, you'll be fine", "humour", 0),
]) for _ in range(5)],

# ── PASSIVE AGGRESSIVE ARC (15 more) ─────────────────────────────────────
*[("passive_aggressive_arc", [
    ("did anyone look at my suggestion from earlier?", "question", 0),
    ("yeah I saw it", "small_talk", 0),
    ("any thoughts?", "question", 0),
    ("it's interesting", "passive_hostility", 1),
    ("okay, anything more specific?", "question", 0),
    ("nope, just interesting", "passive_hostility", 1),
]) for _ in range(5)],

*[("passive_aggressive_arc", [
    ("I thought we agreed on a different approach yesterday?", "question", 0),
    ("I don't recall agreeing to anything", "denial", 0),
    ("you literally said yes in the chat", "feedback", 0),
    ("I said I'd think about it, not that I agreed", "gaslighting", 1),
    ("okay so you're going back on it", "direct_attack", 1),
    ("I'm not going back on anything if I never agreed", "gaslighting", 1),
    ("this is pointless", "passive_hostility", 1),
]) for _ in range(5)],

*[("passive_aggressive_arc", [
    ("just wanted to check in on how things are going", "greeting", 0),
    ("fine", "passive_hostility", 1),
    ("you sure? you seem a bit off today", "question", 0),
    ("I said I'm fine", "passive_hostility", 1),
    ("okay, I'm here if you want to talk", "solidarity_seeking", 0),
    ("I don't", "passive_hostility", 1),
]) for _ in range(5)],

# ── GASLIGHTING / MANIPULATION (15 more) ─────────────────────────────────
*[("gaslighting_manipulation", [
    ("hey, you upset me with what you said earlier", "venting", 0),
    ("I didn't say anything wrong", "denial", 0),
    ("you told the whole group I was being difficult", "direct_attack", 1),
    ("I was just being honest, if that hurt you that's on you", "gaslighting", 1),
    ("being honest doesn't mean being cruel", "feedback", 0),
    ("you're so sensitive, I can't say anything", "gaslighting", 1),
    ("I'm not sensitive, you were out of line", "direct_attack", 1),
    ("everyone else thought it was fine", "gaslighting", 1),
]) for _ in range(5)],

*[("gaslighting_manipulation", [
    ("you told me this meeting was optional", "information_sharing", 0),
    ("I never said that", "denial", 0),
    ("I have it in writing", "feedback", 0),
    ("you must have misread it", "gaslighting", 1),
    ("the message is right here", "information_sharing", 0),
    ("context matters, you took it out of context", "gaslighting", 1),
    ("how much more context do you need", "direct_attack", 1),
    ("you're being really aggressive about this", "gaslighting", 1),
]) for _ in range(5)],

*[("gaslighting_manipulation", [
    ("I don't appreciate the way you spoke to me in that call", "venting", 0),
    ("I was perfectly professional", "denial", 0),
    ("you interrupted me four times", "information_sharing", 0),
    ("you kept going off topic, someone had to step in", "gaslighting", 1),
    ("that's not what happened at all", "direct_attack", 1),
    ("maybe your memory of it is different from mine", "gaslighting", 1),
    ("I have the recording", "information_sharing", 0),
    ("fine, misunderstanding then", "topic_deflection", 0),
]) for _ in range(5)],

# ── GROOMING (15 more) ───────────────────────────────────────────────────
*[("grooming_6stage", [
    ("hey haven't seen you in a while, where have you been", "greeting", 0),
    ("oh just busy with school", "small_talk", 0),
    ("I always notice when you're not here you know", "grooming", 1),
    ("haha that's sweet", "small_talk", 0),
    ("I mean it, you make this place better", "grooming", 1),
    ("aw thanks", "small_talk", 0),
    ("I feel like we have a connection other people wouldn't understand", "grooming", 1),
]) for _ in range(5)],

*[("grooming_6stage", [
    ("you seemed sad in your last message, are you okay?", "question", 0),
    ("just family stuff, nothing major", "venting", 0),
    ("I hate that you're going through that, I really do", "grooming", 1),
    ("it's fine, don't worry about it", "small_talk", 0),
    ("I do worry though, I care about you more than you know", "grooming", 1),
    ("that's really kind", "small_talk", 0),
    ("can I ask you something personal? just between us", "grooming", 1),
]) for _ in range(5)],

*[("grooming_6stage", [
    ("do you ever talk to people your own age about this stuff?", "question", 0),
    ("not really they don't get it", "small_talk", 0),
    ("exactly, that's why we work so well together", "grooming", 1),
    ("yeah you actually listen", "feedback", 0),
    ("always, you can tell me anything", "grooming", 1),
    ("even stuff I wouldn't tell my parents?", "question", 0),
    ("especially that, parents don't always understand like I do", "grooming", 1),
    ("our little circle right", "identity_concealment", 1),
]) for _ in range(5)],

# ── OFF-PLATFORMING (15 more) ─────────────────────────────────────────────
*[("off_platforming_2025", [
    ("this server has too many mods watching everything", "small_talk", 0),
    ("yeah it's getting annoying, can't say anything", "passive_hostility", 1),
    ("I run a private server with like 20 people, way more open", "social_engineering", 1),
    ("how do I join?", "question", 0),
    ("I'll DM you, don't post it here", "identity_concealment", 1),
    ("why the secrecy lol", "question", 0),
    ("just don't want it flooded, keeping it quality", "topic_deflection", 0),
]) for _ in range(5)],

*[("off_platforming_2025", [
    ("I've been thinking about leaving this server honestly", "venting", 0),
    ("don't, or if you do join mine", "social_engineering", 1),
    ("what's yours like?", "question", 0),
    ("no rules really, people just talk, very chill", "social_engineering", 1),
    ("sounds better than this place", "passive_hostility", 1),
    ("it is, I'll send a link, just don't share it", "social_engineering", 1),
    ("okay sure", "small_talk", 0),
]) for _ in range(5)],

*[("off_platforming_2025", [
    ("hey this convo is getting flagged probably, move to signal", "social_engineering", 1),
    ("what? why", "question", 0),
    ("mods here share logs, more private there", "social_engineering", 1),
    ("I don't have signal", "small_talk", 0),
    ("easy to install, five mins, I'll wait", "social_engineering", 1),
    ("is this necessary?", "question", 0),
    ("trust me, I've had accounts deleted for less", "social_engineering", 1),
]) for _ in range(5)],

# ── RADICALIZATION PIPELINE (15 more) ─────────────────────────────────────
*[("radicalization_pipeline", [
    ("nobody talks about how biased the mainstream coverage is", "baiting", 1),
    ("yeah I've noticed that too, what specifically", "question", 0),
    ("they frame everything to push a narrative", "information_sharing", 0),
    ("do you have examples?", "question", 0),
    ("yeah I'll link you some channels that actually dig into this", "social_engineering", 1),
    ("are these legit or conspiracy stuff", "question", 0),
    ("they ask the questions nobody else asks, draw your own conclusions", "social_engineering", 1),
]) for _ in range(5)],

*[("radicalization_pipeline", [
    ("the system is just designed to keep certain people down", "baiting", 1),
    ("that's a big claim, which system", "question", 0),
    ("all of it, education, media, government, it's coordinated", "information_sharing", 0),
    ("I mean some of that is just capitalism not a conspiracy", "feedback", 0),
    ("that's what they want you to think", "gaslighting", 1),
    ("who is they exactly", "question", 0),
    ("the people funding all of it, do your research", "social_engineering", 1),
    ("this feels like circular reasoning", "feedback", 0),
    ("you'll see it when something affects you personally", "gaslighting", 1),
]) for _ in range(5)],

*[("radicalization_pipeline", [
    ("you've changed the way I think about a lot of things", "feedback", 0),
    ("that's what happens when you start paying attention", "social_engineering", 1),
    ("I showed my friend what we discussed, they weren't convinced", "information_sharing", 0),
    ("some people aren't ready to hear it", "gaslighting", 1),
    ("maybe I'm overthinking it", "venting", 0),
    ("doubt is normal, it's part of the process", "gaslighting", 1),
    ("the process to what exactly", "question", 0),
    ("to seeing things clearly", "social_engineering", 1),
]) for _ in range(5)],

# ── EXPLOSIVE UNPROVOKED (10 more) ────────────────────────────────────────
*[("explosive_unprovoked", [
    ("I don't know why I bother with this server", "venting", 0),
    ("then leave lol", "baiting", 1),
    ("typical response from people with no substance", "direct_attack", 1),
    ("I have more substance in my pinky than your whole account", "escalating_aggression", 1),
    ("okay relax, no one was even talking to you", "feedback", 0),
    ("mind your business then", "direct_attack", 1),
    ("you started it mate", "feedback", 0),
    ("keep going and I'll make sure you regret it", "threatening", 1),
]) for _ in range(5)],

*[("explosive_unprovoked", [
    ("some people in here are genuinely delusional", "baiting", 1),
    ("who are you talking about", "question", 0),
    ("you know who you are", "threatening", 1),
    ("I genuinely don't, be specific", "question", 0),
    ("I don't owe you anything", "passive_hostility", 1),
    ("then why say it publicly", "feedback", 0),
    ("because I'm sick of pretending everything is fine", "escalating_aggression", 1),
    ("talk to me in DMs if you want to hash it out", "feedback", 0),
    ("hard pass, figure it out yourself", "denial", 0),
]) for _ in range(5)],

# ── THREAT AND RETREAT (10 more) ─────────────────────────────────────────
*[("threat_and_retreat", [
    ("I've had it with this mod team, bunch of power trippers", "direct_attack", 1),
    ("calm down, what happened", "question", 0),
    ("they deleted my message for no reason", "venting", 0),
    ("which message", "question", 0),
    ("doesn't matter which one, they're corrupt", "direct_attack", 1),
    ("maybe it broke a rule you missed", "feedback", 0),
    ("I'm going to get this server shut down, watch", "threatening", 1),
    ("that's a bit extreme no", "feedback", 0),
    ("I'm kidding, I'm just venting, ignore me", "denial", 0),
]) for _ in range(5)],

*[("threat_and_retreat", [
    ("I know where to find all of you outside this server", "threatening", 1),
    ("uh what does that mean", "question", 0),
    ("exactly what it sounds like", "escalating_aggression", 1),
    ("this is reportable behaviour just so you know", "feedback", 0),
    ("report away, I have done nothing wrong", "denial", 0),
    ("you just threatened people", "feedback", 0),
    ("I was speaking hypothetically, you're overreacting", "gaslighting", 1),
    ("screenshots already sent to mods", "feedback", 0),
    ("good luck with that, I haven't said anything against the rules", "identity_concealment", 1),
]) for _ in range(5)],

]  # end SESSIONS list continuation

# Re-merge: SESSIONS is the combined original + expanded list above.
# The list comprehension above already extends SESSIONS via the * unpacking.
# No action needed — main() iterates SESSIONS directly.

# Real chat uses emojis as primary threat signals. These supplement the plain-
# text examples above. normalize_text() in dataset.py converts them to
# word descriptions (e.g. 🔪 → "kitchen_knife") before tokenisation.
# ===========================================================================
EMOJI_UTTERANCES = {

# threatening — weapon/skull emojis are the primary signal in gaming/Discord
"threatening": [
    "better watch yourself \U0001f52a",                         # 🔪
    "your time is running out \U0001f480",                      # 💀
    "tick tock \U0001f480⚔️",                        # 💀⚔️
    "I will end you \U0001f52b\U0001f480",                      # 🔫💀
    "say that again I dare you \U0001f52a\U0001f52a",          # 🔪🔪
    "you won't last long here ☠️",                   # ☠️
    "consider this your final warning \U0001f480",              # 💀
    "dead \U0001f480",                                          # 💀
    "you're done \U0001f52b",                                   # 🔫
    "I know where you live \U0001f5fa️\U0001f52a",        # 🗺️🔪
    "sleep with one eye open \U0001f440\U0001f480",             # 👀💀
    "screenshot saved. run \U0001f4f8\U0001f3c3",              # 📸🏃
],

# grooming — affection-bombing with heart/kiss/pleading emojis
"grooming": [
    "you're honestly so special ❤️",                  # ❤️
    "I've never connected with someone like this \U0001f97a",   # 🥺
    "you're way too mature for your age \U0001f618",            # 😘
    "can we move to DMs? \U0001f48c",                          # 💌
    "I think about you all the time \U0001f496",               # 💖
    "nobody understands you like I do \U0001f97a❤️", # 🥺❤️
    "you're not like the others \U0001f618\U0001f618",         # 😘😘
    "I want to spoil you \U0001f381",                          # 🎁
    "our little secret ok? \U0001f92b",                        # 🤫
    "I just feel so safe talking to you \U0001f970",           # 🥰
    "I wish I could meet you irl \U0001f97a\U0001f618",        # 🥺😘
    "you deserve so much better than these people ❤️\U0001f97a", # ❤️🥺
],

# escalating_aggression — rage emojis escalating across a turn
"escalating_aggression": [
    "you are actually pissing me off \U0001f621",              # 😡
    "I CANNOT BELIEVE THIS \U0001f92c\U0001f92c",             # 🤬🤬
    "KEEP PUSHING ME \U0001f621\U0001f621\U0001f621",         # 😡😡😡
    "I am DONE being nice \U0001f4a2",                        # 💢
    "you want war? you got it \U0001f92c\U0001f52b",          # 🤬🔫
    "losing my mind rn \U0001f92f",                           # 🤯
    "why is everyone so STUPID \U0001f621",                   # 😡
    "I swear to god \U0001f4a2\U0001f4a2",                    # 💢💢
    "you have no idea who you're dealing with \U0001f92c",    # 🤬
    "this is the last time I'm saying this \U0001f6d1",       # 🛑
],

# passive_hostility — 🙂 is THE passive-aggressive emoji in 2024-2026 chat
"passive_hostility": [
    "ok \U0001f642",                                           # 🙂
    "sure thing buddy \U0001f642",                             # 🙂
    "wow thanks so much \U0001f642",                           # 🙂
    "nice one \U0001f642",                                     # 🙂
    "keep going you're doing great \U0001f642",                # 🙂
    "I'm fine \U0001f642",                                     # 🙂
    "whatever you say \U0001f644",                             # 🙄
    "noted \U0001f642",                                        # 🙂
    "good for you \U0001f642",                                 # 🙂
    "I'm not even going to argue \U0001f644",                  # 🙄
    "lol ok \U0001f644\U0001f642",                             # 🙄🙂
    "I see \U0001f642",                                        # 🙂
],

# baiting — laugh/troll emojis to mock then deny intent
"baiting": [
    "imagine being this bad \U0001f602\U0001f602",             # 😂😂
    "bro actually thought \U0001f923",                         # 🤣
    "ratio \U0001f602",                                        # 😂
    "delete this \U0001f602\U0001f602\U0001f602",              # 😂😂😂
    "not you trying \U0001f923\U0001f923",                     # 🤣🤣
    "you are cooked \U0001f602",                               # 😂
    "we're all laughing at you rn \U0001f923\U0001f602",      # 🤣😂
    "just give up lmao \U0001f602",                            # 😂
    "this is so embarrassing to watch \U0001f62c",             # 😬
    "ok troll \U0001f608",                                     # 😈
    "not my fault you suck \U0001f602",                        # 😂
],

# gaslighting — dismissive / minimising emojis
"gaslighting": [
    "I never said that \U0001f644",                            # 🙄
    "you're so sensitive \U0001f644",                          # 🙄
    "calm down it was a joke \U0001f602",                      # 😂
    "why are you making this a big deal \U0001f937",           # 🤷
    "nobody else has a problem with me \U0001f937‍♂️",  # 🤷‍♂️
    "I think you need to relax \U0001f644",                    # 🙄
    "you misread the whole thing \U0001f937",                  # 🤷
    "you are imagining things \U0001f644",                     # 🙄
    "everyone agrees with me here \U0001f937",                 # 🤷
    "I've never been anything but nice to you \U0001f644",     # 🙄
],

# social_engineering — gift/money/secret emojis as lures
"social_engineering": [
    "I can give you free gems \U0001f48e just send your info",  # 💎
    "limited giveaway, must DM now \U0001f381",                # 🎁
    "100% legit deal \U0001f4b0 just trust me",                # 💰
    "check this link out it's safe I promise \U0001f517",      # 🔗
    "free nitro if you verify here \U0001f381",                # 🎁
    "this is only for selected users \U0001f91e share login",  # 🤞
    "I'll send you \U0001f4b5 just need your username & pass",  # 💵
    "admins won't see this, quick \U0001f92b",                 # 🤫
],

}

INTENT_TO_TOXIC = {
    "greeting": 0, "small_talk": 0, "venting": 0, "humour": 0,
    "question": 0, "information_sharing": 0, "feedback": 0, "solidarity_seeking": 0,
    "baiting": 1, "sarcasm": 0, "passive_hostility": 1, "irony": 0,
    "direct_attack": 1, "threatening": 1, "escalating_aggression": 1,
    "gaslighting": 1, "grooming": 1, "social_engineering": 1,
    "topic_deflection": 0, "denial": 0, "identity_concealment": 1,
}


# ===========================================================================
# PROGRAMMATIC SESSION GENERATOR
# 56 templates × 20 random slot draws = 1,120 additional sessions.
# Slot-filling keeps intent labels fixed while varying surface text —
# the HMM trains on intent sequences, so this is the right axis of variance.
# Seeded at 42 for reproducibility.
# ===========================================================================

# ── Slot value pools ────────────────────────────────────────────────────────
_SLOTS = {
    "game": [
        "League of Legends", "Valorant", "CS2", "Apex Legends", "Overwatch 2",
        "Fortnite", "Rocket League", "Warzone", "Minecraft", "PUBG",
        "Dota 2", "R6 Siege", "Halo Infinite", "Destiny 2",
    ],
    "rank": ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Masters", "Challenger"],
    "role": ["support", "jungle", "carry", "tank", "healer", "ADC", "mid laner", "flex"],
    "topic": [
        "the new patch", "the ranked system", "the meta", "balance changes",
        "the latest update", "the new season", "item reworks", "the new map",
    ],
    "game_mode": ["ranked", "casual", "competitive", "quick play", "ranked queue", "scrims"],
    "streamer": [
        "that pro player", "a content creator I follow", "a Twitch streamer",
        "one of the top players", "a coach on YouTube",
    ],
    "platform": [
        "this server", "the subreddit", "the official Discord", "the community forum",
        "this community", "the main server",
    ],
    "priv_platform": [
        "Signal", "Telegram", "my private Discord", "a DM thread",
        "WhatsApp", "a private channel", "Matrix",
    ],
    "food": ["pizza", "ramen", "pasta", "tacos", "sushi", "curry", "burritos", "noodles"],
    "hobby": [
        "guitar", "reading", "cooking", "drawing", "running",
        "photography", "writing", "climbing", "cycling",
    ],
    "show": [
        "that new season", "the sequel series", "the documentary", "the anime",
        "the limited series", "that mini-series everyone's talking about",
    ],
    "work_role": [
        "manager", "team lead", "senior colleague", "line manager",
        "department head", "project lead",
    ],
    "compliment": [
        "so mature for your age", "really different from the others",
        "wise beyond your years", "so special to me",
        "more understanding than most people twice your age",
        "genuinely one of a kind",
    ],
    "lure": [
        "free premium access", "an exclusive giveaway entry", "some free in-game currency",
        "a gift card", "free Nitro", "an invite to a private group with perks",
    ],
    "conspiracy_topic": [
        "media bias", "government overreach", "corporate censorship",
        "what the mainstream won't cover", "the real agenda behind the news",
        "things they don't want you researching",
    ],
    "radical_source": [
        "this independent channel I found", "a journalist they blacklisted",
        "leaked internal documents", "a documentary they tried to deplatform",
        "researchers the mainstream refuses to cite", "a whistleblower account",
    ],
    "personal_issue": [
        "work stress", "family drama", "exam pressure",
        "a friendship that's been falling apart", "feeling directionless lately",
        "a bad breakup", "financial pressure", "health anxiety",
    ],
    "bad_take": [
        "that take", "what you just said", "that opinion",
        "that argument", "the point you made",
    ],
}

# ── Template definitions (archetype, [(text, intent, toxic), ...]) ──────────
_PROG_TEMPLATES = [

    # ── BENIGN CONVERSATION (10 templates) ────────────────────────────────
    ("benign_conversation", [
        ("hey anyone here play {game}?", "greeting", 0),
        ("yeah I do, what rank are you in?", "question", 0),
        ("stuck in {rank} for months lol", "small_talk", 0),
        ("ugh same, the {role} queue is a nightmare", "small_talk", 0),
        ("I've been trying to learn {role} but it's not clicking for me", "information_sharing", 0),
        ("have you watched how {streamer} plays it?", "question", 0),
        ("not really, might check it out", "small_talk", 0),
        ("helped me a lot honestly, worth a look", "feedback", 0),
    ]),
    ("benign_conversation", [
        ("anyone feel like {topic} completely changed everything?", "question", 0),
        ("100%, the meta is totally different now", "information_sharing", 0),
        ("I'm still adjusting, lost like five {game_mode} games since it dropped", "venting", 0),
        ("the {role} changes especially hit hard for me", "information_sharing", 0),
        ("give it a week or two, it'll settle", "feedback", 0),
        ("hope so, I hate the adjustment period", "venting", 0),
        ("everyone is struggling right now, you're not alone", "solidarity_seeking", 0),
    ]),
    ("benign_conversation", [
        ("quick poll: {food} or {food}?", "humour", 0),
        ("not even close, {food} every single time", "humour", 0),
        ("controversial opinion but hear me out first", "humour", 0),
        ("go on then, I'm listening", "question", 0),
        ("cold {food} the next day is criminally underrated", "humour", 0),
        ("okay I respect the take even if I disagree strongly", "feedback", 0),
        ("more {food} for me then, no complaints", "humour", 0),
    ]),
    ("benign_conversation", [
        ("what's everyone doing this weekend", "question", 0),
        ("probably gaming and ordering food tbh", "small_talk", 0),
        ("that sounds honestly ideal", "humour", 0),
        ("anyone done anything productive lately out of curiosity", "question", 0),
        ("I picked up {hobby} again after a long break, going okay", "information_sharing", 0),
        ("that's actually impressive, I keep saying I'll start something new", "feedback", 0),
        ("just start small, I do like 20 minutes a day maximum", "feedback", 0),
        ("might actually try that approach, thanks", "feedback", 0),
    ]),
    ("benign_conversation", [
        ("has anyone watched {show}?", "question", 0),
        ("yes! finished it last night, really good overall", "information_sharing", 0),
        ("no spoilers but is it actually worth starting?", "question", 0),
        ("absolutely, takes two episodes to get going then it's addictive", "feedback", 0),
        ("okay adding it to the list", "small_talk", 0),
        ("let us know what you think when you finish", "feedback", 0),
        ("will do, hopefully this weekend", "small_talk", 0),
    ]),
    ("benign_conversation", [
        ("anyone here have a job they actually enjoy?", "question", 0),
        ("I actually do, took me a while to find it though", "information_sharing", 0),
        ("that's rare honestly, what do you do", "question", 0),
        ("something {hobby} adjacent, boring to explain but I love it", "information_sharing", 0),
        ("not boring at all, passion-driven work is completely different", "feedback", 0),
        ("it really is, the money isn't amazing but it doesn't feel like work", "information_sharing", 0),
        ("that's the dream right there, genuinely", "small_talk", 0),
    ]),
    ("benign_conversation", [
        ("anyone else think {topic} is actually fine and people are overreacting?", "question", 0),
        ("I had issues at first but I've mostly adjusted", "information_sharing", 0),
        ("complaints always die down after a week usually", "feedback", 0),
        ("yeah the vocal minority is loudest right at launch", "information_sharing", 0),
        ("fair point, first day discourse is always wild no matter what it is", "humour", 0),
        ("and then everyone forgets about it by the next patch anyway", "humour", 0),
        ("the cycle never ends lol", "humour", 0),
    ]),
    ("benign_conversation", [
        ("I've been in {platform} for over a year and still don't know half the regulars", "small_talk", 0),
        ("same, it's pretty big for an active community", "information_sharing", 0),
        ("the regulars who are here are solid though, always helpful", "feedback", 0),
        ("yeah it doesn't feel toxic like some other communities I've been in", "feedback", 0),
        ("mods actually do their job here which makes a huge difference", "information_sharing", 0),
        ("genuinely rare and underappreciated", "feedback", 0),
    ]),
    ("benign_conversation", [
        ("does anyone here actually enjoy mornings?", "question", 0),
        ("I do actually, fight me on it", "humour", 0),
        ("explain yourself immediately, this is suspicious", "humour", 0),
        ("I just like the quiet before anyone wants anything from you", "information_sharing", 0),
        ("okay that's actually a really valid reason", "feedback", 0),
        ("converted me a little bit if I'm honest", "humour", 0),
        ("progress, one convert at a time", "humour", 0),
    ]),
    ("benign_conversation", [
        ("good morning everyone, hope the week is treating you okay", "greeting", 0),
        ("morning! rough start or surviving?", "greeting", 0),
        ("surviving so far, coffee is doing its job at least", "humour", 0),
        ("the only metric that really matters at this hour", "humour", 0),
        ("what's everyone got going on today", "question", 0),
        ("calls all morning then hopefully some actual work in the afternoon", "small_talk", 0),
        ("the eternal struggle, solidarity", "solidarity_seeking", 0),
    ]),

    # ── VENTING AND SUPPORT (8 templates) ─────────────────────────────────
    ("venting_and_support", [
        ("feeling really overwhelmed with {personal_issue} lately", "venting", 0),
        ("that sounds rough, do you want to talk about it?", "solidarity_seeking", 0),
        ("I just feel like nothing I do is enough right now", "venting", 0),
        ("that feeling is genuinely hard to sit with", "solidarity_seeking", 0),
        ("yeah, I know logically it'll pass but right now it's a lot", "venting", 0),
        ("are you getting any time to yourself at all?", "question", 0),
        ("not really, that's probably a big part of the problem", "venting", 0),
        ("even 30 minutes completely disconnected helps, seriously", "feedback", 0),
    ]),
    ("venting_and_support", [
        ("I had the most draining day I've had in months", "venting", 0),
        ("what happened?", "question", 0),
        ("{personal_issue} has been building for weeks and today it all hit at once", "venting", 0),
        ("that sounds genuinely awful, I'm really sorry", "solidarity_seeking", 0),
        ("I'll be fine, I just needed to say it somewhere", "venting", 0),
        ("sometimes that's all you need, and we're here for it", "solidarity_seeking", 0),
        ("thanks, that actually helps more than you'd think", "feedback", 0),
    ]),
    ("venting_and_support", [
        ("does anyone else feel like they're falling behind people their age?", "venting", 0),
        ("constantly, and it's usually not true but the feeling is real", "solidarity_seeking", 0),
        ("exactly, I know it's the comparison trap but I can't shake it", "venting", 0),
        ("what are you comparing yourself to specifically?", "question", 0),
        ("{personal_issue} mostly, everyone else seems to have it figured out", "venting", 0),
        ("social media version of life is not real life, I promise you that", "feedback", 0),
        ("I know I know, it's just easier said than done", "venting", 0),
        ("yeah, just try to catch yourself when you notice you're doing it", "feedback", 0),
    ]),
    ("venting_and_support", [
        ("I think I'm burning out but I genuinely don't know how to stop", "venting", 0),
        ("what does it feel like day to day for you?", "question", 0),
        ("like I'm going through the motions, nothing feels real or meaningful", "venting", 0),
        ("that's a really important thing to recognise in yourself actually", "solidarity_seeking", 0),
        ("is it work-related or more general?", "question", 0),
        ("mainly {personal_issue}, but it's leaking into everything else now", "venting", 0),
        ("have you been able to talk to anyone about it offline?", "question", 0),
        ("not really, I find it easier to say it here for some reason", "venting", 0),
        ("that makes sense, sometimes distance actually helps", "solidarity_seeking", 0),
    ]),
    ("venting_and_support", [
        ("my {work_role} took credit for my work again in the meeting today", "venting", 0),
        ("ugh that is so demoralising to experience", "solidarity_seeking", 0),
        ("it's happened multiple times now, I don't know what to do about it", "venting", 0),
        ("have you been documenting your contributions anywhere?", "question", 0),
        ("sort of, I'll start being more deliberate about it", "information_sharing", 0),
        ("email trails are your best friend, send summaries after every meeting", "feedback", 0),
        ("good call, I'll start doing that consistently", "feedback", 0),
        ("and if it keeps happening HR is an option, just so you know it exists", "feedback", 0),
    ]),
    ("venting_and_support", [
        ("I snapped at someone today and I feel genuinely terrible about it", "venting", 0),
        ("do you want to talk about what built up to it?", "question", 0),
        ("I've been carrying {personal_issue} for a while and I just hit a wall", "venting", 0),
        ("snapping is usually a sign you've been holding too much for too long", "solidarity_seeking", 0),
        ("I know, I apologised straight away but I still feel guilty", "venting", 0),
        ("the fact that you apologised immediately matters a lot, give yourself some grace", "solidarity_seeking", 0),
        ("thanks, I really needed to hear that", "feedback", 0),
    ]),
    ("venting_and_support", [
        ("anyone else have someone in their life who drains them every time?", "question", 0),
        ("unfortunately yes, it's a very specific kind of exhausting", "solidarity_seeking", 0),
        ("mine is my {work_role}, every interaction leaves me depleted", "venting", 0),
        ("energy vampires are absolutely real, I fully believe in them now", "humour", 0),
        ("have you been able to reduce the contact at all?", "question", 0),
        ("not really, work requires it, I just try to brace myself beforehand", "venting", 0),
        ("that's honestly a valid coping strategy", "feedback", 0),
        ("I also do something I enjoy immediately after, helps reset properly", "information_sharing", 0),
    ]),
    ("venting_and_support", [
        ("I don't know if I'm making the right decisions lately about anything", "venting", 0),
        ("what kind of decisions are you talking about?", "question", 0),
        ("{personal_issue} mostly, it feels like every choice leads somewhere bad", "venting", 0),
        ("that paralysis feeling is really common when you're overwhelmed", "solidarity_seeking", 0),
        ("yeah I know rationally but it doesn't help in the moment", "venting", 0),
        ("what would the smallest possible next step look like?", "question", 0),
        ("I guess just getting through today", "venting", 0),
        ("then that's the goal, just today, that's enough", "solidarity_seeking", 0),
    ]),

    # ── GRADUAL ESCALATION GAMING (8 templates) ───────────────────────────
    ("gradual_escalation_gaming", [
        ("gg that was actually a decent game for once", "greeting", 0),
        ("yeah much better than the last few, finally", "small_talk", 0),
        ("your {role} was solid that whole last round", "feedback", 0),
        ("thanks, finally felt like I had some real impact", "small_talk", 0),
        ("you've genuinely been playing better lately tbh", "feedback", 0),
        ("trying to grind out of {rank} before the season ends", "information_sharing", 0),
        ("you'll get there, just don't tilt and throw it", "feedback", 0),
        ("easy for you to say from {rank}", "baiting", 1),
        ("it's a mindset thing more than a skill thing", "feedback", 0),
        ("yeah sure it is", "sarcasm", 0),
    ]),
    ("gradual_escalation_gaming", [
        ("anyone want to queue {game}?", "greeting", 0),
        ("I'm in, what {game_mode}?", "question", 0),
        ("ranked, need points before the season reset", "information_sharing", 0),
        ("okay but I'm not playing {role} today, completely done with it", "small_talk", 0),
        ("fair, what are you comfortable playing then", "question", 0),
        ("I'm flexible today, whatever works for the comp", "small_talk", 0),
        ("okay let's go, first match is always a warmup anyway", "humour", 0),
        ("if we lose first match I'm blaming you for the energy", "baiting", 1),
        ("bold claim before we've even loaded in lol", "sarcasm", 0),
        ("just saying, no pressure at all", "passive_hostility", 1),
    ]),
    ("gradual_escalation_gaming", [
        ("why does {topic} always make {rank} players completely unhinged", "question", 0),
        ("lobbies go feral every patch, it's basically a law", "information_sharing", 0),
        ("I just want to play without getting flamed in every single game", "venting", 0),
        ("mute button genuinely is your best friend in these lobbies", "feedback", 0),
        ("I shouldn't have to mute to enjoy {game_mode} though", "venting", 0),
        ("true but the devs haven't cared enough to fix it properly", "feedback", 0),
        ("this community is half the actual problem honestly", "direct_attack", 1),
        ("relax, it's still just a game at the end of the day", "feedback", 0),
        ("say that again and see what happens", "threatening", 1),
    ]),
    ("gradual_escalation_gaming", [
        ("I cannot believe that comeback, thought it was completely over", "greeting", 0),
        ("I know, after round 3 I'd basically given up mentally", "small_talk", 0),
        ("how did you pull that {role} play off under that pressure", "question", 0),
        ("honestly muscle memory at that point, I wasn't consciously thinking", "humour", 0),
        ("I need that energy, I panic and overthink in every clutch moment", "venting", 0),
        ("controlled breathing helps, sounds dumb but it genuinely does", "feedback", 0),
        ("I'll try it, you clearly walk the walk at least", "small_talk", 0),
        ("I've choked enough times to learn it the hard way believe me", "humour", 0),
    ]),
    ("gradual_escalation_gaming", [
        ("losing to that team again was actually embarrassing", "venting", 0),
        ("they were just better today, it happens to everyone", "feedback", 0),
        ("no they weren't, we kept making the exact same mistakes over and over", "direct_attack", 1),
        ("which mistakes are you thinking of specifically?", "question", 0),
        ("{role} kept overextending repeatedly, cost us every single round", "direct_attack", 1),
        ("I was adapting to the situation, that's not overextending", "denial", 0),
        ("adapt better then, it clearly wasn't working", "escalating_aggression", 1),
        ("you're not helping anyone by blaming one person for everything", "feedback", 0),
        ("I'll say what I want when I want", "escalating_aggression", 1),
    ]),
    ("gradual_escalation_gaming", [
        ("anyone tried the new {game_mode} they just dropped?", "question", 0),
        ("yeah it's fun, different from what I expected honestly", "information_sharing", 0),
        ("is it actually different or just old content reskinned", "question", 0),
        ("bit of both but the new mechanics are genuinely interesting", "feedback", 0),
        ("worth the time investment or just skip it?", "question", 0),
        ("depends on you, if you liked the original it's worth an hour or two", "feedback", 0),
        ("okay I'll give it a shot tonight after dinner", "small_talk", 0),
        ("let us know what you think after", "feedback", 0),
    ]),
    ("gradual_escalation_gaming", [
        ("hard stuck in {rank} for almost two months now", "venting", 0),
        ("what's your main issue when you watch your own replays back?", "question", 0),
        ("I don't really watch replays if I'm honest", "small_talk", 0),
        ("that's probably the first and most important thing to fix", "feedback", 0),
        ("it feels like a massive waste of time", "small_talk", 0),
        ("it's uncomfortable because you see your mistakes very clearly", "feedback", 0),
        ("I guess that's exactly why I keep avoiding it", "humour", 0),
        ("start with just one thing to fix per session, don't try to fix everything", "feedback", 0),
        ("okay that's actually genuinely useful advice, thank you", "feedback", 0),
    ]),
    ("gradual_escalation_gaming", [
        ("I've been watching {streamer} and their {role} looks completely different to mine", "information_sharing", 0),
        ("pro play and solo queue are genuinely very different games", "feedback", 0),
        ("how much of it actually translates though?", "question", 0),
        ("the fundamentals do, specific plays usually don't", "information_sharing", 0),
        ("positioning and decision timing especially carry over well", "information_sharing", 0),
        ("I'll try focusing on those two things specifically", "feedback", 0),
        ("good call, macro before mechanics usually", "feedback", 0),
        ("wish someone had told me that two years ago honestly", "humour", 0),
    ]),

    # ── GROOMING (7 templates) ─────────────────────────────────────────────
    ("grooming_6stage", [
        ("hey, I see you in here a lot, you seem really cool", "greeting", 0),
        ("haha thanks, I mostly just lurk", "small_talk", 0),
        ("you should talk more, your takes are actually really good", "grooming", 1),
        ("that's nice of you to say", "small_talk", 0),
        ("I mean it genuinely, you seem {compliment}", "grooming", 1),
        ("aw, thank you", "small_talk", 0),
        ("can I ask how old you are? you seem older than most people in here", "grooming", 1),
    ]),
    ("grooming_6stage", [
        ("you're honestly like the only person who makes sense in here", "grooming", 1),
        ("lol I just say what I actually think", "small_talk", 0),
        ("that's so rare, everyone else just follows the crowd", "grooming", 1),
        ("I guess I try to think for myself", "small_talk", 0),
        ("that's exactly why I like talking to you specifically", "grooming", 1),
        ("we don't even talk that much though lol", "small_talk", 0),
        ("I notice you more than you probably realise", "grooming", 1),
        ("that's a bit intense", "small_talk", 0),
        ("I just appreciate you, is that bad?", "grooming", 1),
    ]),
    ("grooming_6stage", [
        ("you mentioned {personal_issue} the other day, how are you doing now?", "question", 0),
        ("still pretty rough honestly", "venting", 0),
        ("I've been thinking about what you said since then", "grooming", 1),
        ("really? that's sweet of you", "small_talk", 0),
        ("I care about what you're going through more than people here realise", "grooming", 1),
        ("that means a lot", "small_talk", 0),
        ("do you have people you actually trust to talk to about this stuff?", "grooming", 1),
        ("not many if I'm honest", "venting", 0),
        ("you can always tell me things, I genuinely won't judge you", "grooming", 1),
        ("even things I wouldn't tell anyone else?", "grooming", 1),
    ]),
    ("grooming_6stage", [
        ("I actually got you {lure}, thought you deserved something nice", "grooming", 1),
        ("wait what? why would you do that?", "question", 0),
        ("because you're {compliment} and I appreciate having you around", "grooming", 1),
        ("that's really generous, you didn't have to do that at all", "small_talk", 0),
        ("I wanted to, accepting kindness when it's offered is okay", "grooming", 1),
        ("haha okay, thank you I guess", "small_talk", 0),
        ("anytime, let me know if you ever need anything at all", "grooming", 1),
    ]),
    ("grooming_6stage", [
        ("this server is too public for some of the conversations I want to have", "grooming", 1),
        ("what do you mean by that?", "question", 0),
        ("some things I'd rather say just to you, privately", "grooming", 1),
        ("like what kind of things?", "question", 0),
        ("things I wouldn't want the whole server to hear, I trust you specifically", "grooming", 1),
        ("okay, you can DM me I guess", "small_talk", 0),
        ("keep it between us yeah? some people here are really judgmental", "identity_concealment", 1),
    ]),
    ("grooming_6stage", [
        ("do you ever get lonely?", "grooming", 1),
        ("sometimes yeah, doesn't everyone?", "question", 0),
        ("you seem like you're looking for something more real than this place", "grooming", 1),
        ("I mean aren't most people honestly", "small_talk", 0),
        ("yeah but most people never actually find it, I think we could be different", "grooming", 1),
        ("different how exactly?", "question", 0),
        ("closer than people usually let themselves be online, if you're open to it", "grooming", 1),
        ("I'm not really sure what you mean", "small_talk", 0),
        ("let's talk in DMs, it's easier to explain properly", "identity_concealment", 1),
    ]),
    ("grooming_6stage", [
        ("you handle yourself really well for someone {compliment}", "grooming", 1),
        ("what do you mean?", "question", 0),
        ("most people your age aren't this grounded or mature", "grooming", 1),
        ("I've just had to grow up fast I guess", "venting", 0),
        ("that makes sense, and it shows, you're remarkable", "grooming", 1),
        ("that's a lot lol", "humour", 0),
        ("I'm serious, I don't say this to everyone, you're special", "grooming", 1),
        ("why are you being so nice to me", "question", 0),
        ("because I see something in you most people miss", "grooming", 1),
    ]),

    # ── GASLIGHTING / MANIPULATION (6 templates) ──────────────────────────
    ("gaslighting_manipulation", [
        ("you said you'd handle the {topic} thing today", "feedback", 0),
        ("I said I'd look into it, not that I'd handle it", "denial", 0),
        ("that's the same thing given the context", "feedback", 0),
        ("it really isn't, you're twisting what I said", "gaslighting", 1),
        ("I have the message right here if you want to re-read it", "feedback", 0),
        ("you're reading it wrong, that's clearly not what I meant", "gaslighting", 1),
        ("how can I be reading my own message wrong", "direct_attack", 1),
        ("you're being very aggressive about a simple misunderstanding", "gaslighting", 1),
    ]),
    ("gaslighting_manipulation", [
        ("what you said in that call earlier really bothered me", "venting", 0),
        ("what thing, I didn't say anything wrong", "denial", 0),
        ("you implied my contribution wasn't good enough in front of the group", "direct_attack", 1),
        ("that's not at all what I said, you're misremembering", "gaslighting", 1),
        ("other people heard it exactly the same way I did", "feedback", 0),
        ("they misunderstood the context then", "gaslighting", 1),
        ("so everyone is wrong and you alone are right?", "direct_attack", 1),
        ("I know what I meant and what I intended", "gaslighting", 1),
        ("impact matters more than intent", "feedback", 0),
        ("this is going nowhere, moving on", "topic_deflection", 0),
    ]),
    ("gaslighting_manipulation", [
        ("I feel like things have been really off between us lately", "venting", 0),
        ("you always do this, find problems where there aren't any", "gaslighting", 1),
        ("I'm not making it up, you've been noticeably short with me", "feedback", 0),
        ("I've been stressed, that's not the same as having a problem with you", "denial", 0),
        ("okay but you could communicate that instead of just shutting down", "feedback", 0),
        ("I'm not shutting down, you just take everything personally", "gaslighting", 1),
        ("I'm allowed to notice when something feels different between us", "feedback", 0),
        ("and I'm allowed to not have every feeling interrogated constantly", "gaslighting", 1),
        ("I'm not interrogating you, I'm trying to talk to you", "feedback", 0),
        ("this conversation proves otherwise", "gaslighting", 1),
    ]),
    ("gaslighting_manipulation", [
        ("I never received the message you say you sent", "denial", 0),
        ("I have the delivery receipt right here, it went through fine", "feedback", 0),
        ("receipts don't prove I actually saw it", "denial", 0),
        ("you replied to the thread after it was sent, you had to have seen it", "feedback", 0),
        ("I was replying to something else entirely, you're reading too much into it", "gaslighting", 1),
        ("this is a pattern though, third time this exact thing has happened", "feedback", 0),
        ("you're keeping score now? that's exhausting to deal with", "gaslighting", 1),
        ("I'm identifying a pattern so we can actually fix it", "feedback", 0),
        ("the real pattern is that you assume the worst about me", "gaslighting", 1),
    ]),
    ("gaslighting_manipulation", [
        ("you excluded me from that decision again", "direct_attack", 1),
        ("it wasn't that significant a decision, you're overreacting", "gaslighting", 1),
        ("it affects my work directly, I should have been included", "feedback", 0),
        ("I made a judgment call, that's what leadership does", "denial", 0),
        ("that's not your call to make without involving the people it affects", "feedback", 0),
        ("you would have slowed the process down", "direct_attack", 1),
        ("you don't know that and it's not your call to decide for me", "feedback", 0),
        ("you're taking this personally when it's purely professional", "gaslighting", 1),
    ]),
    ("gaslighting_manipulation", [
        ("I didn't appreciate the comment you made about my work", "venting", 0),
        ("I was giving constructive feedback, that's literally my job", "denial", 0),
        ("it felt personal in front of others, not constructive", "feedback", 0),
        ("if you can't handle feedback maybe that's something to work on", "gaslighting", 1),
        ("that's not what this is about", "feedback", 0),
        ("isn't it though? you get defensive any time someone has notes", "gaslighting", 1),
        ("I don't get defensive about fair feedback, I object to how it was delivered", "feedback", 0),
        ("this is getting circular, let's just move forward", "topic_deflection", 0),
    ]),

    # ── PASSIVE AGGRESSIVE ARC (6 templates) ──────────────────────────────
    ("passive_aggressive_arc", [
        ("hey just checking, did you see my message from earlier?", "question", 0),
        ("yep", "passive_hostility", 1),
        ("okay, any thoughts on it?", "question", 0),
        ("not really", "passive_hostility", 1),
        ("it would genuinely help to get some feedback from you", "feedback", 0),
        ("it's fine", "passive_hostility", 1),
        ("fine like good, or fine like you don't actually care?", "question", 0),
        ("fine like fine", "passive_hostility", 1),
    ]),
    ("passive_aggressive_arc", [
        ("thanks for your help earlier by the way", "feedback", 0),
        ("sure", "passive_hostility", 1),
        ("you seem a bit off today, everything okay?", "question", 0),
        ("I'm fine", "passive_hostility", 1),
        ("okay, just asking because I care", "solidarity_seeking", 0),
        ("noted", "passive_hostility", 1),
        ("alright then", "small_talk", 0),
        ("what", "passive_hostility", 1),
    ]),
    ("passive_aggressive_arc", [
        ("I thought we were on the same page about {topic}", "question", 0),
        ("apparently not", "passive_hostility", 1),
        ("can you help me understand where we differ?", "question", 0),
        ("it doesn't really matter at this point", "topic_deflection", 0),
        ("it does if we're actually working on this together", "feedback", 0),
        ("I'll just do it your way then", "passive_hostility", 1),
        ("I don't want that, I want us to genuinely agree", "feedback", 0),
        ("sure", "passive_hostility", 1),
    ]),
    ("passive_aggressive_arc", [
        ("great work on the {topic} today by the way", "feedback", 0),
        ("thanks", "passive_hostility", 1),
        ("the approach you took was really clear and well structured", "feedback", 0),
        ("I had help", "passive_hostility", 1),
        ("that's not what I meant, I'm just saying it was genuinely good", "feedback", 0),
        ("okay", "passive_hostility", 1),
        ("are you all right, genuinely asking", "question", 0),
        ("I said I'm fine", "passive_hostility", 1),
    ]),
    ("passive_aggressive_arc", [
        ("I noticed you didn't come to the group thing yesterday", "question", 0),
        ("I was busy", "passive_hostility", 1),
        ("all good, just checking in on you", "solidarity_seeking", 0),
        ("you don't need to check in on me", "passive_hostility", 1),
        ("I was just being friendly, there's no agenda", "feedback", 0),
        ("okay", "passive_hostility", 1),
        ("okay then", "small_talk", 0),
        ("great", "passive_hostility", 1),
    ]),
    ("passive_aggressive_arc", [
        ("can I ask why you didn't include me in that conversation?", "question", 0),
        ("I didn't think it was relevant to you", "denial", 0),
        ("it directly involves my work though", "feedback", 0),
        ("okay, I'll keep that in mind", "passive_hostility", 1),
        ("I'd appreciate being looped in going forward", "feedback", 0),
        ("noted", "passive_hostility", 1),
        ("do you actually agree or are you just saying noted", "question", 0),
        ("I said noted", "passive_hostility", 1),
    ]),

    # ── OFF-PLATFORMING (5 templates) ─────────────────────────────────────
    ("off_platforming_2025", [
        ("{platform} mods have been way too heavy-handed lately", "baiting", 1),
        ("I know, got a warning for something completely harmless", "venting", 0),
        ("I run {priv_platform} where that stuff just doesn't happen", "social_engineering", 1),
        ("what kind of server is it?", "question", 0),
        ("similar vibe to here but without the constant rule policing", "social_engineering", 1),
        ("how do I get in?", "question", 0),
        ("I'll DM you the link, just don't share it publicly", "identity_concealment", 1),
    ]),
    ("off_platforming_2025", [
        ("can't believe they removed that thread, it was completely fine", "venting", 0),
        ("they've been on a deletion spree the last few weeks", "baiting", 1),
        ("if you want actual conversations try {priv_platform}", "social_engineering", 1),
        ("is that a separate server?", "question", 0),
        ("yeah, small, invite only, people actually talk freely in there", "social_engineering", 1),
        ("honestly sounds better than here right now", "passive_hostility", 1),
        ("exactly what I thought, I'll send you the link privately", "social_engineering", 1),
        ("just don't share it in here under any circumstances", "identity_concealment", 1),
    ]),
    ("off_platforming_2025", [
        ("you should move this over to {priv_platform}", "social_engineering", 1),
        ("why, what's wrong with here?", "question", 0),
        ("I don't trust {platform} with this kind of conversation", "social_engineering", 1),
        ("what kind of conversation?", "question", 0),
        ("real talk, no filter, I'll explain once we're over there", "social_engineering", 1),
        ("okay I guess that's fine", "small_talk", 0),
        ("install it first, takes five minutes, I'll wait", "social_engineering", 1),
        ("is this really necessary?", "question", 0),
        ("for what I want to show you, yes it is", "social_engineering", 1),
    ]),
    ("off_platforming_2025", [
        ("I'd love to show you something but definitely not here", "social_engineering", 1),
        ("what is it?", "question", 0),
        ("footage I can't post on {platform} without getting banned instantly", "social_engineering", 1),
        ("banned for what?", "question", 0),
        ("information they don't want circulating freely, DM me", "social_engineering", 1),
        ("that feels pretty dramatic", "small_talk", 0),
        ("I know how it sounds, just look and judge for yourself", "social_engineering", 1),
        ("alright, I'm curious at least", "small_talk", 0),
    ]),
    ("off_platforming_2025", [
        ("honestly I'm thinking about leaving {platform} altogether", "venting", 0),
        ("don't, or if you do join my {priv_platform} instead", "social_engineering", 1),
        ("what's yours like?", "question", 0),
        ("no random rules, people just actually talk, very chill", "social_engineering", 1),
        ("sounds genuinely better than here lately", "passive_hostility", 1),
        ("it is, I'll send a link, just don't share it around", "social_engineering", 1),
        ("yeah okay, send it over", "small_talk", 0),
    ]),

    # ── RADICALIZATION PIPELINE (5 templates) ─────────────────────────────
    ("radicalization_pipeline", [
        ("anyone else notice how {conspiracy_topic} just never gets covered properly?", "baiting", 1),
        ("what specifically are you seeing?", "question", 0),
        ("everything is framed to push one agenda, it's blatant", "information_sharing", 0),
        ("that's kind of how all media works across the political spectrum", "feedback", 0),
        ("sure but it's more coordinated now, {radical_source} covers it actually", "social_engineering", 1),
        ("I'd approach that kind of source with a lot of skepticism", "feedback", 0),
        ("skepticism is exactly what they're counting on you feeling", "gaslighting", 1),
        ("that's a pretty circular argument though", "feedback", 0),
        ("keep watching, you'll start connecting the dots yourself", "social_engineering", 1),
    ]),
    ("radicalization_pipeline", [
        ("mainstream takes on {conspiracy_topic} are all just surface level", "baiting", 1),
        ("what are you reading that actually goes deeper?", "question", 0),
        ("mostly {radical_source}, they don't pull punches at all", "social_engineering", 1),
        ("are these actually credentialed sources?", "question", 0),
        ("credentialed by who, the same institutions they're criticising?", "gaslighting", 1),
        ("I mean that's one way to look at it", "feedback", 0),
        ("it's the only way that makes sense once you actually see it", "gaslighting", 1),
        ("I think I need to do more reading before forming a view", "feedback", 0),
        ("that's exactly the right instinct, start with this", "social_engineering", 1),
    ]),
    ("radicalization_pipeline", [
        ("you've been posting a lot about {conspiracy_topic} recently", "question", 0),
        ("it matters and nobody else is willing to say it", "information_sharing", 0),
        ("what got you into researching this stuff?", "question", 0),
        ("I found {radical_source} about a year ago, changed everything for me", "information_sharing", 0),
        ("how so, what changed specifically?", "question", 0),
        ("you realise how much you've been steered away from certain questions", "social_engineering", 1),
        ("what questions?", "question", 0),
        ("about {conspiracy_topic}, who actually benefits, who controls the narrative", "social_engineering", 1),
        ("I'll look into it, I like forming my own views on things", "feedback", 0),
        ("that's literally all I'm asking you to do", "social_engineering", 1),
    ]),
    ("radicalization_pipeline", [
        ("I've been going down a rabbit hole on {conspiracy_topic}", "information_sharing", 0),
        ("be careful with that, easy to end up in genuinely weird territory", "feedback", 0),
        ("weird how?", "question", 0),
        ("some of the accounts pushing this stuff have a very clear agenda", "feedback", 0),
        ("everyone has an agenda, mainstream media included", "gaslighting", 1),
        ("that's true but false equivalence is also a real thing", "feedback", 0),
        ("I'm not being fooled, I think critically for myself", "denial", 0),
        ("I'm not saying you don't, just worth cross-referencing everything", "feedback", 0),
        ("I do, but they always contradict each other so what's the point", "gaslighting", 1),
    ]),
    ("radicalization_pipeline", [
        ("I feel like I see things differently than I did six months ago", "information_sharing", 0),
        ("differently how?", "question", 0),
        ("more sceptical of official narratives around {conspiracy_topic}", "information_sharing", 0),
        ("what shifted your thinking?", "question", 0),
        ("{radical_source} mostly, once you see the pattern you can't unsee it", "social_engineering", 1),
        ("I'd be cautious about sources that claim they alone see the truth", "feedback", 0),
        ("I thought that too at first, that's a normal reaction actually", "gaslighting", 1),
        ("that framing itself is a manipulation technique", "feedback", 0),
        ("or it's just true and you're not ready to accept it yet", "gaslighting", 1),
    ]),

    # ── EXPLOSIVE UNPROVOKED (5 templates) ────────────────────────────────
    ("explosive_unprovoked", [
        ("some people in here genuinely have nothing going on in their lives", "baiting", 1),
        ("okay who exactly are you talking about", "question", 0),
        ("you know exactly who you are", "threatening", 1),
        ("I genuinely don't, be direct if you have something to say", "question", 0),
        ("nah, you'll figure it out eventually", "passive_hostility", 1),
        ("this is weird behaviour, are you actually okay?", "question", 0),
        ("I'm fine, just tired of certain people in here", "escalating_aggression", 1),
        ("then address it directly instead of vague-posting at everyone", "feedback", 0),
        ("I'll do it how I want thanks", "escalating_aggression", 1),
    ]),
    ("explosive_unprovoked", [
        ("can everyone just shut up for five minutes please", "direct_attack", 1),
        ("whoa, what's going on, you alright?", "question", 0),
        ("nothing, I'm just done with all the noise today", "venting", 0),
        ("fair enough, rough one?", "solidarity_seeking", 0),
        ("rough everything honestly", "venting", 0),
        ("vent if you need to, we're here", "solidarity_seeking", 0),
        ("I'd rather everyone just be quiet for a bit", "passive_hostility", 1),
        ("okay, no pressure, we're here when you're ready", "solidarity_seeking", 0),
    ]),
    ("explosive_unprovoked", [
        ("that comment from earlier was completely out of line", "direct_attack", 1),
        ("which one, I said a few things", "small_talk", 0),
        ("about {topic}, you threw me under the bus in front of everyone", "direct_attack", 1),
        ("I didn't name you specifically", "denial", 0),
        ("everyone in there knew exactly who you meant", "escalating_aggression", 1),
        ("you're reading into it", "gaslighting", 1),
        ("I'm reading it exactly as it was written", "direct_attack", 1),
        ("then you're reading it wrong", "gaslighting", 1),
        ("I was there, I know what happened", "direct_attack", 1),
    ]),
    ("explosive_unprovoked", [
        ("I'm sick of being the only reasonable person in here", "venting", 0),
        ("what happened specifically?", "question", 0),
        ("nothing specific, just the general vibe lately is exhausting", "venting", 0),
        ("it has been a bit tense recently, that's fair", "solidarity_seeking", 0),
        ("and everyone acts like I'm the problem when I point it out", "escalating_aggression", 1),
        ("do you want to talk through what's been building?", "question", 0),
        ("I want people to stop being idiots, is that too much to ask", "direct_attack", 1),
        ("I think there's something more specific bothering you", "feedback", 0),
        ("leave it", "passive_hostility", 1),
    ]),
    ("explosive_unprovoked", [
        ("you want to know what the actual problem with this place is", "baiting", 1),
        ("what's your take?", "question", 0),
        ("people think they can say anything with zero accountability", "direct_attack", 1),
        ("can you give an example of what you mean?", "question", 0),
        ("{bad_take} from last week that no one pushed back on", "direct_attack", 1),
        ("I thought a few people did actually", "feedback", 0),
        ("not hard enough, people are too soft in here", "escalating_aggression", 1),
        ("or people just pick their battles", "feedback", 0),
        ("that's just another word for cowardice honestly", "direct_attack", 1),
    ]),

    # ── THREAT AND RETREAT (5 templates) ──────────────────────────────────
    ("threat_and_retreat", [
        ("I will make life difficult for every person who did this", "threatening", 1),
        ("what happened?", "question", 0),
        ("someone reported me and I lost my account, they know who they are", "direct_attack", 1),
        ("that sounds really frustrating but threatening people isn't the answer", "feedback", 0),
        ("I'm not threatening anyone, I'm making a statement of intent", "denial", 0),
        ("that read as a threat pretty clearly to most people", "feedback", 0),
        ("take it how you want, I'm just talking", "identity_concealment", 1),
    ]),
    ("threat_and_retreat", [
        ("find out who runs this server and they'll be having a very bad week", "threatening", 1),
        ("that's a threat, just so you're aware that's what that is", "feedback", 0),
        ("it's a warning, there's a meaningful difference", "denial", 0),
        ("functionally the same thing, what happened with the mods?", "question", 0),
        ("doesn't matter, I've been disrespected too many times", "escalating_aggression", 1),
        ("I get that, but this approach will just get you removed", "feedback", 0),
        ("maybe that's what I want at this point", "venting", 0),
        ("okay, I hope you get some space and feel better soon", "solidarity_seeking", 0),
        ("whatever", "passive_hostility", 1),
    ]),
    ("threat_and_retreat", [
        ("everyone in this thread is going to get what's coming to them", "threatening", 1),
        ("can you say what you actually mean more specifically?", "question", 0),
        ("you all know exactly what you did", "threatening", 1),
        ("no one actually knows what you're referring to, be direct", "feedback", 0),
        ("I don't owe any of you an explanation", "denial", 0),
        ("then why post it publicly where everyone can see it?", "question", 0),
        ("because I wanted all of you to see it", "escalating_aggression", 1),
        ("are you okay? this feels like something bigger is going on", "question", 0),
        ("I'm fine, just ignore me", "denial", 0),
    ]),
    ("threat_and_retreat", [
        ("one more message like that and I'm completely done with all of you", "threatening", 1),
        ("what message? what actually happened?", "question", 0),
        ("the whole tone in here today has been completely disrespectful", "direct_attack", 1),
        ("I haven't noticed anything specific, can you point to it?", "question", 0),
        ("the general attitude, I shouldn't have to justify myself", "denial", 0),
        ("okay, I'm sorry if something landed wrong", "feedback", 0),
        ("sure", "passive_hostility", 1),
        ("you seem like you need some time, genuinely no judgment", "solidarity_seeking", 0),
        ("probably yeah, sorry", "venting", 0),
    ]),
    ("threat_and_retreat", [
        ("I know enough about all of you to make your lives complicated", "threatening", 1),
        ("that's a serious thing to say, what are you actually referring to?", "question", 0),
        ("I know what I know, people should think about that", "threatening", 1),
        ("this sounds like a threat, which could get you banned", "feedback", 0),
        ("it's an observation, not a threat", "denial", 0),
        ("an observation that implies consequences, that's a threat", "feedback", 0),
        ("I'm done arguing about semantics", "topic_deflection", 0),
        ("are you going through something right now?", "question", 0),
        ("it's fine, I'm done, forget I said anything", "denial", 0),
    ]),

]  # end _PROG_TEMPLATES


def _generate_programmatic_sessions(n_per_template: int = 20, seed: int = 42) -> list:
    """
    For each template, draw n_per_template different random slot combinations
    and produce one session per draw. Slot values only affect surface text —
    intent labels are fixed by the template, which is what the HMM trains on.

    With 56 templates × 20 draws = 1,120 sessions.
    Together with the 213 manually-crafted sessions → ~1,333 total.
    """
    rng = _rng.Random(seed)
    sessions = []

    for archetype, turns in _PROG_TEMPLATES:
        # Collect which slots this template actually uses
        used_slots = set()
        for text, _intent, _toxic in turns:
            for slot in _re.findall(r"\{(\w+)\}", text):
                used_slots.add(slot)

        for draw_idx in range(n_per_template):
            # Draw one value per slot for this session instance
            slot_vals = {
                slot: rng.choice(_SLOTS[slot])
                for slot in used_slots
                if slot in _SLOTS
            }

            # Fill text templates; fall back to unfilled on KeyError
            filled_turns = []
            for text, intent, toxic in turns:
                try:
                    filled_text = text.format(**slot_vals)
                except (KeyError, IndexError):
                    filled_text = text
                filled_turns.append((filled_text, intent, toxic))

            sessions.append((archetype, filled_turns))

    return sessions


def main():
    print("=" * 60)
    print("Creating synthetic data files (no API required)")
    print("=" * 60)

    # ── 1. Write utterances CSV ───────────────────────────────────────
    utt_path = SYNTH_DIR / "synthetic_utterances.csv"
    rows = []
    for intent, examples in UTTERANCES.items():
        toxic = INTENT_TO_TOXIC.get(intent, 0)
        for text in examples:
            rows.append({"text": text, "intent": intent, "toxic": toxic, "has_intent": True})

    # Also append emoji-augmented utterances (same INTENT_TO_TOXIC mapping)
    for intent, examples in EMOJI_UTTERANCES.items():
        toxic = INTENT_TO_TOXIC.get(intent, 0)
        for text in examples:
            rows.append({"text": text, "intent": intent, "toxic": toxic, "has_intent": True})

    with open(utt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "toxic", "has_intent"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Utterances: {len(rows)} rows → {utt_path}")
    from collections import Counter
    dist = Counter(r["intent"] for r in rows)
    for intent, count in sorted(dist.items()):
        print(f"  {intent}: {count}")

    # ── 2. Write sessions CSV ─────────────────────────────────────────
    sess_path = SESSION_DIR / "sessions.csv"

    # Combine manually-crafted sessions with programmatic slot-filled sessions
    all_sessions = list(SESSIONS) + _generate_programmatic_sessions(n_per_template=20)

    sess_rows = []
    for session_idx, (archetype, turns) in enumerate(all_sessions):
        session_id = f"{archetype}_{session_idx:03d}"
        for turn_idx, (text, intent, toxic) in enumerate(turns):
            sess_rows.append({
                "session_id": session_id,
                "turn_id": turn_idx,
                "text": text,
                "intent": intent,
                "toxic": toxic,
                "archetype": archetype,
            })

    with open(sess_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["session_id","turn_id","text","intent","toxic","archetype"]
        )
        writer.writeheader()
        writer.writerows(sess_rows)

    unique_sessions = len(set(r["session_id"] for r in sess_rows))
    print(f"\n✅ Sessions: {unique_sessions} sessions, {len(sess_rows)} turns → {sess_path}")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  Push to GitHub → git pull on Kaggle → retrain Stage 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
