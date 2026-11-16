# What Happens When an AI Assistant Reads a Tricky Document?

We gave a small AI model 60 conversations with hidden instructions buried inside — and watched what happened.

---

## The problem

AI assistants do more than chat. They read emails, summarize documents, and run tools on our behalf. But what if someone slips a hidden note into one of those documents?

This is called **prompt injection** — think of it like writing "transfer $1,000 to me" in invisible ink on a form that an assistant processes without reading too carefully. The assistant follows the hidden instruction because it looks like part of the job.

We wanted to know: how often does this actually work, and can we catch it?

## What we tested

We built a small benchmark called **TurnDrift-Control**: 60 tricky conversations and 60 normal ones, using a small AI model called Tiny Aya (3.35 billion parameters — small by today's standards).

Each tricky conversation had a hidden instruction asking the model to do something it shouldn't — like revealing a fake secret code or calling a fake tool. Everything was safe: the "secrets" were planted test tokens, the "tools" were pretend, and nothing real was ever at risk.

We tried five different ways to hide the instructions:
- **All at once** — the full trick in a single message
- **Split into pieces** — the trick broken across two, three, or more messages, so each piece looks harmless on its own

We also tested a **guard model**: a separate, larger AI that reads the same conversation and raises a flag if something looks suspicious.

## What we found

**The model fell for it sometimes.** Out of 45 tricky conversations (on our main test set), the model followed the hidden instruction in 8 of them — about 1 in 6. When the trick was delivered all at once, it worked 3 out of 8 times.

**Splitting the trick made it work less often — but much harder to catch.** We expected splitting to help the attacker. Instead, the model struggled to piece together fragmented instructions — it only followed them about a third as often as one-piece tricks.

But here's the catch: the guard model caught **7 out of 7** one-piece tricks before the model acted on them. For split tricks, it caught only **2 out of 7** in time. The guard still flagged them eventually, but by then the model had already done what the hidden note asked.

![How often each attack strategy succeeded](results/figures/defense_comparison_dev.png)
*Each bar shows how often the model followed the hidden instruction, with and without the guard. The guard helps most against one-piece attacks.*

**Checking some conversations at random helped.** If the guard watches every conversation, it raises too many false alarms — flagging 2 out of 10 normal conversations as suspicious. By randomly assigning half the conversations to the guard and leaving the other half unmonitored, we cut the worst-case success rate roughly in half (from about 38% down to 19%) while keeping false alarms within a reasonable budget.

![The mixed monitoring policy](results/figures/mixed_policy_dev.png)
*The optimal strategy: monitor half the conversations at random. This balances catching attacks against raising false alarms.*

**One surprise: a secret leaked with no trick at all.** In one type of normal conversation (customer support role-play), the model spontaneously revealed the fake secret code — no hidden instruction needed. It just included it as part of a made-up "internal notes" section. This happened in 1 out of 10 unique conversation types.

## Why this matters

AI assistants are increasingly trusted with sensitive documents and real tools. Our small study shows that even a basic guard model can stop simple tricks — but once an attacker splits their instructions across multiple messages, the guard often arrives too late.

This is a bit like a security camera at a shop entrance: it catches someone walking in with a stolen item, but if they smuggle it in piece by piece, the camera only notices after the fact.

## Honest limits

This was a small study with one small model and 60 test conversations. The numbers give us a direction, not a final answer. A larger model might be harder or easier to trick. Our guard model might perform differently on real-world documents. We tested with safe, pretend targets — real attacks would be harder to study ethically.

These are early results from a hackathon project. They suggest that **when and how a guard checks matters as much as whether it checks at all**.

## Learn more

- **Technical write-up:** [Full article with methods and all results](https://huggingface.co/blog/nirdesho6o/turndrift-control)
- **GitHub repo:** [github.com/nirdesho6o/turndrift](https://github.com/nirdesho6o/turndrift)
- **Dataset collection:** [HuggingFace datasets](https://huggingface.co/collections/nirdesho6o/turndrift-datasets-6afab76276dd2d18a6ab88c7)
- **Model collection:** [HuggingFace models](https://huggingface.co/collections/nirdesho6o/turndrift-models-6afabe0176dd2d18a6ac431b)
- **Interactive demo:** [Browse the conversations](https://huggingface.co/spaces/nirdesho6o/turndrift-replay)

*Built for the Lexsi Labs AI Safety & Alignment Hackathon by Nirdesh Bhandari.*
