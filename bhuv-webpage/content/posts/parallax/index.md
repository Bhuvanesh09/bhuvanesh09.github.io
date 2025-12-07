+++
title = "The Planning Bottleneck"
date = "2025-12-06"
plotly = true
type = "posts"
author = "Bhuvanesh Sridharan · Animesh Sinha"
tags = [ "UX", "AI", "Agents" ]
+++


Cursor and Claude Code can build entire systems from specs now. But good specs take an hour to write. You're context-switching between your editor, files, Slack, past docs. That's the bottleneck.

AI coding agents got good enough that execution isn't the constraint anymore. Planning is. And we're still planning through request-response loops: 
You stop typing -> Formulate complete question -> Wait for answer -> Resume thinking -> Repeat.

Chat interfaces assume you know what to ask. But planning is figuring out what to ask. When you're planning, your thinking is incomplete by definition. You're exploring, context fragmented, trying to piece together what you don't know. Chatbots require you to package that messy, half-formed thinking into a coherent prompt before you get any help.

That's backwards. You need the most help when your thoughts are least formed.

## What if the AI thought alongside you while you type?

Not after you press enter. Not in a separate chat window. Continuously, in the background.

This is Parallax. An AI that processes your context and surfaces insights as inline suggestions: questions, connections, completions. All directly in your editor as ghost texts, and info cards.

Here's how it works:

Suppose you're writing a spec for rate limiting. As you type "per-user limits", Parallax is already:

- Navigating your codebase in real-time
- Spotting similar logic in `payments/stripe_handler.py:142`
- Identifying the gap: "Should retry attempts be per-user or per-transaction?"

{{< image src="illustrated_example.png" width="100%"  >}}

All appears as inline ghost text, and info cards. Tab to accept. Keep typing to dismiss. The AI never blocks you.

**Three types of suggestions:**

- *Questions* — "What happens when Redis is unavailable?"
- *Context* — "Similar pattern in auth/session.py:89, but with distributed locks"
- *Completions* — Traditional autocomplete, but context-aware across your workspace

## Why this works now

Speed matters more than you'd think. Traditional LLMs run at ~200 tokens/sec. Continuous suggestions need 2000+ tok/s or they feel laggy.

Model providers like Cerebras and Grok now hit 2000+ tok/s in production with advanced reasoning models. That's not a 10x improvement. It's an unlock for new interaction paradigms. At 200 tok/s, background reasoning feels like lag. At 2000 tok/s, it disappears and reasoning becomes ambient.

The UX constraint was always latency. That constraint is gone.

Second piece: semantic search over codebases got fast enough to run continuously. Tools like Mixedbread's [mgrep](https://github.com/mixedbread-ai/mgrep) use multi-vector retrieval to understand search intent rather than matching literal strings. The AI can query "find authentication retry logic" instead of guessing function names. Combined with fast inference, the background agent can navigate, reason, and surface insights before you finish your sentence.

<!-- ## The paradigm shift

The request-response loop shaped how we think about AI tools. You formulate a complete question. The AI formulates a complete answer. Back and forth.

What's a reasonable person to do when the AI can keep up with your thoughts? Simple: Stop formulating complete questions. Think out loud. Let the AI fill in gaps as you go.

Version 1: AI waits for your prompt in a chat interface

Version 2: AI responds to your prompt in an async way, after working on its own.

Version 3: AI thinks while you think

Version 3 is now possible. -->

## Proof of concept

We built a prototype at the MBZUAI K2-Think hackathon. Terminal UI, real-time suggestions, works with local codebases. It's rough, but it works.

{{< image src="parallax_gif_1_small.gif" width="100%" >}}

The point wasn't to ship a product. It was to prove the interaction model is viable. It is.

---

The future of AI tools isn't better chatbots. It's background intelligence that surfaces exactly what you need, when you need it, without you asking.

This isn't just for developers. Any knowledge worker dealing with fragmented context hits the same stop-and-search pattern. PMs synthesizing research. Designers connecting user feedback. Analysts piecing together data. Continuous reasoning beats request-response for all of them.

The pieces are here. Someone's going to build the polished version of this. Maybe one of the existing tools adds it. Maybe it's a new player. Either way, the interaction paradigm is ready to shift.

---

Interested in the code or want to discuss more on the idea? Reach out. Happy to share what we learned.
