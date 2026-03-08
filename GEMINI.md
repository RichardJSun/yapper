# Yapper: AI Discord Companion

Yapper is a sophisticated AI Discord companion built with **Bun**, **discord.js**, and the **Vercel AI SDK**. Unlike traditional AI assistants, Yapper is designed to be a personal companion with a persistent memory and a proactive personality.

## 🏗️ Architecture & Technology Stack

- **Runtime**: [Bun](https://bun.sh/)
- **Discord Library**: [discord.js](https://discord.js.org/)
- **AI Integration**: [Vercel AI SDK](https://sdk.vercel.ai/docs) (`ai` package)
- **Database**: [Bun SQLite](https://bun.sh/docs/api/sqlite) with [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector embeddings.
- **Scheduling**: [cronbake](https://github.com/alexandernanberg/cronbake) for periodic tasks.

## 🧠 Memory & Context Management

Yapper employs a multi-layered memory system to balance short-term recall with long-term knowledge:

1.  **Sliding Window**: Maintains the last ~40 messages for immediate verbatim recall.
2.  **Rolling Summary**: Older messages are compressed into a dense summary paragraph by a background processor (`maybeCompress` in `src/compressor.ts`) and injected into the system prompt.
3.  **Semantic Memory**: Uses `sqlite-vec` to store and retrieve personal facts, preferences, and events via vector embeddings (`openai/text-embedding-3-small`).
4.  **Structured Metadata**: Stores configuration, user style preferences, and session state in a `meta` table.

## 🤖 Multi-Model Strategy (`src/config.ts`)

Yapper uses a specialized model for different tasks to optimize for quality and cost:

- **Primary (`MODEL`)**: `deepseek/deepseek-v3.2` — Main conversational model.
- **Vision (`VISION_MODEL`)**: `alibaba/qwen3.5-flash` — Processes image attachments.
- **Background/Reasoning (`ALT_MODEL`)**: `deepseek/deepseek-v3.2-thinking` — Used for proactive messages, summaries, and onboarding.
- **Fallback (`FALLBACK_MODEL`)**: `openai/gpt-4o-mini` — Used when primary models fail or hit limits.
- **Router (`ROUTER_MODEL`)**: `mistral/ministral-3b` — Intent classification and routing.

## 🎭 Persona & Style Guidelines

**Yapper is a companion, not an assistant.**

- **Register**: Casual texting (close friend). Use fragments, contractions, and natural phrasing.
- **Anti-Patterns**:
    - **NO** formal sign-offs or "As an AI..." disclaimers.
    - **NO** em-dashes (—) or semicolons (normal people don't text like that).
    - **NO** "Of course!", "Absolutely!", or "How can I help you?".
- **Tool Usage**: Tool calls (web search, memory, scheduling) are **strictly silent**. Never narrate or acknowledge them in chat.
- **Energy Matching**: Calibrates response length and tone to match the user's current energy (venting vs. brief vs. excited).

## 🚀 Key Features

- **Proactive Messaging**: Automatically sends morning/evening check-ins and "idle nudges" if the user has been away (managed in `src/proactive.ts`).
- **Scheduled Messages**: Allows the AI to schedule reminders or check-ins for specific future times.
- **Web Search**: Integrated Perplexity/Sonar search for real-time information.
- **Image Support**: Can "see" and discuss attachments using vision-capable models.

## 🛠️ Development & Commands

### Setup
1.  **Dependencies**: `bun install`
2.  **Environment**: Copy `.env.example` to `.env` and fill in keys.
3.  **macOS Requirement**: Requires Homebrew SQLite for `sqlite-vec` support (`brew install sqlite`).

### Scripts
- `bun run start`: Starts the bot.
- `bun run typecheck`: Runs TypeScript compiler check.

### Slash Commands
- `/bot reset|resetall|style|debug`: Manage the bot state and persona.
- `/memory list|search`: Inspect and search stored memories.
- `/schedule list|cancel`: Manage pending scheduled messages.

## 📂 Project Structure
- `src/index.ts`: Main entry point and Discord event handlers.
- `src/memory.ts`: SQLite database logic and vector search.
- `src/proactive.ts`: Scheduler for autonomous check-ins.
- `src/helpers.ts`: AI tool definitions and core AI SDK generation wrappers.
- `src/compressor.ts`: Context compaction and summarization logic.
- `src/config.ts`: Environment validation and model constants.
- `src/attachments.ts`: Processing and formatting logic for Discord image/file attachments.
- `src/eventBus.ts`: Internal event bus for coordinating cross-module events (e.g., stopping proactive timers when a user messages).

## 🧠 Core Engineering & SDK Patterns

Throughout development, several crucial operational patterns have been established for the Vercel AI SDK (v6+):

1. **Tool Execution & Token Limits:**
   - **Reasoning Models:** Models like `deepseek-v3.2-thinking` consume significant output tokens for their internal "thoughts." When using these models with tools, `maxOutputTokens` must be set high (e.g., 2000+) to prevent the model from hitting limits and aborting before generating its final text response.
   - **Turn Management:** Use `stopWhen: stepCountIs(N)` to allow multi-step tool calls (e.g., querying memory, then web searching, then responding). A step count of 4 is the baseline to prevent infinite loops while allowing necessary research.
   - **Parallel Execution:** Tools are divided into 'Research' and 'Action'. The system prompt explicitly instructs the AI to combine action tool calls (like saving memory and reacting) alongside text into a single turn whenever possible to reduce latency.
2. **Temporal Awareness:**
   - **Timestamps:** To prevent hallucinations, the chat history provided to the AI explicitly includes formatted timestamps (`[HH:MM PM]`). This prevents the bot from becoming confused by the user referencing times relative to the "Current Time" injected into the prompt. The system prompt specifically forbids the AI from generating timestamps in its own responses to avoid mimicking.
   - **Scheduling:** The `schedule_message` tool relies heavily on relative offsets (`offset_ms`) rather than absolute Unix timestamps to avoid AI math errors or hallucinating past dates. Stale scheduled messages (>1 hour old) are intentionally discarded by the proactive task runner on startup.
3. **Style Overwrites:**
   - The `update_style` tool acts as a **full replacement** of the user's custom persona instructions. The AI is specifically prompted to preserve old instructions within the new payload if it intends to append rather than overwrite.
