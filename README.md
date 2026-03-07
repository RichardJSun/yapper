# Yapper

Yapper is an AI Discord Assistant built with `discord.js`, the `ai` SDK, and `sqlite-vec` for memory management. It is designed to run on `bun`. Because who needs real friends when you can build them from scratch? 🥲

## Features

- **Context Compaction**: Automatically compresses conversational context to maintain relevance without exceeding token limits.
- **Memory**: Persistent long-term memory managed via SQLite and `sqlite-vec`.
- **Proactive Messaging**: The bot can initiate conversations or send messages based on triggers rather than just responding to prompts.
- **Scheduled Messaging**: Includes the ability to schedule messages for future delivery.
- **Reactions**: Can react to Discord messages to make interactions feel more lively.
- **Web Search**: Integrates with Perplexity Sonar to search the web for up-to-date information.

## AI Models

- **Primary Model**: The bot uses **Kimi k2.5** by default because it seems to have the best persona results and interactions.
- **Alternative Model**: You can also use **Grok-4.1-fast-reasoning**, except the model is known to be a bit "dumber" compared to Kimi k2.5 for this specific persona use case.

## Setup & Running

Ensure you have [Bun](https://bun.sh/) installed.

1. Clone the repository.
2. Install dependencies:
   ```bash
   bun install
   ```
3. Set up your environment variables (see `.env.example`).
4. Start the bot:
   ```bash
   bun run start
   ```
