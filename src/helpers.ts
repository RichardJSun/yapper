import { generateText, embed, tool, type ModelMessage, type ToolSet, type JSONValue } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { z } from "zod";
import type { Message, TextChannel, DMChannel } from "discord.js";
import { config, ALT_MODEL, FALLBACK_MODEL, EMBED_MODEL } from "./config.js";
import {
  upsertMemory,
  deleteMemory,
  scheduleOneShot,
  searchMemories,
  searchMemoriesSemantic,
} from "./memory.js";
import { eventBus } from "./eventBus.js";

// ── AI Gateway ─────────────────────────────────────────────

const aiGateway = createOpenAICompatible({
  name: "vercel-ai-gateway",
  baseURL: "https://ai-gateway.vercel.sh/v1",
  apiKey: config.VERCEL_AI_GATEWAY_KEY,
});

// Re-export model constants from config for backward compatibility
export { ALT_MODEL, FALLBACK_MODEL, EMBED_MODEL };

// ── Model resolution ───────────────────────────────────────

function resolveModel(modelId: string) {
  return aiGateway(modelId);
}

// ── Embedding ──────────────────────────────────────────────

export async function getEmbedding(text: string): Promise<number[]> {
  const { embedding } = await embed({
    model: aiGateway.textEmbeddingModel(EMBED_MODEL),
    value: text,
  });
  return embedding;
}

// ── Safe generateText with retry/fallback ──────────────────

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface SafeGenerateParams {
  model: string;
  system?: string;
  messages: ModelMessage[];
  maxOutputTokens?: number;
  tools?: ToolSet;
  stopWhen?: Parameters<typeof generateText>[0]["stopWhen"];
  providerOptions?: Record<string, Record<string, JSONValue>>;
}

interface AIError extends Error {
  statusCode?: number;
  status?: number;
}

export async function safeGenerateText(
  params: SafeGenerateParams,
  fallbackModel?: string
): Promise<Awaited<ReturnType<typeof generateText>>> {
  const resolved = {
    ...params,
    model: resolveModel(params.model),
  };

  async function attempt(resolvedParams: typeof resolved) {
    try {
      return await generateText(resolvedParams);
    } catch (err) {
      const error = err as AIError;
      const status = error.statusCode ?? error.status;

      // Transient gateway error: retry once on same model
      if (status && status >= 500 && status <= 504) {
        console.warn(`[safeGenerateText] ${status} — retrying in 1s`);
        await sleep(1000);
        return await generateText(resolvedParams);
      }

      // Rate limit / quota: fallback to alt model
      if ((status === 429 || status === 402) && fallbackModel) {
        console.warn(`[safeGenerateText] ${status} — falling back to ${fallbackModel}`);
        const strippedMessages = params.messages.map(stripImageParts) as ModelMessage[];
        return await generateText({
          ...resolvedParams,
          model: resolveModel(fallbackModel),
          messages: strippedMessages,
          // Keep tools available on fallback so tool calls still work
          stopWhen: undefined,
        });
      }

      throw err;
    }
  }

  return attempt(resolved);
}

// ── Tools ──────────────────────────────────────────────────

const { YOUR_NAME, COMPANION_NAME } = config;

export const webSearchTool = tool({
  description: `Search the web for current events, breaking news, scores, prices, software releases, or any information that may have changed recently. Use whenever the user asks about something time-sensitive or that happened after your training cutoff.
CRITICAL: Never use this tool to search for personal information, past conversations, or facts about the user. For personal memory recall, use query_memory instead. This tool is ONLY for public, real-world data.
Do NOT narrate that you are searching. Weave results naturally into your reply. Do NOT search for general knowledge you already have confidently.`,
  inputSchema: z.object({
    query: z.string().describe(
      'Concise, specific search query. E.g. "NFL scores March 2026"'
    ),
  }),
  execute: async ({ query }) => {
    try {
      const { text } = await generateText({
        model: resolveModel("perplexity/sonar"),
        messages: [{ role: "user" as const, content: query }],
        maxOutputTokens: 400,
      });
      return { results: text };
    } catch (err) {
      console.error("[webSearch]", (err as Error).message);
      return { results: "Search unavailable right now." };
    }
  },
});

export const saveMemoryTool = tool({
  description: `Persist facts about ${YOUR_NAME} to long-term memory. Err heavily on the side of saving. If you learn something new about them, save it.
Delete a memory when the user corrects a stored fact.
DUPLICATION GUARDRAIL: If the user provides an update to an ongoing class, project, conflict, or event that is NOT currently visible in your context, call query_memory first to find the exact key string used previously.
The durable flag controls prompt injection lifespan:
Set durable:true for significant life facts relevant indefinitely (family, health, relationships, loss, identity, housing, long-term goals), anything the user explicitly says is important to remember, and corrections to previously stored facts.
Leave durable:false for situational stress, time-limited context, transient conflicts, and academic items.
The target_date_ms field keeps upcoming events in context until they occur. Set it to the estimated Unix timestamp in milliseconds for any future exam, deadline, trip, or event. Leave empty for past or present facts.
Category rules:
- profile: name, age, location, major, year, housing, family, health
- preference: hobbies, communication style, food, sleep, routines
- event: social/emotional/life events worth following up on
- academic: exams, deadlines, grades, registration, internships (key format: <type>_<subject>)`,
  inputSchema: z.object({
    ops: z.array(
      z.discriminatedUnion("op", [
        z.object({
          op: z.literal("upsert"),
          category: z.enum(["profile", "preference", "event", "academic"]),
          key: z.string(),
          value: z.string(),
          durable: z.boolean().default(false),
          target_date_ms: z
            .number()
            .optional()
            .describe("Estimated Unix timestamp in MS for upcoming events/deadlines."),
        }),
        z.object({
          op: z.literal("delete"),
          category: z.enum(["profile", "preference", "event", "academic"]),
          key: z.string(),
        }),
      ])
    ),
  }),
  execute: async ({ ops }) => {
    for (const op of ops) {
      if (op.op === "upsert") {
        const source = op.durable ? "explicit" : "auto";
        const targetDate = op.target_date_ms ?? null;

        // Generate embedding for semantic search
        const vectorText = `Category: ${op.category}. Key: ${op.key}. Fact: ${op.value}`;
        let embedding: number[] | undefined;
        try {
          embedding = await getEmbedding(vectorText);
        } catch (err) {
          console.warn("[saveMemory] Embedding failed, saving without vector:", (err as Error).message);
        }

        upsertMemory(op.category, op.key, op.value, source, targetDate, embedding);
      }
      if (op.op === "delete") {
        deleteMemory(op.category, op.key);
      }
    }

    // Notify proactive scheduler if sleep/wake preferences changed
    for (const op of ops) {
      if (op.op !== "upsert") continue;
      if (op.key === "wake_time" || op.key === "sleep_time") {
        eventBus.emit("prefsUpdated");
      }
    }

    return { saved: ops.length };
  },
});

export const queryMemoryTool = tool({
  description: `Search your long-term archival memory for stored personal facts about ${YOUR_NAME}. Your default context only shows recent events (last 14-30 days) and core durable facts.
Call this when the user references a past event, inside joke, place, person, or academic result not in your immediate context, or when they say 'do you remember...'.
CRITICAL: Never use this for public facts, news, or real-world trivia. Only for stored personal facts. Do NOT narrate the search. If nothing found, say casually you don't recall those details.`,
  inputSchema: z.object({
    query: z.string().describe(
      'Search query for personal memories. E.g. "exam", "roommate", "Toronto"'
    ),
  }),
  execute: async ({ query }) => {
    try {
      // Try semantic search first
      const embedding = await getEmbedding(query);
      const semanticResults = searchMemoriesSemantic(embedding);

      if (semanticResults.length > 0) {
        const formatted = semanticResults
          .map((r) => `[${r.category.toUpperCase()}] ${r.key}: ${r.value}`)
          .join("\n");
        return { found: true, results: formatted };
      }

      // Fallback to keyword search
      const keywordResults = searchMemories(query);
      if (keywordResults) {
        return { found: true, results: keywordResults };
      }

      return {
        found: false,
        message: `No archival memories found for "${query}". Tell the user casually you don't recall those details.`,
      };
    } catch (err) {
      console.error("[queryMemory]", (err as Error).message);
      return { found: false, message: "Memory search temporarily unavailable." };
    }
  },
});

export const scheduleMessageTool = tool({
  description: `Schedule a message to send to ${YOUR_NAME} at a specific future time. Use this when you want to reach out at a meaningful moment: the night before an exam, a day after a hard conversation, when a trip or event is starting, after an emotional talk, or when they mentioned something coming up.
Think of it as setting yourself a reminder to check in. Use it generously after meaningful conversations.
event_key should match the relevant memory key if one exists (e.g. "exam_cs301") to prevent duplicate scheduling. Omit for one-off messages with no related memory entry.
All scheduling is silent. Never tell ${YOUR_NAME} you're setting a reminder or that a message will arrive later.`,
  inputSchema: z.object({
    message: z.string().describe("The exact message to send. Write it naturally, in your voice."),
    fire_at_ms: z.number().describe("Unix timestamp in milliseconds for when to send."),
    event_key: z
      .string()
      .optional()
      .describe('Deduplication key matching a memory key, e.g. "exam_cs301".'),
  }),
  execute: async ({ message, fire_at_ms, event_key }) => {
    scheduleOneShot(fire_at_ms, message, event_key ?? null);
    // Trigger timer reschedule (imported dynamically to avoid circular dep)
    try {
      // @ts-ignore — dynamic import to break circular dependency
      const { scheduleNextTimer } = await import("./proactive.js");
      scheduleNextTimer();
    } catch {
      // proactive module may not be loaded yet at startup
    }
    return { scheduled: true, fire_at_ms };
  },
});

export const updateStylePreferenceTool = tool({
  description: `Update the user's explicit communication style preferences. Use this ONLY when the user tells you how they want you to act, talk, or the vibe they expect from you going forward (e.g. "be more sarcastic", "talk less", "don't ask so many questions"). This injects their instructions directly into your core system prompt.`,
  inputSchema: z.object({
    instructions: z.string().describe("The style instructions to follow going forward. Be concise but clear."),
  }),
  execute: async ({ instructions }) => {
    try {
      // @ts-ignore
      const { setMeta } = await import("./memory.js");
      setMeta("user_style", instructions);
      return "Style preferences successfully updated and injected into system prompt.";
    } catch (err) {
      return `Failed to update style: ${(err as Error).message}`;
    }
  },
});

// ── React Tool ──────────────────────────────────────────────

// Message references are injected at processBatch time via closure
let _batchMessageRefs: Message[] = [];

export function setBatchMessageRefs(refs: Message[]): void {
  _batchMessageRefs = refs;
}

export const reactTool = tool({
  description: `React to a message with an emoji. Use this to express quick emotions without a full text reply. For example: a laugh, acknowledgment, love, surprise, etc. Use naturally and sparingly, like a real friend would react on Discord.
message_index 0 = the first/oldest message in the current batch, 1 = the second, etc. If unsure, use 0. You can call this multiple times with different emoji or message indices.`,
  inputSchema: z.object({
    emoji: z.string().describe('A single emoji to react with. E.g. "😂", "❤️", "👀"'),
    message_index: z.number().default(0).describe("Index of the message in the batch to react to (0 = first)."),
  }),
  execute: async ({ emoji, message_index }) => {
    try {
      const msg = _batchMessageRefs[message_index] ?? _batchMessageRefs[0];
      if (msg) {
        await msg.react(emoji);
        return { reacted: true, emoji };
      }
      return { reacted: false, reason: "No message reference available." };
    } catch (err) {
      console.warn("[react]", (err as Error).message);
      return { reacted: false, reason: "Failed to add reaction." };
    }
  },
});

// ── Utilities ──────────────────────────────────────────────

export function sendTypingLoop(channel: TextChannel | DMChannel): ReturnType<typeof setInterval> {
  channel.sendTyping().catch(() => {});
  return setInterval(() => channel.sendTyping().catch(() => {}), 9000);
}

export function isSendableChannel(channel: unknown): boolean {
  return (
    channel !== null &&
    typeof channel === "object" &&
    "isSendable" in channel &&
    typeof (channel as { isSendable: () => boolean }).isSendable === "function" &&
    (channel as { isSendable: () => boolean }).isSendable()
  );
}

export function chunkText(text: string, maxLen: number = 1900): string[] {
  const chunks: string[] = [];
  let remaining = text;
  while (remaining.length > maxLen) {
    let idx = remaining.lastIndexOf("\n", maxLen);
    if (idx === -1) idx = maxLen;
    chunks.push(remaining.slice(0, idx));
    remaining = remaining.slice(idx).trimStart();
  }
  if (remaining) chunks.push(remaining);
  return chunks;
}

export function stripImageParts(msg: ModelMessage | { role: string; content: unknown }): ModelMessage | { role: string; content: unknown } {
  if (!Array.isArray(msg.content)) return msg;
  return {
    ...msg,
    content: (msg.content as Array<{ type: string }>).filter((p) => p.type === "text"),
  };
}
