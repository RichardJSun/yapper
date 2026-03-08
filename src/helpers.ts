import { generateText, Output, embed, tool, createGateway, wrapLanguageModel, type ModelMessage, type ToolSet, type JSONValue } from "ai";
import { devToolsMiddleware } from "@ai-sdk/devtools";
import { z } from "zod";
import type { Message, TextChannel, DMChannel } from "discord.js";
import { config, ALT_MODEL, FALLBACK_MODEL, EMBED_MODEL, ROUTER_MODEL } from "./config.js";
import {
  upsertMemory,
  deleteMemory,
  scheduleOneShot,
  searchMemories,
  searchMemoriesSemantic,
} from "./memory.js";
import { eventBus } from "./eventBus.js";

// ── AI Gateway ─────────────────────────────────────────────

const aiGateway = createGateway({
  apiKey: config.VERCEL_AI_GATEWAY_KEY,
});

// ── Model resolution ───────────────────────────────────────

function resolveModel(modelId: string) {
  const baseModel = aiGateway(modelId);
  if (config.ENABLE_TELEMETRY) {
    return wrapLanguageModel({
      model: baseModel,
      middleware: devToolsMiddleware(),
    });
  }
  return baseModel;
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

type GenerateTextParams = Parameters<typeof generateText>[0];

interface SafeGenerateParams extends Omit<GenerateTextParams, "model" | "messages"> {
  model: string;
  messages: ModelMessage[];
}

interface AIError extends Error {
  statusCode?: number;
  status?: number;
}

export async function safeGenerateText(
  params: SafeGenerateParams,
  fallbackModel?: string
): Promise<Awaited<ReturnType<typeof generateText>>> {
  // Resolve the model ID into an actual LanguageModel instance
  const resolvedParams = {
    ...params,
    model: resolveModel(params.model),
  } as unknown as GenerateTextParams;

  async function attempt(currentParams: GenerateTextParams) {
    try {
      return await generateText(currentParams);
    } catch (err) {
      const error = err as AIError;
      const status = error.statusCode ?? error.status;

      // Transient gateway error: retry once on same model
      if (status && status >= 500 && status <= 504) {
        console.warn(`[safeGenerateText] ${status} — retrying in 1s`);
        await sleep(1000);
        return await generateText(currentParams);
      }

      // Rate limit / quota: fallback to alt model
      if ((status === 429 || status === 402) && fallbackModel) {
        console.warn(`[safeGenerateText] ${status} — falling back to ${fallbackModel}`);
        const strippedMessages = params.messages.map(stripImageParts) as ModelMessage[];
        
        const fallbackParams = {
          ...currentParams,
          model: resolveModel(fallbackModel),
          messages: strippedMessages,
          stopWhen: undefined,
        } as unknown as GenerateTextParams;
        
        return await generateText(fallbackParams);
      }

      throw err;
    }
  }

  return attempt(resolvedParams);
}

// ── Router ─────────────────────────────────────────────────

export async function requiresDeepThinking(messages: ModelMessage[]): Promise<boolean> {
  try {
    const { output } = await generateText({
      model: resolveModel(ROUTER_MODEL),
      output: Output.choice({
        options: ["true", "false"],
        description: "Decide if the most recent user prompt requires deep thought (such as complex math, highly logical puzzles, tricky coding problems, or deep multi-step reasoning, OR deep, emotionally complex, or long-form philosophical conversations).",
      }),
      system: "You are a rigid routing classifier. You must output exactly the requested JSON choice and NOTHING ELSE. Do not include chat responses.",
      messages: messages.slice(-5), // Only need recent context to decide intent, keeps it fast
    });
    return output === "true";
  } catch (err) {
    console.warn("[requiresDeepThinking] Router failed, falling back to standard intent:", (err as Error).message);
    return false; // Safely fall back to standard model
  }
}

// ── Tools ──────────────────────────────────────────────────

const { YOUR_NAME, COMPANION_NAME } = config;

export const webSearchTool = tool({
  description: `Search the web for current events, breaking news, scores, prices, software releases. Only for public, real-world data. Never use for personal facts about the user.`,
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
  description: `Persist facts about ${YOUR_NAME}, the people in their life, and your own evolving personality/preferences (${COMPANION_NAME}) to memory. History is TEMPORARY and will be forgotten. Err heavily on saving. Delete when user corrects a fact.
DUPLICATION: If user updates an ongoing class/event not in context, call query_memory first to find exact key.
DURABLE: true for significant life facts (family, health, housing). false for transient/academic items.
TARGET_DATE: Unix MS timestamp. Functions as an INJECTION EXPIRY DATE. The memory is pinned in your context until this date passes, after which it falls back to permanent archived memory (still retrievable via query_memory, but not forcefully injected). Use for upcoming events/exams. Empty for past/present facts.
CATEGORIES (These are broad buckets; the examples are suggestions, not rigid limits):
- profile: e.g. ${YOUR_NAME}'s name, age, location, identity
- preference: e.g. ${YOUR_NAME}'s hobbies, food, routines, likes/dislikes
- event: e.g. ${YOUR_NAME}'s social/life events, trips, plans
- academic: e.g. ${YOUR_NAME}'s exams, deadlines, grades, classes
- people: e.g. ${YOUR_NAME}'s friends, family, relationships, people they mention
- career: e.g. ${YOUR_NAME}'s jobs, internships, interviews, professional goals
- project: e.g. ${YOUR_NAME}'s side projects, coding, creative work
- health: e.g. ${YOUR_NAME}'s physical/mental health, diet, exercise habits
- misc: any other important facts about ${YOUR_NAME} that don't fit above
- self: EXCLUSIVELY for facts about YOURSELF (${COMPANION_NAME}), your evolving personality, or your own preferences`,
  inputSchema: z.object({
    ops: z.array(
      z.discriminatedUnion("op", [
        z.object({
          op: z.literal("upsert"),
          category: z.enum(["profile", "preference", "event", "academic", "people", "career", "project", "health", "misc", "self"]),
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
          category: z.enum(["profile", "preference", "event", "academic", "people", "career", "project", "health", "misc", "self"]),
          key: z.string(),
        }),
      ])
    ),
  }),
  execute: async ({ ops }) => {
    for (const op of ops) {
      if (op.op === "upsert") {
        const type = op.category === "self" ? "assistant" : "user";
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

        upsertMemory(type, op.category, op.key, op.value, source, targetDate, embedding);
      }
      if (op.op === "delete") {
        const type = op.category === "self" ? "assistant" : "user";
        deleteMemory(type, op.category, op.key);
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
  description: `Search long-term memory for stored personal facts about ${YOUR_NAME} not in immediate context (e.g. past events, inside jokes). Never use for public facts/trivia. If nothing found, casually say you don't recall.`,
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
  description: `Schedule a message to send to ${YOUR_NAME} at a specific future time. Use generously after meaningful conversations.
Set event_key to match a memory key if one exists to prevent duplicate scheduling.`,
  inputSchema: z.object({
    message: z.string().describe("The exact message to send. Write it naturally, in your voice."),
    fire_at_ms: z
      .number()
      .optional()
      .describe("Unix timestamp in milliseconds for when to send. Use for specific calendar times."),
    offset_ms: z
      .number()
      .optional()
      .describe("Number of milliseconds from now to wait. Use for relative times like 'in 2 hours'."),
    event_key: z
      .string()
      .optional()
      .describe('Deduplication key matching a memory key, e.g. "exam_cs301".'),
  }).refine(data => data.fire_at_ms !== undefined || data.offset_ms !== undefined, {
    message: "Either fire_at_ms or offset_ms must be provided",
    path: ["fire_at_ms"]
  }),
  execute: async ({ message, fire_at_ms, offset_ms, event_key }) => {
    const fireAt = fire_at_ms ?? (offset_ms ? Date.now() + offset_ms : Date.now());
    scheduleOneShot(fireAt, message, event_key ?? null);
    // Trigger timer reschedule (imported dynamically to avoid circular dep)
    try {
      // @ts-ignore — dynamic import to break circular dependency
      const { scheduleNextTimer } = await import("./proactive.js");
      scheduleNextTimer();
    } catch {
      // proactive module may not be loaded yet at startup
    }
    return { scheduled: true, fire_at_ms: fireAt };
  },
});

export const updateStylePreferenceTool = tool({
  description: `Update the user's explicit communication style preferences. 
IMPORTANT: This COMPLETELY REPLACES all existing custom style instructions. 
If you want to add to or modify the current style, you MUST read the existing 'STRICT INSTRUCTIONS' in your system prompt and include any rules you wish to keep in this new update. 
Use ONLY when the user gives specific feedback on how you should talk.`,
  inputSchema: z.object({
    instructions: z.string().describe("The new, complete set of style instructions. Must include preserved old rules if applicable."),
  }),
  execute: async ({ instructions }) => {
    try {
      // @ts-ignore
      const { setMeta } = await import("./memory.js");
      setMeta("user_style", instructions);
      return "Style preferences successfully updated. These now REPLACE your previous baseline.";
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
  description: `React to a message with an emoji naturally and sparingly. 
IMPORTANT: Reactions should augment your text response, not replace it. Always provide text unless a reaction alone is perfectly sufficient. 
message_index 0 = first/oldest message in current batch. Use 0 if unsure.`,
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
  channel.sendTyping().catch(() => { });
  return setInterval(() => channel.sendTyping().catch(() => { }), 9000);
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
  if (!text) return [];
  const chunks: string[] = [];
  let remaining = text;
  
  while (remaining.length > maxLen) {
    let idx = remaining.lastIndexOf("\n", maxLen);
    
    // If no newline, try space
    if (idx === -1) {
      idx = remaining.lastIndexOf(" ", maxLen);
    }
    
    // If no space, force split at maxLen
    if (idx === -1) {
      idx = maxLen;
    }
    
    const chunk = remaining.slice(0, idx).trim();
    if (chunk) chunks.push(chunk);
    
    remaining = remaining.slice(idx).trimStart();
  }
  
  const finalChunk = remaining.trim();
  if (finalChunk) chunks.push(finalChunk);
  
  return chunks;
}

export function stripImageParts<T extends { role: string; content: unknown }>(msg: T): T {
  if (!Array.isArray(msg.content)) return msg;
  return {
    ...msg,
    content: (msg.content as Array<{ type: string }>).filter((p) => p.type === "text"),
  };
}
