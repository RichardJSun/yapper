import { z } from "zod";

const envSchema = z.object({
  DISCORD_TOKEN: z.string().min(1, "DISCORD_TOKEN is required"),
  VERCEL_AI_GATEWAY_KEY: z.string().min(1, "VERCEL_AI_GATEWAY_KEY is required"),
  MY_DISCORD_ID: z.string().min(1, "MY_DISCORD_ID is required"),
  YOUR_NAME: z.string().min(1, "YOUR_NAME is required"),
  COMPANION_NAME: z.string().min(1, "COMPANION_NAME is required"),
  TZ: z.string().default("America/New_York"),
  ENABLE_TELEMETRY: z.string().optional().transform(v => v === "true"),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error("❌ Invalid environment variables:");
  for (const issue of parsed.error.issues) {
    console.error(`   ${issue.path.join(".")}: ${issue.message}`);
  }
  process.exit(1);
}

export const config = parsed.data;

// ── Model constants (single source of truth) ───────────────
// Primary conversational model used for responding to user messages
export const MODEL = "deepseek/deepseek-v3.2";
// Used for responding when there are image attachments
export const VISION_MODEL = "alibaba/qwen3.5-flash";
// Used for lower-priority background tasks: proactive messages, summaries, and onboarding
export const ALT_MODEL = "deepseek/deepseek-v3.2-thinking";
// Used as a reliable fallback when the primary model hits rate limits or API errors
export const FALLBACK_MODEL = "openai/gpt-4o-mini";
// Used to generate vector embeddings for semantic memory search
export const EMBED_MODEL = "openai/text-embedding-3-small";
