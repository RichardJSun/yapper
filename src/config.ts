import { z } from "zod";

const envSchema = z.object({
  DISCORD_TOKEN: z.string().min(1, "DISCORD_TOKEN is required"),
  VERCEL_AI_GATEWAY_KEY: z.string().min(1, "VERCEL_AI_GATEWAY_KEY is required"),
  MY_DISCORD_ID: z.string().min(1, "MY_DISCORD_ID is required"),
  YOUR_NAME: z.string().min(1, "YOUR_NAME is required"),
  COMPANION_NAME: z.string().min(1, "COMPANION_NAME is required"),
  TZ: z.string().default("America/New_York"),
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
export const MODEL = "moonshotai/kimi-k2.5";
export const VISION_MODEL = "moonshotai/kimi-k2.5";
export const ALT_MODEL = "deepseek/deepseek-v3.2-thinking";
export const FALLBACK_MODEL = "openai/gpt-4o-mini";
export const EMBED_MODEL = "openai/text-embedding-3-small";
