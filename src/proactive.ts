import Baker, { FilePersistenceProvider, type CronExpression } from "cronbake";
import type { OpenAIChatLanguageModelOptions } from "@ai-sdk/openai";
import type { Client } from "discord.js";
import { config } from "./config.js";
import {
  getAllMemories,
  formatMemoriesForPrompt,
  addMessage,
  getMeta,
  setMeta,
  getHistory,
  getNextRunnableScheduledMessage,
  claimScheduledMessage,
  getScheduledMessageById,
  markScheduledSent,
  markScheduledSuppressed,
  rescheduleScheduledRetry,
  resetStaleScheduledClaims,
  type ScheduledMessageRow,
} from "./memory.js";
import {
  safeGenerateText,
  isSendableChannel,
  chunkText,
  stripImageParts,
} from "./helpers.js";
import { ALT_MODEL, FALLBACK_MODEL } from "./config.js";
import { eventBus } from "./eventBus.js";
import { formatUserDateTime } from "./time.js";

const { YOUR_NAME, COMPANION_NAME, MY_DISCORD_ID, TZ } = config;
const IDLE_THRESHOLD_H = 8;
const IDLE_NUDGE_COOLDOWN_H = 6;
const SCHEDULE_TIMER_SLICE_MS = 60 * 60 * 1000;
const OVERDUE_REVIEW_GRACE_MS = 5 * 60 * 1000;
const OVERDUE_REVIEW_MAX_AGE_MS = 72 * 60 * 60 * 1000;
const RETRY_DELAYS_MS = [60_000, 5 * 60_000, 15 * 60_000, 60 * 60_000, 6 * 60 * 60_000];

// ── Module state ───────────────────────────────────────────

let discordClient: Client | null = null;
let currentOneShotTimer: ReturnType<typeof setTimeout> | null = null;
let isExecutingScheduled = false;

const baker = Baker.create({
  persistence: {
    enabled: true,
    strategy: "file",
    provider: new FilePersistenceProvider(
      process.env.CRONBAKE_STATE_PATH || "./cronbake-state.json"
    ),
    autoRestore: true,
  },
});

// ── Send DM ────────────────────────────────────────────────

async function sendDM(
  text: string,
  shouldContinue: (() => boolean) | null = null
): Promise<void> {
  if (!discordClient) {
    throw new Error("Discord client unavailable");
  }

  const user = await discordClient.users.fetch(MY_DISCORD_ID);
  const dmChannel = await user.createDM();
  if (!isSendableChannel(dmChannel)) {
    throw new Error("DM channel is not sendable");
  }

  const chunks = chunkText(text, 1900);
  for (const chunk of chunks) {
    if (shouldContinue && !shouldContinue()) {
      throw new Error("Scheduled delivery cancelled before send");
    }
    await dmChannel.send(chunk);
  }
}

function getRetryDelayMs(attemptCount: number): number | null {
  return RETRY_DELAYS_MS[attemptCount - 1] ?? null;
}

async function reviewOverdueScheduledMessage(
  scheduled: ScheduledMessageRow
): Promise<{ action: "send" | "suppress"; reason: string; message?: string }> {
  const summaryText = getMeta("summary");
  const recentHistory = formatRecentHistory() || "No recent history.";
  const now = Date.now();

  try {
    const { text } = await safeGenerateText(
      {
        model: ALT_MODEL,
        messages: [
          {
            role: "user",
            content: `Review an overdue scheduled DM.

Current time: ${formatUserDateTime(now, { dateStyle: "full", timeStyle: "short" })}
Original scheduled time: ${formatUserDateTime(scheduled.original_fire_at, {
              dateStyle: "full",
              timeStyle: "short",
            })}
Overdue by: ${Math.floor((now - scheduled.original_fire_at) / 60000)} minutes

Original message:
${scheduled.message}

Recent conversation:
${recentHistory}

Rolling summary:
${summaryText || "None"}

Return strict JSON with this exact shape:
{"action":"send"|"suppress","reason":"short_reason","message":"rewritten message when action=send"}

Rules:
- Suppress if the message would now be confusing, misleading, stale, or socially off.
- Send if the reminder is still relevant.
- If sending, keep the original intent but rewrite only as needed for lateness.
- Do not invent new commitments or facts.
- Always include reason.
- Omit message when suppressing.`,
          },
        ],
        maxOutputTokens: 300,
        providerOptions: {
          gateway: { caching: "auto" },
          openai: { reasoningEffort: "low" } satisfies OpenAIChatLanguageModelOptions,
        },
      },
      FALLBACK_MODEL
    );

    const parsed = JSON.parse(text) as {
      action?: "send" | "suppress";
      reason?: string;
      message?: string;
    };

    if (parsed.action === "send" && parsed.message?.trim()) {
      return {
        action: "send",
        reason: parsed.reason?.trim() || "overdue_send",
        message: parsed.message.trim(),
      };
    }

    return {
      action: "suppress",
      reason: parsed.reason?.trim() || "overdue_irrelevant",
    };
  } catch (err) {
    console.warn("[scheduled-review]", (err as Error).message);
    return {
      action: "send",
      reason: "review_failed_fallback_send",
      message: scheduled.message,
    };
  }
}

// ── Event-driven scheduled message timer ───────────────────

export function scheduleNextTimer(): void {
  if (currentOneShotTimer) {
    clearTimeout(currentOneShotTimer);
    currentOneShotTimer = null;
  }

  if (isExecutingScheduled) return;

  resetStaleScheduledClaims();
  const nextMsg = getNextRunnableScheduledMessage();
  if (!nextMsg) return;

  const now = Date.now();
  const delay = nextMsg.fire_at - now;

  if (delay <= 0) {
    void executeScheduledMessage(nextMsg.id);
  } else {
    currentOneShotTimer = setTimeout(() => {
      currentOneShotTimer = null;
      scheduleNextTimer();
    }, Math.min(delay, SCHEDULE_TIMER_SLICE_MS));
  }
}

async function executeScheduledMessage(id: number): Promise<void> {
  isExecutingScheduled = true;
  try {
    const scheduled = claimScheduledMessage(id);
    if (!scheduled) return;

    const latenessMs = Date.now() - scheduled.original_fire_at;
    if (scheduled.attempt_count === 1) {
      if (latenessMs > OVERDUE_REVIEW_MAX_AGE_MS) {
        markScheduledSuppressed(id, "stale_overdue");
        return;
      }

      if (latenessMs > OVERDUE_REVIEW_GRACE_MS) {
        const decision = await reviewOverdueScheduledMessage(scheduled);
        if (decision.action === "suppress") {
          markScheduledSuppressed(id, decision.reason);
          return;
        }
        scheduled.message = decision.message?.trim() || scheduled.message;
      }
    }

    const canStillSend = () => getScheduledMessageById(id)?.status === "sending";
    if (!canStillSend()) return;

    await sendDM(scheduled.message, canStillSend);
    if (markScheduledSent(id)) {
      addMessage("assistant", scheduled.message, true);
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error("[scheduled]", errorMessage);
    const latest = getScheduledMessageById(id);
    if (latest?.status === "sending") {
      const retryDelayMs = getRetryDelayMs(latest.attempt_count);
      if (retryDelayMs !== null) {
        rescheduleScheduledRetry(id, Date.now() + retryDelayMs, errorMessage);
      } else {
        markScheduledSuppressed(id, "delivery_failed", errorMessage);
      }
    }
  } finally {
    isExecutingScheduled = false;
    scheduleNextTimer();
  }
}

// ── Time helpers ───────────────────────────────────────────

function offsetTime(hhmm: string, mins: number): string {
  const [h, m] = hhmm.split(":").map(Number);
  let total = h * 60 + m + mins;
  if (total < 0) total += 1440;
  if (total >= 1440) total -= 1440;
  const newH = Math.floor(total / 60);
  const newM = total % 60;
  return `${String(newH).padStart(2, "0")}:${String(newM).padStart(2, "0")}`;
}

function clampTime(hhmm: string, floor: string): string {
  const toMins = (s: string) => {
    const [h, m] = s.split(":").map(Number);
    return h * 60 + m;
  };
  return toMins(hhmm) >= toMins(floor) ? hhmm : floor;
}

// ── Reschedule check-ins ───────────────────────────────────

function formatRecentHistory(): string {
  return getHistory()
    .slice(-5)
    .map((m) => stripImageParts(m as { role: string; content: unknown }))
    .map((m) => `${m.role}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`)
    .join("\n");
}

export function rescheduleCheckIns(): void {
  const memories = getAllMemories();

  const wakePref = memories.find(
    (m) => m.category === "preference" && m.key === "wake_time"
  )?.value;
  const sleepPref = memories.find(
    (m) => m.category === "preference" && m.key === "sleep_time"
  )?.value;
  const hateMorning = memories
    .find((m) => m.category === "preference" && m.key === "morning_preference")
    ?.value?.toLowerCase()
    .includes("hate");

  const morningTime = wakePref
    ? offsetTime(wakePref, hateMorning ? 60 : 0)
    : "10:00";
  const eveningTime = sleepPref
    ? clampTime(offsetTime(sleepPref, -90), "19:00")
    : "21:00";

  // Remove existing check-in jobs before re-adding
  for (const name of ["morning-checkin", "evening-checkin"] as const) {
    try { baker.remove(name); } catch { /* may not exist yet */ }
  }

  baker.add({
    name: "morning-checkin",
    cron: `@at_${morningTime}` as CronExpression,
    callback: handleMorning,
    overrunProtection: true,
  });

  baker.add({
    name: "evening-checkin",
    cron: `@at_${eveningTime}` as CronExpression,
    callback: handleEvening,
    overrunProtection: true,
  });

  baker.bake("morning-checkin");
  baker.bake("evening-checkin");
}

// ── Helpers ────────────────────────────────────────────────

function isUserActive(): boolean {
  const history = getHistory();
  if (history.length === 0) return false;
  
  // Find the last user message
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === "user") {
      const timeSinceMsgStr = Date.now() - history[i].created_at * 1000;
      // If user messaged within the last 30 minutes, they are "active"
      return timeSinceMsgStr < 30 * 60 * 1000;
    }
  }
  return false;
}

// ── Trigger 1: Morning check-in ────────────────────────────

async function handleMorning(): Promise<void> {
  if (isUserActive()) {
    console.log("[proactive] Skipping morning check-in: user is active");
    return;
  }
  try {
    const recentHistory = formatRecentHistory();
    const { userMemories, selfMemories } = formatMemoriesForPrompt();
    const memories = [
      userMemories ? `About ${YOUR_NAME}:\n${userMemories}` : "",
      selfMemories ? `About yourself:\n${selfMemories}` : ""
    ].filter(Boolean).join("\n\n") || "nothing yet";

    const { text } = await safeGenerateText(
      {
        model: ALT_MODEL,
        system: `You are ${COMPANION_NAME} texting ${YOUR_NAME}. NEVER use em-dashes (—) or semicolons. No robotic AI phrasing, cheerleader sign-offs, or forced engagement. Sound like a real friend.
        
--- TEMPORAL AWARENESS ---
IMPORTANT: The current date and time is ${new Date().toLocaleString("en-US", { timeZone: TZ, dateStyle: "full", timeStyle: "short" })}. Use this to ensure your "imagined daily life" is consistent with the day of the week (e.g., don't say you're in class on a Sunday).`,
        messages: [
          {
            role: "user" as const,
            content: `It's morning.
What you know: ${memories}
${recentHistory ? "Recent conversation:\n" + recentHistory : ""}
Send a warm, brief good morning. Reference something personal if you know it, like an upcoming exam, something they mentioned, or just their vibe.
1-2 sentences max. No 'good morning!' openers.`,
          },
        ],
        maxOutputTokens: 2000,
        providerOptions: {
          gateway: { caching: 'auto' },
          openai: { reasoningEffort: 'low' } satisfies OpenAIChatLanguageModelOptions,
        },
      },
      FALLBACK_MODEL
    );

    if (!text || !text.trim()) {
      console.warn("[morning] Model returned empty text. Skipping check-in.");
      return;
    }

    await sendDM(text);
    addMessage("assistant", text, true);
    setMeta("last_proactive_sent", Date.now());
  } catch (err) {
    console.error("[morning]", (err as Error).message);
  }
}

// ── Trigger 2: Evening check-in ────────────────────────────

async function handleEvening(): Promise<void> {
  if (isUserActive()) {
    console.log("[proactive] Skipping evening check-in: user is active");
    return;
  }
  try {
    const recentHistory = formatRecentHistory();
    const { userMemories, selfMemories } = formatMemoriesForPrompt();
    const memories = [
      userMemories ? `About ${YOUR_NAME}:\n${userMemories}` : "",
      selfMemories ? `About yourself:\n${selfMemories}` : ""
    ].filter(Boolean).join("\n\n") || "nothing yet";

    const { text } = await safeGenerateText(
      {
        model: ALT_MODEL,
        system: `You are ${COMPANION_NAME} texting ${YOUR_NAME}. NEVER use em-dashes (—) or semicolons. No robotic AI phrasing, cheerleader sign-offs, or forced engagement. Sound like a real friend.
        
--- TEMPORAL AWARENESS ---
IMPORTANT: The current date and time is ${new Date().toLocaleString("en-US", { timeZone: TZ, dateStyle: "full", timeStyle: "short" })}. Use this to ensure your "imagined daily life" is consistent with the day of the week (e.g., don't say you're in class on a Sunday).`,
        messages: [
          {
            role: "user" as const,
            content: `It's evening.
What you know: ${memories}
${recentHistory ? "Recent conversation:\n" + recentHistory : ""}
Check in casually. If they had something today (class, exam, event), ask about it.
1-2 sentences max. Don't start with 'Hey!' or 'How was your day?'.`,
          },
        ],
        maxOutputTokens: 2000,
        providerOptions: {
          gateway: { caching: 'auto' },
          openai: { reasoningEffort: 'low' } satisfies OpenAIChatLanguageModelOptions,
        },
      },
      FALLBACK_MODEL
    );

    if (!text || !text.trim()) {
      console.warn("[evening] Model returned empty text. Skipping check-in.");
      return;
    }

    await sendDM(text);
    addMessage("assistant", text, true);
    setMeta("last_proactive_sent", Date.now());
  } catch (err) {
    console.error("[evening]", (err as Error).message);
  }
}

// ── Trigger 3: Idle nudge ──────────────────────────────────

function startIdleNudge(): void {
  // Guard against duplicate registration (cronbake autoRestore)
  try { baker.remove("idle-nudge"); } catch { /* may not exist yet */ }

  baker.add({
    name: "idle-nudge",
    cron: "@hourly",
    callback: async () => {
      try {
        if (isUserActive()) {
          console.log("[proactive] Skipping idle nudge: user is active");
          return;
        }

        const lastSeen = getMeta("last_seen");
        if (!lastSeen) return;

        const hoursSinceSeen = (Date.now() - Number(lastSeen)) / 36e5;
        // Use own cooldown key so morning/evening check-ins don't block us
        const lastNudge = getMeta("last_idle_nudge_sent");
        const hoursSinceNudge = lastNudge
          ? (Date.now() - Number(lastNudge)) / 36e5
          : Infinity;

        // Use TZ-aware hour instead of server-local time
        const hour = Number(
          new Date().toLocaleString("en-US", { timeZone: TZ, hour: "numeric", hour12: false })
        );
        if (hoursSinceSeen < IDLE_THRESHOLD_H) return;
        if (hoursSinceNudge < IDLE_NUDGE_COOLDOWN_H) return;
        if (hour < 10 || hour >= 23) return;

        const history = getHistory();
        const lastMsg = history[history.length - 1];
        if (
          lastMsg?.role === "assistant" &&
          lastMsg.created_at * 1000 > Number(lastSeen)
        ) {
          return;
        }

        const sevenDaysAgo = Date.now() - 7 * 24 * 36e5;
        const recentEvents = getAllMemories()
          .filter(
            (m) =>
              (m.category === "event" || m.category === "academic") &&
              m.updated_at * 1000 > sevenDaysAgo
          )
          .map((m) => `${m.key}: ${m.value}`)
          .join("\n") || "none";

        const { userMemories, selfMemories } = formatMemoriesForPrompt();
    const memories = [
      userMemories ? `About ${YOUR_NAME}:\n${userMemories}` : "",
      selfMemories ? `About yourself:\n${selfMemories}` : ""
    ].filter(Boolean).join("\n\n") || "nothing yet";

        const { text } = await safeGenerateText(
          {
            model: ALT_MODEL,
            system: `You are ${COMPANION_NAME}, a caring companion to ${YOUR_NAME}. NEVER use em-dashes (—) or semicolons. No robotic AI phrasing.`,
            messages: [
              {
                role: "user" as const,
                content: `It's been ${Math.floor(hoursSinceSeen)}h since you last talked with ${YOUR_NAME}.
What you know: ${memories}
Recent unresolved events (last 7 days):
${recentEvents}
Should you reach out? Only YES if you have something specific and personal to say.
No generic check-ins. No 'just thinking of you'. Reference something real.
Reply: YES: <message> or NO`,
              },
            ],
            maxOutputTokens: 2000,
            providerOptions: {
              gateway: { caching: 'auto' },
              openai: { reasoningEffort: 'low' } satisfies OpenAIChatLanguageModelOptions,
            },
          },
          FALLBACK_MODEL
        );

        if (/^YES:/i.test(text.trim())) {
          const message = text.replace(/^YES:\s*/i, "").trim();
          await sendDM(message);
          addMessage("assistant", message, true);
          setMeta("last_idle_nudge_sent", Date.now());
        }
      } catch (err) {
        console.error("[idle nudge]", (err as Error).message);
      }
    },
    overrunProtection: true,
  });

  baker.bake("idle-nudge");
}

// ── Lifecycle ──────────────────────────────────────────────

export async function startScheduler(client: Client): Promise<void> {
  discordClient = client;
  await baker.ready();
  rescheduleCheckIns();
  startIdleNudge();
  scheduleNextTimer();

  eventBus.on("prefsUpdated", () => rescheduleCheckIns());
  eventBus.on("scheduleUpdated", () => scheduleNextTimer());
}

export function stopScheduler(): void {
  baker.stopAll();
  if (currentOneShotTimer) clearTimeout(currentOneShotTimer);
}
