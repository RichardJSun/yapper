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
  getNextScheduledMessage,
  markSent,
} from "./memory.js";
import {
  safeGenerateText,
  isSendableChannel,
  chunkText,
  stripImageParts,
} from "./helpers.js";
import { ALT_MODEL, FALLBACK_MODEL } from "./config.js";
import { eventBus } from "./eventBus.js";

const { YOUR_NAME, COMPANION_NAME, MY_DISCORD_ID, TZ } = config;
const IDLE_THRESHOLD_H = 8;
const IDLE_NUDGE_COOLDOWN_H = 6;

// ── Module state ───────────────────────────────────────────

let discordClient: Client | null = null;
let currentOneShotTimer: ReturnType<typeof setTimeout> | null = null;
let isExecutingScheduled = false;

const baker = Baker.create({
  persistence: {
    enabled: true,
    strategy: "file",
    provider: new FilePersistenceProvider("./cronbake-state.json"),
    autoRestore: true,
  },
});

// ── Send DM ────────────────────────────────────────────────

async function sendDM(text: string): Promise<void> {
  if (!discordClient) return;
  try {
    const user = await discordClient.users.fetch(MY_DISCORD_ID);
    const dmChannel = await user.createDM();
    if (!isSendableChannel(dmChannel)) return;
    const chunks = chunkText(text, 1900);
    for (const chunk of chunks) {
      await dmChannel.send(chunk);
    }
  } catch (err) {
    console.error("[sendDM]", (err as Error).message);
  }
}

// ── Event-driven scheduled message timer ───────────────────

export function scheduleNextTimer(): void {
  if (currentOneShotTimer) {
    clearTimeout(currentOneShotTimer);
    currentOneShotTimer = null;
  }

  // If we're already executing a scheduled message, don't re-enter.
  if (isExecutingScheduled) return;

  const nextMsg = getNextScheduledMessage();
  if (!nextMsg) return;

  const now = Date.now();
  const delay = nextMsg.fire_at - now;

  if (delay <= 0) {
    // If the message is older than 1 hour, it's likely stale or from a hallucinated past timestamp.
    // Mark it as sent to clear it from the queue without sending.
    if (Math.abs(delay) > 60 * 60 * 1000) {
      console.warn(`[proactive] Skipping stale scheduled message (ID: ${nextMsg.id}, fire_at: ${nextMsg.fire_at})`);
      markSent(nextMsg.id);
      scheduleNextTimer();
      return;
    }
    executeScheduledMessage(nextMsg.id, nextMsg.message);
  } else {
    currentOneShotTimer = setTimeout(
      () => executeScheduledMessage(nextMsg.id, nextMsg.message),
      delay
    );
  }
}

async function executeScheduledMessage(id: number, message: string): Promise<void> {
  // Claim-before-execute: mark sent first to prevent duplicate delivery
  // if scheduleNextTimer is called re-entrantly during the async DM send.
  isExecutingScheduled = true;
  markSent(id);
  try {
    await sendDM(message);
    addMessage("assistant", message, true);
  } catch (err) {
    console.error("[scheduled]", (err as Error).message);
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
        system: `You are ${COMPANION_NAME} texting ${YOUR_NAME}. NEVER use em-dashes (—) or semicolons. No robotic AI phrasing, cheerleader sign-offs, or forced engagement. Sound like a real friend.`,
        messages: [
          {
            role: "user" as const,
            content: `It's morning.
Current time: ${new Date().toLocaleString("en-US", { timeZone: TZ, dateStyle: "full", timeStyle: "short" })}
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
        system: `You are ${COMPANION_NAME} texting ${YOUR_NAME}. NEVER use em-dashes (—) or semicolons. No robotic AI phrasing, cheerleader sign-offs, or forced engagement. Sound like a real friend.`,
        messages: [
          {
            role: "user" as const,
            content: `It's evening.
Current time: ${new Date().toLocaleString("en-US", { timeZone: TZ, dateStyle: "full", timeStyle: "short" })}
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
