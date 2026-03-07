import { getSummary, setSummary, archiveOldestMessages, type HistoryEntry } from "./memory.js";
import { safeGenerateText, ALT_MODEL, FALLBACK_MODEL, stripImageParts } from "./helpers.js";

const COMPRESS_THRESHOLD = 30;
const TOKEN_THRESHOLD = 6000;

export async function maybeCompress(history: HistoryEntry[]): Promise<HistoryEntry[]> {
  const estimated = JSON.stringify(history).length / 4;

  if (history.length < COMPRESS_THRESHOLD && estimated < TOKEN_THRESHOLD) {
    return history;
  }

  const mid = Math.floor(history.length / 2);
  const oldHalf = history.slice(0, mid);
  const recentHalf = history.slice(mid);
  const prevSummary = getSummary();

  const content = prevSummary
    ? `Previous summary:\n${prevSummary}\n\nNow summarize the following new conversation. Preserve key facts, preferences, decisions, and important context.\n\n${JSON.stringify(oldHalf)}`
    : `Summarize this conversation concisely. Preserve key facts, preferences, decisions, and important context.\n\n${JSON.stringify(oldHalf)}`;

  try {
    const { text } = await safeGenerateText(
      {
        model: ALT_MODEL,
        messages: [{ role: "user", content }],
        maxOutputTokens: 400,
      },
      FALLBACK_MODEL
    );

    setSummary(text);
    archiveOldestMessages(mid);
    return recentHalf;
  } catch (err) {
    console.error("[compressor]", (err as Error).message);
    return history;
  }
}
