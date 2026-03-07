import { getSummary, setSummary, archiveOldestMessages, type HistoryEntry } from "./memory.js";
import { safeGenerateText, stripImageParts } from "./helpers.js";
import { config, ALT_MODEL, FALLBACK_MODEL } from "./config.js";
import { encode } from "@toon-format/toon";

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

  const transcriptData = oldHalf
    .map((msg) => {
      let text = "";
      if (typeof msg.content === "string") {
        text = msg.content;
      } else if (Array.isArray(msg.content)) {
        text = msg.content
          .filter((p: any) => p?.type === "text")
          .map((p: any) => p.text)
          .join(" ");
      } else {
        text = JSON.stringify(msg.content);
      }

      const dateStr = new Date(msg.created_at).toLocaleString("en-US", {
        timeZone: config.TZ,
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });

      return {
        time: dateStr,
        role: msg.role === "user" ? "USER" : "ASSISTANT",
        text,
      };
    });

  const transcript = encode(transcriptData);

  const content = prevSummary
    ? `Previous summary:\n${prevSummary}\n\nNow summarize the following new conversation. Preserve key facts, preferences, decisions, and important context.\n\n${transcript}`
    : `Summarize this conversation concisely. Preserve key facts, preferences, decisions, and important context.\n\n${transcript}`;

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
