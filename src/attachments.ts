import type { Collection, Attachment } from "discord.js";

const SUPPORTED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "image/avif",
]);

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const FETCH_TIMEOUT = 10_000; // 10 seconds

export interface ImagePart {
  type: "image";
  image: string;
}

export async function processAttachments(
  attachments: Collection<string, Attachment>
): Promise<ImagePart[]> {
  const parts: ImagePart[] = [];

  for (const [, attachment] of attachments) {
    if (!attachment.contentType || !SUPPORTED_TYPES.has(attachment.contentType)) {
      continue;
    }
    if (attachment.size > MAX_FILE_SIZE) {
      console.warn(`[attachments] Skipping ${attachment.name}: exceeds 20MB`);
      continue;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

    try {
      const response = await fetch(attachment.url, { signal: controller.signal });
      clearTimeout(timeoutId);
      const buffer = await response.arrayBuffer();
      const base64 = Buffer.from(buffer).toString("base64");
      parts.push({
        type: "image",
        image: `${attachment.contentType};base64,${base64}`,
      });
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof Error && err.name === "AbortError") {
        console.warn(`[attachments] Timeout fetching ${attachment.name}`);
      } else {
        console.warn(`[attachments] Error fetching ${attachment.name}:`, err);
      }
    }
  }

  return parts;
}
