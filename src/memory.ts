import { randomUUID } from "node:crypto";
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { existsSync } from "node:fs";
import { config } from "./config.js";
import { encode } from "@toon-format/toon";
import { formatUserDate, formatUserDateTime } from "./time.js";

export type QueueStatus = "pending" | "processing" | "failed";
export type ScheduledMessageStatus =
  | "pending"
  | "sending"
  | "sent"
  | "cancelled"
  | "suppressed";
export type StoredContentKind = "text" | "json" | "legacy";

export const STALE_QUEUE_CLAIM_MS = 5 * 60 * 1000;
export const STALE_SCHEDULED_CLAIM_MS = 5 * 60 * 1000;

// ── Row types ──────────────────────────────────────────────

export interface MessageRow {
  id: number;
  role: string;
  content: string;
  content_kind: StoredContentKind;
  is_archived: number;
  created_at: number;
  is_proactive: number;
}

export interface QueueRow {
  id: number;
  channel_id: string;
  message_id: string;
  text_content: string | null;
  image_parts: string;
  status: QueueStatus;
  claim_token: string | null;
  claimed_at: number | null;
  attempt_count: number;
  last_error: string | null;
  created_at: number;
}

export interface MemoryRow {
  id: number;
  type: "user" | "assistant";
  category: string;
  key: string;
  value: string;
  source: string;
  target_date: number | null;
  updated_at: number;
}

export interface ScheduledMessageRow {
  id: number;
  fire_at: number;
  original_fire_at: number;
  message: string;
  event_key: string | null;
  sent: number;
  status: ScheduledMessageStatus;
  attempt_count: number;
  last_attempt_at: number | null;
  sent_at: number | null;
  cancelled_at: number | null;
  suppressed_at: number | null;
  last_error: string | null;
  suppression_reason: string | null;
}

export interface VecSearchRow extends MemoryRow {
  distance: number;
}

export interface HistoryEntry {
  role: string;
  content: unknown;
  created_at: number;
  is_proactive: boolean;
}

export interface ClaimedQueueItem {
  id: number;
  channel_id: string;
  message_id: string;
  text_content: string | null;
  image_parts: unknown[];
  created_at: number;
  attempt_count: number;
}

export interface ClaimedQueueBatch {
  claim_token: string;
  channel_id: string;
  items: ClaimedQueueItem[];
}

// ── Load custom SQLite on macOS (required for extension support) ──

if (process.platform === "darwin") {
  const armPath = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
  const intelPath = "/usr/local/opt/sqlite3/lib/libsqlite3.dylib";
  const libPath = existsSync(armPath) ? armPath : intelPath;

  if (existsSync(libPath)) {
    Database.setCustomSQLite(libPath);
  } else {
    console.warn(
      "[memory] macOS detected but Homebrew SQLite not found. Run: brew install sqlite"
    );
    console.warn("[memory] Vector search will not work without it.");
  }
}

const dbPath = process.env.MEMORY_DB_PATH || "memory.db";
const db = new Database(dbPath, { create: true });
db.exec("PRAGMA journal_mode = WAL;");
db.exec("PRAGMA foreign_keys = ON;");
sqliteVec.load(db);

db.exec(`
  CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_kind  TEXT NOT NULL DEFAULT 'legacy',
    is_archived   INTEGER DEFAULT 0,
    created_at    INTEGER DEFAULT (strftime('%s','now')),
    is_proactive  INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS inbound_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id    TEXT NOT NULL,
    message_id    TEXT NOT NULL,
    text_content  TEXT,
    image_parts   TEXT,
    status        TEXT DEFAULT 'pending',
    claim_token   TEXT DEFAULT NULL,
    claimed_at    INTEGER DEFAULT NULL,
    attempt_count INTEGER DEFAULT 0,
    last_error    TEXT DEFAULT NULL,
    created_at    INTEGER DEFAULT (strftime('%s','now'))
  );

  CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT DEFAULT 'user',
    category    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT DEFAULT 'auto',
    target_date INTEGER DEFAULT NULL,
    updated_at  INTEGER DEFAULT (strftime('%s','now')),
    UNIQUE(type, category, key)
  );

  CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    id        INTEGER PRIMARY KEY,
    embedding float[1536]
  );

  CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS scheduled_messages (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_at            INTEGER NOT NULL,
    original_fire_at   INTEGER DEFAULT NULL,
    message            TEXT NOT NULL,
    event_key          TEXT,
    sent               INTEGER DEFAULT 0,
    status             TEXT NOT NULL DEFAULT 'pending',
    attempt_count      INTEGER NOT NULL DEFAULT 0,
    last_attempt_at    INTEGER DEFAULT NULL,
    sent_at            INTEGER DEFAULT NULL,
    cancelled_at       INTEGER DEFAULT NULL,
    suppressed_at      INTEGER DEFAULT NULL,
    last_error         TEXT DEFAULT NULL,
    suppression_reason TEXT DEFAULT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_messages_archived ON messages(is_archived);
  CREATE INDEX IF NOT EXISTS idx_queue_status ON inbound_queue(status, created_at);
  CREATE INDEX IF NOT EXISTS idx_scheduled_due ON scheduled_messages(status, fire_at);
`);

function hasColumn(table: string, column: string): boolean {
  const rows = db
    .query<{ name: string }, []>(`PRAGMA table_info(${table})`)
    .all();
  return rows.some((row) => row.name === column);
}

function ensureColumn(table: string, definition: string): void {
  const columnName = definition.trim().split(/\s+/)[0];
  if (!hasColumn(table, columnName)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${definition}`);
  }
}

function runMigrations(): void {
  ensureColumn("messages", "content_kind TEXT NOT NULL DEFAULT 'legacy'");

  ensureColumn("inbound_queue", "claim_token TEXT DEFAULT NULL");
  ensureColumn("inbound_queue", "claimed_at INTEGER DEFAULT NULL");
  ensureColumn("inbound_queue", "attempt_count INTEGER NOT NULL DEFAULT 0");
  ensureColumn("inbound_queue", "last_error TEXT DEFAULT NULL");

  ensureColumn("scheduled_messages", "original_fire_at INTEGER DEFAULT NULL");
  ensureColumn("scheduled_messages", "status TEXT NOT NULL DEFAULT 'pending'");
  ensureColumn("scheduled_messages", "attempt_count INTEGER NOT NULL DEFAULT 0");
  ensureColumn("scheduled_messages", "last_attempt_at INTEGER DEFAULT NULL");
  ensureColumn("scheduled_messages", "sent_at INTEGER DEFAULT NULL");
  ensureColumn("scheduled_messages", "cancelled_at INTEGER DEFAULT NULL");
  ensureColumn("scheduled_messages", "suppressed_at INTEGER DEFAULT NULL");
  ensureColumn("scheduled_messages", "last_error TEXT DEFAULT NULL");
  ensureColumn("scheduled_messages", "suppression_reason TEXT DEFAULT NULL");

  db.exec(
    `UPDATE messages
     SET content_kind = CASE
       WHEN content_kind IN ('text', 'json') THEN content_kind
       ELSE 'legacy'
     END`
  );
  db.exec(
    `UPDATE scheduled_messages
     SET original_fire_at = fire_at
     WHERE original_fire_at IS NULL`
  );
  db.exec(
    `UPDATE scheduled_messages
     SET status = CASE
       WHEN status IN ('pending', 'sending', 'sent', 'cancelled', 'suppressed') THEN status
       WHEN sent = 1 THEN 'sent'
       ELSE 'pending'
     END`
  );
}

runMigrations();

export function rerunMigrationsForTests(): void {
  runMigrations();
}

const MAX_MESSAGES = 40;

function decodeLegacyContent(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) return raw;
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }
  return raw;
}

function decodeMessageContent(row: Pick<MessageRow, "content" | "content_kind">): unknown {
  if (row.content_kind === "text") {
    return row.content;
  }
  if (row.content_kind === "json") {
    try {
      return JSON.parse(row.content);
    } catch {
      return row.content;
    }
  }
  return decodeLegacyContent(row.content);
}

function coerceQueueImages(imageParts: string): unknown[] {
  try {
    const parsed = JSON.parse(imageParts);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    throw new Error("Invalid queued image payload");
  }
}

// ── Messages ───────────────────────────────────────────────

export function addMessage(
  role: "user" | "assistant",
  content: string | unknown,
  isProactive: boolean = false
): void {
  const contentKind: StoredContentKind =
    typeof content === "string" ? "text" : "json";
  const text = typeof content === "string" ? content : JSON.stringify(content);

  db.query(
    `INSERT INTO messages (role, content, content_kind, is_proactive)
     VALUES ($role, $content, $contentKind, $isProactive)`
  ).run({
    $role: role,
    $content: text,
    $contentKind: contentKind,
    $isProactive: isProactive ? 1 : 0,
  });

  const { count } = db.query<{ count: number }, []>(
    `SELECT COUNT(*) as count FROM messages WHERE is_archived = 0`
  ).get()!;

  if (count > MAX_MESSAGES) {
    db.query(
      `UPDATE messages SET is_archived = 1 WHERE id IN (
        SELECT id FROM messages WHERE is_archived = 0 ORDER BY id ASC LIMIT $n
      )`
    ).run({ $n: count - MAX_MESSAGES });
  }
}

export function getHistory(): HistoryEntry[] {
  const rows = db.query<MessageRow, []>(
    `SELECT role, content, content_kind, created_at, is_proactive
     FROM messages
     WHERE is_archived = 0
     ORDER BY id ASC`
  ).all();

  return rows.map((row) => ({
    role: row.role,
    content: decodeMessageContent(row),
    created_at: row.created_at,
    is_proactive: row.is_proactive === 1,
  }));
}

export function archiveOldestMessages(n: number): void {
  db.query(
    `UPDATE messages SET is_archived = 1 WHERE id IN (
      SELECT id FROM messages WHERE is_archived = 0 ORDER BY id ASC LIMIT $n
    )`
  ).run({ $n: n });
}

export function deleteOldestMessages(n: number): void {
  db.query(
    `DELETE FROM messages WHERE id IN (
      SELECT id FROM messages ORDER BY id ASC LIMIT $n
    )`
  ).run({ $n: n });
}

// ── Summary ────────────────────────────────────────────────

export function getSummary(): string | null {
  return getMeta("summary") || null;
}

export function setSummary(text: string): void {
  setMeta("summary", text);
}

// ── Memories ───────────────────────────────────────────────

export function upsertMemory(
  type: "user" | "assistant",
  category: string,
  key: string,
  value: string,
  source: string = "auto",
  targetDate: number | null = null,
  embeddingArray?: number[]
): void {
  db.transaction(() => {
    const result = db.query<
      { id: number },
      [string, string, string, string, string, number | null]
    >(
      `INSERT INTO memories (type, category, key, value, source, target_date, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
       ON CONFLICT(type, category, key) DO UPDATE SET
         value = excluded.value,
         source = excluded.source,
         target_date = excluded.target_date,
         updated_at = strftime('%s','now')
       RETURNING id`
    ).get(type, category, key, value, source, targetDate);

    if (result && embeddingArray) {
      db.query(`DELETE FROM vec_memories WHERE id = ?`).run(result.id);
      db.query(
        `INSERT INTO vec_memories (id, embedding) VALUES (?, ?)`
      ).run(result.id, new Float32Array(embeddingArray));
    }
  })();
}

export function deleteMemory(
  type: "user" | "assistant",
  category: string,
  key: string
): void {
  const row = db.query<{ id: number }, [string, string, string]>(
    `SELECT id FROM memories WHERE type = ? AND category = ? AND key = ?`
  ).get(type, category, key);

  db.query(
    `DELETE FROM memories WHERE type = ? AND category = ? AND key = ?`
  ).run(type, category, key);

  if (row) {
    db.query(`DELETE FROM vec_memories WHERE id = ?`).run(row.id);
  }
}

export function getAllMemories(): MemoryRow[] {
  return db.query<MemoryRow, []>(
    `SELECT * FROM memories ORDER BY type, category, key`
  ).all();
}

export function hasCategoryMemory(category: string): boolean {
  return (
    db.query<{ id: number }, [string]>(
      `SELECT id FROM memories WHERE category = ? LIMIT 1`
    ).get(category) !== null
  );
}

export function searchMemories(term: string): string | null {
  const wildcard = `%${term}%`;
  const rows = db.query<MemoryRow, [string, string]>(
    `SELECT type, category, key, value, updated_at, source, target_date
     FROM memories
     WHERE key LIKE ? OR value LIKE ?
     ORDER BY updated_at DESC
     LIMIT 10`
  ).all(wildcard, wildcard);

  if (rows.length === 0) return null;

  return rows
    .map((r) => {
      const owner = r.type === "assistant" ? "[SELF]" : `[${r.category.toUpperCase()}]`;
      return `${owner} ${r.key}: ${r.value} (Saved: ${formatUserDate(
        r.updated_at * 1000
      )})`;
    })
    .join("\n");
}

export function searchMemoriesSemantic(embeddingArray: number[]): VecSearchRow[] {
  return db.query<VecSearchRow, [Float32Array]>(`
    SELECT m.type, m.category, m.key, m.value, m.updated_at, m.source, m.target_date, v.distance
    FROM vec_memories v
    JOIN memories m ON v.id = m.id
    WHERE v.embedding MATCH ? AND k = 5
    ORDER BY distance ASC
  `).all(new Float32Array(embeddingArray));
}

export function formatMemoriesForPrompt(): {
  userMemories: string | null;
  selfMemories: string | null;
} {
  const nowMs = Date.now();
  const nowSec = Math.floor(nowMs / 1000);
  const cutoff14d = nowSec - 14 * 86400;
  const cutoff30d = nowSec - 30 * 86400;

  const filtered = db.query<MemoryRow, [number, number, number]>(`
    SELECT * FROM memories
    WHERE source = 'explicit'
       OR (target_date IS NOT NULL AND target_date > ?1)
       OR (type = 'user' AND (
            category = 'profile'
         OR category = 'preference'
         OR category = 'people'
         OR category = 'career'
         OR category = 'project'
         OR category = 'health'
         OR (category = 'misc'     AND source = 'auto' AND updated_at > ?3)
         OR (category = 'academic' AND source = 'auto' AND updated_at > ?2)
         OR (category = 'event'    AND source = 'auto' AND updated_at > ?3)
       ))
       OR (type = 'assistant' AND updated_at > ?3)
    ORDER BY type, category, key
  `).all(nowMs, cutoff14d, cutoff30d);

  if (filtered.length === 0) return { userMemories: null, selfMemories: null };

  const userData: Record<string, string[]> = {};
  const selfData: string[] = [];

  for (const m of filtered) {
    const suffixes: string[] = [];
    if (m.source === "explicit") suffixes.push("durable");
    if (m.target_date !== null && m.target_date > nowMs) suffixes.push("future");
    const suffix = suffixes.length > 0 ? ` (${suffixes.join(", ")})` : "";
    const line = `${m.key}: ${m.value}${suffix}`;

    if (m.type === "assistant") {
      selfData.push(line);
    } else {
      const cat = m.category.toUpperCase();
      if (!userData[cat]) userData[cat] = [];
      userData[cat].push(line);
    }
  }

  return {
    userMemories: Object.keys(userData).length > 0 ? encode(userData) : null,
    selfMemories: selfData.length > 0 ? encode(selfData) : null,
  };
}

// ── Meta ───────────────────────────────────────────────────

export function getMeta(key: string): string | null {
  const row = db.query<{ value: string }, [string]>(
    `SELECT value FROM meta WHERE key = ?`
  ).get(key);
  return row?.value ?? null;
}

export function setMeta(key: string, value: string | number): void {
  db.query(`INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`).run(
    key,
    String(value)
  );
}

// ── Inbound Queue ──────────────────────────────────────────

export function pushToQueue(
  channelId: string,
  messageId: string,
  text: string | null,
  images: unknown[]
): void {
  db.query(
    `INSERT INTO inbound_queue (channel_id, message_id, text_content, image_parts, status)
     VALUES (?, ?, ?, ?, 'pending')`
  ).run(channelId, messageId, text, JSON.stringify(images));
}

export function resetStaleQueueClaims(now: number = Date.now()): void {
  db.query(
    `UPDATE inbound_queue
     SET status = 'pending',
         claim_token = NULL,
         claimed_at = NULL,
         last_error = COALESCE(last_error, 'Recovered stale queue claim')
     WHERE status = 'processing'
       AND claimed_at IS NOT NULL
       AND claimed_at <= ?`
  ).run(now - STALE_QUEUE_CLAIM_MS);
}

export function claimNextQueueBatch(): ClaimedQueueBatch | null {
  resetStaleQueueClaims();

  let claimToken = "";
  let channelId = "";
  let rows: QueueRow[] = [];

  db.transaction(() => {
    const next = db.query<{ channel_id: string }, []>(
      `SELECT channel_id
       FROM inbound_queue
       WHERE status = 'pending'
       ORDER BY created_at ASC, id ASC
       LIMIT 1`
    ).get();

    if (!next) return;

    claimToken = randomUUID();
    channelId = next.channel_id;
    const claimedAt = Date.now();

    db.query(
      `UPDATE inbound_queue
       SET status = 'processing',
           claim_token = ?,
           claimed_at = ?,
           attempt_count = attempt_count + 1,
           last_error = NULL
       WHERE channel_id = ?
         AND status = 'pending'`
    ).run(claimToken, claimedAt, channelId);

    rows = db.query<QueueRow, [string]>(
      `SELECT *
       FROM inbound_queue
       WHERE claim_token = ?
       ORDER BY created_at ASC, id ASC`
    ).all(claimToken);
  })();

  if (!claimToken) return null;

  try {
    return {
      claim_token: claimToken,
      channel_id: channelId,
      items: rows.map((row) => ({
        id: row.id,
        channel_id: row.channel_id,
        message_id: row.message_id,
        text_content: row.text_content,
        image_parts: coerceQueueImages(row.image_parts),
        created_at: row.created_at,
        attempt_count: row.attempt_count,
      })),
    };
  } catch (err) {
    markQueueBatchFailed(
      claimToken,
      err instanceof Error ? err.message : "Failed to decode queue payload"
    );
    return null;
  }
}

export function markQueueBatchDone(claimToken: string): void {
  db.query(`DELETE FROM inbound_queue WHERE claim_token = ?`).run(claimToken);
}

export function markQueueBatchRetryable(
  claimToken: string,
  error: string
): void {
  db.query(
    `UPDATE inbound_queue
     SET status = 'pending',
         claim_token = NULL,
         claimed_at = NULL,
         last_error = ?
     WHERE claim_token = ?`
  ).run(error, claimToken);
}

export function markQueueBatchFailed(claimToken: string, error: string): void {
  db.query(
    `UPDATE inbound_queue
     SET status = 'failed',
         claim_token = NULL,
         claimed_at = NULL,
         last_error = ?
     WHERE claim_token = ?`
  ).run(error, claimToken);
}

export function getPendingQueueCount(): number {
  return db.query<{ c: number }, []>(
    `SELECT COUNT(*) as c FROM inbound_queue WHERE status = 'pending'`
  ).get()!.c;
}

// ── Scheduled Messages ─────────────────────────────────────

export function scheduleOneShot(
  fireAtMs: number,
  message: string,
  eventKey: string | null
): void {
  if (eventKey) {
    const existing = db.query<{ id: number }, [string]>(
      `SELECT id
       FROM scheduled_messages
       WHERE event_key = ?
         AND status = 'pending'`
    ).get(eventKey);
    if (existing) return;
  } else {
    const duplicate = db.query<{ id: number }, [string]>(
      `SELECT id
       FROM scheduled_messages
       WHERE message = ?
         AND status = 'pending'`
    ).get(message);
    if (duplicate) return;

    const latest = db.query<{ fire_at: number | null }, []>(
      `SELECT MAX(fire_at) as fire_at
       FROM scheduled_messages
       WHERE status = 'pending'
         AND event_key IS NULL`
    ).get();
    if (
      latest?.fire_at &&
      Math.abs(fireAtMs - latest.fire_at) < 4 * 60 * 60 * 1000
    ) {
      fireAtMs = latest.fire_at + 4 * 60 * 60 * 1000;
    }
  }

  db.query(
    `INSERT INTO scheduled_messages (
      fire_at,
      original_fire_at,
      message,
      event_key,
      sent,
      status
    ) VALUES (?, ?, ?, ?, 0, 'pending')`
  ).run(fireAtMs, fireAtMs, message, eventKey);
}

export function resetStaleScheduledClaims(now: number = Date.now()): void {
  db.query(
    `UPDATE scheduled_messages
     SET status = 'pending',
         last_error = COALESCE(last_error, 'Recovered stale scheduled delivery claim')
     WHERE status = 'sending'
       AND last_attempt_at IS NOT NULL
       AND last_attempt_at <= ?`
  ).run(now - STALE_SCHEDULED_CLAIM_MS);
}

export function getScheduledMessageById(id: number): ScheduledMessageRow | null {
  return (
    db.query<ScheduledMessageRow, [number]>(
      `SELECT *
       FROM scheduled_messages
       WHERE id = ?`
    ).get(id) ?? null
  );
}

export function getNextRunnableScheduledMessage(): ScheduledMessageRow | null {
  return (
    db.query<ScheduledMessageRow, []>(
      `SELECT *
       FROM scheduled_messages
       WHERE status = 'pending'
       ORDER BY fire_at ASC, id ASC
       LIMIT 1`
    ).get() ?? null
  );
}

export function claimScheduledMessage(id: number): ScheduledMessageRow | null {
  const attemptedAt = Date.now();
  const result = db.query(
    `UPDATE scheduled_messages
     SET status = 'sending',
         attempt_count = attempt_count + 1,
         last_attempt_at = ?,
         last_error = NULL
     WHERE id = ?
       AND status = 'pending'`
  ).run(attemptedAt, id) as { changes?: number };

  if (!result.changes) return null;
  return getScheduledMessageById(id);
}

export function markScheduledSent(id: number): boolean {
  const result = db.query(
    `UPDATE scheduled_messages
     SET status = 'sent',
         sent = 1,
         sent_at = ?,
         last_error = NULL,
         suppression_reason = NULL
     WHERE id = ?
       AND status = 'sending'`
  ).run(Date.now(), id) as { changes?: number };

  return Boolean(result.changes);
}

export function markScheduledCancelled(id: number): boolean {
  const result = db.query(
    `UPDATE scheduled_messages
     SET status = 'cancelled',
         sent = 1,
         cancelled_at = ?,
         last_error = NULL
     WHERE id = ?
       AND status IN ('pending', 'sending')`
  ).run(Date.now(), id) as { changes?: number };

  return Boolean(result.changes);
}

export function markScheduledSuppressed(
  id: number,
  reason: string,
  error: string | null = null
): void {
  db.query(
    `UPDATE scheduled_messages
     SET status = 'suppressed',
         sent = 1,
         suppressed_at = ?,
         suppression_reason = ?,
         last_error = COALESCE(?, last_error)
     WHERE id = ?`
  ).run(Date.now(), reason, error, id);
}

export function rescheduleScheduledRetry(
  id: number,
  nextFireAt: number,
  error: string
): void {
  db.query(
    `UPDATE scheduled_messages
     SET status = 'pending',
         fire_at = ?,
         last_error = ?
     WHERE id = ?`
  ).run(nextFireAt, error, id);
}

export function listPendingScheduledMessages(): ScheduledMessageRow[] {
  return db.query<ScheduledMessageRow, []>(
    `SELECT *
     FROM scheduled_messages
     WHERE status = 'pending'
     ORDER BY fire_at ASC, id ASC`
  ).all();
}

export function getUnsentMessages(): ScheduledMessageRow[] {
  return listPendingScheduledMessages();
}

export function getNextScheduledMessage(): ScheduledMessageRow | null {
  return getNextRunnableScheduledMessage();
}

export function markSent(id: number): void {
  markScheduledSent(id);
}

export function deleteScheduledMessage(id: number): void {
  markScheduledCancelled(id);
}

export function formatScheduledMessagesForPrompt(): string | null {
  const messages = listPendingScheduledMessages();
  if (messages.length === 0) return null;

  const data = messages.map((m) => {
    const dateStr = formatUserDateTime(m.fire_at, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
    return `ID ${m.id}: [${dateStr}] "${m.message}"${
      m.event_key ? ` (key: ${m.event_key})` : ""
    }`;
  });

  return encode({ PENDING: data });
}

// ── Resets ──────────────────────────────────────────────────

export function clearAll(): void {
  db.run(`DELETE FROM messages`);
  db.run(`DELETE FROM inbound_queue`);
}

export function resetAll(): void {
  db.run(`DELETE FROM messages`);
  db.run(`DELETE FROM memories`);
  db.run(`DELETE FROM vec_memories`);
  db.run(`DELETE FROM meta`);
  db.run(`DELETE FROM scheduled_messages`);
  db.run(`DELETE FROM inbound_queue`);
}

export function closeDb(): void {
  db.close();
}
