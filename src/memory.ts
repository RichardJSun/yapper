import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { existsSync } from "node:fs";
import { config } from "./config.js";
import { encode } from "@toon-format/toon";

// ── Row types ──────────────────────────────────────────────

export interface MessageRow {
  id: number;
  role: string;
  content: string;
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
  status: string;
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
  message: string;
  event_key: string | null;
  sent: number;
}

export interface VecSearchRow extends MemoryRow {
  distance: number;
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

const db = new Database("memory.db", { create: true });
db.exec("PRAGMA journal_mode = WAL;");
db.exec("PRAGMA foreign_keys = ON;");
sqliteVec.load(db);

db.exec(`
  CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    is_archived INTEGER DEFAULT 0,
    created_at  INTEGER DEFAULT (strftime('%s','now')),
    is_proactive INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS inbound_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id   TEXT NOT NULL,
    message_id   TEXT NOT NULL,
    text_content TEXT,
    image_parts  TEXT,
    status       TEXT DEFAULT 'pending',
    created_at   INTEGER DEFAULT (strftime('%s','now'))
  );


  CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT DEFAULT 'user', -- 'user' or 'assistant'
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
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_at   INTEGER NOT NULL,
    message   TEXT NOT NULL,
    event_key TEXT,
    sent      INTEGER DEFAULT 0
  );

  CREATE INDEX IF NOT EXISTS idx_messages_archived ON messages(is_archived);
  CREATE INDEX IF NOT EXISTS idx_queue_status ON inbound_queue(status);
  CREATE INDEX IF NOT EXISTS idx_scheduled_due ON scheduled_messages(sent, fire_at);
`);

const MAX_MESSAGES = 40;

// ── Messages ───────────────────────────────────────────────

export function addMessage(role: "user" | "assistant", content: string | unknown, isProactive: boolean = false): void {
  const text = typeof content === "string" ? content : JSON.stringify(content);
  db.query(`INSERT INTO messages (role, content, is_proactive) VALUES ($role, $content, $isProactive)`)
    .run({ $role: role, $content: text, $isProactive: isProactive ? 1 : 0 });

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

export interface HistoryEntry {
  role: string;
  content: unknown;
  created_at: number;
  is_proactive: boolean;
}

export function getHistory(): HistoryEntry[] {
  const rows = db.query<MessageRow, []>(
    `SELECT role, content, created_at, is_proactive FROM messages WHERE is_archived = 0 ORDER BY id ASC`
  ).all();

  return rows.map((row) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(row.content);
    } catch {
      parsed = row.content;
    }
    return { role: row.role, content: parsed, created_at: row.created_at, is_proactive: row.is_proactive === 1 };
  });
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
    const result = db.query<{ id: number }, [string, string, string, string, string, number | null]>(
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
      db.query(
        `INSERT OR REPLACE INTO vec_memories (id, embedding) VALUES (?, ?)`
      ).run(result.id, new Float32Array(embeddingArray));
    }
  })();
}

export function deleteMemory(type: "user" | "assistant", category: string, key: string): void {
  const row = db.query<{ id: number }, [string, string, string]>(
    `SELECT id FROM memories WHERE type = ? AND category = ? AND key = ?`
  ).get(type, category, key);

  db.query(`DELETE FROM memories WHERE type = ? AND category = ? AND key = ?`).run(type, category, key);

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
  return db.query<{ id: number }, [string]>(
    `SELECT id FROM memories WHERE category = ? LIMIT 1`
  ).get(category) !== null;
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
      const date = new Date(r.updated_at * 1000).toLocaleDateString();
      const owner = r.type === "assistant" ? "[SELF]" : `[${r.category.toUpperCase()}]`;
      return `${owner} ${r.key}: ${r.value} (Saved: ${date})`;
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

export function formatMemoriesForPrompt(): { userMemories: string | null; selfMemories: string | null } {
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

  // Group by type and category
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

  const userMemories = Object.keys(userData).length > 0 ? encode(userData) : null;
  const selfMemories = selfData.length > 0 ? encode(selfData) : null;

  return {
    userMemories,
    selfMemories,
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
  db.query(
    `INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`
  ).run(key, String(value));
}

// ── Inbound Queue ──────────────────────────────────────────

export interface ClaimedQueueItem {
  id: number;
  channel_id: string;
  message_id: string;
  text_content: string | null;
  image_parts: unknown[];
  created_at: number;
}

export function pushToQueue(
  channelId: string,
  messageId: string,
  text: string | null,
  images: unknown[]
): void {
  db.query(
    `INSERT INTO inbound_queue (channel_id, message_id, text_content, image_parts) VALUES (?, ?, ?, ?)`
  ).run(channelId, messageId, text, JSON.stringify(images));
}

export function claimPendingQueue(): ClaimedQueueItem[] {
  db.run(`UPDATE inbound_queue SET status = 'processing' WHERE status = 'pending'`);
  const rows = db.query<QueueRow, []>(
    `SELECT * FROM inbound_queue WHERE status = 'processing' ORDER BY created_at ASC`
  ).all();
  return rows.map((r) => ({
    ...r,
    image_parts: JSON.parse(r.image_parts) as unknown[],
  }));
}

export function clearProcessingQueue(): void {
  db.run(`DELETE FROM inbound_queue WHERE status = 'processing'`);
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
      `SELECT id FROM scheduled_messages WHERE event_key = ? AND sent = 0`
    ).get(eventKey);
    if (existing) return; // Dedupe: skip if same event_key pending
  } else {
    // Dedupe: skip if an unsent message with identical text already exists
    const duplicate = db.query<{ id: number }, [string]>(
      `SELECT id FROM scheduled_messages WHERE message = ? AND sent = 0`
    ).get(message);
    if (duplicate) return;

    const latest = db.query<{ fire_at: number }, []>(
      `SELECT MAX(fire_at) as fire_at FROM scheduled_messages WHERE sent = 0 AND event_key IS NULL`
    ).get();
    if (latest?.fire_at && Math.abs(fireAtMs - latest.fire_at) < 4 * 60 * 60 * 1000) {
      fireAtMs = latest.fire_at + 4 * 60 * 60 * 1000; // 4h cluster guard
    }
  }

  db.query(
    `INSERT INTO scheduled_messages (fire_at, message, event_key) VALUES (?, ?, ?)`
  ).run(fireAtMs, message, eventKey);
}

export function getDueMessages(): ScheduledMessageRow[] {
  return db.query<ScheduledMessageRow, [number]>(
    `SELECT * FROM scheduled_messages WHERE sent = 0 AND fire_at <= ? ORDER BY fire_at ASC`
  ).all(Date.now());
}

export function getNextScheduledMessage(): ScheduledMessageRow | null {
  return db.query<ScheduledMessageRow, []>(
    `SELECT * FROM scheduled_messages WHERE sent = 0 ORDER BY fire_at ASC LIMIT 1`
  ).get() ?? null;
}

export function markSent(id: number): void {
  db.query(`UPDATE scheduled_messages SET sent = 1 WHERE id = ?`).run(id);
}

export function getUnsentMessages(): ScheduledMessageRow[] {
  return db.query<ScheduledMessageRow, []>(
    `SELECT id, fire_at, message, event_key FROM scheduled_messages WHERE sent = 0 ORDER BY fire_at ASC`
  ).all();
}

// ── Resets ──────────────────────────────────────────────────

export function clearAll(): void {
  db.run(`DELETE FROM messages`);
  db.run(`DELETE FROM meta WHERE key = 'summary'`);
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
