import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

process.env.DISCORD_TOKEN = "test-token";
process.env.VERCEL_AI_GATEWAY_KEY = "test-key";
process.env.MY_DISCORD_ID = "123";
process.env.YOUR_NAME = "R";
process.env.COMPANION_NAME = "Buffalo";
process.env.TZ = "America/New_York";
process.env.ENABLE_TELEMETRY = "false";
process.env.MEMORY_DB_PATH = join(mkdtempSync(join(tmpdir(), "yapper-test-")), "memory.db");

const memory = await import("../src/memory.js");
const time = await import("../src/time.js");

beforeAll(() => {
  memory.resetAll();
});

afterEach(() => {
  memory.resetAll();
});

afterAll(() => {
  memory.closeDb();
});

describe("message storage", () => {
  test("round-trips JSON-looking plain text as strings", () => {
    memory.addMessage("user", "123");
    memory.addMessage("user", "true");
    memory.addMessage("user", "null");

    const history = memory.getHistory();
    expect(history[0]?.content).toBe("123");
    expect(history[1]?.content).toBe("true");
    expect(history[2]?.content).toBe("null");
  });

  test("round-trips structured content as structured JSON", () => {
    const content = [{ type: "text", text: "hello" }, { type: "image", image: "png" }];
    memory.addMessage("user", content);

    const [entry] = memory.getHistory();
    expect(entry?.content).toEqual(content);
  });
});

describe("resets", () => {
  test("clearAll preserves summary but removes messages", () => {
    memory.setSummary("kept");
    memory.addMessage("assistant", "hello");

    memory.clearAll();

    expect(memory.getSummary()).toBe("kept");
    expect(memory.getHistory()).toHaveLength(0);
  });
});

describe("queue batching", () => {
  test("claims one channel at a time and isolates failed batches", () => {
    memory.pushToQueue("channel-a", "m1", "first", []);
    memory.pushToQueue("channel-a", "m2", "second", []);
    memory.pushToQueue("channel-b", "m3", "third", []);

    const firstBatch = memory.claimNextQueueBatch();
    expect(firstBatch?.channel_id).toBe("channel-a");
    expect(firstBatch?.items).toHaveLength(2);

    memory.markQueueBatchFailed(firstBatch!.claim_token, "fetch failed");

    const secondBatch = memory.claimNextQueueBatch();
    expect(secondBatch?.channel_id).toBe("channel-b");
    expect(secondBatch?.items).toHaveLength(1);
  });

  test("stale queue claims can be recovered", () => {
    memory.pushToQueue("channel-a", "m1", "first", []);
    const batch = memory.claimNextQueueBatch();
    expect(batch).not.toBeNull();

    memory.resetStaleQueueClaims(Date.now() + memory.STALE_QUEUE_CLAIM_MS + 1);

    const recovered = memory.claimNextQueueBatch();
    expect(recovered?.channel_id).toBe("channel-a");
    expect(recovered?.items).toHaveLength(1);
  });

  test("retryable queue failures can be reclaimed", () => {
    memory.pushToQueue("channel-a", "m1", "first", []);
    const batch = memory.claimNextQueueBatch();
    expect(batch).not.toBeNull();

    memory.markQueueBatchRetryable(batch!.claim_token, "temporary discord failure");

    const retried = memory.claimNextQueueBatch();
    expect(retried?.channel_id).toBe("channel-a");
    expect(retried?.items).toHaveLength(1);
    expect(retried?.items[0]?.attempt_count).toBe(2);
  });
});

describe("scheduled messages", () => {
  test("cancellation removes pending scheduled messages", () => {
    memory.scheduleOneShot(Date.now() + 60_000, "check in", "event_1");

    const [scheduled] = memory.getUnsentMessages();
    expect(scheduled).toBeTruthy();

    const cancelled = memory.markScheduledCancelled(scheduled!.id);
    expect(cancelled).toBe(true);
    expect(memory.getUnsentMessages()).toHaveLength(0);
  });

  test("stale sending claims are reset to pending", () => {
    memory.scheduleOneShot(Date.now() + 60_000, "check in", "event_1");
    const [scheduled] = memory.getUnsentMessages();
    const claimed = memory.claimScheduledMessage(scheduled!.id);
    expect(claimed?.status).toBe("sending");

    memory.resetStaleScheduledClaims(
      Date.now() + memory.STALE_SCHEDULED_CLAIM_MS + 1
    );

    const recovered = memory.getScheduledMessageById(scheduled!.id);
    expect(recovered?.status).toBe("pending");
  });

  test("migration preserves cancelled and suppressed statuses", async () => {
    memory.scheduleOneShot(Date.now() + 60_000, "check in", "cancel_me");
    memory.scheduleOneShot(Date.now() + 120_000, "follow up", "suppress_me");

    const pending = memory.getUnsentMessages();
    const cancelledId = pending.find((row) => row.event_key === "cancel_me")!.id;
    const suppressedId = pending.find((row) => row.event_key === "suppress_me")!.id;

    expect(memory.markScheduledCancelled(cancelledId)).toBe(true);
    memory.markScheduledSuppressed(suppressedId, "irrelevant_now");

    const cancelledBefore = memory.getScheduledMessageById(cancelledId);
    const suppressedBefore = memory.getScheduledMessageById(suppressedId);
    expect(cancelledBefore?.status).toBe("cancelled");
    expect(suppressedBefore?.status).toBe("suppressed");

    memory.rerunMigrationsForTests();

    const cancelledAfter = memory.getScheduledMessageById(cancelledId);
    const suppressedAfter = memory.getScheduledMessageById(suppressedId);
    expect(cancelledAfter?.status).toBe("cancelled");
    expect(suppressedAfter?.status).toBe("suppressed");
  });
});

describe("timezone formatting", () => {
  test("formats user-facing timestamps with configured TZ", () => {
    const formatted = time.formatUserDateTime(Date.UTC(2026, 2, 31, 15, 0, 0), {
      hour: "numeric",
      minute: "2-digit",
      hour12: false,
    });

    expect(formatted).toContain("11:00");
  });
});
