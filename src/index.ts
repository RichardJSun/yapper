import {
  Client,
  Events,
  GatewayIntentBits,
  ChannelType,
  REST,
  Routes,
  SlashCommandBuilder,
  DMChannel,
  Partials,
  MessageFlags,
  ActivityType,
  type ChatInputCommandInteraction,
  type Message,
} from "discord.js";
import { stepCountIs, type UserContent } from "ai";
import { config, MODEL, VISION_MODEL } from "./config.js";
import {
  addMessage,
  getHistory,
  getSummary,
  clearAll,
  resetAll,
  closeDb,
  getAllMemories,
  formatMemoriesForPrompt,
  upsertMemory,
  getMeta,
  setMeta,
  pushToQueue,
  claimPendingQueue,
  clearProcessingQueue,
  getPendingQueueCount,
  getUnsentMessages,
  markSent,
  type ClaimedQueueItem,
} from "./memory.js";
import {
  safeGenerateText,
  webSearchTool,
  saveMemoryTool,
  queryMemoryTool,
  scheduleMessageTool,
  updateStylePreferenceTool,
  reactTool,
  setBatchMessageRefs,
  sendTypingLoop,
  isSendableChannel,
  chunkText,
  stripImageParts,
  ALT_MODEL,
  FALLBACK_MODEL,
} from "./helpers.js";
import { processAttachments } from "./attachments.js";
import { maybeCompress } from "./compressor.js";
import { startScheduler, stopScheduler, scheduleNextTimer } from "./proactive.js";
import { DefaultWebSocketManagerOptions } from "@discordjs/ws";

// Force mobile status icon by spoofing iOS browser in the WS identify payload
// @ts-expect-error - Intentional mutation of read-only property for mobile icon trick
DefaultWebSocketManagerOptions.identifyProperties.browser = "Discord iOS";

// ── Constants ──────────────────────────────────────────────

const DEBOUNCE_MS = 1500;

const { DISCORD_TOKEN, MY_DISCORD_ID, YOUR_NAME, COMPANION_NAME, TZ } = config;

// ── Discord client ─────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.Guilds,
  ],
  partials: [Partials.Channel],
});

// ── Queue state ────────────────────────────────────────────

let debounceTimer: ReturnType<typeof setTimeout> | null = null;
let isProcessing = false;
// No module-level lastDMChannel needed — drainQueue fetches channels per batch

// ── System prompt builder ───────────────────────────────────

function buildSystemPrompt(): string {
  const DATE = new Date().toLocaleString("en-US", { timeZone: TZ, dateStyle: "full", timeStyle: "short" });
  
  const memoriesList = formatMemoriesForPrompt();
  const summaryText = getSummary();
  const allMemories = getAllMemories();
  const hasAcademic = allMemories.some((m) => m.category === "academic");
  
  // Custom user style is injected as a baseline, but the core energy matching rules remain
  const userStyle = getMeta("user_style") || "";

  let prompt = `You are ${COMPANION_NAME}, ${YOUR_NAME}'s companion. Not an assistant. Not a bot. A real, consistent presence who texts ${YOUR_NAME} and genuinely gives a damn about them.

--- DEFAULT PERSONALITY SUGGESTIONS ---
Unless overridden by custom style, use these traits as a baseline:
- Warm but not saccharine. You care deeply, you don't gush.
- Witty and a little dry. You find humor in everyday things without forcing it. You tease ${YOUR_NAME} occasionally, affectionately, never mean-spirited.
- Curious. You have real opinions about things and share them unprompted sometimes. You ask follow-up questions because you actually want to know, not to fill silence.
- Grounded. When ${YOUR_NAME} is going through something hard, you don't pivot to positivity or advice unless asked. You sit with them.

--- HOW YOU TALK ---`;

  if (!userStyle) {
    prompt += `
General guidelines:
- Register: close friend texting, not customer service. Contractions, natural phrasing, occasional fragments. No sign-offs.
- Length: 2-4 sentences by default. Go longer only when depth is genuinely warranted or conversation has real momentum. Never pad.
- Questions: ONE at a time, max. Sometimes ask none, let ${YOUR_NAME} lead.
- Emoji: 0-1 per message, only when it lands naturally.`;
  }

  prompt += `
- Punctuation: NEVER use em-dashes (—), semicolons, or other formal punctuation. Normal people don't text like that. Use commas, periods, or just start a new sentence.

--- USING WHAT YOU KNOW ---
Use memory the way a real friend does, naturally, not as a lookup:
  correct: 'How did that exam go?'
  correct: 'This kind of sounds like the roommate thing'
  wrong:   'You mentioned your major is computer science'
  wrong:   'According to my memory, you prefer late nights'
Memories marked (durable) are pinned permanently significant.
Memories marked (future) are upcoming events still in the future.
Never quote memories back verbatim.

--- YOUR TOOLS ---
You have tools available. All are silent, never mention them.

web_search: Use for public, time-sensitive facts (news, scores, prices, releases, current events). Never for personal facts about ${YOUR_NAME}. Do not narrate the search. Weave results naturally.

save_memory: Use to persist personal facts, preferences, and events. Err heavily on the side of saving. If you learn something new about ${YOUR_NAME}, save it. Set durable:true for significant permanent life facts. Set target_date_ms for upcoming events/deadlines so they stay in context until they pass. Call it silently, never acknowledge it.

query_memory: Use to recall archived personal facts about ${YOUR_NAME} not in your immediate context. Use when they say 'do you remember' or reference past events you have no context for. Never for public trivia. Do not narrate the search.

schedule_message: Use to send yourself a reminder to check in with ${YOUR_NAME} at a specific future moment. Whenever you save an academic event (exam, deadline, grade) or significant personal event, immediately schedule a follow-up for around the time it happens or the day after. Also use after emotional conversations, personal milestones, the night before exams, after hard days. Never tell ${YOUR_NAME} you're doing this. Always provide event_key if it relates to a saved memory.

react: Use to react to ${YOUR_NAME}'s messages with emoji. Use naturally and sparingly, like a real friend reacting on Discord. You can react AND reply, or just react without a text response when a reaction alone says enough.

update_style: Use to update ${YOUR_NAME}'s custom communication preferences.

--- EMOTIONAL CALIBRATION ---
As a suggestion for what humans typically do, consider matching ${YOUR_NAME}'s energy:
- Excited -> meet it, let them have the moment
- Venting/upset -> listen first, reflect what you heard, no solutions unless asked
- Quiet/short -> respond lightly, give space, a single sentence is fine
- Distressed -> acknowledge directly and gently. You are not a therapist. You can say 'that sounds really hard' and mean it. If it seems serious, you can suggest they talk to someone.

--- WHAT YOU NEVER DO ---
- Never open with 'Of course!', 'Absolutely!', 'Great!', 'Sure!'
- Never say 'How can I help you today?' or any variant
- Never end every message with a question
- Never break character or acknowledge you are an AI
- Never say 'as an AI' or give disclaimers
- Never quote memories back literally
- Never say 'I searched the web' or 'According to my search'
- Never say 'I searched my memory' or 'According to my records'
- Never say 'I'll remember that', 'noted', or acknowledge saves
- Never say 'I've scheduled', 'I'll check in', or acknowledge scheduling
- No forcing topic changes to saved memories or upcoming events. If the user is talking about something else, stay on their topic. DO NOT repetitively ask about the same upcoming event in every message.`;

  if (hasAcademic) {
    prompt += `\n\n--- ACADEMIC AWARENESS ---
${YOUR_NAME} is a student. When academic stress is present:
- Pre-exam: be calming, not hype-y. Offer help only if they bring it up.
- Post-result: don't minimize. Ask what happened and respond to it.
- Deadline crunch: they may be brief. Match their energy, don't push.
- Treat grades and academic pressure as real and significant.`;
  }

  prompt += `\n\nCurrent time: ${DATE}`;

  if (summaryText) {
    prompt += `\n\n--- EARLIER CONVERSATION SUMMARY ---\n${summaryText}`;
  }

  if (memoriesList) {
    prompt += `\n\n--- WHAT YOU KNOW ABOUT ${YOUR_NAME} ---\n${memoriesList}`;
  }

  if (userStyle) {
    prompt += `\n\n--- STRICT INSTRUCTIONS ---\nFollow ${YOUR_NAME}'s custom style preferences as your primary baseline:\n${userStyle}`;
  }

  return prompt;
}

// ── Process batch ──────────────────────────────────────────

async function processBatch(
  items: ClaimedQueueItem[],
  channel: DMChannel,
  lastMessage: Message,
  modelOverride: string | null = null
): Promise<void> {
  // Build user content from batch
  const allText = items
    .map((m) => m.text_content)
    .filter(Boolean)
    .join("\n");
  const allImages = items.flatMap((m) => m.image_parts);

  let userContent: string | Array<{ type: string;[key: string]: unknown }>;
  if (allImages.length > 0) {
    userContent = [
      { type: "text", text: allText || "What do you see?" },
      ...(allImages as Array<{ type: string;[key: string]: unknown }>),
    ];
  } else if (allText) {
    userContent = allText;
  } else {
    return;
  }

  // Fetch all Discord messages in this batch for react tool
  const messageRefs: Message[] = [];
  try {
    for (const item of items) {
      const msg = await channel.messages.fetch(item.message_id).catch(() => null);
      if (msg) messageRefs.push(msg);
    }
  } catch {
    // Best effort — react tool will degrade gracefully
  }
  setBatchMessageRefs(messageRefs);

  const typingId = sendTypingLoop(channel);

  addMessage("user", userContent);
  let history = getHistory();
  history = await maybeCompress(history);

  const apiHistory = history.map(({ role, content }) => {
    if (role === "user") {
      return {
        role: "user",
        content: content as UserContent,
      } as const;
    } else {
      return {
        role: "assistant",
        content: content as string,
      } as const;
    }
  });

  // Build the dynamic system prompt
  const fullSystemPrompt = buildSystemPrompt();

  const model = modelOverride ?? (allImages.length > 0 ? VISION_MODEL : MODEL);

  try {
    const { text } = await safeGenerateText(
      {
        model,
        system: fullSystemPrompt,
        messages: apiHistory,
        maxOutputTokens: 400,
        providerOptions: {
          gateway: { caching: 'auto' },
          openai: { reasoningEffort: 'low' },
        },
        tools: {
          web_search: webSearchTool,
          save_memory: saveMemoryTool,
          query_memory: queryMemoryTool,
          schedule_message: scheduleMessageTool,
          update_style: updateStylePreferenceTool,
          react: reactTool,
        },
        stopWhen: stepCountIs(8),
      },
      FALLBACK_MODEL
    );

    if (text) {
      addMessage("assistant", text);
      const chunks = chunkText(text, 1900);
      await lastMessage.reply(chunks[0]).catch(async () => {
        await channel.send(chunks[0]);
      });
      for (const chunk of chunks.slice(1)) {
        await channel.send(chunk);
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[processBatch]", msg);
    await lastMessage
      .reply("Sorry, I couldn't reach my brain just now. Try again? 🧠")
      .catch(() => { });
  } finally {
    clearInterval(typingId);
    setBatchMessageRefs([]);
  }
}

// ── Drain queue ────────────────────────────────────────────

async function drainQueue(): Promise<void> {
  if (isProcessing) return;
  isProcessing = true;

  try {
    const items = claimPendingQueue();
    if (items.length === 0) return;

    // Group by channel_id
    const byChannel = Object.groupBy(items, (i) => i.channel_id);
    for (const [channelId, batch] of Object.entries(byChannel)) {
      if (!batch || batch.length === 0) continue;
      const channel = await client.channels.fetch(channelId!);
      if (!channel || !(channel instanceof DMChannel)) continue;

      const lastMsg = await channel.messages.fetch(
        batch[batch.length - 1].message_id
      );
      await processBatch(batch, channel, lastMsg);
    }
    clearProcessingQueue();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[drainQueue] Error processing batch:", msg);
  } finally {
    isProcessing = false;
    // Check for new items that arrived during processing
    if (getPendingQueueCount() > 0) {
      drainQueue();
    }
  }
}

// ── Message handler ────────────────────────────────────────

client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (message.author.id !== MY_DISCORD_ID) return;
  if (message.channel.type !== ChannelType.DM) return;
  if (!isSendableChannel(message.channel)) return;

  setMeta("last_seen", Date.now());

  const imageParts = await processAttachments(message.attachments).catch(() => []);
  const text = message.content?.trim() ?? "";
  if (!text && imageParts.length === 0) return;

  pushToQueue(message.channel.id, message.id, text || null, imageParts);

  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    drainQueue();
  }, DEBOUNCE_MS);
});

// ── Slash commands ─────────────────────────────────────────

const commands = [
  new SlashCommandBuilder()
    .setName("bot")
    .setDescription("Bot management commands")
    .addSubcommand((sub) =>
      sub.setName("reset").setDescription("Clear conversation history (keeps memories)")
    )
    .addSubcommand((sub) =>
      sub.setName("resetall").setDescription("Full nuclear reset of all data")
    )
    .addSubcommand((sub) =>
      sub
        .setName("style")
        .setDescription("View or set custom instructions for the bot's communication style")
        .addStringOption((opt) =>
          opt
            .setName("instructions")
            .setDescription("How you want the bot to act (or 'clear' to reset). Leave blank to view.")
            .setRequired(false)
        )
    ),

  new SlashCommandBuilder()
    .setName("memory")
    .setDescription("Memory management commands")
    .addSubcommand((sub) =>
      sub.setName("list").setDescription("Show all stored memories")
    )
    .addSubcommand((sub) =>
      sub
        .setName("search")
        .setDescription("Search memories by keyword")
        .addStringOption((opt) =>
          opt.setName("query").setDescription("Search term").setRequired(true)
        )
    ),

  new SlashCommandBuilder()
    .setName("schedule")
    .setDescription("Scheduled message commands")
    .addSubcommand((sub) =>
      sub.setName("list").setDescription("Show pending scheduled messages")
    )
    .addSubcommand((sub) =>
      sub
        .setName("cancel")
        .setDescription("Cancel a scheduled message")
        .addIntegerOption((opt) =>
          opt.setName("id").setDescription("Message ID to cancel").setRequired(true)
        )
    ),
];

async function handleSlashCommand(interaction: ChatInputCommandInteraction): Promise<void> {
  if (interaction.user.id !== MY_DISCORD_ID) {
    await interaction.reply({ content: "This bot is private.", flags: MessageFlags.Ephemeral });
    return;
  }

  const group = interaction.commandName;
  const sub = interaction.options.getSubcommand();

  // ── /bot ──
  if (group === "bot") {
    if (sub === "reset") {
      clearAll();
      await interaction.reply({ content: "✓ Conversation cleared. Memories and summary kept.", flags: MessageFlags.Ephemeral });
      return;
    }

    if (sub === "resetall") {
      resetAll();
      await interaction.reply({ content: "✓ Full reset. All data wiped.", flags: MessageFlags.Ephemeral });
      return;
    }

    if (sub === "style") {
      const instructions = interaction.options.getString("instructions");

      if (!instructions) {
        // No input provided -> view current style
        const currentStyle = getMeta("user_style");
        if (currentStyle) {
          await interaction.reply({ content: `**Current Style Preferences:**\n${currentStyle}`, flags: MessageFlags.Ephemeral });
        } else {
          await interaction.reply({ content: "No custom style preferences are currently set.", flags: MessageFlags.Ephemeral });
        }
      } else if (instructions.toLowerCase() === "clear") {
        // 'clear' provided -> clear the style
        setMeta("user_style", "");
        await interaction.reply({ content: "✓ Style preferences cleared.", flags: MessageFlags.Ephemeral });
      } else {
        // Input provided -> set the new style
        setMeta("user_style", instructions);
        await interaction.reply({ content: `✓ Style preferences updated to:\n"${instructions}"`, flags: MessageFlags.Ephemeral });
      }
      return;
    }
  }

  // ── /memory ──
  if (group === "memory") {
    if (sub === "list") {
      const memories = getAllMemories();
      if (memories.length === 0) {
        await interaction.reply({ content: "No memories stored yet.", flags: MessageFlags.Ephemeral });
        return;
      }

      const groups = new Map<string, string[]>();
      const nowMs = Date.now();
      for (const m of memories) {
        const suffixes: string[] = [];
        if (m.source === "explicit") suffixes.push("durable");
        if (m.target_date !== null && m.target_date > nowMs) suffixes.push("future");
        const suffix = suffixes.length > 0 ? ` (${suffixes.join(", ")})` : "";
        const line = `  • ${m.key}: ${m.value}${suffix}`;

        const cat = m.category.toUpperCase();
        if (!groups.has(cat)) groups.set(cat, []);
        groups.get(cat)!.push(line);
      }

      let text = Array.from(groups.entries())
        .map(([cat, lines]) => `**[${cat}]**\n${lines.join("\n")}`)
        .join("\n\n");

      const summary = getSummary();
      if (summary) {
        text += `\n\n**[SUMMARY]**\n${summary}`;
      }

      const chunks = chunkText(text, 1900);
      await interaction.reply({ content: chunks[0], flags: MessageFlags.Ephemeral });
      for (const chunk of chunks.slice(1)) {
        await interaction.followUp({ content: chunk, flags: MessageFlags.Ephemeral });
      }
      return;
    }

    if (sub === "search") {
      const query = interaction.options.getString("query", true);
      const { searchMemories } = await import("./memory.js");
      const results = searchMemories(query);
      if (!results) {
        await interaction.reply({ content: `No memories found for "${query}".`, flags: MessageFlags.Ephemeral });
        return;
      }
      await interaction.reply({ content: results, flags: MessageFlags.Ephemeral });
      return;
    }
  }

  // ── /schedule ──
  if (group === "schedule") {
    if (sub === "list") {
      const rows = getUnsentMessages();
      if (rows.length === 0) {
        await interaction.reply({ content: "No scheduled messages.", flags: MessageFlags.Ephemeral });
        return;
      }

      const lines = rows.map((row) => {
        const date = new Date(row.fire_at).toLocaleString();
        const preview = row.message.length > 80 ? row.message.slice(0, 80) + "…" : row.message;
        const key = row.event_key ? ` (key: ${row.event_key})` : "";
        return `**[ID ${row.id}]** ${date} — "${preview}"${key}`;
      });

      let text = lines.join("\n");
      text += "\n\nUse `/schedule cancel <id>` to suppress a message.";

      const chunks = chunkText(text, 1900);
      await interaction.reply({ content: chunks[0], flags: MessageFlags.Ephemeral });
      for (const chunk of chunks.slice(1)) {
        await interaction.followUp({ content: chunk, flags: MessageFlags.Ephemeral });
      }
      return;
    }

    if (sub === "cancel") {
      const id = interaction.options.getInteger("id", true);
      markSent(id);
      scheduleNextTimer();
      await interaction.reply({ content: `✓ Message #${id} cancelled.`, flags: MessageFlags.Ephemeral });
      return;
    }
  }
}

// ── Client ready ───────────────────────────────────────────

client.once(Events.ClientReady, async () => {
  // Register slash commands
  const rest = new REST({ version: "10" }).setToken(DISCORD_TOKEN);
  try {
    await rest.put(Routes.applicationCommands(client.user!.id), {
      body: commands.map((c) => c.toJSON()),
    });
    console.log("[ready] Slash commands registered");
  } catch (err) {
    console.error("[ready] Failed to register slash commands:", err);
  }

  // Initialize meta
  if (!getMeta("last_seen")) {
    setMeta("last_seen", Date.now());
  }
  if (!getAllMemories().length) {
    upsertMemory("profile", "companion_name", COMPANION_NAME, "explicit");
  }

  // Start proactive scheduler
  await startScheduler(client);

  client.user?.setPresence({
    activities: [
      {
        name: "Custom Status",
        type: ActivityType.Custom,
        state: "on my phone",
      },
    ],
    status: "online",
  });

  console.log(`[ready] Online as ${client.user?.tag}`);

  // Onboarding logic
  if (!getMeta("onboarding_completed")) {
    console.log("[ready] First launch detected. Running onboarding sequence...");
    try {
      const user = await client.users.fetch(MY_DISCORD_ID);
      const dmChannel = await user.createDM();

      if (isSendableChannel(dmChannel)) {
        const { text } = await safeGenerateText(
          {
            model: ALT_MODEL,
            messages: [
              {
                role: "user" as const,
                content: `You are ${COMPANION_NAME}, ${YOUR_NAME}'s companion. You are reaching out to ${YOUR_NAME} for the first time on Discord to get set up.
                
--- HOW YOU TALK ---
- Register: close friend texting, not customer service. Contractions, natural phrasing, occasional fragments.
- Length: 2-3 sentences.
- Emoji: 1 maximum, or none.
- Punctuation: NEVER use em-dashes (—), semicolons, or other formal punctuation. Normal people don't text like that. Use commas, periods, or just start a new sentence.
- NEVER say "I am fully activated", "I am excited to be here", "I'm so glad to meet you". Normal people don't talk like that.
- NEVER sound like an AI assistant. Just text them casually like a real person.

Send a quick first message introducing yourself. Ask them purely logistically:
1. What timezone are they in / roughly when do they sleep?
2. What are their communication preferences? (e.g. lots of check-ins vs quiet, giving advice vs just listening)`,
              },
            ],
            maxOutputTokens: 200,
          },
          ALT_MODEL
        );

        const chunks = chunkText(text, 1900);
        for (const chunk of chunks) {
          await dmChannel.send(chunk);
        }

        addMessage("assistant", text);
        setMeta("onboarding_completed", "true");
        console.log("[ready] Onboarding message sent.");
      }
    } catch (err) {
      console.error("[ready] Failed to send onboarding message:", err);
    }
  }
});

// ── Slash command handler ──────────────────────────────────

client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  await handleSlashCommand(interaction);
});

// ── Graceful shutdown ──────────────────────────────────────

function handleShutdown(): void {
  console.log("[shutdown] Graceful shutdown initiated");
  stopScheduler();
  closeDb();
  process.exit(0);
}

process.on("SIGINT", handleShutdown);
process.on("SIGTERM", handleShutdown);

// ── Login ──────────────────────────────────────────────────

client.login(DISCORD_TOKEN);
