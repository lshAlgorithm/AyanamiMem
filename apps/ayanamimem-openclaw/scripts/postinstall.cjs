#!/usr/bin/env node
/* eslint-disable */
"use strict";

/**
 * AyanamiMem OpenClaw Plugin — postinstall script
 * Mirrors memos-local-openclaw/scripts/postinstall.cjs pattern.
 */

const fs   = require("node:fs");
const path = require("node:path");
const os   = require("node:os");
const { execSync, spawnSync } = require("node:child_process");

const OK   = "✅";
const WARN = "⚠️ ";
const ERR  = "❌";

console.log("\n╔══════════════════════════════════════════════════╗");
console.log("║   AyanamiMem OpenClaw Plugin — post-install       ║");
console.log("╚══════════════════════════════════════════════════╝\n");

let hasErrors = false;

// ── Step 1: Node.js version ───────────────────────────────────────────────────

const nodeVer = parseInt(process.versions.node.split(".")[0], 10);
if (nodeVer >= 18) {
  console.log(`${OK} Node.js ${process.versions.node}`);
} else {
  console.log(`${ERR} Node.js >= 18 required (found ${process.versions.node})`);
  hasErrors = true;
}

// ── Step 2: Check @huggingface/transformers ───────────────────────────────────

try {
  require.resolve("@huggingface/transformers");
  console.log(`${OK} @huggingface/transformers available`);
} catch {
  console.log(`${WARN} @huggingface/transformers not resolved yet`);
  console.log("     Run: npm install  (inside this plugin directory)");
}

// ── Step 3: Check js-yaml ─────────────────────────────────────────────────────

try {
  require.resolve("js-yaml");
  console.log(`${OK} js-yaml available`);
} catch {
  console.log(`${WARN} js-yaml not resolved yet`);
  console.log("     Run: npm install  (inside this plugin directory)");
}

// ── Step 4: Print required env vars ──────────────────────────────────────────

console.log("\nRequired environment variables:");
const OPENAI_KEY  = process.env.OPENAI_API_KEY;
const OPENAI_BASE = process.env.OPENAI_BASE_URL ?? process.env.OPENAI_API_BASE;

if (OPENAI_KEY) {
  console.log(`${OK} OPENAI_API_KEY is set`);
} else {
  console.log(`${WARN} OPENAI_API_KEY not set — LLM memory extraction will fail`);
  console.log("     Add to ~/.openclaw/.env or your shell profile:");
  console.log("       OPENAI_API_KEY=sk-...");
  console.log("       OPENAI_BASE_URL=https://openrouter.ai/api/v1");
}

if (OPENAI_BASE) {
  console.log(`${OK} OPENAI_BASE_URL: ${OPENAI_BASE}`);
} else {
  console.log(`${WARN} OPENAI_BASE_URL not set — defaults to OpenAI`);
}

// ── Step 5: Patch ~/.openclaw/openclaw.json ───────────────────────────────────

const openclawCfgPath = path.join(os.homedir(), ".openclaw", "openclaw.json");
if (fs.existsSync(openclawCfgPath)) {
  try {
    const cfg = JSON.parse(fs.readFileSync(openclawCfgPath, "utf8"));
    const tools = cfg.tools ?? {};
    const allow = Array.isArray(tools.allow) ? tools.allow : [];
    if (!allow.includes("group:plugins")) {
      allow.push("group:plugins");
      tools.allow = allow;
      cfg.tools = tools;
      const tmp = openclawCfgPath + ".tmp";
      fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2), "utf8");
      fs.renameSync(tmp, openclawCfgPath);
      console.log(`\n${OK} Patched ~/.openclaw/openclaw.json: added "group:plugins" to tools.allow`);
    } else {
      console.log(`\n${OK} ~/.openclaw/openclaw.json already has "group:plugins" in tools.allow`);
    }
  } catch (e) {
    console.log(`\n${WARN} Could not patch ~/.openclaw/openclaw.json: ${e.message}`);
  }
} else {
  console.log(`\n${WARN} ~/.openclaw/openclaw.json not found (gateway not yet started?)`);
  console.log("     After first gateway start, re-run: node scripts/postinstall.cjs");
}

// ── Step 6: Install bundled skill guide ───────────────────────────────────────

try {
  const skillSrc  = path.join(__dirname, "..", "skill", "ayanamimem-guide", "SKILL.md");
  const skillDest = path.join(os.homedir(), ".openclaw", "workspace", "skills", "ayanamimem-guide");
  if (fs.existsSync(skillSrc)) {
    fs.mkdirSync(skillDest, { recursive: true });
    fs.copyFileSync(skillSrc, path.join(skillDest, "SKILL.md"));
    console.log(`${OK} Skill guide installed: ${skillDest}/SKILL.md`);
  }
} catch (e) {
  console.log(`${WARN} Skill guide install skipped: ${e.message}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log("\n" + "─".repeat(52));
if (hasErrors) {
  console.log(`${ERR} Setup incomplete — fix errors above then restart gateway`);
} else {
  console.log(`${OK} Setup complete!`);
  console.log("\nNext steps:");
  console.log("  1. Set OPENAI_API_KEY and OPENAI_BASE_URL in your environment");
  console.log("  2. Restart the OpenClaw gateway: openclaw gateway restart");
  console.log("  3. Memory files will be stored in: ~/.ayanamimem/sessions/");
  console.log("  4. Open any .md file there in a text editor to inspect memories\n");
}
