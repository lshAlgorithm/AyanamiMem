/** Session manager — _sessions.json registry with atomic writes.
 *
 * Port of Python src/memos/mem_user/session_manager.py
 * Single-threaded Node.js: uses fs.renameSync for atomic writes.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Grant, Session, SessionsFile } from "./types.js";

const SESSIONS_FILENAME = "_sessions.json";
const PUBLIC_DIR = "_public";

export class SessionManager {
  readonly rootDir: string;
  private filepath: string;

  constructor(rootDir: string) {
    this.rootDir = path.resolve(rootDir);
    this.filepath = path.join(this.rootDir, SESSIONS_FILENAME);
    fs.mkdirSync(this.rootDir, { recursive: true });
  }

  // ── Session lifecycle ───────────────────────────────────────────────────────

  registerSession(sessionId: string, userId = "default"): void {
    const data = this._read();
    if (!data.sessions[sessionId]) {
      data.sessions[sessionId] = {
        sessionId,
        userId,
        createdAt: new Date().toISOString(),
        isActive: true,
      };
    } else {
      data.sessions[sessionId].userId = userId;
      data.sessions[sessionId].isActive = true;
    }
    this._write(data);
    fs.mkdirSync(this.sessionDir(sessionId), { recursive: true });
  }

  deactivateSession(sessionId: string): void {
    const data = this._read();
    if (data.sessions[sessionId]) {
      data.sessions[sessionId].isActive = false;
      this._write(data);
    }
  }

  deleteSession(sessionId: string): void {
    const data = this._read();
    delete data.sessions[sessionId];
    data.grants = data.grants.filter(
      (g) => g.grantingSession !== sessionId && g.granteeSession !== sessionId,
    );
    this._write(data);
    const dir = this.sessionDir(sessionId);
    if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
  }

  listSessions(): Session[] {
    return Object.values(this._read().sessions);
  }

  sessionExists(sessionId: string): boolean {
    return sessionId in this._read().sessions;
  }

  sessionDir(sessionId: string): string {
    return path.join(this.rootDir, sessionId);
  }

  publicDir(): string {
    const p = path.join(this.rootDir, PUBLIC_DIR);
    fs.mkdirSync(p, { recursive: true });
    return p;
  }

  // ── Grants ──────────────────────────────────────────────────────────────────

  grantAccess(
    grantingSession: string,
    granteeSession: string,
    scope: "read" = "read",
    subtreePath?: string,
  ): void {
    const data = this._read();
    // Idempotent
    const exists = data.grants.some(
      (g) =>
        g.grantingSession === grantingSession &&
        g.granteeSession  === granteeSession &&
        g.subtreePath     === subtreePath &&
        g.scope           === scope,
    );
    if (!exists) {
      const grant: Grant = {
        grantingSession,
        granteeSession,
        scope,
        subtreePath,
        grantedAt: new Date().toISOString(),
      };
      data.grants.push(grant);
      this._write(data);
    }
  }

  revokeAccess(
    grantingSession: string,
    granteeSession: string,
    subtreePath?: string,
  ): void {
    const data = this._read();
    data.grants = data.grants.filter(
      (g) =>
        !(
          g.grantingSession === grantingSession &&
          g.granteeSession  === granteeSession &&
          (subtreePath === undefined || g.subtreePath === subtreePath)
        ),
    );
    this._write(data);
  }

  getGrantsFor(granteeSession: string): Grant[] {
    return this._read().grants.filter((g) => g.granteeSession === granteeSession);
  }

  getVisiblePaths(sessionId: string): string[] {
    const paths: string[] = [this.sessionDir(sessionId), this.publicDir()];
    for (const grant of this.getGrantsFor(sessionId)) {
      const base = this.sessionDir(grant.grantingSession);
      const full = grant.subtreePath ? path.join(base, grant.subtreePath) : base;
      if (!paths.includes(full)) paths.push(full);
    }
    return paths;
  }

  // ── Persistence ─────────────────────────────────────────────────────────────

  private _read(): SessionsFile {
    if (!fs.existsSync(this.filepath)) {
      return { sessions: {}, grants: [] };
    }
    try {
      return JSON.parse(fs.readFileSync(this.filepath, "utf8")) as SessionsFile;
    } catch {
      return { sessions: {}, grants: [] };
    }
  }

  private _write(data: SessionsFile): void {
    const tmp = this.filepath + ".tmp";
    fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
    fs.renameSync(tmp, this.filepath);
  }
}
