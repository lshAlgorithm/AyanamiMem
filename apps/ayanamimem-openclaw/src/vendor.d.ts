/** Minimal type shims for packages without bundled @types in dev environment. */

declare module "js-yaml" {
  export function load(str: string, options?: Record<string, unknown>): unknown;
  export function dump(obj: unknown, options?: Record<string, unknown>): string;
}
