import { config } from "./config.js";

type DateTimeOptions = Intl.DateTimeFormatOptions;

export function formatUserDateTime(
  ms: number,
  options: DateTimeOptions = {}
): string {
  return new Date(ms).toLocaleString("en-US", {
    timeZone: config.TZ,
    ...options,
  });
}

export function formatUserTime(
  ms: number,
  options: DateTimeOptions = {}
): string {
  return new Date(ms).toLocaleTimeString("en-US", {
    timeZone: config.TZ,
    hour: "2-digit",
    minute: "2-digit",
    ...options,
  });
}

export function formatUserDate(
  ms: number,
  options: DateTimeOptions = {}
): string {
  return new Date(ms).toLocaleDateString("en-US", {
    timeZone: config.TZ,
    month: "short",
    day: "numeric",
    year: "numeric",
    ...options,
  });
}
