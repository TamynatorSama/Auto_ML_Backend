/** Numbers and dates as the console writes them. */

export function count(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toLocaleString("en-GB");
}

/** 4.21 M, 812 K — the short form beside the exact one. */
export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  if (value >= 1e3) return `${Math.round(value / 1e3)} K`;
  return String(value);
}

export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size < 10 && unit > 0 ? size.toFixed(1) : Math.round(size)} ${units[unit]}`;
}

export function percent(value: number | null | undefined, places = 1): string {
  return value === null || value === undefined ? "—" : `${Number(value).toFixed(places)}%`;
}

/** "for 7 more days" — how long something lasts, rounded up so the last day still counts. */
export function until(iso: string | null | undefined): string {
  if (!iso) return "—";
  const days = Math.ceil((new Date(iso).getTime() - Date.now()) / 86_400_000);
  if (days <= 0) return "no longer";
  return days === 1 ? "for one more day" : `for ${days} more days`;
}

/** "4 minutes ago", "in 7 days" — whichever side of now it falls. */
export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000;
  const ahead = seconds > 0;
  const spans: [number, string][] = [[60, "second"], [60, "minute"], [24, "hour"], [7, "day"], [4.35, "week"],
                                     [12, "month"]];
  let size = Math.abs(seconds);
  let name = "year";
  for (const [step, label] of spans) {
    if (size < step) { name = label; break; }
    size /= step;
  }
  const whole = Math.max(Math.floor(size), 1);
  const plural = `${whole} ${name}${whole === 1 ? "" : "s"}`;
  return ahead ? `in ${plural}` : `${plural} ago`;
}
