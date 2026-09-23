import { Audio, Video } from "@remotion/media";
import type { ReactNode } from "react";
import { AbsoluteFill, Easing, Img, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";

const colors = {
  bg: "#050807",
  panel: "#0d1210",
  line: "#26312d",
  text: "#f3f7f5",
  muted: "#9aaba5",
  teal: "#20d1aa",
  tealDark: "#12876f",
  mint: "#8fffe2",
};

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

function Atmosphere({ intensity = 1 }: { intensity?: number }) {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ background: colors.bg, overflow: "hidden" }}>
    <AbsoluteFill style={{
      backgroundImage: "radial-gradient(circle at 50% 25%, #123f34 0%, #07100d 34%, #050807 68%)",
      opacity: intensity,
    }} />
    <div style={{ position: "absolute", width: 1100, height: 1100, left: -430, top: -520,
      borderRadius: "50%", background: "radial-gradient(circle, #20d1aa30 0%, transparent 68%)",
      translate: `${interpolate(frame, [0, 180], [0, 160], clamp)}px 0`, opacity: 0.9 }} />
    <div style={{ position: "absolute", width: 900, height: 900, right: -390, bottom: -500,
      borderRadius: "50%", background: "radial-gradient(circle, #48d8bc24 0%, transparent 67%)",
      translate: `${interpolate(frame, [0, 180], [0, -120], clamp)}px 0` }} />
    <div style={{ position: "absolute", left: -120, right: -120, height: 760, bottom: -520,
      backgroundImage: "linear-gradient(#20d1aa20 1px, transparent 1px), linear-gradient(90deg, #20d1aa20 1px, transparent 1px)",
      backgroundSize: "58px 58px", perspective: 800, rotate: "1 0 0 67deg",
      translate: `0 ${interpolate(frame, [0, 180], [0, -45], clamp)}px`, opacity: 0.45 }} />
  </AbsoluteFill>;
}

function BrandBug() {
  return <div style={{ position: "absolute", top: 35, left: 46, display: "flex", alignItems: "center", gap: 10, zIndex: 20 }}>
    <Img src={staticFile("automl-mark.png")} style={{ width: 34, height: 34, objectFit: "contain" }} />
    <span style={{ color: colors.text, fontSize: 22, fontWeight: 700 }}>AutoML</span>
  </div>;
}

function BrowserFrame({ children, tilt = -1.5, badge, compact = false }: {
  children: ReactNode; tilt?: number; badge?: string; compact?: boolean;
}) {
  const frame = useCurrentFrame();
  return <div style={{ position: "absolute", left: compact ? 130 : 90, top: compact ? 177 : 162,
    width: compact ? 1020 : 1100, height: compact ? 490 : 528, perspective: 1600,
    opacity: interpolate(frame, [2, 15], [0, 1], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) }),
    translate: `0 ${interpolate(frame, [2, 18], [42, 0], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) })}px`,
    scale: interpolate(frame, [0, 90], [0.91, 1.015], clamp),
    rotate: `0 1 0 ${tilt}deg`, zIndex: 5 }}>
    <div style={{ position: "absolute", inset: 0, borderRadius: 9, overflow: "hidden",
      border: "1px solid #49635a", background: colors.panel,
      boxShadow: "0 45px 100px #000b, 0 0 54px #20d1aa22" }}>
      <div style={{ height: 28, background: "#111715", borderBottom: `1px solid ${colors.line}`,
        display: "flex", alignItems: "center", gap: 7, padding: "0 12px" }}>
        <i style={{ width: 7, height: 7, borderRadius: "50%", background: "#60716b" }} />
        <i style={{ width: 7, height: 7, borderRadius: "50%", background: "#60716b" }} />
        <i style={{ width: 7, height: 7, borderRadius: "50%", background: colors.tealDark }} />
        <span style={{ marginLeft: 10, color: "#6f827b", fontSize: 10, fontFamily: "monospace" }}>automl / workspace</span>
      </div>
      <div style={{ width: "100%", height: "calc(100% - 28px)", overflow: "hidden" }}>{children}</div>
    </div>
    {badge && <div style={{ position: "absolute", right: -24, top: 54, padding: "10px 15px",
      borderRadius: 5, background: "#0b1713e8", border: `1px solid ${colors.tealDark}`,
      boxShadow: "0 16px 45px #000a, 0 0 24px #20d1aa29", color: colors.mint,
      fontSize: 13, fontWeight: 700, letterSpacing: 0.4,
      opacity: interpolate(frame, [22, 34], [0, 1], clamp),
      scale: interpolate(frame, [22, 36], [0.76, 1], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) }) }}>{badge}</div>}
  </div>;
}

function Headline({ eyebrow, line1, line2, align = "left" }: {
  eyebrow: string; line1: string; line2?: string; align?: "left" | "center";
}) {
  const frame = useCurrentFrame();
  return <div style={{ position: "absolute", zIndex: 10, left: align === "left" ? 64 : 80,
    right: align === "left" ? 64 : 80, top: 77, textAlign: align,
    opacity: interpolate(frame, [0, 12], [0, 1], clamp),
    translate: `0 ${interpolate(frame, [0, 14], [22, 0], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) })}px` }}>
    <div style={{ color: colors.teal, fontSize: 13, fontWeight: 800, letterSpacing: 2.2, marginBottom: 8 }}>{eyebrow}</div>
    <div style={{ color: colors.text, fontSize: 51, lineHeight: 1.02, fontWeight: 750 }}>{line1}{line2 && <><br /><span style={{ color: colors.mint }}>{line2}</span></>}</div>
  </div>;
}

function IntroScene() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif" }}>
    <Atmosphere intensity={1} />
    <BrandBug />
    <div style={{ position: "absolute", left: 62, right: 62, top: 206, zIndex: 4 }}>
      <div style={{ color: colors.text, fontSize: 88, lineHeight: 0.98, fontWeight: 780,
        opacity: interpolate(frame, [3, 18], [0, 1], clamp),
        translate: `${interpolate(frame, [3, 18], [-34, 0], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) })}px 0` }}>One dataset.</div>
      <div style={{ color: colors.mint, fontSize: 88, lineHeight: 1.02, fontWeight: 780, marginTop: 12,
        opacity: interpolate(frame, [14, 30], [0, 1], clamp),
        translate: `${interpolate(frame, [14, 30], [42, 0], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) })}px 0` }}>One clear winner.</div>
      <div style={{ width: interpolate(frame, [26, 57], [0, 560], clamp), height: 3,
        background: colors.teal, marginTop: 34, boxShadow: "0 0 18px #20d1aa88" }} />
    </div>
    <div style={{ position: "absolute", right: 20, top: 95, width: 470, height: 226, opacity: 0.18,
      rotate: "-5deg", scale: interpolate(frame, [0, 75], [0.96, 1.08], clamp), filter: "blur(2px)",
      border: "1px solid #20d1aa55", borderRadius: 8, overflow: "hidden" }}>
      <Video src={staticFile("run/train-late.mp4")} muted playbackRate={1.5} objectFit="cover" style={{ width: "100%", height: "100%" }} />
    </div>
  </AbsoluteFill>;
}

function DataScene() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif" }}>
    <Atmosphere intensity={0.88} /><BrandBug />
    <Headline eyebrow="PROFILE" line1="Every row." line2="No blind spots." />
    <BrowserFrame tilt={-1.8} badge="29,451 ROWS · PROFILED">
      <Video src={staticFile("run/data.mp4")} muted playbackRate={1.5} objectFit="cover"
        style={{ width: "100%", height: "100%", scale: interpolate(frame, [0, 105], [1.02, 1.08], clamp) }} />
    </BrowserFrame>
  </AbsoluteFill>;
}

function SchemaScene() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif" }}>
    <Atmosphere intensity={0.84} /><BrandBug />
    <Headline eyebrow="CONTROL" line1="You set the target." line2="AutoML locks the contract." />
    <BrowserFrame tilt={1.7} badge="TARGET · PRICE_IN_LACS">
      <Video src={staticFile("run/schema.mp4")} muted playbackRate={1.5} objectFit="cover"
        style={{ width: "100%", height: "100%", scale: interpolate(frame, [0, 105], [1.04, 1.1], clamp),
          translate: `${interpolate(frame, [0, 105], [0, -16], clamp)}px 0` }} />
    </BrowserFrame>
  </AbsoluteFill>;
}

function MetricChip({ children, left, delay }: { children: ReactNode; left: number; delay: number }) {
  const frame = useCurrentFrame();
  return <div style={{ position: "absolute", left, top: 129, zIndex: 15, borderRadius: 4,
    border: "1px solid #2e6658", background: "#0b1713e8", color: colors.mint,
    padding: "7px 12px", fontSize: 12, fontWeight: 700, boxShadow: "0 12px 30px #0008",
    opacity: interpolate(frame, [delay, delay + 10], [0, 1], clamp),
    scale: interpolate(frame, [delay, delay + 12], [0.72, 1], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) }) }}>{children}</div>;
}

function TrainingScene() {
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif" }}>
    <Atmosphere intensity={0.94} /><BrandBug />
    <Headline eyebrow="LIVE TRAINING" line1="Models compete." line2="Every attempt stays visible." />
    <MetricChip left={855} delay={22}>LIVE SCORES</MetricChip>
    <MetricChip left={962} delay={30}>FAILURES</MetricChip>
    <MetricChip left={1052} delay={38}>COST</MetricChip>
    <BrowserFrame tilt={-0.8} compact>
      <Sequence durationInFrames={45} name="Configure the run">
        <Video src={staticFile("run/configure.mp4")} muted playbackRate={2} objectFit="cover" style={{ width: "100%", height: "100%" }} />
      </Sequence>
      <Sequence from={45} durationInFrames={60} name="Training begins">
        <Video src={staticFile("run/train-start.mp4")} muted playbackRate={1.7} objectFit="cover" style={{ width: "100%", height: "100%" }} />
      </Sequence>
      <Sequence from={105} durationInFrames={60} name="Models improve">
        <Video src={staticFile("run/train-mid.mp4")} muted playbackRate={1.7} objectFit="cover" style={{ width: "100%", height: "100%" }} />
      </Sequence>
      <Sequence from={165} durationInFrames={60} name="Leaderboard fills">
        <Video src={staticFile("run/train-late.mp4")} muted playbackRate={1.7} objectFit="cover" style={{ width: "100%", height: "100%" }} />
      </Sequence>
    </BrowserFrame>
  </AbsoluteFill>;
}

function ResultsScene() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif" }}>
    <Atmosphere intensity={1} /><BrandBug />
    <Headline eyebrow="THE HANDOFF" line1="The winner" line2="explains itself." />
    <BrowserFrame tilt={1.1} badge="MODEL + SCRIPT · READY">
      <Video src={staticFile("run/results.mp4")} muted playbackRate={1.6} objectFit="cover"
        style={{ width: "100%", height: "100%", scale: interpolate(frame, [0, 135], [1.03, 1.11], clamp),
          translate: `${interpolate(frame, [0, 135], [0, -22], clamp)}px 0` }} />
    </BrowserFrame>
    <div style={{ position: "absolute", zIndex: 14, right: 72, bottom: 42, color: colors.muted,
      fontSize: 13, letterSpacing: 1.1 }}>COMPARE · INSPECT · DOWNLOAD</div>
  </AbsoluteFill>;
}

function OutroScene() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ fontFamily: "Inter, Segoe UI, Arial, sans-serif", alignItems: "center", justifyContent: "center" }}>
    <Atmosphere intensity={1} />
    <div style={{ position: "relative", zIndex: 5, display: "flex", alignItems: "center", gap: 18,
      opacity: interpolate(frame, [0, 13], [0, 1], clamp), scale: interpolate(frame, [0, 18], [0.86, 1], { ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1) }) }}>
      <Img src={staticFile("automl-mark.png")} style={{ width: 72, height: 72, objectFit: "contain" }} />
      <span style={{ color: colors.text, fontSize: 62, fontWeight: 760 }}>AutoML</span>
    </div>
    <div style={{ position: "relative", zIndex: 5, color: colors.mint, fontSize: 32, marginTop: 30,
      fontWeight: 600, opacity: interpolate(frame, [14, 29], [0, 1], clamp),
      translate: `0 ${interpolate(frame, [14, 29], [22, 0], clamp)}px` }}>From CSV to a model you can ship.</div>
    <div style={{ position: "absolute", zIndex: 5, bottom: 72, width: interpolate(frame, [28, 57], [0, 520], clamp),
      height: 2, background: colors.teal, boxShadow: "0 0 20px #20d1aa88" }} />
  </AbsoluteFill>;
}

function Flash() {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ pointerEvents: "none", zIndex: 50,
    background: "linear-gradient(100deg, transparent 0%, #8fffe288 44%, #ffffffcc 50%, #20d1aa55 57%, transparent 100%)",
    opacity: interpolate(frame, [0, 3, 9], [0, 0.9, 0], clamp),
    translate: `${interpolate(frame, [0, 9], [-700, 700], clamp)}px 0` }} />;
}

export const CinematicVideo: React.FC = () => <AbsoluteFill style={{ background: colors.bg }}>
  <Sequence durationInFrames={75} name="Hook"><IntroScene /></Sequence>
  <Sequence from={75} durationInFrames={105} name="Profile data"><DataScene /></Sequence>
  <Sequence from={180} durationInFrames={105} name="Set target"><SchemaScene /></Sequence>
  <Sequence from={285} durationInFrames={225} name="Train models"><TrainingScene /></Sequence>
  <Sequence from={510} durationInFrames={135} name="Understand results"><ResultsScene /></Sequence>
  <Sequence from={645} durationInFrames={75} name="Close"><OutroScene /></Sequence>

  <Sequence from={70} durationInFrames={10}><Flash /></Sequence>
  <Sequence from={175} durationInFrames={10}><Flash /></Sequence>
  <Sequence from={280} durationInFrames={10}><Flash /></Sequence>
  <Sequence from={505} durationInFrames={10}><Flash /></Sequence>
  <Sequence from={640} durationInFrames={10}><Flash /></Sequence>

  <Audio src={staticFile("audio/score.wav")} volume={0.72} />
  <Sequence from={66} durationInFrames={28}><Audio src={staticFile("audio/whoosh.wav")} volume={0.32} /></Sequence>
  <Sequence from={171} durationInFrames={28}><Audio src={staticFile("audio/whoosh.wav")} volume={0.32} /></Sequence>
  <Sequence from={276} durationInFrames={28}><Audio src={staticFile("audio/switch.wav")} volume={0.28} /></Sequence>
  <Sequence from={330} durationInFrames={20}><Audio src={staticFile("audio/click.wav")} volume={0.22} /></Sequence>
  <Sequence from={390} durationInFrames={20}><Audio src={staticFile("audio/click.wav")} volume={0.22} /></Sequence>
  <Sequence from={501} durationInFrames={28}><Audio src={staticFile("audio/whoosh.wav")} volume={0.34} /></Sequence>
  <Sequence from={534} durationInFrames={45}><Audio src={staticFile("audio/ding.wav")} volume={0.2} /></Sequence>
  <Sequence from={636} durationInFrames={28}><Audio src={staticFile("audio/switch.wav")} volume={0.25} /></Sequence>
</AbsoluteFill>;
