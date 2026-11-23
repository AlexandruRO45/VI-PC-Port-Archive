"use client";

import { useCallback, useEffect, useState } from "react";
import { PROJECT_SNAPSHOT, type ProjectData } from "./lib/project-data";

const REPO = "https://github.com/tudorek12345/Leonida";
const UPSTREAM = "https://github.com/ItsNotRexed/Leonida";
const REPORT = "https://github.com/ItsNotRexed/Leonida/blob/8e3c7c72b79cabfbe8a2059fce56a5b99d6ddb51/docs/retail_boot_chain.md";
const CLONE = "git clone --recursive https://github.com/tudorek12345/Leonida.git";

const fallback = PROJECT_SNAPSHOT;

function Icon({ name, size = 18 }: { name: "github" | "branch" | "copy" | "check" | "refresh" | "code" | "menu" | "close" | "download"; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    github: <><path d="M9 19c-4.3 1.3-4.3-2.2-6-2.7M15 22v-3.9c0-1.1-.1-1.5-.5-2.1 3.3-.4 6.8-1.6 6.8-7.5 0-1.6-.6-3-1.5-4.1.2-.4.7-2-.2-4.1 0 0-1.3-.4-4.3 1.6A15.2 15.2 0 0 0 7.7 2C4.7 0 3.4.4 3.4.4c-.9 2.1-.4 3.7-.2 4.1A6 6 0 0 0 1.7 8.6c0 5.8 3.5 7.1 6.8 7.5-.4.5-.7 1.1-.7 2.1V22" transform="translate(1 1) scale(.9)" /></>,
    branch: <><circle cx="6" cy="5" r="3"/><circle cx="18" cy="7" r="3"/><circle cx="6" cy="19" r="3"/><path d="M6 8v8m0-4h5a7 7 0 0 0 7-2"/></>,
    copy: <><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 6a8 8 0 0 1 13 3M17.9 18a8 8 0 0 1-13-3"/></>,
    code: <><path d="m8 5-6 7 6 7m8-14 6 7-6 7m-3-17-2 20"/></>,
    menu: <path d="M3 6h18M3 12h18M3 18h18"/>,
    close: <path d="m5 5 14 14M19 5 5 19"/>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

const lanes = [
  { id: "runtime", title: "Runtime & boot", tag: "GET TO THE NEXT CALL", short: "Executable loading, console services, and the first meaningful failure.", heading: "Follow the real executable.", body: "Map the x86-64 executable, resolve the console imports, and use startup traces to find the earliest reproducible blocker. Extend the loader and shims already in Leonida.", tasks: ["Reconcile the fork with upstream boot work", "Verify TLS, storage, and XMem behavior", "Capture repeatable startup traces"], source: `${UPSTREAM}/blob/main/docs/boot.md` },
  { id: "graphics", title: "Memory & graphics", tag: "MAKE THE DATA REACH THE GPU", short: "Xbox D3D12 behavior, descriptor addresses, and Vulkan resource backing.", heading: "Correct memory. Then pixels.", body: "The pinned report’s descriptor-heap assertion is a concrete starting point; check newer upstream work before reproducing it. A useful fix must preserve the relationship between the game's CPU writes, virtual addresses, and GPU-visible resources.", tasks: ["Investigate descriptor-heap address expectations", "Connect physical-page mapping to resource backing", "Trace the first real draw and presentation"], source: REPORT },
  { id: "handoff", title: "Evidence & handoff", tag: "KEEP THE NEXT SESSION MOVING", short: "Small experiments, recorded assumptions, and a reproducible next step.", heading: "Leave a trail someone can follow.", body: "Every experiment should carry its commit, environment, observed result, and next failure. Keep upstream findings distinct from results reproduced in this fork.", tasks: ["Record the exact revision and environment", "Add focused tests for each semantic fix", "Keep the contributor handoff current"], source: "/handoff.md" },
];

function dateLabel(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" }).format(date);
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const [activeLane, setActiveLane] = useState(0);
  const [project, setProject] = useState<ProjectData>(fallback);
  const [loading, setLoading] = useState(true);
  const [activityMessage, setActivityMessage] = useState("Checking repository activity…");

  const loadProject = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const response = await fetch("/api/project", { signal, headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Repository response unavailable");
      const result = await response.json() as ProjectData;
      if (!result.repo || !Array.isArray(result.commits)) throw new Error("Incomplete repository response");
      setProject(result);
      setActivityMessage(result.source === "github" ? "Fetched from the public GitHub repository." : (result.notice || "Showing a saved repository snapshot."));
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") return;
      setProject(fallback);
      setActivityMessage("GitHub activity is temporarily unavailable. Showing the checked 23 November snapshot.");
    } finally { if (!signal?.aborted) setLoading(false); }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadProject(controller.signal);
    return () => controller.abort();
  }, [loadProject]);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2500);
    return () => clearTimeout(timer);
  }, [copied]);

  const copyClone = async () => {
    try { await navigator.clipboard.writeText(CLONE); setCopied(true); setCopyError(false); }
    catch { setCopyError(true); }
  };
  const lane = lanes[activeLane];

  return (
    <div className="site-shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <a className="wordmark" href="#" aria-label="Leonida home"><span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span>LEONIDA<span className="wordmark-suffix">/ HK</span></a>
        <nav className={`main-nav ${menuOpen ? "is-open" : ""}`} aria-label="Main navigation">
          {[ ["The project", "#about"], ["Current work", "#work"], ["Activity", "#activity"] ].map(([label, href]) => <a key={href} href={href} onClick={() => setMenuOpen(false)}>{label}</a>)}
          <a className="nav-github" href={REPO} target="_blank" rel="noopener noreferrer"><Icon name="github"/> View fork</a>
        </nav>
        <button className="menu-toggle" aria-label={menuOpen ? "Close navigation" : "Open navigation"} aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}><Icon name={menuOpen ? "close" : "menu"}/></button>
      </header>

      <main id="main">
        <section className="hero section-wrap" aria-labelledby="hero-title">
          <div className="hero-topline"><span className="eyebrow">HEPERKOO / EXPERIMENTAL FORK</span><span className="hero-platform">XBOX SERIES · x86-64 · WINDOWS</span></div>
          <div className="hero-grid">
            <div className="hero-copy">
              <h1 id="hero-title">Toward the<br/>first <span>real frame.</span></h1>
              <p className="hero-description">A hands-on effort toward running <strong>GTA VI natively on PC</strong>. Building on Leonida, one measured experiment and one runtime blocker at a time.</p>
              <div className="hero-actions"><a className="button button-primary" href={REPO} target="_blank" rel="noopener noreferrer"><Icon name="github"/> Explore the fork</a><a className="text-link" href="#work">See what needs to work</a></div>
              <p className="experimental-note"><span className="status-dot"/> Experimental. No verified rendered gameplay in this fork.</p>
            </div>
            <div className="hero-visual" aria-label="An architectural illustration of the native runtime. This is not a game screenshot.">
              <div className="visual-label"><span>RUNTIME / FIRST FRAME</span><span>RESEARCH BUILD</span></div>
              <div className="frame-art" aria-hidden="true">
                <div className="orbital orbital-one"/><div className="orbital orbital-two"/>
                <div className="wireframe-grid"/>
                <div className="display-frame"><span className="frame-corner corner-tl"/><span className="frame-corner corner-tr"/><span className="frame-corner corner-bl"/><span className="frame-corner corner-br"/><div className="frame-inner-grid"/><div className="frame-glyph">L<span>/</span></div><div className="scan-line"/><div className="frame-await"><span className="pulse-square"/> AWAITING PRESENT</div></div>
                <span className="coordinate coordinate-one">CPU / x86-64</span><span className="coordinate coordinate-two">GPU / VULKAN</span>
                <span className="crosshair crosshair-one">+</span><span className="crosshair crosshair-two">+</span>
              </div>
              <div className="visual-footer"><span className="mono">draw_calls: <strong>not demonstrated</strong></span><span className="art-caption">Architecture illustration</span></div>
            </div>
          </div>
          <div className="hero-bottomline"><span>Built on <a href={UPSTREAM} target="_blank" rel="noopener noreferrer">ItsNotRexed / Leonida</a></span><span>Open development. Shared evidence.</span><a href="/handoff.md" download><Icon name="download" size={16}/> Developer handoff</a></div>
        </section>

        <section id="about" className="about-section section-wrap section-pad">
          <div className="section-kicker"><span className="eyebrow">THE PROJECT</span><span className="section-rule"/></div>
          <div className="about-grid"><h2>Same ambition.<br/><span className="muted-heading">A focused place<br/>to experiment.</span></h2><div className="about-copy"><p>Leonida is working toward executing the Xbox Series game’s x86-64 code on a PC, with console services reimplemented and GPU programs translated for Vulkan.</p><p>This is my working fork for reverse engineering, integration, and collaboration. The goal is the earliest <em>reproducible, game-generated frame</em> — with enough evidence for another developer to pick up exactly where an experiment stopped.</p><div className="attribution"><Icon name="branch" size={22}/><div><strong>Standing on shared work.</strong><span>The original Leonida project and its contributors provide the foundation. This site tracks Heperkoo’s experimental fork.</span></div></div></div></div>
          <div className="architecture-strip" aria-label="Architecture: game CPU code, console runtime services, Xbox graphics interface, Vulkan backend">
            <div><span className="mono architecture-label">EXECUTION</span><strong>x86-64 game code</strong><span>Runs on the PC CPU</span></div><i aria-hidden="true"/><div><span className="mono architecture-label">COMPATIBILITY</span><strong>Console runtime</strong><span>Loader · TLS · GDK · XMem</span></div><i aria-hidden="true"/><div><span className="mono architecture-label">GRAPHICS</span><strong>Xbox D3D12</strong><span>ABI · resources · descriptors</span></div><i aria-hidden="true"/><div><span className="mono architecture-label">BACKEND</span><strong>Vulkan</strong><span>SPIR-V · submission · present</span></div>
          </div>
        </section>

        <section className="evidence-section" id="status">
          <div className="section-wrap section-pad">
            <div className="evidence-heading"><div><span className="eyebrow">DOCUMENTED UPSTREAM CHECKPOINT</span><h2>A measured starting point.<br/>A frame still to verify.</h2></div><a className="small-link" href={REPORT} target="_blank" rel="noopener noreferrer">Read the upstream report</a></div>
            <div className="milestone-track">
              <div className="milestone reached"><span className="milestone-marker"><Icon name="check" size={17}/></span><span className="mono milestone-state">REPORTED</span><h3>Executable startup</h3><p>Real executable execution advanced through early runtime setup.</p></div>
              <div className="milestone reached"><span className="milestone-marker"><Icon name="check" size={17}/></span><span className="mono milestone-state">REPORTED</span><h3>Graphics setup</h3><p>Device, queues, fences, and command lists were created.</p></div>
              <div className="milestone current"><span className="milestone-marker"><span/></span><span className="mono milestone-state">RECORDED CHECKPOINT</span><h3>Descriptor heap</h3><p>An assertion about the heap’s CPU handle stopped this specific measured run.</p></div>
              <div className="milestone pending"><span className="milestone-marker"/><span className="mono milestone-state">NOT DEMONSTRATED</span><h3>Draw & present</h3><p>No draw or present was demonstrated in this report. A frame remains this fork’s target.</p></div>
            </div>
            <div className="evidence-footnote"><span className="evidence-tag">UPSTREAM EVIDENCE</span><p>Recorded 23 November 2026 on one Windows 11 / RTX 4090 setup with locally combined changes and stand-ins. This is a published observation, not a result independently reproduced by this fork. Later upstream branches may be further along. Verify current code before choosing the next blocker.</p></div>
          </div>
        </section>

        <section id="work" className="work-section section-wrap section-pad">
          <div className="section-kicker"><span className="eyebrow">CURRENT WORK</span><span className="section-rule"/></div>
          <div className="section-heading"><h2>A shorter path<br/>through the unknowns.</h2><p>Priorities for the fork’s next development passes.<br/>Follow the failure. Fix the behavior. Run it again.</p></div>
          <div className="work-grid">
            <div className="work-tabs" role="tablist" aria-label="Development focus" aria-orientation="vertical">
              {lanes.map((item, index) => <button key={item.id} id={`tab-${item.id}`} className={`work-tab ${activeLane === index ? "active" : ""}`} role="tab" aria-selected={activeLane === index} aria-controls={`panel-${item.id}`} tabIndex={activeLane === index ? 0 : -1} onClick={() => setActiveLane(index)} onKeyDown={event => { if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); const next = (index + (event.key === "ArrowDown" ? 1 : -1) + lanes.length) % lanes.length; setActiveLane(next); document.getElementById(`tab-${lanes[next].id}`)?.focus(); } }}><span className="tab-heading"><Icon name={index === 0 ? "code" : index === 1 ? "branch" : "copy"}/>{item.title}</span><span className="tab-description">{item.short}</span><span className="tab-indicator" aria-hidden="true"/></button>)}
            </div>
            <div id={`panel-${lane.id}`} role="tabpanel" aria-labelledby={`tab-${lane.id}`} className="work-detail" tabIndex={0}><span className="mono detail-kicker">{lane.tag}</span><h3>{lane.heading}</h3><p>{lane.body}</p><ul>{lane.tasks.map(task => <li key={task}><span aria-hidden="true"/>{task}</li>)}</ul><a className="small-link" href={lane.source} target={lane.source.startsWith("http") ? "_blank" : undefined} rel={lane.source.startsWith("http") ? "noopener noreferrer" : undefined}>{activeLane === 2 ? "Open the developer handoff" : "Read the technical context"}</a></div>
          </div>
        </section>

        <section id="activity" className="activity-section section-wrap section-pad">
          <div className="activity-heading"><div><span className="eyebrow">IN THE REPOSITORY</span><h2>Work leaves a trace.</h2></div><button className="button button-outline refresh-button" disabled={loading} onClick={() => void loadProject()}><span className={loading ? "spin" : ""}><Icon name="refresh" size={16}/></span>{loading ? "Checking GitHub" : "Refresh activity"}</button></div>
          <div className="activity-source"><span className={`source-label ${project.source === "github" ? "live" : ""}`}>{loading ? "CHECKING" : project.source === "github" ? "GITHUB RESPONSE" : "SAVED SNAPSHOT"}</span><p role="status" aria-live="polite">{activityMessage}</p><a href={`${REPO}/commits/main`} target="_blank" rel="noopener noreferrer">Full history</a></div>
          <div className="commit-list">{project.commits.slice(0, 5).map(commit => <a className="commit-row" href={commit.url} key={commit.sha} target="_blank" rel="noopener noreferrer"><span className="commit-icon"><Icon name="branch" size={19}/></span><div className="commit-copy"><h3>{commit.message.split("\n")[0]}</h3><p>{commit.author} <span aria-hidden="true">/</span> {dateLabel(commit.date)}</p></div><code>{commit.sha.slice(0, 7)}</code></a>)}{project.commits.length === 0 && <p className="empty-activity">No recent commits returned. The repository’s full history is available on GitHub.</p>}</div>
          <p className="activity-caveat">Commit activity includes inherited upstream work. It does not by itself establish runtime compatibility or playable gameplay.</p>
        </section>

        <section className="maintainer-section section-wrap section-pad" id="maintainer"><div className="section-kicker"><span className="eyebrow">BEHIND THE FORK</span><span className="section-rule"/></div><div className="maintainer-grid"><div className="maintainer-identity"><div className="monogram" aria-hidden="true">HK<span>_</span></div><div><h2>Heperkoo</h2><a href="https://github.com/tudorek12345" target="_blank" rel="noopener noreferrer">@tudorek12345</a></div></div><div className="maintainer-copy"><p>I build multiplayer mods, reverse-engineering tools, and porting experiments. I work in VS Code with Codex, connecting analysis to changes that can actually be tested.</p><p>My ongoing work includes <strong>My Winter Car Multiplayer</strong>, a <strong>Batman Telltale port</strong>, and <strong>Call of Duty 2 co-op</strong>. This fork brings that practical workflow to Leonida: investigate, implement, test, document, and collaborate.</p><div className="experience-tags"><span>Reverse engineering</span><span>Multiplayer systems</span><span>Runtime integration</span><span>AI-assisted development</span></div><a className="small-link" href="https://github.com/tudorek12345" target="_blank" rel="noopener noreferrer">Explore my GitHub projects</a></div></div></section>

        <section className="collaborate-section" id="collaborate"><div className="section-wrap collaborate-inner"><div><span className="eyebrow">BUILD IT TOGETHER</span><h2>Bring a trace.<br/><span>Move the next blocker.</span></h2><p>Runtime, graphics, tooling, or careful testing — useful collaboration starts with a reproducible result. Read the handoff, check ongoing upstream work, and bring a focused change to the fork.</p><div className="collaborate-actions"><a className="button button-primary" href={`${REPO}/pulls`} target="_blank" rel="noopener noreferrer"><Icon name="github"/> Collaborate on GitHub</a><a className="button button-outline" href="/handoff.md" download><Icon name="download"/> Get the handoff</a></div></div><div className="clone-card"><span className="mono">YOUR STARTING POINT</span><div className="clone-repo"><Icon name="branch" size={25}/><span>tudorek12345<br/><strong>Leonida</strong></span></div><code>{CLONE}</code><button className="copy-button" onClick={copyClone}><Icon name={copied ? "check" : "copy"} size={16}/>{copied ? "Copied to clipboard" : "Copy clone command"}</button>{copyError && <p role="status" className="copy-feedback">Select and copy the command above. Clipboard access is unavailable in this browser.</p>}<div className="clone-footnote">Experimental source code. Bring your own locally readable game files.</div></div></div></section>
      </main>
      <footer className="site-footer section-wrap"><div className="footer-top"><a className="wordmark" href="#"><span className="brand-symbol" aria-hidden="true"><i/><i/><i/></span>LEONIDA<span className="wordmark-suffix">/ HK</span></a><span>AN EXPERIMENTAL FORK BY HEPERKOO</span><a href={UPSTREAM} target="_blank" rel="noopener noreferrer">Original Leonida project</a></div><div className="footer-bottom"><p>Independent research project. Not affiliated with Rockstar Games, Take-Two Interactive, or Microsoft. GTA VI and related names belong to their respective owners.</p><a href={`${REPO}/blob/main/LICENSE`} target="_blank" rel="noopener noreferrer">GPL-3.0</a><span>Made for the next handoff.</span></div></footer>
    </div>
  );
}
