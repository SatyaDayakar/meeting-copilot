"""
app.py — Meeting Intelligence Copilot
A multi-tab Streamlit dashboard with streaming progress, rich analytics,
chat interface, and one-click content export.
"""

import streamlit as st
import tempfile
import os
import json
import time
from datetime import datetime

from transcribe import transcribe_audio, format_transcript_for_llm, TranscriptResult
from extract import extract_insights, insights_to_dict, MeetingInsights

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Meeting Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1040 50%, #0d1f3c 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    border: 1px solid rgba(139, 92, 246, 0.3);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: #ffffff;
    margin: 0; letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem; color: rgba(255,255,255,0.55);
    margin-top: 6px;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.25);
    color: #c4b5fd;
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    background: #13131f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    flex: 1; min-width: 140px;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; line-height: 1; }
.metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.45); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Section cards ── */
.section-card {
    background: #13131f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 0.8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: rgba(255,255,255,0.4);
    margin-bottom: 14px;
}

/* ── Priority badges ── */
.badge {
    display: inline-block; border-radius: 6px;
    padding: 2px 8px; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-HIGH   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-MEDIUM { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-LOW    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-BLOCKER       { background: rgba(239,68,68,0.15);  color: #f87171; }
.badge-OPEN_QUESTION { background: rgba(59,130,246,0.15); color: #60a5fa; }
.badge-DEPENDENCY    { background: rgba(245,158,11,0.15); color: #fbbf24; }
.badge-RISK          { background: rgba(168,85,247,0.15); color: #c084fc; }

/* ── Action item row ── */
.action-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.action-row:last-child { border-bottom: none; }
.action-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: white; flex-shrink: 0;
}
.action-task { font-size: 0.88rem; color: rgba(255,255,255,0.85); margin-bottom: 4px; }
.action-meta { font-size: 0.75rem; color: rgba(255,255,255,0.4); }

/* ── Health score ring ── */
.health-ring-wrap { text-align: center; padding: 10px 0; }
.health-score-num { font-size: 3rem; font-weight: 700; }
.health-label { font-size: 0.75rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; }

/* ── Speaker sentiment ── */
.sentiment-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
}
.sentiment-row:last-child { border-bottom: none; }
.sent-POSITIVE { color: #4ade80; }
.sent-NEUTRAL  { color: #94a3b8; }
.sent-NEGATIVE { color: #f87171; }

/* ── Code / export blocks ── */
.export-block {
    background: #0a0a14;
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #c4b5fd;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 260px;
    overflow-y: auto;
}

/* ── Transcript segment ── */
.seg-row { display: flex; gap: 12px; margin-bottom: 10px; }
.seg-speaker {
    font-size: 0.72rem; font-weight: 600; color: #a78bfa;
    min-width: 90px; padding-top: 2px;
    text-transform: uppercase; letter-spacing: 0.4px;
}
.seg-time { font-size: 0.68rem; color: rgba(255,255,255,0.3); min-width: 44px; padding-top: 3px; }
.seg-text { font-size: 0.85rem; color: rgba(255,255,255,0.8); line-height: 1.55; }
.seg-conf-high { border-left: 3px solid #4ade80; padding-left: 10px; }
.seg-conf-mid  { border-left: 3px solid #fbbf24; padding-left: 10px; }
.seg-conf-low  { border-left: 3px solid #f87171; padding-left: 10px; }

/* ── Chat ── */
.chat-msg-user { background: rgba(139,92,246,0.15); border-radius: 12px 12px 2px 12px; padding: 10px 14px; margin: 8px 0; text-align: right; }
.chat-msg-ai   { background: #1a1a2e; border-radius: 12px 12px 12px 2px; padding: 10px 14px; margin: 8px 0; }

/* ── Ticket card ── */
.ticket-card {
    background: #0f0f1e;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.ticket-title { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
.ticket-desc  { font-size: 0.78rem; color: rgba(255,255,255,0.5); line-height: 1.5; }

/* ── Topic pill ── */
.topic-pill {
    display: inline-block;
    background: rgba(139,92,246,0.12);
    color: #a78bfa;
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin: 3px;
}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #13131f; border-radius: 8px;
    color: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(139,92,246,0.2) !important;
    color: #c4b5fd !important;
    border-color: rgba(139,92,246,0.4) !important;
}
div[data-testid="stFileUploader"] {
    background: #13131f;
    border: 2px dashed rgba(139,92,246,0.35);
    border-radius: 14px;
    padding: 20px;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "transcript_result": None,
        "insights": None,
        "chat_history": [],
        "processing": False,
        "whisper_model": "base",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=2,
        help="Larger = more accurate, slower"
    )
    st.markdown("---")
    st.markdown("**Agents running:**")
    for agent in ["🔍 Summarizer", "✅ Action Extractor", "⚠️ Risk Detector",
                   "💬 Sentiment Analyzer", "📝 Content Generator"]:
        st.markdown(f"- {agent}")

    if st.session_state.insights:
        st.markdown("---")
        st.markdown("**Export**")
        insights_dict = insights_to_dict(st.session_state.insights)
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(insights_dict, indent=2),
            file_name=f"meeting_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )


# ─── Hero ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-badge">Gen AI Copilot</div>
  <div class="hero-title">🧠 Meeting Intelligence Copilot</div>
  <div class="hero-sub">Upload a recording → multi-agent pipeline → summary, action items, risks, tickets, and a follow-up draft. All in one click.</div>
</div>
""", unsafe_allow_html=True)


# ─── Upload ───────────────────────────────────────────────────────────────────

audio_file = st.file_uploader(
    "Drop your meeting audio here",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="Supports MP3, WAV, M4A, OGG, FLAC. Up to ~2 hours."
)

if audio_file and not st.session_state.insights:
    if st.button("🚀 Run Copilot Pipeline"):
        st.session_state.processing = True
        suffix = os.path.splitext(audio_file.name)[1]
        tmp_path = os.path.join(tempfile.gettempdir(), f"meeting_audio{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())

        # ── Transcription ──
        status = st.status("Running pipeline...", expanded=True)
        with status:
            st.write("🎙️ Transcribing audio...")
            try:
                result: TranscriptResult = transcribe_audio(tmp_path, model_size=st.session_state.whisper_model)
                st.session_state.transcript_result = result
                st.write(f"✅ Transcript ready — {len(result.segments)} segments, {result.speaker_count} speaker(s) detected")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

            # ── Multi-agent extraction ──
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(msg: str, pct: int):
                progress_bar.progress(pct)
                progress_text.write(f"🤖 {msg}")

            labeled_transcript = format_transcript_for_llm(result)

            try:
                insights: MeetingInsights = extract_insights(
                    labeled_transcript,
                    progress_callback=on_progress,
                )
                st.session_state.insights = insights
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

            progress_bar.progress(100)
            progress_text.write("✅ Pipeline complete!")
            status.update(label="✅ Pipeline complete!", state="complete")

        os.remove(tmp_path)
        st.session_state.processing = False
        st.rerun()


# ─── Results ─────────────────────────────────────────────────────────────────

if st.session_state.insights:
    ins: MeetingInsights = st.session_state.insights
    tr: TranscriptResult = st.session_state.transcript_result

    # ── Top metrics ──
    def fmt_duration(secs):
        m, s = int(secs // 60), int(secs % 60)
        return f"{m}m {s}s" if m else f"{s}s"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{len(ins.action_items)}</div>
        <div class="metric-label">Action Items</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{len(ins.risks)}</div>
        <div class="metric-label">Risks / Blockers</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{tr.speaker_count if tr else "—"}</div>
        <div class="metric-label">Speakers</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{fmt_duration(tr.duration) if tr else "—"}</div>
        <div class="metric-label">Duration</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{ins.health_score}/10</div>
        <div class="metric-label">Meeting Health</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{ins.processing_time_ms // 1000}s</div>
        <div class="metric-label">Pipeline Time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab_overview, tab_actions, tab_risks, tab_content, tab_tickets, tab_transcript, tab_chat = st.tabs([
        "📊 Overview", "✅ Actions", "⚠️ Risks", "📧 Content", "🎫 Tickets", "📄 Transcript", "💬 Chat"
    ])

    # ═══════════ OVERVIEW ═══════════
    with tab_overview:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:rgba(255,255,255,0.8);font-size:0.9rem;line-height:1.7">{ins.executive_summary}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Key Decisions</div>', unsafe_allow_html=True)
            for d in ins.key_decisions:
                st.markdown(f'<p style="color:rgba(255,255,255,0.75);font-size:0.85rem;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)">◆ {d}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Topics Discussed</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="topic-pill">{t}</span>' for t in ins.topics)
            st.markdown(pills, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Meeting health
            score = ins.health_score
            color = "#4ade80" if score >= 7 else "#fbbf24" if score >= 4 else "#f87171"
            st.markdown(f"""
            <div class="section-card" style="text-align:center">
              <div class="section-title">Meeting Health</div>
              <div style="font-size:3.5rem;font-weight:700;color:{color};line-height:1">{score}</div>
              <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:4px">out of 10</div>
              <div style="font-size:0.7rem;color:rgba(255,255,255,0.3);margin-top:8px">{ins.meeting_type}</div>
            </div>
            """, unsafe_allow_html=True)

            # Sentiment
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Speaker Sentiment</div>', unsafe_allow_html=True)
            for s in ins.speaker_sentiments:
                icon = "🟢" if s.sentiment == "POSITIVE" else "🟡" if s.sentiment == "NEUTRAL" else "🔴"
                st.markdown(f"""
                <div class="sentiment-row">
                  <div>
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.8);font-weight:500">{icon} {s.speaker}</div>
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.35);margin-top:2px">"{s.notable_quote}"</div>
                  </div>
                  <div style="font-size:0.72rem;color:rgba(255,255,255,0.4)">{s.engagement}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Open questions
            if ins.open_questions:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Open Questions</div>', unsafe_allow_html=True)
                for q in ins.open_questions:
                    st.markdown(f'<p style="font-size:0.8rem;color:rgba(255,255,255,0.6);margin:6px 0">❓ {q}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════ ACTIONS ═══════════
    with tab_actions:
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_actions = sorted(ins.action_items, key=lambda x: priority_order.get(x.priority, 2))

        # Filter by priority
        col_f, col_s = st.columns([2, 2])
        with col_f:
            filter_priority = st.multiselect("Filter by priority", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
        with col_s:
            filter_cat = st.multiselect("Filter by category", list({a.category for a in ins.action_items}), default=list({a.category for a in ins.action_items}))

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        displayed = 0
        for action in sorted_actions:
            if action.priority not in filter_priority:
                continue
            if action.category not in filter_cat:
                continue
            initials = "".join(w[0].upper() for w in action.person.split()[:2])
            st.markdown(f"""
            <div class="action-row">
              <div class="action-avatar">{initials}</div>
              <div style="flex:1">
                <div class="action-task">{action.task}</div>
                <div class="action-meta">
                  👤 {action.person} &nbsp;·&nbsp; 📅 {action.deadline} &nbsp;·&nbsp; 🏷️ {action.category}
                  &nbsp;&nbsp;<span class="badge badge-{action.priority}">{action.priority}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            displayed += 1
        if displayed == 0:
            st.info("No action items match the current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════ RISKS ═══════════
    with tab_risks:
        if not ins.risks:
            st.success("🎉 No blockers or risks detected in this meeting.")
        else:
            for risk in sorted(ins.risks, key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.severity, 1)):
                st.markdown(f"""
                <div class="section-card" style="border-left:3px solid {'#f87171' if risk.severity=='HIGH' else '#fbbf24' if risk.severity=='MEDIUM' else '#4ade80'}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span class="badge badge-{risk.type}">{risk.type.replace('_',' ')}</span>
                    <span class="badge badge-{risk.severity}">{risk.severity}</span>
                  </div>
                  <div style="color:rgba(255,255,255,0.85);font-size:0.88rem">{risk.description}</div>
                  <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;margin-top:6px">Owner: {risk.owner}</div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════ CONTENT ═══════════
    with tab_content:
        col_email, col_slack = st.columns(2)
        with col_email:
            st.markdown("#### 📧 Follow-up Email")
            email_display = ins.email.replace("\\n", "\n")
            st.markdown(f'<div class="export-block">{email_display}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Copy Email", data=email_display, file_name="followup_email.txt", mime="text/plain")
        with col_slack:
            st.markdown("#### 💬 Slack Message")
            slack_display = ins.slack_message.replace("\\n", "\n")
            st.markdown(f'<div class="export-block">{slack_display}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Copy Slack Msg", data=slack_display, file_name="slack_message.txt", mime="text/plain")

    # ═══════════ TICKETS ═══════════
    with tab_tickets:
        if not ins.ticket_drafts:
            st.info("No engineering/design tasks detected for ticketing.")
        else:
            st.markdown("*Auto-generated JIRA-style tickets from action items. Review before importing.*")
            for ticket in ins.ticket_drafts:
                labels_html = "".join(f'<span class="topic-pill">{l}</span>' for l in ticket.get("labels", []))
                st.markdown(f"""
                <div class="ticket-card">
                  <div class="ticket-title">🎫 {ticket.get('title','')}</div>
                  <div class="ticket-desc">{ticket.get('description','')}</div>
                  <div style="margin-top:8px">
                    {labels_html}
                    <span style="font-size:0.72rem;color:rgba(255,255,255,0.35);margin-left:8px">👤 {ticket.get('assignee','Unassigned')}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            ticket_export = json.dumps(ins.ticket_drafts, indent=2)
            st.download_button("⬇️ Export Tickets JSON", data=ticket_export, file_name="tickets.json", mime="application/json")

    # ═══════════ TRANSCRIPT ═══════════
    with tab_transcript:
        tr = st.session_state.transcript_result
        if tr:
            st.markdown(f"**Language:** `{tr.language}` &nbsp;|&nbsp; **Duration:** `{fmt_duration(tr.duration)}` &nbsp;|&nbsp; **Speakers:** `{tr.speaker_count}`")
            speaker_filter = st.multiselect(
                "Filter by speaker",
                list({s.speaker for s in tr.segments}),
                default=list({s.speaker for s in tr.segments}),
            )
            st.markdown('<div class="section-card" style="max-height:500px;overflow-y:auto">', unsafe_allow_html=True)
            for seg in tr.segments:
                if seg.speaker not in speaker_filter:
                    continue
                mins, secs = int(seg.start // 60), int(seg.start % 60)
                conf_class = "seg-conf-high" if seg.confidence > 0.7 else "seg-conf-mid" if seg.confidence > 0.4 else "seg-conf-low"
                st.markdown(f"""
                <div class="seg-row">
                  <div class="seg-time">{mins:02d}:{secs:02d}</div>
                  <div class="seg-speaker">{seg.speaker}</div>
                  <div class="seg-text {conf_class}">{seg.text}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Transcript not available.")

    # ═══════════ CHAT ═══════════
    with tab_chat:
        st.markdown("**Ask anything about your meeting.** The Copilot has full context of the transcript and all extracted insights.")

        # Display history
        for msg in st.session_state.chat_history:
            role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f'<div class="{role_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        user_q = st.chat_input("Ask about decisions, owners, risks, or anything from the meeting…")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            from groq import Groq as _G
            _client = _G(api_key=os.environ.get("GROQ_API_KEY"))

            context = f"""You are a Meeting Copilot assistant with full access to the meeting analysis below.
Answer concisely and helpfully. Use bullet points where appropriate.

=== MEETING CONTEXT ===
Type: {ins.meeting_type}
Summary: {ins.executive_summary}
Key Decisions: {'; '.join(ins.key_decisions)}
Action Items: {json.dumps([{'person': a.person, 'task': a.task, 'deadline': a.deadline, 'priority': a.priority} for a in ins.action_items], ensure_ascii=False)}
Risks: {json.dumps([{'type': r.type, 'desc': r.description, 'owner': r.owner} for r in ins.risks], ensure_ascii=False)}
Open Questions: {'; '.join(ins.open_questions)}
Health Score: {ins.health_score}/10
Transcript (excerpt): {(st.session_state.transcript_result.full_text if st.session_state.transcript_result else '')[:2000]}
========================
"""
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": context},
                    *st.session_state.chat_history,
                ],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

elif not audio_file:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:rgba(255,255,255,0.3)">
      <div style="font-size:3rem;margin-bottom:16px">🎙️</div>
      <div style="font-size:1rem;font-weight:500">Upload a meeting recording to begin</div>
      <div style="font-size:0.8rem;margin-top:8px">Supports MP3, WAV, M4A, OGG, FLAC</div>
    </div>
    """, unsafe_allow_html=True)