import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_BASE_URL || "";

function errorDetail(payload, fallback) {
  if (!payload?.detail) return fallback;
  if (typeof payload.detail === "string") return payload.detail;
  return fallback;
}

async function readPayload(response, fallback) {
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      if (!response.ok) throw new Error(text.slice(0, 240) || fallback);
      throw new Error(fallback);
    }
  }
  if (!response.ok) throw new Error(errorDetail(payload, fallback));
  return payload;
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [brief, setBrief] = useState("");
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refreshDocs() {
    const response = await fetch(`${API}/api/documents`);
    const payload = await readPayload(response, "Could not load the document library.");
    setDocuments(payload.documents || []);
  }

  useEffect(() => {
    refreshDocs().catch((err) => setError(err.message));
  }, []);

  async function loadSample() {
    setError("");
    try {
      const response = await fetch(`${API}/api/sample`);
      const payload = await readPayload(response, "Could not load the sample brief.");
      setBrief(payload.brief);
    } catch (err) {
      setError(err.message);
    }
  }

  async function seedPolicy() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/seed`, { method: "POST" });
      await readPayload(response, "Seed failed");
      await refreshDocs();
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Could not add the sample policy. Wait a few seconds and try Seed sample again."
          : err.message,
      );
    } finally {
      setBusy(false);
    }
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API}/api/documents`, { method: "POST", body });
      await readPayload(response, "Upload failed");
      setFileName(file.name);
      await refreshDocs();
    } catch (err) {
      setFileName("");
      setError(err.message === "Failed to fetch" ? "Upload failed. The API could not index that file." : err.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function investigate() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brief }),
      });
      const payload = await readPayload(response, "Investigate failed");
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <section className="hero">
        <h1>Doc Investigator</h1>
        <p className="muted">
          Upload a policy. Paste a brief. The agent searches the library, reads passages, and
          keeps only claims that quote the source.
        </p>
      </section>

      {error ? <p className="error">{error}</p> : null}

      <div className="layout">
        <aside className="card">
          <h2>Library</h2>
          <label className="file-btn">
            Choose PDF or .txt
            <input type="file" accept=".pdf,.txt" onChange={upload} disabled={busy} />
          </label>
          {fileName ? <p className="muted">Uploaded {fileName}</p> : null}
          <div className="row">
            <a className="button secondary" href={`${API}/api/sample-file`}>
              Download sample
            </a>
            <button className="secondary" type="button" onClick={seedPolicy} disabled={busy}>
              Seed sample
            </button>
          </div>
          {documents.length === 0 ? (
            <p className="muted">No documents yet. Seed or upload a policy.</p>
          ) : (
            documents.map((doc) => (
              <div className="doc" key={doc.id}>
                {doc.filename}
                <div className="muted">
                  {doc.chunk_count} passages · {doc.page_count} pages
                </div>
              </div>
            ))
          )}
        </aside>

        <div>
          <section className="card">
            <h2>Investigation brief</h2>
            <textarea
              value={brief}
              onChange={(event) => setBrief(event.target.value)}
              placeholder="What should the agent check in these documents?"
            />
            <div className="row">
              <button className="secondary" type="button" onClick={loadSample} disabled={busy}>
                Load sample brief
              </button>
              <button
                type="button"
                onClick={investigate}
                disabled={busy || brief.trim().length < 12 || documents.length === 0}
              >
                {busy ? "Investigating…" : "Run agent"}
              </button>
            </div>
          </section>

          {result ? (
            <section className="card">
              <h2>Findings</h2>
              <p>{result.summary || "No summary."}</p>
              <p className="muted">{result.step_count} tool calls</p>
              {(result.findings || []).length === 0 ? (
                <p className="muted">No grounded findings. The agent refused unsupported claims.</p>
              ) : (
                result.findings.map((item, index) => (
                  <div className="finding" key={`${item.chunk_id}-${index}`}>
                    <strong>{item.claim}</strong>
                    <blockquote>{item.quote}</blockquote>
                    <p className="muted">
                      {item.document} {item.chunk_id ? `· ${item.chunk_id}` : ""}
                    </p>
                  </div>
                ))
              )}
              <h3>Tool trace</h3>
              <pre>{(result.trace || []).join("\n") || "No tools were logged."}</pre>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
