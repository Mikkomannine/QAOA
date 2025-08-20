import { useState, useEffect } from "react";
import "./css/QAOA.css";
import ReactMarkdown from "react-markdown";
import "highlight.js/styles/github.css";
import rehypeHighlight from "rehype-highlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

export default function QAOASimulator() {
  const [consoleOutput, setConsoleOutput] = useState("");
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [iterations, setIterations] = useState([]);
  const [optimizedParams, setOptimizedParams] = useState([]);
  const [bestBitstring, setBestBitstring] = useState("");
  const [loading, setLoading] = useState(false);
  const [markdownContent, setMarkdownContent] = useState("");
  const [intro, setIntro] = useState("");

  const API_BASE = process.env.REACT_APP_API_BASE || "https://qaoa.onrender.com";

  const runQAOA = async () => {
    setLoading(true);
    setImage1(null);
    setImage2(null);
    setIterations([]);
    setOptimizedParams([]);
    setBestBitstring("");
    setConsoleOutput("");

    // Abort if the network is stuck (helps mobile)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    try {
      const response = await fetch(`${API_BASE}/api/qaoa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`API ${response.status}: ${text || "Request failed"}`);
      }

      const data = await response.json();

      //URL-based images returned by the backend
      if (data.graph_url && data.hist_url) {
        const graphUrl =
          data.graph_url.startsWith("http")
            ? data.graph_url
            : `${API_BASE}${data.graph_url}`;
        const histUrl =
          data.hist_url.startsWith("http")
            ? data.hist_url
            : `${API_BASE}${data.hist_url}`;

        // Setting key forces React to replace <img> nodes between runs
        setImage1(graphUrl);
        setImage2(histUrl);
      } else if (data.graph && data.hist) {

        // Backward-compat (if server still returns base64)
        setImage1(`data:image/png;base64,${data.graph}`);
        setImage2(`data:image/png;base64,${data.hist}`);
      } else {
        throw new Error("API did not return image URLs or base64 data");
      }

      setIterations(Array.isArray(data.iterations) ? data.iterations : []);
      setOptimizedParams(
        Array.isArray(data.optimized_params) ? data.optimized_params : []
      );
      setBestBitstring(data.most_freq_bitstring || "");
      setConsoleOutput(data.console_output || "Console Output:");
    } catch (err) {
      console.error("Error running QAOA:", err);
      setConsoleOutput(
        `Error: ${err?.message || "Failed to run QAOA. Please try again."}`
      );
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchMarkdown = async () => {
      try {
        const response = await fetch("/README.md");
        if (!response.ok) throw new Error("Failed to fetch markdown content");
        const text = await response.text();
        setMarkdownContent(text);
      } catch (error) {
        console.error("Error fetching markdown:", error);
      }
    };
    fetchMarkdown();
  }, []);

  useEffect(() => {
    fetch("/intro.md")
      .then((res) => res.text())
      .then(setIntro)
      .catch((err) => console.error("Failed to load markdown:", err));
  }, []);

  return (
    <div className="container">
      <h1 className="header">QAOA for MaxCut</h1>

      <div className="markdown">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {intro}
        </ReactMarkdown>
      </div>

      <button
        onClick={runQAOA}
        disabled={loading}
        className={`button ${loading ? "pulsing" : ""}`}
      >
        {loading ? "Running..." : "Run QAOA"}
      </button>

      {loading && (
        <div className="loading">
          <img src="/images/qubitBR.png" alt="Qubit" className="qubit-img" />
        </div>
      )}

      {image1 && (
        <img
          key={image1}
          src={image1}
          alt="Graph Partition"
          className="graph"
          loading="lazy"
          decoding="async"
        />
      )}

      {image2 && (
        <img
          key={image2}
          src={image2}
          alt="Bitstring Histogram"
          className="histogram"
          loading="lazy"
          decoding="async"
        />
      )}

      {(consoleOutput ||
        bestBitstring ||
        iterations.length > 0 ||
        optimizedParams.length > 0) && (
        <div className="logs">
          <div className="wrapper">
            <h2 className="text-2xl font-bold">{consoleOutput}</h2>

            {bestBitstring && (
              <div className="bitstring">
                <h3>Most Frequently Measured Bitstring:</h3>
                <div className="text-xl font-mono">{bestBitstring}</div>
              </div>
            )}

            {iterations.length > 0 && (
              <div className="iterations">
                <h3>iterations:</h3>
                <pre>{iterations.join("\n")}</pre>
              </div>
            )}
          </div>

          {optimizedParams.length > 0 && (
            <div className="params">
              <h3>Optimized Parameters for Each Layer:</h3>
              <pre className="whitespace-pre-wrap text-sm">
                {JSON.stringify(optimizedParams, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      <div className="markdown">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {markdownContent}
        </ReactMarkdown>
      </div>
    </div>
  );
}
