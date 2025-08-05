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

  const runQAOA = async () => {
    setLoading(true);
    setImage1(null);
    setImage2(null);
    setIterations([]);
    setOptimizedParams([]);
    setBestBitstring("");
    setConsoleOutput("");

    try {
      const response = await fetch(`https://qaoa.onrender.com/api/qaoa`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: "{}",
      });
      const data = await response.json();
      setImage1(`data:image/png;base64,${data.graph}`);
      setImage2(`data:image/png;base64,${data.hist}`);
      setIterations(data.iterations);
      setOptimizedParams(data.optimized_params);
      setBestBitstring(data.most_freq_bitstring);
      setConsoleOutput(data.console_output);
    } catch (err) {
      console.error("Error running QAOA:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchMarkdown = async () => {
      try {
        const response = await fetch("/README.md");
        if (!response.ok) {
          throw new Error("Failed to fetch markdown content");
        }
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
          rehypePlugins={[rehypeKatex, rehypeHighlight]}>
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
          <img
            src="/images/qubitBR.png"
            alt="Qubit"
            className="qubit-img"
          />
        </div>
      )}
        {image1 && (
        <img
          src={image1}
          alt="Graph Partition"
          className="graph"
        />
      )}
      {image2 && (
        <img
          src={image2}
          alt="Bitstring Histogram"
          className="histogram"
        />
      )}
      {(consoleOutput || bestBitstring || iterations.length > 0 || optimizedParams.length > 0) && (
        
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
          <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(optimizedParams, null, 2)}</pre>
        </div>
      )}
      </div>
      )}
      <div className="markdown">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}>
          {markdownContent}
        </ReactMarkdown>
        </div>
    </div>
  );
}
