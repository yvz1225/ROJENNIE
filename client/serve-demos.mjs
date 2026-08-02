import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const port = Number(process.env.PORT || 5174);

const routes = new Map([
  ["/flow", "flow.html"],
  ["/flow.html", "flow.html"],
  ["/agent1", "demo/agent1-case-builder-live.html"],
  ["/agent1.html", "demo/agent1-case-builder-live.html"],
  ["/agent2", "demo/agent2-hybrid-retriever-live.html"],
  ["/agent2.html", "demo/agent2-hybrid-retriever-live.html"],
  ["/agent3", "demo/agent3-logic-verification-live.html"],
  ["/agent3.html", "demo/agent3-logic-verification-live.html"],
  ["/agent4", "demo/agent4-response-composer-live.html"],
  ["/agent4.html", "demo/agent4-response-composer-live.html"],
  ["/demo-agent1", "demo/agent1-case-builder.html"],
  ["/demo-agent2", "demo/agent2-hybrid-retriever.html"],
  ["/demo-agent3", "demo/agent3-logic-verification.html"],
  ["/demo-agent4", "demo/agent4-response-composer.html"],
  ["/demo-flow", "demo/kb-key-buddy-mobile-flow.html"],
]);

const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
};

function safePath(pathname) {
  let relativePath = routes.get(pathname) || pathname.replace(/^\/+/, "");

  if (relativePath === "") return null;
  if (relativePath.startsWith("demo/assets/")) {
    relativePath = relativePath.replace(/^demo\/assets\//, "assets/");
  }
  if (relativePath === "demo/agent-api.js") {
    relativePath = "agent-api.js";
  }

  const resolved = resolve(root, normalize(relativePath));
  if (resolved !== root.slice(0, -1) && !resolved.startsWith(root)) return null;
  return resolved;
}

function sendIndex(res) {
  res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  res.end(`<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>KB Key Buddy Local Demos</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; background: #fdf9ef; color: #1a1512; }
      .page { width: min(880px, calc(100vw - 40px)); padding: 8px 4px; }
      .meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0 0 14px; font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase; color: #857a6c; }
      .meta .divider { color: #cbbfa9; }
      .meta .team-name { color: #8a5710; font-weight: 700; }
      h1 { margin: 0 0 10px; font-size: 30px; font-weight: 800; letter-spacing: -0.01em; }
      h1 .brand-kb { color: #1a1512; }
      h1 .brand-key { color: #b8791c; }
      p.desc { margin: 0 0 30px; color: #6b6256; font-size: 15px; }
      .card { position: relative; display: flex; flex-direction: column; gap: 4px; padding: 18px 20px; border-radius: 14px; background: #ffffff; box-shadow: 0 8px 22px rgba(74, 54, 20, 0.12); color: #3b2b00; font-weight: 800; text-decoration: none; transition: transform 180ms ease, box-shadow 180ms ease; }
      .card:hover { transform: translateY(-4px); box-shadow: 0 16px 32px rgba(74, 54, 20, 0.20); }
      .card-badge { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #b8791c; font-weight: 700; }
      .card-label { font-size: 16px; }
      .flow-section { margin: 0 0 28px; }
      .card-flow { border-left: 4px solid #d4941a; }
      .agent-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; align-items: stretch; }
      .agent-card { border-left: 5px solid transparent; }
      .agent-sub { font-size: 12.5px; font-weight: 600; color: #948a7c; }
      .agent-1 { border-left-color: #e4572e; }
      .agent-1:hover { box-shadow: 0 16px 32px rgba(228, 87, 46, 0.32); }
      .agent-1 .agent-sub { color: #c7431d; }
      .agent-2 { border-left-color: #e8b93e; }
      .agent-2:hover { box-shadow: 0 16px 32px rgba(232, 185, 62, 0.32); }
      .agent-2 .agent-sub { color: #8a6408; }
      .agent-3 { border-left-color: #3f9c6b; }
      .agent-3:hover { box-shadow: 0 16px 32px rgba(63, 156, 107, 0.32); }
      .agent-3 .agent-sub { color: #2c7a52; }
      .agent-4 { border-left-color: #3e7cb1; }
      .agent-4:hover { box-shadow: 0 16px 32px rgba(62, 124, 177, 0.32); }
      .agent-4 .agent-sub { color: #2e6090; }
      @media (max-width: 480px) {
        .agent-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <div class="meta">
        <span class="event-name">제8회 Future Finance AI Challenge</span>
        <span class="divider">·</span>
        <span class="team-name">Team ROJENNIE</span>
      </div>
      <h1><span class="brand-kb">KB</span> <span class="brand-key">Key Buddy</span> Local Demos</h1>
      <p class="desc">서버가 켜져 있는 동안 아래 링크로 Agent 1-4와 유저 플로우를 볼 수 있습니다.</p>
      <div class="flow-section">
        <a href="/flow" class="card card-flow">
          <span class="card-badge">전체 흐름</span>
          <span class="card-label">User Flow</span>
        </a>
      </div>
      <div class="agent-grid">
        <a href="/agent1" class="card agent-card agent-1">
          <span class="card-label">Agent 1</span>
          <span class="agent-sub">Case Builder</span>
        </a>
        <a href="/agent2" class="card agent-card agent-2">
          <span class="card-label">Agent 2</span>
          <span class="agent-sub">Hybrid Retriever</span>
        </a>
        <a href="/agent3" class="card agent-card agent-3">
          <span class="card-label">Agent 3</span>
          <span class="agent-sub">Logic Verification</span>
        </a>
        <a href="/agent4" class="card agent-card agent-4">
          <span class="card-label">Agent 4</span>
          <span class="agent-sub">Response Composer</span>
        </a>
      </div>
    </div>
  </body>
</html>`);
}

function serveFile(filePath, res) {
  if (!filePath || !existsSync(filePath) || !statSync(filePath).isFile()) {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
    return;
  }

  res.writeHead(200, {
    "content-type": mime[extname(filePath).toLowerCase()] || "application/octet-stream",
    "cache-control": "no-store",
  });
  createReadStream(filePath).pipe(res);
}

createServer((req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const pathname = decodeURIComponent(url.pathname);

  if (pathname === "/" || pathname === "/index") {
    sendIndex(res);
    return;
  }

  serveFile(safePath(pathname), res);
}).listen(port, "127.0.0.1", () => {
  console.log(`KB Key Buddy demos: http://127.0.0.1:${port}/`);
});
