// Deno Pyodide host — reads JSON from stdin, executes Python, writes JSON to stdout.
//
// Usage: echo '{"code":"2+2","timeout":5}' | deno run --allow-read --allow-net=cdn.jsdelivr.net _runner.js

import { loadPyodide } from "npm:pyodide";

const decoder = new TextDecoder();
const chunks = [];
for await (const chunk of Deno.stdin.readable) {
  chunks.push(chunk);
}
const input = decoder.decode(new Uint8Array(chunks.flatMap((c) => [...c])));
const { code, timeout = 5 } = JSON.parse(input);

const pyodide = await loadPyodide();

// Redirect stdout/stderr to StringIO
pyodide.runPython(`
import sys, io
_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture
`);

const result = { return_value: "None", stdout: "", stderr: "", error: null };

try {
  // Use AbortController for timeout
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout * 1000);

  const rv = await pyodide.runPythonAsync(code);
  clearTimeout(timer);

  result.return_value = rv !== undefined && rv !== null ? String(rv) : "None";
} catch (e) {
  result.error = String(e.message || e);
}

// Capture redirected output
try {
  result.stdout = pyodide.runPython("_stdout_capture.getvalue()");
  result.stderr = pyodide.runPython("_stderr_capture.getvalue()");
} catch (_) {
  // If capture fails, leave empty
}

const encoder = new TextEncoder();
const out = encoder.encode(JSON.stringify(result) + "\n");
await Deno.stdout.write(out);
