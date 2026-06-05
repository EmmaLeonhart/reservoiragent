// Renderer: connect to the Python reservoir server, stream the agent's output into the
// left pane, send the user's injections + the live reservoir-gain knob.
const params = new URLSearchParams(location.search);
const PORT = params.get('port') || '8765';
const URL = `ws://127.0.0.1:${PORT}`;

const $ = (id) => document.getElementById(id);
const agentStream = $('agentStream');
const youHistory = $('youHistory');
const userInput = $('userInput');
const scroll = (el) => { el.parentElement.scrollTop = el.parentElement.scrollHeight; };

let ws = null;
let lastKind = 'idle';

function setLive(on, text) {
  $('liveDot').classList.toggle('live', on);
  $('connText').textContent = on ? 'live' : 'offline';
  if (text) $('statusText').textContent = text;
}

function connect() {
  ws = new WebSocket(URL);
  ws.onopen = () => setLive(true, 'connected');
  ws.onclose = () => { setLive(false, 'server offline — retrying…'); setTimeout(connect, 1000); };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
  ws.onmessage = (e) => {
    let m;
    try { m = JSON.parse(e.data); } catch (_) { return; }
    if (m.type === 'status') {
      $('statusText').textContent = m.text;
    } else if (m.type === 'telemetry') {
      lastKind = m.kind;
      $('tick').textContent = m.tick;
      $('kind').textContent = m.kind;
      $('gate').textContent = m.emit ? 'open' : 'silent';
      $('entropy').textContent = (m.entropy ?? 0).toFixed(3);
      $('cos').textContent = (m.state_cos ?? 0).toFixed(3);
      // when a new pass begins, break the stream so bursts read as separate thoughts
      if (agentStream.lastChild && agentStream.textContent &&
          !agentStream.textContent.endsWith('\n\n')) {
        agentStream.appendChild(document.createTextNode('\n'));
      }
    } else if (m.type === 'token') {
      agentStream.appendChild(document.createTextNode(m.text));
      scroll(agentStream);
    }
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    const div = document.createElement('div');
    div.className = 'msg you';
    div.textContent = text;
    youHistory.appendChild(div);
    scroll(youHistory);
    send({ type: 'inject', text });
    userInput.value = '';
  }
});

const scaleEl = $('scale');
scaleEl.addEventListener('input', () => {
  const v = parseFloat(scaleEl.value);
  $('scaleVal').textContent = v.toFixed(2);
  send({ type: 'set_scale', value: v });
});

connect();
