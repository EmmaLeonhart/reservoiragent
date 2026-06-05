// Electron main process: spawn the Python reservoir server, open the two-pane window.
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const PORT = process.env.RESERVOIR_PORT || '8765';
let py = null;

function startServer() {
  const python = process.env.RESERVOIR_PYTHON || 'python';
  const repoRoot = path.join(__dirname, '..', '..');
  const serverPath = path.join(__dirname, '..', 'server', 'server.py');
  py = spawn(python, [serverPath], {
    cwd: repoRoot,
    env: { ...process.env, RESERVOIR_PORT: PORT, PYTHONUTF8: '1' },
  });
  py.stdout.on('data', (d) => process.stdout.write('[py] ' + d));
  py.stderr.on('data', (d) => process.stderr.write('[py] ' + d));
  py.on('exit', (code) => console.log('[py] server exited with code', code));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1180,
    height: 760,
    backgroundColor: '#14110e',
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  win.loadFile('index.html', { query: { port: PORT } });
}

app.whenReady().then(() => {
  startServer();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

function killServer() {
  if (py) { try { py.kill(); } catch (_) {} py = null; }
}

app.on('window-all-closed', () => { killServer(); app.quit(); });
app.on('quit', killServer);
process.on('exit', killServer);
