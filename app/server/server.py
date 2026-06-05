"""WebSocket server for the real-time reservoir agent.

Owns one :class:`AliveEngine` over a reservoir-injected chat model. A background thread
runs the continuous tick loop (the engine is a *process*, not a request handler) and
pushes events onto a queue; the asyncio side broadcasts those events to every connected
client and feeds client messages (user injections, the live reservoir-gain knob) into the
engine. The model + engine outlive client connections, so the reservoir state persists
across reconnects — the whole point.

Run standalone:  python app/server/server.py     (env: RESERVOIR_PORT, RESERVOIR_MODEL,
                                                   RESERVOIR_SCALE, RESERVOIR_DT)
Requires the project installed (``reservoir`` importable) + the ``models`` extra + a CUDA
or CPU torch. Named plainly: this serves the *untrained* substrate live — the agent's
behaviour is not yet a trained policy (see FINDINGS / the loss-function design).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time

import websockets

HOST = os.environ.get("RESERVOIR_HOST", "127.0.0.1")
PORT = int(os.environ.get("RESERVOIR_PORT", "8765"))
MODEL = os.environ.get("RESERVOIR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SCALE = float(os.environ.get("RESERVOIR_SCALE", "0.2"))
DT = float(os.environ.get("RESERVOIR_DT", "0.4"))

_clients: set = set()
_events: "queue.Queue[dict]" = queue.Queue()
_last_status = {"type": "status", "text": "starting…"}
_engine = None
_engine_lock = threading.Lock()


def _build_engine():
    """Load the model + reservoir and construct the engine. Heavy; runs in the loop thread."""
    global _engine
    from reservoir.kv_live import TorchReservoirPrefixInjectedLM
    from reservoir.alive import AliveEngine

    _events.put({"type": "status", "text": f"loading {MODEL} …"})
    lm = TorchReservoirPrefixInjectedLM(MODEL, n_reservoir=512, n_prefix=8,
                                        dtype="bfloat16", seed=0)
    lm.readout_scale = SCALE
    eng = AliveEngine(lm, burst_tokens=24)
    with _engine_lock:
        _engine = eng
    _events.put({"type": "status",
                 "text": f"ready · {MODEL} on {lm.device} · UNTRAINED substrate "
                         f"(behaviour not yet trained)"})


def _engine_loop():
    """Continuous tick loop: build the engine, then step forever, emitting events."""
    try:
        _build_engine()
    except Exception as e:  # surface load failures to the UI instead of dying silently
        _events.put({"type": "status", "text": f"error loading model: {e!r}"})
        return
    while True:
        with _engine_lock:
            eng = _engine
        for kind, payload in eng.step():
            if kind == "telemetry":
                _events.put({"type": "telemetry", **payload})
            else:
                _events.put({"type": "token", "text": payload})
        time.sleep(DT)


async def _broadcast_pump():
    """Drain the engine's event queue and fan each event out to all clients."""
    global _last_status
    loop = asyncio.get_event_loop()
    while True:
        ev = await loop.run_in_executor(None, _events.get)
        if ev.get("type") == "status":
            _last_status = ev
        msg = json.dumps(ev)
        for ws in list(_clients):
            try:
                await ws.send(msg)
            except Exception:
                _clients.discard(ws)


async def _handler(ws, *args):
    """One client: replay the latest status, then route its messages to the engine.

    (``*args`` absorbs the legacy ``path`` positional that older websockets versions pass.)
    """
    _clients.add(ws)
    try:
        await ws.send(json.dumps(_last_status))
        async for raw in ws:
            try:
                m = json.loads(raw)
            except (ValueError, TypeError):
                continue
            with _engine_lock:
                eng = _engine
            if eng is None:
                continue
            kind = m.get("type")
            if kind == "inject":
                eng.inject(m.get("text", ""))
            elif kind == "set_scale":
                eng.set_readout_scale(m.get("value", SCALE))
    finally:
        _clients.discard(ws)


async def _main():
    threading.Thread(target=_engine_loop, daemon=True).start()
    async with websockets.serve(_handler, HOST, PORT):
        print(f"reservoir server on ws://{HOST}:{PORT}", flush=True)
        await _broadcast_pump()


if __name__ == "__main__":
    asyncio.run(_main())
