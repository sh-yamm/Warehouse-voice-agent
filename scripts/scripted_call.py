"""Dial the agent over WebRTC like the browser does, speak scripted lines, record and time the agent's speech.

    python scripts/scripted_call.py --url http://localhost:7860/api/offer --out call.wav --duration 120 \
        "20:Yeah, that's me." "34:Tomorrow evening works for me."
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import time

import av
import httpx
import numpy as np
import soundfile as sf
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from scipy.signal import resample_poly

RATE = 48000
FRAME = 960  # 20 ms at 48 kHz


class ScriptedSpeaker(MediaStreamTrack):
    kind = "audio"

    def __init__(self, cues: list[tuple[float, np.ndarray, str]]):
        super().__init__()
        self.cues, self.t0, self.pts, self.queue = cues, None, 0, np.zeros(0, np.int16)

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.time()
        await asyncio.sleep(max(0.0, self.t0 + self.pts / RATE - time.time()))
        now = time.time() - self.t0
        while self.cues and self.cues[0][0] <= now:
            _, pcm, text = self.cues.pop(0)
            self.queue = np.concatenate([self.queue, pcm])
            print(f"[{now:6.1f}s] customer: {text}", flush=True)
        chunk, self.queue = self.queue[:FRAME], self.queue[FRAME:]
        frame = av.AudioFrame.from_ndarray(np.pad(chunk, (0, FRAME - len(chunk))).reshape(1, -1), format="s16",
                                           layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = RATE, self.pts, fractions.Fraction(1, RATE)
        self.pts += FRAME
        return frame


def synthesize(lines: list[str]) -> list[np.ndarray]:
    from kokoro import KModel, KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M",
                         model=KModel(repo_id="hexgrad/Kokoro-82M").to("cpu").eval())
    out = []
    for line in lines:
        audio = np.concatenate([r.audio.numpy() for r in pipeline(line, voice="am_michael")])
        out.append((np.clip(resample_poly(audio, 2, 1), -1, 1) * 32767).astype(np.int16))  # 24 kHz -> 48 kHz
    return out


async def call(url: str, out: str, duration: float, cues: list[tuple[float, str]]) -> None:
    pcms = synthesize([text for _, text in cues])
    pc = RTCPeerConnection()
    pc.addTrack(ScriptedSpeaker([(t, pcm, text) for (t, text), pcm in zip(cues, pcms)]))
    received: list[tuple[float, np.ndarray, int]] = []
    started = time.time()

    @pc.on("track")
    def on_track(track):
        async def pull():
            while True:
                try:
                    frame = await track.recv()
                except Exception:
                    return
                received.append((time.time() - started, frame.to_ndarray().astype(np.float32).mean(axis=0) / 32768,
                                 frame.sample_rate))
        asyncio.ensure_future(pull())

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=120) as client:
        answer = (await client.post(url, json={"sdp": pc.localDescription.sdp,
                                               "type": pc.localDescription.type})).json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print(f"[{time.time() - started:6.1f}s] connected", flush=True)
    await asyncio.sleep(duration)
    await pc.close()
    if not received:
        print("no agent audio received")
        return
    sf.write(out, np.concatenate([r[1] for r in received]), received[0][2])
    spans, begin = [], None
    for t, audio, _ in received:
        loud = float(np.sqrt(np.mean(audio ** 2))) > 0.01
        if loud and begin is None:
            begin = t
        elif not loud and begin is not None:
            if t - begin > 0.3:
                spans.append(f"{begin:.1f}-{t:.1f}")
            begin = None
    print("agent speaking (s):", ", ".join(spans))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/api/offer")
    parser.add_argument("--out", default="call.wav")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("cues", nargs="+", help='"seconds:text" spoken by the customer')
    args = parser.parse_args()
    cues = sorted((float(c.split(":", 1)[0]), c.split(":", 1)[1]) for c in args.cues)
    asyncio.run(call(args.url, args.out, args.duration, cues))


if __name__ == "__main__":
    main()
