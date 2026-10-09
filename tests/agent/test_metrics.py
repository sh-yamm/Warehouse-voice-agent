import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import BotStartedSpeakingFrame, MetricsFrame, VADUserStoppedSpeakingFrame
from pipecat.metrics.metrics import TTFBMetricsData

from voiceagent.agent.metrics import BargeInObserver, BargeInTimer, CallRecorder, ResponseLatencyObserver

from conftest import NOW


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_barge_in_timer_measures_interruptions_only():
    clock = FakeClock()
    timer = BargeInTimer(now=clock)
    timer.user_started()  # bot silent: not an interruption
    timer.bot_started()
    assert timer.bot_stopped() is None
    timer.bot_started()
    clock.t = 1.0
    timer.user_started()
    clock.t = 1.15
    assert round(timer.bot_stopped(), 1) == 150.0


def push(observer, frame):
    asyncio.run(observer.on_push_frame(SimpleNamespace(frame=frame)))


def test_response_latency_adds_vad_wait_and_collects_ttfb():
    clock, seen = FakeClock(), []

    async def on_response(total_ms, breakdown):
        seen.append((total_ms, breakdown))

    observer = ResponseLatencyObserver(on_response=on_response, vad_stop_secs=0.2, now=clock)
    stop = VADUserStoppedSpeakingFrame()
    push(observer, stop)
    clock.t = 0.1
    push(observer, stop)  # the same frame seen again on the next hop must not restart the clock
    push(observer, MetricsFrame(data=[TTFBMetricsData(processor="GreedyWhisperSTTService#0", value=0.15)]))
    clock.t = 0.5
    started = BotStartedSpeakingFrame()
    push(observer, started)
    push(observer, started)
    assert len(seen) == 1
    total, breakdown = seen[0]
    assert total == 700.0
    assert breakdown["contributions"] == [["endpointing_wait", "silence wait (VAD)", 200.0],
                                          ["after_silence", "turn detection + STT + intent + first audio", 500.0]]
    assert breakdown["ttfb"] == [["GreedyWhisperSTTService#0", 150.0]]


def test_first_speech_after_connect_is_the_greeting():
    clock, greetings = FakeClock(), []

    async def on_greeting(ms):
        greetings.append(ms)

    async def ignore(*args):
        pass

    observer = ResponseLatencyObserver(on_response=ignore, on_greeting=on_greeting, now=clock)
    observer.mark_connected()
    clock.t = 0.04
    push(observer, BotStartedSpeakingFrame())
    push(observer, BotStartedSpeakingFrame())
    assert [round(g) for g in greetings] == [40]


def test_record_response_and_greeting(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.record_response(700.0, {"contributions": [["endpointing_wait", "silence wait (VAD)", 200.0]]})
    recorder.record_greeting(40.0)
    recorder.record_barge_in(142.0)
    metrics = repo.call_metrics(recorder.call_id)
    assert [m["kind"] for m in metrics] == ["response", "greeting", "barge_in"]
    assert metrics[0]["total_ms"] == 700.0 and metrics[0]["breakdown"]["contributions"][0][2] == 200.0


def test_finish_keeps_spoken_turns_only(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.finish("scheduled", [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "Hi, is this Priya?"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "{}"},
    ])
    call = repo.get_call(recorder.call_id)
    assert call["outcome"] == "scheduled"
    assert call["transcript"] == [{"role": "assistant", "content": "Hi, is this Priya?"},
                                  {"role": "user", "content": "Yes."}]


def test_finish_without_outcome_is_abandoned(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.finish(None, [])
    assert repo.get_call(recorder.call_id)["outcome"] == "abandoned"


def test_observers(world, repo):
    observers = CallRecorder(repo, 1, clock=lambda: NOW).observers()
    assert [type(o) for o in observers] == [ResponseLatencyObserver, BargeInObserver]


def test_finish_is_idempotent(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.finish("scheduled", [{"role": "user", "content": "Yes."}])
    recorder.finish(None, [])
    call = repo.get_call(recorder.call_id)
    assert call["outcome"] == "scheduled" and len(call["transcript"]) == 1
