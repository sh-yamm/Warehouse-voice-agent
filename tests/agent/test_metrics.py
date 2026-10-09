from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown, LatencyContribution, LatencyOwnerKind, MeasuredFrom, TTFBBreakdownMetrics,
    UserBotLatencyObserver)

from voiceagent.agent.metrics import BargeInObserver, BargeInTimer, CallRecorder

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


def breakdown(measured_from=MeasuredFrom.USER_SILENCE):
    return LatencyBreakdown(
        contributions=[
            LatencyContribution(key="vad", label="endpointing wait", owner="config: VAD stop_secs",
                                owner_kind=LatencyOwnerKind.SETTING, start_time=0.0, duration_secs=0.2),
            LatencyContribution(key="llm", label="LLM inference", owner="OpenAILLMService#0",
                                owner_kind=LatencyOwnerKind.SERVICE, start_time=0.2, duration_secs=0.55),
        ],
        ttfb=[TTFBBreakdownMetrics(processor="OpenAILLMService#0", start_time=0.2, duration_secs=0.3)],
        measured_from=measured_from, total_secs=0.75, user_turn_secs=0.26)


def test_record_breakdown(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.record_breakdown(breakdown())
    recorder.record_breakdown(breakdown(MeasuredFrom.CLIENT_CONNECTED))
    recorder.record_barge_in(142.0)
    metrics = repo.call_metrics(recorder.call_id)
    assert [m["kind"] for m in metrics] == ["response", "greeting", "barge_in"]
    assert metrics[0]["total_ms"] == 750.0
    assert metrics[0]["breakdown"]["contributions"] == [["vad", "endpointing wait", 200.0],
                                                       ["llm", "LLM inference", 550.0]]
    assert metrics[0]["breakdown"]["ttfb"] == [["OpenAILLMService#0", 300.0]]
    assert metrics[0]["breakdown"]["user_turn_ms"] == 260.0


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
    assert [type(o) for o in observers] == [UserBotLatencyObserver, BargeInObserver]
