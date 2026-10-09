"""Delivery caller agent.

    llama-server:  bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf   (serves the intent reader only)
    agent:         python -m voiceagent.agent.bot -t webrtc --order-id 1
    browser:       http://localhost:7860/client  (click Connect; allow the microphone)
    dashboard:     python -m voiceagent.agent.dashboard  ->  http://localhost:7861
"""
from __future__ import annotations

import argparse

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from voiceagent.agent.dialog import ACKS, DialogManager
from voiceagent.agent.intent import IntentClassifier
from voiceagent.agent.metrics import CallRecorder
from voiceagent.agent.processor import DialogProcessor
from voiceagent.agent.services import GreedyWhisperSTTService, KokoroTorchTTSService, make_turn_analyzer
from voiceagent.agent.tools import CallSession
from voiceagent.db.repository import Repository

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warehouse delivery caller agent")
    parser.add_argument("--order-id", type=int, default=1, help="order to call the customer about")
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1")
    return parser


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    args = runner_args.cli_args
    repo = Repository(args.db)
    session = CallSession(repo, args.order_id)
    manager = DialogManager(session)
    classifier = IntentClassifier(args.llm_url)
    recorder = CallRecorder(repo, args.order_id)

    stt = GreedyWhisperSTTService(device="cuda", compute_type="float16",
                                  settings=GreedyWhisperSTTService.Settings(model="distil-small.en"))
    tts = KokoroTorchTTSService(voice="af_heart", cache_phrases=(*ACKS, manager.greeting()))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=make_turn_analyzer())],
            ),
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        DialogProcessor(manager, classifier),
        tts,
        transport.output(),
    ])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[*recorder.observers(), TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Calling about order {args.order_id} ({manager.customer_name})")
        await worker.queue_frames([TTSSpeakFrame(manager.greeting())])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        recorder.finish(session.outcome, manager.transcript)
        logger.info(f"Call {recorder.call_id} ended: {session.outcome or 'abandoned'}")
        await runner.cancel()

    try:
        await runner.run()
    finally:
        # An EndTaskFrame (call finished) stops the pipeline without a client disconnect; close the row either way.
        recorder.finish(session.outcome, manager.transcript)
        await classifier.aclose()
        repo.close()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import main

    # The runner prints a Unicode banner; Windows consoles default to cp1252 and would crash on it.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main(build_parser())
