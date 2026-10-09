"""Delivery caller agent — July-2025 stack (Pipecat 0.0.77, Smart Turn v2, Qwen3 intent reader).

    llama-server:  LLAMA_DIR=llama.cpp-b6050 bash scripts/llama_server.sh models/jul2025/llm/Qwen3-1.7B-Q4_K_M.gguf
    agent:         .venv-jul2025/Scripts/python -m voiceagent.agent.bot --order-id 1
    browser:       http://localhost:7860/client  (click Connect; allow the microphone)
    dashboard:     python -m voiceagent.agent.dashboard  ->  http://localhost:7861
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from voiceagent.agent.dialog import ACKS, DialogManager
from voiceagent.agent.intent import IntentClassifier
from voiceagent.agent.metrics import CallRecorder
from voiceagent.agent.processor import DialogProcessor
from voiceagent.agent.services import GreedyWhisperSTTService, KokoroTorchTTSService, make_turn_analyzer
from voiceagent.agent.tools import CallSession
from voiceagent.db.repository import Repository

VAD_STOP_SECS = 0.2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warehouse delivery caller agent", add_help=False)
    parser.add_argument("--order-id", type=int, default=1, help="order to call the customer about")
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1")
    return parser


# Pipecat 0.0.77's runner parses sys.argv itself and rejects unknown flags, so ours are split off first.
ARGS = build_parser().parse_known_args([])[0]


async def run_bot(transport: BaseTransport, args: argparse.Namespace):
    repo = Repository(args.db)
    session = CallSession(repo, args.order_id)
    manager = DialogManager(session)
    classifier = IntentClassifier(args.llm_url)
    recorder = CallRecorder(repo, args.order_id)

    stt = GreedyWhisperSTTService(model="distil-small.en", device="cuda", compute_type="float16")
    tts = KokoroTorchTTSService(voice="af_heart", cache_phrases=(*ACKS, manager.greeting()))
    user_aggregator = LLMUserContextAggregator(OpenAILLMContext())

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        DialogProcessor(manager, classifier),
        tts,
        transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[*recorder.observers(), TranscriptionLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Calling about order {args.order_id} ({manager.customer_name})")
        recorder.latency.mark_connected()
        await task.queue_frames([TTSSpeakFrame(manager.greeting())])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        recorder.finish(session.outcome, manager.transcript)
        logger.info(f"Call {recorder.call_id} ended: {session.outcome or 'abandoned'}")
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        # An EndTaskFrame (call finished) stops the pipeline without a client disconnect; close the row either way.
        recorder.finish(session.outcome, manager.transcript)
        await classifier.aclose()
        repo.close()


async def bot(runner_args: RunnerArguments):
    turn_analyzer = await make_turn_analyzer()
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
            turn_analyzer=turn_analyzer,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, ARGS)


if __name__ == "__main__":
    from pipecat.runner.run import main

    # The runner prints emoji; Windows consoles default to cp1252 and would crash on them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ARGS, rest = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    main()
