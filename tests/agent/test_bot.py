from voiceagent.agent.bot import build_parser


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert (args.order_id, args.db, args.llm_url) == (1, "data/warehouse.db", "http://127.0.0.1:8080/v1")
    assert build_parser().parse_args(["--order-id", "42"]).order_id == 42
