import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from voiceagent.agent.flow import DeliveryFlow
from voiceagent.agent.tools import CallSession

from conftest import NOW


@pytest.fixture
def clock():
    return {"now": NOW}


@pytest.fixture
def flow(world, repo, clock):
    return DeliveryFlow(CallSession(repo, 1, clock=lambda: clock["now"]))


def call(node_or_list, name, **kwargs):
    functions = node_or_list["functions"] if isinstance(node_or_list, dict) else node_or_list
    fn = next(f for f in functions if f.__name__ == name)
    return asyncio.run(fn(SimpleNamespace(state={}), **kwargs))


def names(node):
    return {f.__name__ for f in node["functions"]}


def test_role_carries_facts_and_rules(flow):
    assert "CALL FACTS" in flow.role
    assert "- 2 x Amul milk [MILK1]: ready" in flow.role
    assert "one or two short spoken sentences" in flow.role


def test_greet_node_speaks_fixed_greeting_then_waits(flow):
    node = flow.greet_node()
    assert node["name"] == "greet"
    assert node["respond_immediately"] is False
    assert node["pre_actions"][0]["type"] == "tts_say"
    assert "Priya" in node["pre_actions"][0]["text"]
    assert names(node) == {"confirm_identity", "wrong_person"}
    assert node["role_message"] == flow.role


def test_global_functions(flow):
    assert {f.__name__ for f in flow.global_functions} == {"cancel_order", "callback_later", "add_item"}


def test_confirm_identity_moves_to_order(flow):
    result, node = call(flow.greet_node(), "confirm_identity")
    assert node["name"] == "present_order"


def test_wrong_person_closes(flow):
    result, node = call(flow.greet_node(), "wrong_person")
    assert node["name"] == "close" and flow.session.outcome == "wrong_person"
    assert node["post_actions"] == [{"type": "end_conversation"}]


def test_items_confirmed_blocked_by_shortage(flow):
    result, node = call(flow.order_node(), "items_confirmed")
    assert result == {"status": "shortage_unresolved", "skus": ["BREAD1"]} and node is None


def test_resolving_last_shortage_moves_to_schedule(flow):
    result, node = call(flow.order_node(), "resolve_shortage", sku="BREAD1", choice="substitute",
                        substitute_sku="BREAD2")
    assert result["status"] == "done" and node["name"] == "schedule"


def test_schedule_then_details_then_confirm(flow):
    schedule = flow.schedule_node()
    result, node = call(schedule, "slot_agreed")
    assert result == {"status": "no_slot_held"} and node is None
    result, node = call(schedule, "check_slot", preferred_time="tomorrow after 5")
    assert result["status"] == "held" and node is None
    result, details = call(schedule, "slot_agreed")
    assert details["name"] == "details"
    assert "12 MG Road, Bengaluru" in details["task_messages"][0]["content"]
    result, confirm = call(details, "details_done")
    assert confirm["name"] == "confirm"
    assert "tomorrow, 5 to 6 PM" in confirm["task_messages"][0]["content"]


def test_confirm_booking_closes_call(flow, repo):
    call(flow.schedule_node(), "check_slot", preferred_time="tomorrow after 5")
    result, node = call(flow.confirm_node(), "confirm_booking")
    assert result["status"] == "booked" and node["name"] == "close"
    assert repo.get_order(1).status == "scheduled"


def test_confirm_after_hold_expiry_returns_to_schedule(flow, clock):
    call(flow.schedule_node(), "check_slot", preferred_time="tomorrow after 5")
    clock["now"] = NOW + timedelta(minutes=6)
    result, node = call(flow.confirm_node(), "confirm_booking")
    assert result["status"] == "failed" and node["name"] == "schedule"


def test_global_cancel_and_callback_close(flow):
    result, node = call(flow.global_functions, "callback_later", when="tomorrow morning")
    assert result["callback_at"] == "tomorrow 8 AM" and node["name"] == "close"
    result, node = call(flow.global_functions, "cancel_order")
    assert node["name"] == "close"


def test_global_add_item_stays_in_node(flow):
    result, node = call(flow.global_functions, "add_item", product_name="nandini milk", quantity=1)
    assert result["status"] == "added" and node is None


def test_order_node_accepts_an_early_delivery_time(flow):
    node = flow.order_node()
    assert "check_slot" in names(node)
    result, next_node = call(node, "check_slot", preferred_time="tomorrow after 5")
    assert result["status"] == "held" and next_node is None


def test_callback_tool_says_not_for_delivery_times(flow):
    callback = next(f for f in flow.global_functions if f.__name__ == "callback_later")
    assert "not for delivery times" in callback.__doc__


def test_tools_are_logged(flow):
    from loguru import logger
    lines = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        call(flow.schedule_node(), "check_slot", preferred_time="tomorrow after 5")
    finally:
        logger.remove(sink)
    assert any("check_slot" in line and "tomorrow after 5" in line and "held" in line for line in lines)
