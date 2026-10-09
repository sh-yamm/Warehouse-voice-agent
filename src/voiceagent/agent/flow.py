from __future__ import annotations

from pipecat.flows import FlowManager, NodeConfig

from voiceagent.agent.tools import CallSession

AGENT_NAME = "Asha"
COMPANY = "QuickMart"

ROLE_TEMPLATE = """You are {agent}, calling customers on behalf of {company} to schedule delivery of their grocery order.
You are on a live phone call and everything you write is spoken aloud.
Rules:
- Reply in one or two short spoken sentences. No lists, markdown, symbols or emojis.
- Say times the way people speak them, for example "five to six PM".
- Only state items, quantities, prices, slots and dates that appear in the call facts below or in a function result. Never guess.
- Use the functions for every action: checking or booking slots, changing the order, cancelling, or arranging a callback.
- If the customer asks about something unrelated to this delivery, say you can only help with this delivery.

CALL FACTS
{facts}"""


class DeliveryFlow:
    """The call's conversation graph (spec section 3), built on a CallSession."""

    def __init__(self, session: CallSession):
        self.session = session
        ctx = session.context()
        self.customer_name = ctx.customer_name
        self.first_name = ctx.customer_name.split()[0]
        self.role = ROLE_TEMPLATE.format(agent=AGENT_NAME, company=COMPANY, facts=ctx.prompt_text)
        self.global_functions = self._global_functions()

    def greeting(self) -> str:
        return (f"Hi, this is {AGENT_NAME} calling from {COMPANY} about your grocery order. "
                f"Am I speaking with {self.first_name}?")

    def _node(self, name: str, task: str, functions: list, **extra) -> NodeConfig:
        return NodeConfig(name=name, role_message=self.role,
                          task_messages=[{"role": "developer", "content": task}], functions=functions, **extra)

    def close_node(self, instruction: str) -> NodeConfig:
        return self._node("close", instruction, [], post_actions=[{"type": "end_conversation"}])

    def _global_functions(self) -> list:
        session = self.session

        async def cancel_order(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """Cancel the whole order. Use only when the customer clearly asks to cancel."""
            return session.cancel_order(), self.close_node(
                "The order is cancelled. Confirm the cancellation and say goodbye in one sentence.")

        async def callback_later(flow_manager: FlowManager, when: str) -> tuple[dict, NodeConfig]:
            """Arrange a callback only when the customer is busy or asks to be called back; not for delivery times.

            Args:
                when (str): When to call back in the customer's words, for example "tomorrow morning". Use "" if they did not say.
            """
            return session.schedule_callback(when), self.close_node(
                "A callback is arranged. Tell the customer when you will call back and say goodbye.")

        async def add_item(flow_manager: FlowManager, product_name: str, quantity: int) -> tuple[dict, None]:
            """Add a product to the order when the customer asks for something extra.

            Args:
                product_name (str): The product the customer asked for, in their words.
                quantity (int): How many units to add.
            """
            return session.add_item(product_name, quantity), None

        return [cancel_order, callback_later, add_item]

    def greet_node(self) -> NodeConfig:
        session = self.session

        async def confirm_identity(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The person on the phone confirmed they are the customer."""
            return {"status": "confirmed"}, self.order_node()

        async def wrong_person(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The person on the phone is not the customer."""
            return session.wrong_person(), self.close_node(
                "You reached the wrong person. Apologize briefly, say you will try again later, and say goodbye.")

        task = (f'You already said: "{self.greeting()}" Wait for the answer. '
                f"If they confirm they are {self.customer_name}, call confirm_identity. "
                "If not, call wrong_person. If they cannot talk now, call callback_later.")
        return self._node("greet", task, [confirm_identity, wrong_person],
                          pre_actions=[{"type": "tts_say", "text": self.greeting()}], respond_immediately=False)

    def order_node(self) -> NodeConfig:
        session = self.session

        async def resolve_shortage(flow_manager: FlowManager, sku: str, choice: str,
                                   substitute_sku: str) -> tuple[dict, NodeConfig | None]:
            """Record how the customer wants a short item handled.

            Args:
                sku (str): SKU of the short item from the call facts, for example "BAK040".
                choice (str): One of "substitute", "partial" or "wait".
                substitute_sku (str): SKU of the substitute when choice is "substitute", otherwise "".
            """
            result = session.resolve_shortage(sku, choice, substitute_sku)
            done = result["status"] == "done" and not result["remaining_shortages"]
            return result, self.schedule_node() if done else None

        async def items_confirmed(flow_manager: FlowManager) -> tuple[dict, NodeConfig | None]:
            """The customer is happy with the items and no shortage is left unresolved."""
            remaining = [s.sku for s in session.inventory.shortages(session.order_id)]
            if remaining:
                return {"status": "shortage_unresolved", "skus": remaining}, None
            return {"status": "ok"}, self.schedule_node()

        task = ("Tell the customer in one or two sentences what their order contains. "
                "If an item is SHORT, explain it and offer the options: a listed substitute, sending what is "
                "available, or waiting for the restock. Call resolve_shortage only after the customer says which option "
                "they want; never choose for them. "
                "If nothing is short, call items_confirmed once they are happy. "
                "If they already mention a delivery time, call check_slot with their words.")
        return self._node("present_order", task, [resolve_shortage, items_confirmed, self._check_slot_function()])

    def _check_slot_function(self):
        session = self.session

        async def check_slot(flow_manager: FlowManager, preferred_time: str) -> tuple[dict, None]:
            """Check the customer's preferred delivery time and hold the slot if it is free.

            Args:
                preferred_time (str): The customer's own words, for example "tomorrow after 5 PM".
            """
            return session.check_slot(preferred_time), None

        return check_slot

    def schedule_node(self) -> NodeConfig:
        session = self.session
        check_slot = self._check_slot_function()

        async def choose_slot(flow_manager: FlowManager, slot_id: int) -> tuple[dict, None]:
            """Hold an alternative slot the customer picked.

            Args:
                slot_id (int): The slot id from a check_slot result or the call facts.
            """
            return session.choose_slot(slot_id), None

        async def slot_agreed(flow_manager: FlowManager) -> tuple[dict, NodeConfig | None]:
            """The customer agreed to the slot that is currently held."""
            if session.held_slot_id is None:
                return {"status": "no_slot_held"}, None
            return {"status": "ok"}, self.details_node()

        task = ("Ask when they would like the delivery and call check_slot with their exact words. "
                "If a slot is held, ask them to confirm that time. If it is full, offer the alternatives and call "
                "choose_slot for the one they pick. When they agree to the held slot, call slot_agreed.")
        return self._node("schedule", task, [check_slot, choose_slot, slot_agreed])

    def details_node(self) -> NodeConfig:
        session = self.session
        address = session.repo.get_order(session.order_id).address

        async def update_address(flow_manager: FlowManager, address: str) -> tuple[dict, None]:
            """Change the delivery address.

            Args:
                address (str): The full new address as the customer said it.
            """
            return session.update_address(address), None

        async def add_note(flow_manager: FlowManager, note: str) -> tuple[dict, None]:
            """Save a delivery instruction, for example where to leave the order.

            Args:
                note (str): The instruction in a few words.
            """
            return session.add_note(note), None

        async def details_done(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The address is correct and any delivery instructions are saved."""
            return {"status": "ok"}, self.confirm_node()

        task = (f"Check that the delivery address is still {address}; if it changed, call update_address. "
                "Ask if there are any delivery instructions and save them with add_note. Then call details_done.")
        return self._node("details", task, [update_address, add_note, details_done])

    def confirm_node(self) -> NodeConfig:
        session = self.session
        summary = session.order_summary()

        async def confirm_booking(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer said yes to the read-back."""
            result = session.confirm_booking()
            if result["status"] != "booked":
                return result, self.schedule_node()
            return result, self.close_node("The delivery is booked. Thank the customer and say goodbye in one sentence.")

        async def change_time(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer wants a different delivery time."""
            return {"status": "ok"}, self.schedule_node()

        async def change_address(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer wants to change the address or delivery instructions."""
            return {"status": "ok"}, self.details_node()

        task = ("Read this back in one or two sentences and ask the customer to confirm: "
                f"items {'; '.join(summary['items'])}; delivery {summary['slot']}; address {summary['address']}. "
                "If they say yes, call confirm_booking.")
        return self._node("confirm", task, [confirm_booking, change_time, change_address])
