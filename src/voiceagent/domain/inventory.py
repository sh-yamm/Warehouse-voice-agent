from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from voiceagent.db.repository import Repository
from voiceagent.domain.models import OrderLine, Product


@dataclass(frozen=True)
class ProductMatch:
    sku: str
    name: str
    spoken_name: str
    available: int
    unit_price: float
    score: float


@dataclass(frozen=True)
class Shortage:
    sku: str
    name: str
    wanted: int
    reserved: int
    restock_eta: datetime | None
    substitutes: list[ProductMatch]


@dataclass(frozen=True)
class AddItemResult:
    status: Literal["added", "insufficient"]
    sku: str
    name: str
    qty: int
    available: int


def _tokens(text: str) -> list[str]:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in text.lower().split()]


def _score(query: str, product: Product) -> float:
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    hay = set(_tokens(f"{product.spoken_name} {product.name}"))
    overlap = sum(t in hay for t in q_tokens) / len(q_tokens)
    q = query.lower().strip()
    ratio = max(difflib.SequenceMatcher(None, q, product.spoken_name.lower()).ratio(),
                difflib.SequenceMatcher(None, q, product.name.lower()).ratio())
    return round(0.6 * overlap + 0.4 * ratio, 3)


class InventoryService:
    MATCH_THRESHOLD = 0.45

    def __init__(self, repo: Repository):
        self.repo = repo

    def _match(self, warehouse_id: int, product: Product, score: float) -> ProductMatch:
        return ProductMatch(product.sku, product.name, product.spoken_name,
                            self.repo.available_stock(warehouse_id, product.sku), product.unit_price, score)

    def _line(self, order_id: int, sku: str) -> OrderLine | None:
        return next((l for l in self.repo.get_order_lines(order_id) if l.sku == sku), None)

    def find_products(self, warehouse_id: int, query: str, limit: int = 3) -> list[ProductMatch]:
        matches = []
        for product in self.repo.list_products():
            score = _score(query, product)
            if score >= self.MATCH_THRESHOLD:
                matches.append(self._match(warehouse_id, product, score))
        matches.sort(key=lambda m: (-m.score, m.sku))
        return matches[:limit]

    def shortages(self, order_id: int) -> list[Shortage]:
        order = self.repo.get_order(order_id)
        if order is None:
            return []
        out = []
        for line in self.repo.get_order_lines(order_id):
            if line.reserved_qty >= line.qty or line.resolution is not None:
                continue
            product = self.repo.get_product(line.sku)
            missing = line.qty - line.reserved_qty
            substitutes = [
                self._match(order.warehouse_id, p, 1.0)
                for p in self.repo.list_products()
                if p.category == product.category and p.sku != product.sku
                and self.repo.available_stock(order.warehouse_id, p.sku) >= missing
            ][:2]
            out.append(Shortage(line.sku, product.spoken_name, line.qty, line.reserved_qty,
                                self.repo.restock_eta(order.warehouse_id, line.sku), substitutes))
        return out

    def add_item(self, order_id: int, sku: str, qty: int) -> AddItemResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        order, product = self.repo.get_order(order_id), self.repo.get_product(sku)
        if order is None or product is None:
            raise ValueError(f"unknown order {order_id} or sku {sku}")
        if not order.is_open:
            raise ValueError(f"order {order_id} is {order.status}")
        with self.repo.transaction():
            available = self.repo.available_stock(order.warehouse_id, sku)
            if available < qty:
                return AddItemResult("insufficient", sku, product.spoken_name, qty, available)
            self.repo.reserve_stock(order.warehouse_id, sku, qty)
            existing = self._line(order_id, sku)
            if existing:
                self.repo.update_order_line(order_id, sku, qty=existing.qty + qty,
                                            reserved_qty=existing.reserved_qty + qty)
            else:
                self.repo.insert_order_line(order_id, sku, qty, qty)
        return AddItemResult("added", sku, product.spoken_name, qty, available - qty)

    def _short_line(self, order_id: int, sku: str) -> OrderLine | None:
        order = self.repo.get_order(order_id)
        if order is None or not order.is_open:
            return None
        line = self._line(order_id, sku)
        if line is None or line.reserved_qty >= line.qty or line.resolution is not None:
            return None
        return line

    def apply_substitution(self, order_id: int, sku: str, substitute_sku: str) -> bool:
        order, line = self.repo.get_order(order_id), self._short_line(order_id, sku)
        if order is None or line is None or substitute_sku == sku:
            return False
        missing = line.qty - line.reserved_qty
        with self.repo.transaction():
            if self.repo.available_stock(order.warehouse_id, substitute_sku) < missing:
                return False
            self.repo.reserve_stock(order.warehouse_id, substitute_sku, missing)
            self.repo.update_order_line(order_id, sku, substitute_sku=substitute_sku,
                                        substitute_qty=missing, resolution="substitute")
        return True

    def set_partial(self, order_id: int, sku: str) -> bool:
        if self._short_line(order_id, sku) is None:
            return False
        self.repo.update_order_line(order_id, sku, resolution="partial")
        return True

    def wait_restock(self, order_id: int, sku: str) -> datetime | None:
        order, line = self.repo.get_order(order_id), self._short_line(order_id, sku)
        if order is None or line is None:
            return None
        eta = self.repo.restock_eta(order.warehouse_id, sku)
        if eta is None:
            return None
        self.repo.update_order_line(order_id, sku, resolution="wait")
        return eta

    def earliest_delivery(self, order_id: int) -> datetime | None:
        order = self.repo.get_order(order_id)
        if order is None:
            return None
        etas = [self.repo.restock_eta(order.warehouse_id, l.sku)
                for l in self.repo.get_order_lines(order_id) if l.resolution == "wait"]
        etas = [e for e in etas if e is not None]
        return max(etas) if etas else None
