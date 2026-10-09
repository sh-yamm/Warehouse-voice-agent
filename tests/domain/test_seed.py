from voiceagent.db.repository import Repository
from voiceagent.db.seed import generate_world


def _fresh():
    r = Repository(":memory:")
    r.init_schema()
    return r


def test_seed_is_deterministic(now):
    a, b = _fresh(), _fresh()
    generate_world(a, now, seed=7)
    generate_world(b, now, seed=7)
    assert a.dump() == b.dump()


def test_different_seed_differs(now):
    a, b = _fresh(), _fresh()
    generate_world(a, now, seed=7)
    generate_world(b, now, seed=8)
    assert a.dump() != b.dump()


def test_seed_invariants(now):
    r = _fresh()
    generate_world(r, now, seed=7)
    q = lambda sql: r.conn.execute(sql).fetchone()[0]
    assert q("SELECT COUNT(*) FROM products") == 50
    assert q("SELECT COUNT(*) FROM orders") == 300
    # stock.reserved equals the sum of line reservations per warehouse and sku
    mismatches = q(
        """SELECT COUNT(*) FROM stock s WHERE s.reserved != (
               SELECT COALESCE(SUM(l.reserved_qty), 0) FROM order_lines l
               JOIN orders o ON o.id = l.order_id
               WHERE o.warehouse_id = s.warehouse_id AND l.sku = s.sku)"""
    )
    assert mismatches == 0
    assert q("SELECT COUNT(*) FROM order_lines WHERE reserved_qty < qty") >= 10
    assert q("SELECT COUNT(*) FROM delivery_slots WHERE booked = capacity") >= 5
    assert q("SELECT COUNT(*) FROM stock WHERE on_hand = 0 AND restock_eta IS NOT NULL") >= 5


def test_seed_has_a_week_of_slots(now):
    r = _fresh()
    generate_world(r, now, seed=7)
    days = r.conn.execute("SELECT COUNT(DISTINCT substr(start_ts, 1, 10)) FROM delivery_slots").fetchone()[0]
    assert days == 7
