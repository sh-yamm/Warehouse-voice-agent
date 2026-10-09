import pytest

from voiceagent.domain.inventory import InventoryService


@pytest.fixture
def inv(world, repo):
    return InventoryService(repo)


def test_find_products_ranks_best_match_first(inv):
    assert inv.find_products(1, "amul milk")[0].sku == "MILK1"
    assert inv.find_products(1, "nandini")[0].sku == "MILK2"
    assert inv.find_products(1, "Eggs")[0].sku == "EGGS12"


def test_find_products_reports_availability(inv):
    match = inv.find_products(1, "amul milk")[0]
    assert match.available == 8


def test_find_products_needs_token_overlap(inv):
    assert inv.find_products(1, "toothpaste") == []
    assert inv.find_products(1, "") == []


def test_shortages_lists_short_lines_with_substitutes(inv, now):
    [shortage] = inv.shortages(1)
    assert (shortage.sku, shortage.wanted, shortage.reserved) == ("BREAD1", 1, 0)
    assert shortage.restock_eta.day == 13 and shortage.restock_eta.hour == 9
    assert [s.sku for s in shortage.substitutes] == ["BREAD2"]


def test_add_new_item_reserves_stock(inv, repo):
    result = inv.add_item(1, "MILK2", 2)
    assert (result.status, result.available) == ("added", 3)
    assert repo.available_stock(1, "MILK2") == 3
    assert [l.sku for l in repo.get_order_lines(1)][-1] == "MILK2"


def test_add_existing_item_increases_quantity(inv, repo):
    inv.add_item(1, "MILK1", 1)
    line = next(l for l in repo.get_order_lines(1) if l.sku == "MILK1")
    assert (line.qty, line.reserved_qty) == (3, 3)


def test_add_item_insufficient_changes_nothing(inv, repo):
    result = inv.add_item(1, "RICE5", 5)
    assert (result.status, result.available) == ("insufficient", 2)
    assert all(l.sku != "RICE5" for l in repo.get_order_lines(1))
    assert repo.available_stock(1, "RICE5") == 2


def test_add_item_rejects_bad_input(inv):
    with pytest.raises(ValueError):
        inv.add_item(1, "MILK1", 0)
    with pytest.raises(ValueError):
        inv.add_item(1, "NOPE", 1)


def test_substitution_reserves_substitute(inv, repo):
    assert inv.apply_substitution(1, "BREAD1", "BREAD2") is True
    line = next(l for l in repo.get_order_lines(1) if l.sku == "BREAD1")
    assert (line.substitute_sku, line.substitute_qty, line.resolution) == ("BREAD2", 1, "substitute")
    assert repo.available_stock(1, "BREAD2") == 3
    assert inv.shortages(1) == []


def test_substitution_fails_without_stock_or_twice(inv, repo):
    assert inv.apply_substitution(1, "BREAD1", "RICE5") is True
    assert inv.apply_substitution(1, "BREAD1", "BREAD2") is False
    assert inv.apply_substitution(1, "MILK1", "MILK2") is False  # not short


def test_partial_marks_line(inv, repo):
    assert inv.set_partial(1, "BREAD1") is True
    assert next(l for l in repo.get_order_lines(1) if l.sku == "BREAD1").resolution == "partial"
    assert inv.shortages(1) == []


def test_wait_restock_sets_earliest_delivery(inv):
    eta = inv.wait_restock(1, "BREAD1")
    assert eta.day == 13 and eta.hour == 9
    assert inv.earliest_delivery(1) == eta


def test_wait_restock_on_line_that_is_not_short_returns_none(inv):
    assert inv.wait_restock(1, "MILK1") is None
    assert inv.earliest_delivery(1) is None


def test_closed_order_rejects_inventory_changes(inv, repo):
    from voiceagent.domain.orders import OrderService
    OrderService(repo).cancel_order(1)
    with pytest.raises(ValueError):
        inv.add_item(1, "MILK2", 2)
    assert inv.apply_substitution(1, "MILK1", "MILK2") is False
    assert inv.set_partial(1, "BREAD1") is False
    assert inv.wait_restock(1, "BREAD1") is None
    assert repo.available_stock(1, "MILK2") == 5


def test_substitutes_prefer_the_most_similar_product(inv, repo):
    repo.insert_product("BREAD3", "Modern Brown Bread 400g", "Modern brown bread", "bakery", 47.0)
    repo.insert_stock(1, "BREAD3", 6)
    [shortage] = inv.shortages(1)
    assert [s.sku for s in shortage.substitutes] == ["BREAD3", "BREAD2"]
