from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

from voiceagent.db.repository import Repository

CATALOG = [  # (category, item, unit, base_price)
    ("dairy", "Toned Milk", "1L", 50), ("dairy", "Curd", "400g", 35), ("dairy", "Paneer", "200g", 90),
    ("dairy", "Butter", "100g", 56), ("bakery", "Brown Bread", "400g", 45), ("bakery", "Pav Buns", "6 pack", 30),
    ("eggs", "Farm Eggs", "12 pack", 90), ("staples", "Basmati Rice", "5kg", 520), ("staples", "Atta", "5kg", 260),
    ("staples", "Toor Dal", "1kg", 160), ("staples", "Sugar", "1kg", 48), ("oils", "Sunflower Oil", "1L", 150),
    ("oils", "Mustard Oil", "1L", 170), ("snacks", "Potato Chips", "150g", 50), ("snacks", "Salted Peanuts", "200g", 60),
    ("beverages", "Tea Powder", "500g", 280), ("beverages", "Instant Coffee", "100g", 300),
    ("beverages", "Orange Juice", "1L", 120), ("produce", "Onions", "1kg", 40), ("produce", "Tomatoes", "1kg", 35),
    ("produce", "Bananas", "1 dozen", 60), ("household", "Dishwash Liquid", "500ml", 110),
    ("household", "Detergent Powder", "1kg", 140), ("personal", "Toothpaste", "150g", 95),
    ("personal", "Bath Soap", "4 pack", 160),
]
BRANDS = {
    "dairy": ["Amul", "Nandini"], "bakery": ["Modern", "Harvest"], "eggs": ["Eggoz", "Country"],
    "staples": ["Aashirvaad", "Tata"], "oils": ["Fortune", "Saffola"], "snacks": ["Lays", "Haldiram"],
    "beverages": ["Tata", "Bru"], "produce": ["Fresh", "Farm"], "household": ["Vim", "Surf"],
    "personal": ["Colgate", "Dove"],
}
FIRST_NAMES = ["Priya", "Arjun", "Ananya", "Rahul", "Kavya", "Vikram", "Sneha", "Rohan", "Meera", "Aditya",
               "Divya", "Karthik", "Pooja", "Suresh", "Neha", "Manoj", "Lakshmi", "Imran", "Fatima", "John"]
LAST_NAMES = ["Sharma", "Rao", "Iyer", "Reddy", "Nair", "Patel", "Khan", "Gupta", "Menon", "Das"]
STREETS = ["MG Road", "Church Street", "Indiranagar 100 Feet Road", "HSR Layout Sector 2",
           "Koramangala 5th Block", "Jayanagar 4th Block", "Whitefield Main Road", "BTM Layout 2nd Stage"]
WAREHOUSES = [(1, "Koramangala Hub", "south"), (2, "Indiranagar Hub", "east")]


def generate_world(repo: Repository, now: datetime, seed: int = 7, n_customers: int = 200,
                   n_orders: int = 300, days: int = 7) -> None:
    rng = random.Random(seed)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    with repo.transaction():
        for warehouse in WAREHOUSES:
            repo.insert_warehouse(*warehouse)

        skus = []
        for i, (category, item, unit, price) in enumerate(CATALOG):
            for j, brand in enumerate(BRANDS[category]):
                sku = f"{category[:3].upper()}{i:02d}{j}"
                repo.insert_product(sku, f"{brand} {item} {unit}", f"{brand} {item.lower()}", category,
                                    float(price + 5 * j))
                skus.append(sku)

        for warehouse_id, _, _ in WAREHOUSES:
            for sku in skus:
                if rng.random() < 0.12:
                    eta = today + timedelta(days=rng.randint(1, 3), hours=9)
                    repo.insert_stock(warehouse_id, sku, 0, eta)
                else:
                    repo.insert_stock(warehouse_id, sku, rng.randint(2, 40))

        addresses = {}
        for customer_id in range(1, n_customers + 1):
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            addresses[customer_id] = f"{rng.randint(1, 250)} {rng.choice(STREETS)}, Bengaluru"
            repo.insert_customer(customer_id, name, f"+9198{customer_id:08d}", addresses[customer_id])

        for order_id in range(1, n_orders + 1):
            customer_id = rng.randint(1, n_customers)
            warehouse_id = rng.choice(WAREHOUSES)[0]
            repo.insert_order(order_id, customer_id, warehouse_id, addresses[customer_id],
                              now - timedelta(minutes=rng.randint(5, 600)))
            for sku in rng.sample(skus, rng.randint(1, 4)):
                qty = rng.randint(1, 3)
                reserved = min(qty, repo.available_stock(warehouse_id, sku))
                repo.insert_order_line(order_id, sku, qty, reserved)
                if reserved:
                    repo.reserve_stock(warehouse_id, sku, reserved)

        slot_id = 1
        for warehouse_id, _, _ in WAREHOUSES:
            for day in range(days):
                for hour in range(8, 21):
                    start = today + timedelta(days=day, hours=hour)
                    capacity = rng.randint(3, 6)
                    booked = capacity if rng.random() < 0.2 else rng.randint(0, capacity - 1)
                    repo.insert_slot(slot_id, warehouse_id, start, start + timedelta(hours=1), capacity, booked)
                    slot_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic synthetic warehouse database")
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--now", help="ISO datetime used as 'now' (default: current minute)")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now().replace(second=0, microsecond=0)
    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    repo = Repository(path)
    repo.init_schema()
    generate_world(repo, now, seed=args.seed)
    q = lambda sql: repo.conn.execute(sql).fetchone()[0]
    print(f"wrote {path}: {q('SELECT COUNT(*) FROM orders')} orders, "
          f"{q('SELECT COUNT(*) FROM order_lines WHERE reserved_qty < qty')} short lines, "
          f"{q('SELECT COUNT(*) FROM delivery_slots WHERE booked = capacity')} full slots")
    repo.close()


if __name__ == "__main__":
    main()
