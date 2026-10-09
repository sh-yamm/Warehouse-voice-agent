CREATE TABLE IF NOT EXISTS warehouses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    zone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    spoken_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stock (
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    sku TEXT NOT NULL REFERENCES products(sku),
    on_hand INTEGER NOT NULL CHECK (on_hand >= 0),
    reserved INTEGER NOT NULL DEFAULT 0 CHECK (reserved >= 0 AND reserved <= on_hand),
    restock_eta TEXT,
    PRIMARY KEY (warehouse_id, sku)
);
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    default_address TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_slots (
    id INTEGER PRIMARY KEY,
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    booked INTEGER NOT NULL DEFAULT 0 CHECK (booked >= 0 AND booked <= capacity)
);
CREATE INDEX IF NOT EXISTS idx_slots_wh_start ON delivery_slots(warehouse_id, start_ts);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    status TEXT NOT NULL DEFAULT 'pending_schedule',
    address TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    slot_id INTEGER REFERENCES delivery_slots(id),
    callback_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_lines (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    sku TEXT NOT NULL REFERENCES products(sku),
    qty INTEGER NOT NULL CHECK (qty > 0),
    reserved_qty INTEGER NOT NULL CHECK (reserved_qty >= 0 AND reserved_qty <= qty),
    substitute_sku TEXT REFERENCES products(sku),
    substitute_qty INTEGER NOT NULL DEFAULT 0,
    resolution TEXT,
    PRIMARY KEY (order_id, sku)
);
CREATE TABLE IF NOT EXISTS slot_holds (
    slot_id INTEGER NOT NULL REFERENCES delivery_slots(id),
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT,
    transcript_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL REFERENCES calls(id),
    kind TEXT NOT NULL,
    total_ms REAL NOT NULL,
    breakdown_json TEXT NOT NULL DEFAULT '{}',
    recorded_at TEXT NOT NULL
);
