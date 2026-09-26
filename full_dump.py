import asyncio, json
from composer import compose_async, COMPOSE_CACHE
from judge_simulator import DatasetLoader, DATASET_DIR

data = DatasetLoader(DATASET_DIR)
data.load()

async def main():
    rows = []
    for tid, trigger in data.triggers.items():
        merchant = data.merchants.get(trigger.get("merchant_id"))
        if not merchant:
            continue
        category = data.categories.get(merchant.get("category_slug"))
        customer = data.customers.get(trigger.get("customer_id")) if trigger.get("customer_id") else None
        COMPOSE_CACHE.clear()
        action = await compose_async(category, merchant, trigger, customer)
        rows.append({
            "trigger_id": tid,
            "kind": trigger.get("kind"),
            "merchant": merchant.get("identity", {}).get("name"),
            "body": action.get("body"),
            "rationale": action.get("rationale"),
        })
    with open("full_dump.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} rows to full_dump.json")

asyncio.run(main())