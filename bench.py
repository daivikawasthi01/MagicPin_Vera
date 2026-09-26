"""
Benchmarking & Evaluation Harness for Vera Bot.
Executes compose() on all 30 canonical pairs from expanded/test_pairs.json,
runs an all-kinds sweep across all 25 seed trigger types,
validates output compliance (11 required keys, 0 URLs, 0 internal jargon leaks),
and generates submission.jsonl.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from composer import compose
from validator import (
    REQUIRED_ACTION_KEYS,
    strip_urls,
    JARGON_TERMS,
    check_no_unresolved_template_values,
)
import re

EXPANDED_DIR = Path(__file__).parent / "expanded"
DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_SUBMISSION = Path(__file__).parent / "submission.jsonl"

def load_json_directory(dir_path: Path, key_field: str) -> Dict[str, Any]:
    store = {}
    if not dir_path.exists():
        return store
    for f in dir_path.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    k = data.get(key_field) or f.stem
                    store[k] = data
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return store

def run_all_kinds_sweep():
    """Validates 100% trigger kind coverage across all 25 seed triggers in dataset/triggers_seed.json."""
    print("\n" + "=" * 60)
    print("ALL-KINDS SEED TRIGGER COVERAGE SWEEP (25 TRIGGERS)")
    print("=" * 60)

    # Load seed dataset
    categories = load_json_directory(DATASET_DIR / "categories", "slug")
    
    merchants_seed_path = DATASET_DIR / "merchants_seed.json"
    merchants = {}
    if merchants_seed_path.exists():
        with open(merchants_seed_path, "r", encoding="utf-8") as f:
            m_list = json.load(f).get("merchants", [])
            for m in m_list:
                merchants[m.get("merchant_id")] = m

    customers_seed_path = DATASET_DIR / "customers_seed.json"
    customers = {}
    if customers_seed_path.exists():
        with open(customers_seed_path, "r", encoding="utf-8") as f:
            c_list = json.load(f).get("customers", [])
            for c in c_list:
                customers[c.get("customer_id")] = c

    triggers_seed_path = DATASET_DIR / "triggers_seed.json"
    if not triggers_seed_path.exists():
        print(f"Warning: {triggers_seed_path} not found.")
        return

    with open(triggers_seed_path, "r", encoding="utf-8") as f:
        trig_list = json.load(f).get("triggers", [])

    passed = 0
    total = len(trig_list)

    for idx, trig in enumerate(trig_list, 1):
        m_id = trig.get("merchant_id")
        c_id = trig.get("customer_id")
        t_id = trig.get("id")

        merchant = merchants.get(m_id)
        customer = customers.get(c_id) if c_id else None
        cat_slug = merchant.get("category_slug") if merchant else None
        category = categories.get(cat_slug) if cat_slug else None
        if not merchant or not category or (c_id and not customer):
            print(f"S{idx:02d} [FAIL] | missing required context for trigger {t_id}")
            continue

        action = compose(category=category, merchant=merchant, trigger=trig, customer=customer)

        has_all_keys = all(k in action for k in REQUIRED_ACTION_KEYS)
        body = action.get("body", "")
        has_url = bool(re.search(r"https?://|www\.", body))
        has_jargon = any(re.search(pat, body, re.IGNORECASE) for pat in JARGON_TERMS)
        body_non_empty = len(body.strip()) > 0
        no_unresolved_template_values = check_no_unresolved_template_values(body)

        is_clean = (
            has_all_keys and not has_url and not has_jargon
            and body_non_empty and no_unresolved_template_values
        )
        if is_clean:
            passed += 1
        status_str = "[PASS]" if is_clean else "[FAIL]"
        print(f"S{idx:02d} {status_str} | {action.get('send_as'):<18} | {trig.get('kind', 'generic'):<24} | {body[:50]}...")

    print("-" * 60)
    print(f"SEED SWEEP COMPLETED: {passed}/{total} trigger kinds verified 100% clean.")
    print("=" * 60 + "\n")

def main():
    print("=" * 60)
    print("VERA BOT — CANONICAL 30-PAIR BENCHMARK HARNESS")
    print("=" * 60)

    # 1. Load context datasets
    categories = load_json_directory(EXPANDED_DIR / "categories", "slug")
    merchants = load_json_directory(EXPANDED_DIR / "merchants", "merchant_id")
    customers = load_json_directory(EXPANDED_DIR / "customers", "customer_id")
    triggers = load_json_directory(EXPANDED_DIR / "triggers", "id")

    test_pairs_path = EXPANDED_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"Error: {test_pairs_path} not found. Run dataset generator first.")
        return

    with open(test_pairs_path, "r", encoding="utf-8") as f:
        test_pairs_data = json.load(f)

    pairs = test_pairs_data.get("pairs", [])
    print(f"Loaded {len(pairs)} canonical test pairs.")
    print(f"Available: {len(categories)} categories, {len(merchants)} merchants, {len(customers)} customers, {len(triggers)} triggers.")
    print("-" * 60)

    results: List[Dict[str, Any]] = []
    penalties_detected = 0
    passed_validation = 0

    for idx, pair in enumerate(pairs, 1):
        t_id = pair.get("trigger_id")
        m_id = pair.get("merchant_id")
        c_id = pair.get("customer_id")
        test_id = pair.get("test_id", f"T{idx:02d}")

        trigger = triggers.get(t_id)
        merchant = merchants.get(m_id)
        customer = customers.get(c_id) if c_id else None

        cat_slug = merchant.get("category_slug") if merchant else None
        category = categories.get(cat_slug) if cat_slug else None
        if not trigger or not merchant or not category or (c_id and not customer):
            penalties_detected += 1
            print(f"{test_id} [FAIL] | missing required context for pair {t_id}/{m_id}")
            continue

        action = compose(category=category, merchant=merchant, trigger=trigger, customer=customer)
        action["test_id"] = test_id

        # Validation Checks
        has_all_keys = all(k in action for k in REQUIRED_ACTION_KEYS)
        body = action.get("body", "")
        has_url = bool(re.search(r"https?://|www\.", body))
        has_jargon = any(re.search(pat, body, re.IGNORECASE) for pat in JARGON_TERMS)
        body_non_empty = len(body.strip()) > 0
        no_unresolved_template_values = check_no_unresolved_template_values(body)

        is_clean = (
            has_all_keys and not has_url and not has_jargon
            and body_non_empty and no_unresolved_template_values
        )
        if is_clean:
            passed_validation += 1
        else:
            penalties_detected += 1

        status_str = "[PASS]" if is_clean else "[FAIL]"
        print(f"{test_id} {status_str} | {action.get('send_as'):<18} | {trigger.get('kind', 'generic'):<24} | {body[:50]}...")

        results.append(action)

    # Export submission.jsonl
    with open(OUTPUT_SUBMISSION, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"BENCHMARK COMPLETED: {passed_validation}/{len(pairs)} passed 100% clean.")
    print(f"Penalties / Issues: {penalties_detected}")
    print(f"Submission generated at: {OUTPUT_SUBMISSION} ({len(results)} lines)")
    print("=" * 60)

    # Run the comprehensive seed sweep
    run_all_kinds_sweep()

if __name__ == "__main__":
    main()
