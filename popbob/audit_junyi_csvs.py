"""Independently audit the prepared CSVs against the original Junyi log."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

import build_junyi_csvs as build


def read_output(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == build.OUTPUT_COLUMNS, (path, reader.fieldnames)
        rows = list(reader)
    assert all(row["user_id"] for row in rows)
    assert all(row["skill_code"] in build.SKILLS for row in rows)
    assert all(row["y"] in {"0", "1"} for row in rows)
    assert all(row["time_done"].isdigit() for row in rows)
    assert all(
        (a["user_id"], int(a["time_done"]))
        <= (b["user_id"], int(b["time_done"]))
        for a, b in zip(rows, rows[1:])
    ), f"Rows out of user/time order in {path}"
    return rows


def key(row):
    return row["user_id"], row["skill_code"], row["time_done"], row["y"]


def main():
    mapping = build.load_map()
    train = read_output(build.TRAIN)
    test = read_output(build.TEST)
    assert len(train) == build.TRAIN_LIMIT
    assert len(test) == build.TEST_LIMIT
    train_users = {row["user_id"] for row in train}
    test_users = {row["user_id"] for row in test}
    assert train_users.isdisjoint(test_users), "Student leakage"

    expected = Counter(map(key, train + test))
    selected_users = train_users | test_users
    source_map_counts = Counter()
    for line, raw, idx in build.log_rows():
        exercise = raw[idx["exercise"]]
        problem_type = raw[idx["problem_type"]]
        skill = build.mapped_skill(mapping, exercise, problem_type)
        if skill is None:
            continue
        map_key = (exercise, problem_type) if (exercise, problem_type) in mapping else (exercise, "")
        source_map_counts[map_key] += 1
        user = raw[idx["user_id"]]
        if user not in selected_users:
            continue
        y = "1" if raw[idx["correct"]].strip().lower() == "true" else "0"
        item = user, skill, raw[idx["time_done"]], y
        if expected[item] <= 0:
            raise AssertionError(f"Selected source row absent from outputs at line {line}: {item}")
        expected[item] -= 1
    assert not +expected, f"Output rows absent from source: {sum(expected.values())}"

    print("PASS: exact 80,000/20,000 output rows; disjoint student IDs")
    print("PASS: headers, labels, skill codes, timestamps, and user/time order")
    print("PASS: every mapped source answer for each selected student appears exactly once")
    for name, rows in (("train", train), ("test", test)):
        by_skill = defaultdict(Counter)
        for row in rows:
            by_skill[row["skill_code"]][row["y"]] += 1
        assert set(by_skill) == build.SKILLS
        assert all(by_skill[skill]["0"] and by_skill[skill]["1"] for skill in build.SKILLS)
        print(f"{name}: {len(rows):,} rows; {len({r['user_id'] for r in rows}):,} students")
        print("  by skill (wrong, correct):", {
            skill: (by_skill[skill]["0"], by_skill[skill]["1"])
            for skill in sorted(build.SKILLS)
        })
    print("Source answers per mapping rule:")
    for map_key in mapping:
        print(f"  {map_key}: {source_map_counts[map_key]:,}")


if __name__ == "__main__":
    main()
