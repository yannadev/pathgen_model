"""Build student-disjoint, size-capped Junyi BKT CSVs from a reviewed skill map.

Run from this directory after skill_map.csv has been reviewed:
    python build_junyi_csvs.py

The script scans the original log twice, so it never loads the 2.7 GB source
file into memory. It keeps each selected student's complete mapped history.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXERCISES = ROOT / "junyi_Exercise_table.csv"
LOG = ROOT / "junyi_ProblemLog_original.csv"
MAP = ROOT / "skill_map.csv"
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
TRAIN_LIMIT = 80_000
TEST_LIMIT = 20_000
SEED = 42
SKILLS = {f"S{i}" for i in range(1, 8)}
OUTPUT_COLUMNS = ["user_id", "skill_code", "time_done", "y"]


def load_map() -> dict[tuple[str, str], str]:
    with EXERCISES.open(encoding="utf-8-sig", newline="") as handle:
        exercise_names = {row["name"] for row in csv.DictReader(handle)}

    mapping: dict[tuple[str, str], str] = {}
    with MAP.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["exercise", "problem_type", "skill_code"]:
            raise ValueError("skill_map.csv must have exercise,problem_type,skill_code columns")
        for line_number, row in enumerate(reader, start=2):
            exercise = row["exercise"].strip()
            problem_type = row["problem_type"].strip()
            skill = row["skill_code"].strip()
            if exercise not in exercise_names:
                raise ValueError(f"Unknown exercise on map line {line_number}: {exercise}")
            if skill not in SKILLS:
                raise ValueError(f"Invalid skill on map line {line_number}: {skill}")
            key = exercise, problem_type
            if key in mapping:
                raise ValueError(f"Duplicate exercise/problem_type on map line {line_number}: {key}")
            mapping[key] = skill
    if not mapping:
        raise ValueError("skill_map.csv has no mapped exercises")
    return mapping


def mapped_skill(mapping: dict[tuple[str, str], str], exercise: str, problem_type: str):
    return mapping.get((exercise, problem_type), mapping.get((exercise, "")))


def log_rows():
    with LOG.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        needed = ["user_id", "exercise", "problem_type", "time_done", "correct"]
        missing = set(needed) - set(header)
        if missing:
            raise ValueError(f"Missing log columns: {sorted(missing)}")
        idx = {name: header.index(name) for name in needed}
        for source_line, row in enumerate(reader, start=2):
            yield source_line, row, idx


def scan_counts(mapping: dict[tuple[str, str], str]) -> tuple[Counter, Counter]:
    user_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    for line, row, idx in log_rows():
        skill = mapped_skill(mapping, row[idx["exercise"]], row[idx["problem_type"]])
        if skill is None:
            continue
        user = row[idx["user_id"]]
        label = row[idx["correct"]].strip().lower()
        if not user or label not in {"true", "false"}:
            raise ValueError(f"Invalid mapped row at source line {line}")
        try:
            int(row[idx["time_done"]])
        except ValueError as error:
            raise ValueError(f"Invalid time_done at source line {line}") from error
        user_counts[user] += 1
        skill_counts[skill] += 1
    return user_counts, skill_counts


def select_users(counts: Counter[str]) -> tuple[set[str], set[str]]:
    users = sorted(counts)
    random.Random(SEED).shuffle(users)
    test_cutoff = len(users) // 5
    test_pool = users[:test_cutoff]
    train_pool = users[test_cutoff:]

    def take(pool: list[str], limit: int) -> set[str]:
        chosen: set[str] = set()
        used = 0
        for user in pool:
            n = counts[user]
            if used + n <= limit:
                chosen.add(user)
                used += n
                if used == limit:
                    break
        return chosen

    train_users = take(train_pool, TRAIN_LIMIT)
    test_users = take(test_pool, TEST_LIMIT)
    assert train_users.isdisjoint(test_users)
    return train_users, test_users


def collect_rows(mapping: dict[tuple[str, str], str], train_users: set[str], test_users: set[str]):
    train_rows = []
    test_rows = []
    chosen = train_users | test_users
    for line, row, idx in log_rows():
        user = row[idx["user_id"]]
        if user not in chosen:
            continue
        skill = mapped_skill(mapping, row[idx["exercise"]], row[idx["problem_type"]])
        if skill is None:
            continue
        y = 1 if row[idx["correct"]].strip().lower() == "true" else 0
        item = (user, skill, int(row[idx["time_done"]]), y, line)
        (train_rows if user in train_users else test_rows).append(item)
    train_rows.sort(key=lambda r: (r[0], r[2], r[4]))
    test_rows.sort(key=lambda r: (r[0], r[2], r[4]))
    return train_rows, test_rows


def write_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows((user, skill, time_done, y) for user, skill, time_done, y, _ in rows)


def report(name: str, rows: list[tuple]) -> None:
    per_skill = Counter(row[1] for row in rows)
    per_user_skill = defaultdict(Counter)
    for user, skill, *_ in rows:
        per_user_skill[skill][user] += 1
    repeated = {
        skill: sum(n >= 2 for n in per_user_skill[skill].values())
        for skill in sorted(SKILLS)
    }
    print(f"{name}: {len(rows):,} rows, {len({r[0] for r in rows}):,} students")
    print(f"  rows by skill: {dict(sorted(per_skill.items()))}")
    print(f"  students with 2+ answers per skill: {repeated}")


def main() -> None:
    mapping = load_map()
    counts, available = scan_counts(mapping)
    print(f"Mapped exercises: {len(mapping)}; eligible students: {len(counts):,}")
    print(f"Available mapped rows by skill: {dict(sorted(available.items()))}")
    train_users, test_users = select_users(counts)
    train_rows, test_rows = collect_rows(mapping, train_users, test_users)
    if len(train_rows) > TRAIN_LIMIT or len(test_rows) > TEST_LIMIT:
        raise AssertionError("Row cap exceeded")
    if {r[0] for r in train_rows} & {r[0] for r in test_rows}:
        raise AssertionError("Student leakage")
    report("train", train_rows)
    report("test", test_rows)
    write_csv(TRAIN, train_rows)
    write_csv(TEST, test_rows)


if __name__ == "__main__":
    main()
