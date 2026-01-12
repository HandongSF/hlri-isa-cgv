#!/usr/bin/env python3
import csv
from pathlib import Path


# CSV 파일 위치와 4회 반복 묶음 크기 설정
CSV_PATH = Path(__file__).with_name("objnav_hm3d.csv")
GROUP_SIZE = 4


def read_rows(path: Path):
    """CSV에서 필요한 컬럼만 읽어 목록으로 반환한다."""
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 빈 줄이 들어있을 수 있으므로 episode 값이 없으면 건너뛴다.
            if not row or not row.get("episode"):
                continue
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "success": float(row["success"]),
                    "start_distance": float(row["start_distance_to_goal"]),
                }
            )
    return rows


def main():
    rows = read_rows(CSV_PATH)

    total_eps = len(rows)
    total_success = sum(int(round(r["success"])) for r in rows)
    success_rate = (total_success / total_eps * 100) if total_eps else 0.0

    print(f"총 에피소드 수: {total_eps}")
    print(f"총 성공 횟수: {total_success}")
    print(f"총 성공률: {success_rate:.1f}%")
    print()

    print("에피소드 묶음별(동일 출발/목표) 성공 현황:")
    for idx, start in enumerate(range(0, total_eps, GROUP_SIZE), start=1):
        block = rows[start : start + GROUP_SIZE]
        block_success = sum(int(round(r["success"])) for r in block)
        block_rate = (block_success / len(block) * 100) if block else 0.0
        start_dist = block[0]["start_distance"] if block else 0.0
        print(
            f"  묶음 {idx}: 성공 {block_success}/{len(block)}회 "
            f"(성공률 {block_rate:.1f}%), 시작-목표 거리 약 {start_dist:.2f}m"
        )


if __name__ == "__main__":
    main()
