"""탐색 스크립트 01~06을 순서대로 실행하고, 결과를 results/EDA_REPORT.md 한 파일로 모은다.

실행 (프로젝트 폴더에서):  py eda/run_all.py
하나만 다시 실행:          py eda/04_weather_correlation.py
"""

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eda_utils import RESULTS, SECTIONS, ensure_dirs

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_check_blanks.py",
    "02_describe_data.py",
    "03_level_change.py",
    "04_weather_correlation.py",
    "05_compare_feature_sets.py",
    "06_plots.py",
]


def versions() -> str:
    lines = [f"- Python {platform.python_version()} ({sys.executable})",
             f"- OS: {platform.platform()}",
             f"- CPU: {platform.processor() or '확인 불가'}"]
    for mod in ("pandas", "numpy", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            lines.append(f"- {mod} {m.__version__}")
        except ImportError:
            lines.append(f"- {mod} (설치 안 됨)")
    return "\n".join(lines)


def main():
    ensure_dirs()
    for script in SCRIPTS:
        print(f"\n===== {script} =====")
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if result.returncode != 0:
            print(f"\n[중단] {script} 실행 중 오류가 났습니다. 위 메시지를 확인하세요.")
            sys.exit(result.returncode)

    parts = [
        "# 탐색(EDA) 결과 보고서",
        f"생성 시각: {datetime.now():%Y-%m-%d %H:%M}",
        "이 파일은 `py eda/run_all.py`가 자동으로 만든다. 직접 고치지 말고, 판단은 `eda/DECISIONS.md`에 적는다.",
        "### 실행 환경",
        versions(),
    ]
    for script in SCRIPTS:
        name = {
            "01_check_blanks.py": "01_blanks",
            "02_describe_data.py": "02_describe",
            "03_level_change.py": "03_level_change",
            "04_weather_correlation.py": "04_correlation",
            "05_compare_feature_sets.py": "05_feature_sets",
            "06_plots.py": "06_plots",
        }[script]
        parts.append((SECTIONS / f"{name}.md").read_text(encoding="utf-8"))

    report = RESULTS / "EDA_REPORT.md"
    report.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"\n[완료] 보고서: {report}")


if __name__ == "__main__":
    main()
