from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="机器人状态曲线查看器")
    parser.add_argument("path", nargs="?", help="启动时直接打开的日志文件路径")
    args = parser.parse_args(argv)

    initial_path = Path(args.path).expanduser() if args.path else None

    try:
        from .main_window import launch_app
    except ImportError as exc:
        print(
            "缺少 GUI 依赖，请先安装 requirements.txt 中的依赖后再启动。\n"
            f"详细错误: {exc}",
            file=sys.stderr,
        )
        return 1

    return launch_app(initial_path=initial_path)


if __name__ == "__main__":
    raise SystemExit(main())
