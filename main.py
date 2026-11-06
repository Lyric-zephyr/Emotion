"""统一入口：默认启动网页，也可运行 train / predict 子命令。"""

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="动物情绪识别系统统一入口")
    commands = parser.add_subparsers(dest="command", required=True)

    web = commands.add_parser("web", help="启动网页界面（默认）")
    web.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    web.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    web.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")

    # 保留原脚本的参数解析和帮助信息。
    commands.add_parser("train", add_help=False, help="训练模型，参数同 train.py")
    commands.add_parser("predict", add_help=False, help="预测图片，参数同 predict.py")

    arguments = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = parser.parse_known_args(arguments or ["web"])

    if args.command != "web":
        script = PROJECT_DIR / f"{args.command}.py"
        return subprocess.run([sys.executable, str(script), *forwarded]).returncode

    if forwarded:
        parser.error("未识别的网页启动参数：" + " ".join(forwarded))

    try:
        from app import demo
    except ModuleNotFoundError as error:
        print(f"缺少运行依赖：{error.name}", file=sys.stderr)
        print(
            '请在当前 Python 环境执行：python -m pip install '
            f'-r "{PROJECT_DIR / "requirements.txt"}"',
            file=sys.stderr,
        )
        return 1

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
