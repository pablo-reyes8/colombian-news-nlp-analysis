from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import load_settings


def build_argparser() -> argparse.ArgumentParser:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Levanta la API FastAPI para sentimiento y predicción")
    p.add_argument("--host", default=settings.api_host)
    p.add_argument("--port", type=int, default=settings.api_port)
    p.add_argument("--reload", action="store_true", default=settings.api_reload)
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
