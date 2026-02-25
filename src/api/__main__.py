from __future__ import annotations

import uvicorn

from src.api.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
