import asyncio
import cv2
import numpy as np

from src.core.config import Config
from src.cognitive.llm_client import LLMClient


async def main() -> None:
    cfg = Config()
    cfg.llm.model_name = "gemma4:e4b"
    client = LLMClient(cfg)

    ok = await client.health_check()
    print("HEALTH_CHECK=", ok)

    text = await client.generate("Reponds en 4 mots: systeme operationnel?")
    print("TEXT_RESPONSE=", text[:220].replace("\n", " "))

    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:, :] = (30, 30, 30)
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 255), -1)
    ok_img, buf = cv2.imencode(".jpg", img)
    if not ok_img:
        raise RuntimeError("encode image failed")

    multi = await client.generate(
        "Decris tres brievement limage en francais.",
        image=buf.tobytes(),
    )
    print("MULTIMODAL_RESPONSE=", multi[:220].replace("\n", " "))

    print(
        "METRICS_LAST_MS=", round(client.metrics.last_response_ms, 2),
        "AVG_MS=", round(client.metrics.avg_response_ms, 2),
    )


asyncio.run(main())
