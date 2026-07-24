import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("uvicorn.error")

LOKI_URL = os.environ.get("LOKI_URL", "http://localhost:3100")
LOKI_QUERY = os.environ.get("LOKI_QUERY", '{cluster=~".+"}')
LOKI_MODE = os.environ.get("LOKI_MODE", "mock")  # "mock" | "live"
LOKI_TIMEOUT_SECONDS = float(os.environ.get("LOKI_TIMEOUT_SECONDS", "10"))

POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "30"))
QUERY_WINDOW_SECONDS = float(os.environ.get("QUERY_WINDOW_SECONDS", "300"))

ANALYZE_API_URL = os.environ.get("ANALYZE_API_URL", "http://localhost:8000/analyze")
ANALYZE_TIMEOUT_SECONDS = float(os.environ.get("ANALYZE_TIMEOUT_SECONDS", "120"))

# Lambda(Slack 알림) 엔드포인트. 없으면 알림 자체를 건너뜀 (기본값 없음)
LAMBDA_NOTIFY_URL = os.environ.get("LAMBDA_NOTIFY_URL", "")
LAMBDA_NOTIFY_TIMEOUT_SECONDS = float(os.environ.get("LAMBDA_NOTIFY_TIMEOUT_SECONDS", "3"))

FILTER_KEYWORDS = ("error", "warning", "oomkilled", "crashloopbackoff")
_KEYWORD_PATTERN = re.compile("|".join(FILTER_KEYWORDS), re.IGNORECASE)


@dataclass
class LogEntry:
    cluster_name: str
    namespace: str
    pod: str
    container: str
    timestamp_ns: int
    message: str


# Loki 실 서버 없이 개발/테스트할 수 있도록 준비한 GET /loki/api/v1/query_range 응답 mock.
# eks-a/eks-b/onprem 세 클러스터를 섞어서, cluster_name 라벨 파싱과 키워드 필터링을 함께 검증할 수 있게 구성.
MOCK_QUERY_RANGE_RESPONSE = {
    "status": "success",
    "data": {
        "resultType": "streams",
        "result": [
            {
                "stream": {
                    "cluster": "eks-a",
                    "namespace": "payment",
                    "pod": "payment-service-7d8f9c-x2k1p",
                    "container": "payment-api",
                },
                "values": [
                    [
                        "1752054194000000000",
                        '2026-07-09T10:23:14Z [Warning] Reason: OOMKilled Exit Code: 137 Memory Limit: 512Mi',
                    ],
                    [
                        "1752054190000000000",
                        "2026-07-09T10:23:10Z INFO Processing batch job #4821, records=50000",
                    ],
                ],
            },
            {
                "stream": {
                    "cluster": "eks-b",
                    "namespace": "order",
                    "pod": "order-service-9f7d2a-m4x8k",
                    "container": "order-worker",
                },
                "values": [
                    [
                        "1752057932000000000",
                        "2026-07-09T11:05:32Z [Warning] Reason: CrashLoopBackOff Restart Count: 7 "
                        "Last State: Terminated (Exit Code: 1)",
                    ],
                ],
            },
            {
                "stream": {
                    "cluster": "onprem",
                    "namespace": "default",
                    "pod": "web-frontend-abc123-9z8y7",
                    "container": "nginx",
                },
                "values": [
                    [
                        "1752054200000000000",
                        "2026-07-09T10:23:20Z INFO GET /healthz 200 OK",
                    ],
                ],
            },
        ],
    },
}


def _query_loki_mock(start_ns: int, end_ns: int) -> dict:
    return MOCK_QUERY_RANGE_RESPONSE


async def _query_loki_live(start_ns: int, end_ns: int) -> dict:
    params = {
        "query": LOKI_QUERY,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": "1000",
    }
    async with httpx.AsyncClient(timeout=LOKI_TIMEOUT_SECONDS) as client:
        resp = await client.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params)
        resp.raise_for_status()
        return resp.json()


async def _query_loki(start_ns: int, end_ns: int) -> dict | None:
    try:
        if LOKI_MODE == "live":
            return await _query_loki_live(start_ns, end_ns)
        return _query_loki_mock(start_ns, end_ns)
    except httpx.ConnectError as e:
        logger.warning("Loki 연결 실패, 이번 폴링 사이클 건너뜀: %s", e)
    except httpx.TimeoutException as e:
        logger.warning("Loki 응답 시간 초과, 이번 폴링 사이클 건너뜀: %s", e)
    except httpx.HTTPStatusError as e:
        logger.error(
            "Loki가 오류 상태코드를 반환(%s), 이번 폴링 사이클 건너뜀: %s",
            e.response.status_code,
            e,
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.error("Loki 응답 파싱 실패, 이번 폴링 사이클 건너뜀: %s", e)
    return None


def _parse_response(data: dict) -> list[LogEntry]:
    entries = []
    for stream in data.get("data", {}).get("result", []):
        labels = stream.get("stream", {})
        # cluster_name은 실제 Loki에서는 "cluster" 라벨(또는 배포 시 정해질 라벨 키)에서 가져온다.
        cluster_name = labels.get("cluster", "unknown")
        namespace = labels.get("namespace", "unknown")
        pod = labels.get("pod", "unknown")
        container = labels.get("container", "unknown")
        for ts_str, line in stream.get("values", []):
            entries.append(
                LogEntry(
                    cluster_name=cluster_name,
                    namespace=namespace,
                    pod=pod,
                    container=container,
                    timestamp_ns=int(ts_str),
                    message=line,
                )
            )
    return entries


def _matches_filter(entry: LogEntry) -> bool:
    return bool(_KEYWORD_PATTERN.search(entry.message))


_last_seen_ns = 0


def _select_new_matches(entries: list[LogEntry]) -> list[LogEntry]:
    global _last_seen_ns
    new_matches = [
        e for e in entries if e.timestamp_ns > _last_seen_ns and _matches_filter(e)
    ]
    if entries:
        _last_seen_ns = max(_last_seen_ns, max(e.timestamp_ns for e in entries))
    return new_matches


def _to_log_text(entry: LogEntry) -> str:
    return (
        f"cluster: {entry.cluster_name}\n"
        f"namespace: {entry.namespace}\n"
        f"pod: {entry.pod}\n"
        f"container: {entry.container}\n"
        f"{entry.message}"
    )


def _matched_keyword(entry: LogEntry) -> str:
    match = _KEYWORD_PATTERN.search(entry.message)
    return match.group(0) if match else "unknown"


async def _notify_error_detected(entry: LogEntry) -> None:
    if not LAMBDA_NOTIFY_URL:
        return

    payload = {
        "notification_type": "error_detected",
        "cluster_name": entry.cluster_name,
        "failure_keyword": _matched_keyword(entry),
        "log_snippet": entry.message[:500],
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=LAMBDA_NOTIFY_TIMEOUT_SECONDS) as client:
            resp = await client.post(LAMBDA_NOTIFY_URL, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError as e:
        logger.warning(
            "Slack 알림(에러 감지) 연결 실패, 무시하고 계속 진행 (cluster=%s pod=%s): %s",
            entry.cluster_name,
            entry.pod,
            e,
        )
    except httpx.TimeoutException as e:
        logger.warning(
            "Slack 알림(에러 감지) 응답 시간 초과, 무시하고 계속 진행 (cluster=%s pod=%s): %s",
            entry.cluster_name,
            entry.pod,
            e,
        )
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Slack 알림(에러 감지) 호출 실패(HTTP %s), 무시하고 계속 진행 (cluster=%s pod=%s): %s",
            e.response.status_code,
            entry.cluster_name,
            entry.pod,
            e,
        )


async def _send_to_analyze(entry: LogEntry) -> dict:
    async with httpx.AsyncClient(timeout=ANALYZE_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            ANALYZE_API_URL,
            json={"log_text": _to_log_text(entry), "cluster_name": entry.cluster_name},
        )
        resp.raise_for_status()
        return resp.json()


async def poll_once() -> list[tuple[LogEntry, dict]]:
    now_ns = time.time_ns()
    start_ns = now_ns - int(QUERY_WINDOW_SECONDS * 1e9)

    data = await _query_loki(start_ns, now_ns)
    if data is None:
        return []

    try:
        entries = _parse_response(data)
    except (ValueError, KeyError, TypeError) as e:
        logger.error("Loki 응답 파싱 실패, 이번 폴링 사이클 건너뜀: %s", e)
        return []

    new_matches = _select_new_matches(entries)

    results = []
    for entry in new_matches:
        logger.warning(
            "새 장애 로그 감지 (cluster=%s pod=%s): %s",
            entry.cluster_name,
            entry.pod,
            entry.message[:120],
        )

        await _notify_error_detected(entry)

        try:
            analyzed = await _send_to_analyze(entry)
        except httpx.ConnectError as e:
            logger.error(
                "analyze 서버 연결 실패, 해당 로그 건 건너뜀 (cluster=%s pod=%s): %s",
                entry.cluster_name,
                entry.pod,
                e,
            )
            continue
        except httpx.TimeoutException as e:
            logger.error(
                "analyze 응답 시간 초과, 해당 로그 건 건너뜀 (cluster=%s pod=%s): %s",
                entry.cluster_name,
                entry.pod,
                e,
            )
            continue
        except httpx.HTTPStatusError as e:
            logger.error(
                "analyze 호출 실패(HTTP %s), 해당 로그 건 건너뜀 (cluster=%s pod=%s): %s",
                e.response.status_code,
                entry.cluster_name,
                entry.pod,
                e,
            )
            continue
        results.append((entry, analyzed))
    return results


async def poll_loop() -> None:
    logger.info(
        "Loki 폴러 시작 (mode=%s, interval=%ss, query_window=%ss, analyze_url=%s)",
        LOKI_MODE,
        POLL_INTERVAL_SECONDS,
        QUERY_WINDOW_SECONDS,
        ANALYZE_API_URL,
    )
    while True:
        try:
            await poll_once()
        except Exception:
            logger.exception("폴링 중 오류 발생")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(poll_loop())
