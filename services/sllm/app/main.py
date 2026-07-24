import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

# Lambda(Slack 알림) 엔드포인트. 없으면 알림 자체를 건너뜀 (기본값 없음)
LAMBDA_NOTIFY_URL = os.environ.get("LAMBDA_NOTIFY_URL", "")
LAMBDA_NOTIFY_TIMEOUT_SECONDS = float(os.environ.get("LAMBDA_NOTIFY_TIMEOUT_SECONDS", "3"))

# dr_action_needed 판단 기준 (팀 확정, 7/24 변경): OOMKilled/DiskPressure/NetworkTimeout이면 DR 전환 필요
_DR_ACTION_FAILURE_TYPES = {"OOMKilled", "DiskPressure", "NetworkTimeout"}

FAILURE_TYPES = [
    "OOMKilled",
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "DBConnectionFailure",
    "LivenessProbeFailure",
    "DiskPressure",
    "NetworkTimeout",
    "ConfigurationError",
    "Unknown",
]

PROMPT_TEMPLATE = """당신은 SRE 장애 분석 어시스턴트입니다. 아래 로그를 분석해서 반드시 아래 JSON 형식으로만 답변하세요. JSON 외의 다른 텍스트, 설명, 마크다운은 절대 포함하지 마세요.

반드시 한국어로만 답변하세요. 중국어, 영어를 포함한 어떤 다른 언어도 절대 사용하지 마세요. JSON의 모든 값(failure_type, root_cause, checks, actions)은 예외 없이 한국어 문장이어야 합니다.

단, PostgreSQL, Kubernetes 같은 기술 용어와 고유명사는 번역하거나 음차 표기하지 말고 원문 영문 그대로 표기하세요. 문장의 나머지 부분은 한국어로 작성하세요.

failure_type은 반드시 아래 목록 중 하나의 값을 그대로(변형하거나 번역하지 말고) 사용하세요. 로그 내용이 목록의 어느 항목과도 명확히 일치하지 않으면 "Unknown"을 사용하세요.
{failure_types}

{{
  "failure_type": "위 목록 중 하나",
  "root_cause": "추정 원인 (2줄 이내)",
  "checks": ["확인해야 할 점검 항목 1", "점검 항목 2", "점검 항목 3"],
  "actions": ["권장 조치 1", "권장 조치 2"]
}}

로그:
{log_text}

checks와 actions의 각 항목은 완전한 한국어 문장으로 작성하세요. 셸 변수(예: $POD_NAME), 코드 조각, 한자를 문장 안에 섞지 마세요.

답변 시 다음을 반드시 지키세요:
1. 로그에 등장하는 구체적인 이름(Pod 이름, 이미지 경로, 도메인, 서비스명, 포트 번호 등)은 반드시 root_cause 또는 checks에서 그대로 인용하세요. 일반적인 표현("이미지", "서버")으로 뭉뚱그리지 마세요.
2. checks와 actions의 각 항목은 반드시 로그에 등장한 구체적인 내용에 근거해야 합니다. 로그에 없는 일반적인 조언은 포함하지 마세요.
3. 위 지시를 어기고 로그와 무관한 내용을 생성할 바에는, 해당 항목 수를 줄이더라도 정확한 항목만 포함하세요.

다시 한번 강조합니다: failure_type은 반드시 목록 중 하나여야 합니다. root_cause, checks, actions의 모든 값은 기술 용어/고유명사(예: PostgreSQL, Kubernetes)를 제외하고 반드시 한국어로만 작성하세요. 중국어나 불필요한 영어 단어, 한자, 셸 변수, 코드 조각을 섞지 마세요."""

CJK_PATTERN = re.compile(r"[一-鿿]")
CODE_PATTERN = re.compile(r'="|\$\w|\$\{')

FALLBACK_ROOT_CAUSE = "분석 실패 - 원본 로그 직접 확인 필요"
FALLBACK_ITEM = "자동 분석 실패 - 수동 확인 필요"

app = FastAPI(title="sLLM 장애 로그 분석 API")

_fallback_count = 0


class AnalyzeRequest(BaseModel):
    log_text: str
    cluster_name: str | None = None


class AnalyzeResponse(BaseModel):
    failure_type: str
    root_cause: str
    checks: list[str]
    actions: list[str]
    degraded: bool = False


def _is_bad_text(text: str) -> bool:
    return bool(CJK_PATTERN.search(text) or CODE_PATTERN.search(text))


def _contains_bad_pattern(value: object) -> bool:
    if isinstance(value, str):
        return _is_bad_text(value)
    if isinstance(value, list):
        return any(_contains_bad_pattern(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_bad_pattern(v) for v in value.values())
    return False


def _has_bad_pattern(result: AnalyzeResponse) -> bool:
    return _contains_bad_pattern(result.model_dump())


def _build_degraded_response(
    first: AnalyzeResponse, retry: AnalyzeResponse
) -> AnalyzeResponse:
    global _fallback_count
    _fallback_count += 1
    logger.warning(
        "degraded 폴백 발동 (누적 %d회) | first=%s | retry=%s",
        _fallback_count,
        first.model_dump(),
        retry.model_dump(),
    )
    return AnalyzeResponse(
        failure_type=first.failure_type,
        root_cause=FALLBACK_ROOT_CAUSE if _is_bad_text(retry.root_cause) else retry.root_cause,
        checks=[FALLBACK_ITEM if _is_bad_text(item) else item for item in retry.checks],
        actions=[FALLBACK_ITEM if _is_bad_text(item) else item for item in retry.actions],
        degraded=True,
    )


async def _call_ollama(prompt: str) -> AnalyzeResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama 응답 시간 초과.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama 오류 응답: {e.response.text}")

    raw_text = resp.json().get("response", "")

    try:
        parsed = json.loads(raw_text)
        return AnalyzeResponse(**parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error("모델 출력 파싱 실패: %s | raw=%s", e, raw_text)
        raise HTTPException(
            status_code=502,
            detail=f"모델 출력이 기대한 JSON 형식이 아닙니다: {e}",
        )


async def _notify_analysis_complete(
    cluster_name: str, result: AnalyzeResponse, response_time_sec: float
) -> None:
    if not LAMBDA_NOTIFY_URL:
        return

    payload = {
        "notification_type": "analysis_complete",
        "cluster_name": cluster_name,
        "failure_type": result.failure_type,
        "dr_action_needed": result.failure_type in _DR_ACTION_FAILURE_TYPES,
        "summary": result.root_cause,
        "response_time_sec": round(response_time_sec, 2),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=LAMBDA_NOTIFY_TIMEOUT_SECONDS) as client:
            resp = await client.post(LAMBDA_NOTIFY_URL, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError as e:
        logger.warning("Slack 알림(분석 완료) 연결 실패, 무시: %s", e)
    except httpx.TimeoutException as e:
        logger.warning("Slack 알림(분석 완료) 응답 시간 초과, 무시: %s", e)
    except httpx.HTTPStatusError as e:
        logger.warning("Slack 알림(분석 완료) 호출 실패(HTTP %s), 무시: %s", e.response.status_code, e)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    start = time.monotonic()

    prompt = PROMPT_TEMPLATE.format(
        log_text=req.log_text,
        failure_types=", ".join(FAILURE_TYPES),
    )

    first = await _call_ollama(prompt)

    if not _has_bad_pattern(first):
        result = first
    else:
        logger.warning("응답 필드에 한자·코드성 패턴 감지, 재시도: %s", first.model_dump())
        retry = await _call_ollama(prompt)
        result = retry if not _has_bad_pattern(retry) else _build_degraded_response(first, retry)

    elapsed = time.monotonic() - start
    await _notify_analysis_complete(req.cluster_name or "unknown", result, elapsed)

    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
