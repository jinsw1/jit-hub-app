# 시연용 장애 로그 (jithub_test_demo_logs.log)

이 디렉토리의 `jithub_test_demo_logs.log`는 sLLM 장애 분석 파이프라인 시연에 사용할
로그 원문입니다. 시연용 pod가 `/var/log/jithub_test.log`에 이 내용을 write하면,
promtail → Loki → poller → `/analyze` 순으로 흘러가는 전체 파이프라인을 검증할 수 있습니다.

## 포함된 장애 유형 (3종)
- OOMKilled
- DiskPressure
- NetworkTimeout
