# Unified Dashboard Plan (3 Servers)

## Goal
Use ready-made tools to manage all three servers from one place without frequent SSH sessions.

## Current Servers
1. `2x5090`:
- Proxy
- Qwen3.5 9B
- Qwen3.5 4B
- UI

2. `2x3090`:
- Ministral-3-14B AWQ 8bit
- Qwen3 8B Embedding

3. `EPYC 9654`:
- Qwen3.5 9B Q4_K_M
- Qwen3.5 9B Q6_K

## Chosen Tools
1. `Portainer CE`:
- One UI for Docker environments on all servers
- Container lifecycle control (start/stop/restart)
- Logs and exec from one interface

2. `Grafana + Loki + Promtail`:
- Centralized logs from all servers
- Filtering by server/service/container
- Dashboards and alerting

## Deployment Layout
1. Main server (`2x5090`):
- Portainer Server
- Grafana
- Loki

2. Other servers (`2x3090`, `EPYC 9654`):
- Portainer Agent
- Promtail

## Labels / Metadata Convention
Add labels for easier filtering and dashboards:
- `server=2x5090|2x3090|epyc9654`
- `role=proxy|chat|embed|cpu`

## Expected Outcome
1. One dashboard with separate visibility per server.
2. Unified container management from one place.
3. Centralized logs for all model services.
4. Fewer SSH sessions and less manual operations.

## Next Session Plan
1. Deploy Portainer first (fastest operational win).
2. Deploy Loki + Promtail + Grafana.
3. Add basic alerts for model/container availability.
4. Validate daily workflow without multi-SSH monitoring.
