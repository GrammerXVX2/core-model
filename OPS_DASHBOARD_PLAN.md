# K3s Migration Plan (Agent-Driven Runbook)

## Purpose
Migrate from Docker Compose operations to a manageable multi-server K3s platform in controlled phases without breaking current model-serving workloads.

This file is the single source of truth for execution status and can be used directly by the agent.

## Operating Model
- Current Compose setup remains production fallback until final cutover.
- Migration is phased, reversible, and validated at each gate.
- No local Qwen3.5-4B restore on this server unless explicitly requested.

## Status Legend
- TODO: not started
- IN_PROGRESS: currently executing
- DONE: completed and validated
- BLOCKED: blocked, waiting for user action

## Environment Inventory (fill once)
- SERVER_MAIN_2x5090: artem-MS-7E48 (10.77.166.161)
- SERVER_GPU_2x3090: dewiar-super-server (10.77.166.156)
- SERVER_CPU_EPYC9654: AI-machine (10.77.163.177)
- SERVER_K3S_CONTROL_PLANE: AI-machine (10.77.163.177)
- ADMIN_USER_PER_SERVER: artem (main); TODO username for dewiar-super-server and AI-machine
- PRIVATE_NETWORK_CIDR: 10.77.0.0/16 (verify)
- K3S_VERSION_TARGET: v1.34.4+k3s1 (detected)

## Remote Execution Protocol
Because the agent has no direct guaranteed access to external servers by default, execution is done as follows:
1. Agent provides exact commands per server.
2. User runs commands on target server via SSH.
3. User sends command output back.
4. Agent validates output and advances status.

Optional mode:
- If your local workstation already has SSH key-based access to those servers, the agent can issue ssh commands from your local terminal session.
- This still uses your machine identity and keys; the agent does not hold independent server credentials.

## Phase 0 - Preconditions and Freeze
Status: IN_PROGRESS

Tasks:
- [ ] Confirm Compose stack is stable and documented rollback works.
- [ ] Snapshot current env files and compose manifests.
- [ ] Confirm model routing map and public aliases are current.
- [ ] Define maintenance windows and rollback owner.

Validation:
- [ ] /api/models healthy in current Compose.
- [ ] /api/chat and /api/embeddings smoke pass.

Execution notes:
- 2026-03-12: Phase 0 started.
- Waiting for server inventory outputs (hostname, OS, CPU/GPU, docker, network reachability).
- 2026-03-12: Inventory received for dewiar-super-server and AI-machine.
- 2026-03-12: Need one follow-up value from dewiar-super-server: free -h memory line output.

## Phase 1 - K3s Control Plane Bootstrap (No Production Traffic)
Status: DONE

Target:
- K3s server on main node (2x5090), no model traffic cutover yet.

Tasks:
- [ ] Install K3s server on main node.
- [ ] Secure kubeconfig and restrict access.
- [ ] Install kubectl + basic RBAC baseline.
- [ ] Enable metrics-server.

Suggested commands (run on SERVER_MAIN_2x5090):
```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="${K3S_VERSION_TARGET}" sh -s - server --write-kubeconfig-mode 644
sudo kubectl get nodes -o wide
sudo kubectl get pods -A
```

Validation:
- [x] Node Ready (control-plane node AI-machine).
- [x] System pods healthy (kube-system + installed platform pods running).

Execution notes:
- 2026-03-12: k3s binary already present on main server.
- Detected version: v1.34.4+k3s1.
- 2026-03-12: Control-plane confirmed on AI-machine (10.77.163.177).
- 2026-03-12: Existing platform components detected: ArgoCD, Argo Workflows, GPU Operator, Traefik, metrics-server.

## Phase 2 - Add Workers and Networking Baseline
Status: DONE

Tasks:
- [ ] Join 2x3090 as worker.
- [ ] Join EPYC9654 as worker.
- [ ] Add deterministic node labels.
- [ ] Define taints for GPU scheduling policy.

Suggested labels/taints:
- node-role=main|gpu|cpu
- gpu=true on GPU nodes
- taint examples for dedicated workloads

Validation:
- [x] All nodes Ready.
- [x] Labels and taints visible.

Execution notes:
- 2026-03-12: Worker artem-ms-7e48 is already joined but currently NotReady.
- 2026-03-12: dewiar-super-server worker not joined yet.
- 2026-03-12: artem-ms-7e48 agent repeatedly fails CA fetch via local LB (127.0.0.1:6444), control-plane endpoint https://10.77.163.177:6443.
- 2026-03-12: Node shows taints node.kubernetes.io/unreachable and stale heartbeat.
- Next: verify control-plane reachability from artem-ms-7e48, then force clean rejoin of k3s-agent.
- 2026-03-12: VPN split/bypass rules updated on artem-ms-7e48 to exclude K3s control-plane/corporate IPs.
- 2026-03-12: Connectivity to 10.77.163.177:6443 restored; k3s-agent restarted.
- 2026-03-12: artem-ms-7e48 status recovered to Ready.
- 2026-03-12: dewiar-super-server joined and Ready.
- 2026-03-12: Final node labels confirmed.
- Result: ai-machine role=control-plane gpu=false ready=True; artem-ms-7e48 role=gpu gpu=true ready=True; dewiar-super-server role=gpu gpu=true ready=True.

## Phase 3 - Platform Services (Observability + Ingress + Secrets)
Status: DONE

Tasks:
- [x] Install ingress (Traefik default or NGINX).
- [x] Install cert-manager.
- [x] Install Prometheus + Grafana.
- [x] Install Loki + Promtail.
- [x] Configure central dashboards and alerts.

Validation:
- [x] Metrics scrape works.
- [x] Logs visible per namespace/pod.
- [x] TLS issuance tested on one endpoint.

Execution notes:
- 2026-03-12: Phase 3 started.
- Existing components already observed in cluster: Traefik, metrics-server, ArgoCD, Argo Workflows, GPU Operator.
- Next: verify cert-manager, Prometheus/Grafana, and Loki/Promtail status; install missing pieces only.
- 2026-03-12: Audit results confirmed:
- Ingress present: Traefik LoadBalancer on 10.77.163.177,10.77.166.156,10.77.166.161.
- Missing components: cert-manager, Prometheus, Grafana, Loki, Promtail (not found in CRD/deploy).
- Operational issue found: argocd-repo-server pod is 0/1 Unknown.
- 2026-03-12: argocd-repo-server recovered after pod recreation (1/1 Running).
- 2026-03-12: cert-manager install attempt failed due to Helm using localhost:8080 (missing KUBECONFIG context); retry with explicit k3s kubeconfig.
- 2026-03-12: cert-manager installed successfully; all controller/webhook pods Running and CRDs present.
- 2026-03-12: kube-prometheus-stack installed successfully in namespace monitoring.
- 2026-03-12: Prometheus/Grafana/Alertmanager running; monitoring CRDs present.
- 2026-03-12: issue detected: kube-prom-stack-prometheus-node-exporter CrashLoopBackOff on dewiar-super-server.
- 2026-03-12: node-exporter conflict resolved on dewiar-super-server (host port 9100 conflict removed).
- 2026-03-12: all node-exporter pods Running across all nodes.
- 2026-03-12: helm release kube-prom-stack status is deployed.
- 2026-03-12: Loki deployed in logging namespace (single binary mode), status deployed.
- 2026-03-12: Promtail deployed as DaemonSet on all nodes, status deployed.
- 2026-03-12: logging services loki/loki-gateway available in cluster.
- 2026-03-12: Grafana Loki datasource added successfully (Alertmanager/Loki/Prometheus datasources visible).
- 2026-03-12: Prometheus rules API responds with status=success.
- Pending: apply and verify custom PrometheusRule object platform-basic-alerts.
- 2026-03-12: PrometheusRule platform-basic-alerts present and validated by prometheus-operator.
- 2026-03-12: TLS issuance test started in monitoring namespace; Certificate created and CertificateRequest emitted.
- 2026-03-12: Current state is expected transitional Issuing/DoesNotExist before final Secret appears.
- 2026-03-12: ClusterIssuer selfsigned-local created and Ready=True.
- 2026-03-12: Test certificate phase3-tls-test issued successfully (Ready=True), TLS secret phase3-tls-test-secret present.

## Phase 4 - GitOps and Deployment Hygiene
Status: DONE

Tasks:
- [ ] Install ArgoCD or Flux.
- [ ] Create repo structure for manifests/helm values.
- [ ] Add environments: dev/stage/prod overlays.
- [ ] Enforce image tag pinning and rollout strategy.

Validation:
- [x] One non-critical app reconciles successfully.
- [x] Rollback tested via Git revision.

Execution notes:
- 2026-03-12: Initial GitOps scaffold created in repository under k8s/base, k8s/overlays/{dev,stage,prod}, and k8s/argocd.
- Pending: commit/push scaffold and create ArgoCD Application core-model-dev for first reconciliation test.
- 2026-03-13: BLOCKED: core-model-dev application sync status is Unknown due to argocd-repo-server connectivity error (service endpoint 10.43.116.34:8081 connection refused).
- 2026-03-13: BLOCKER RESOLVED: core-model-dev reconciled successfully; status Synced/Healthy.
- 2026-03-13: ArgoCD operation completed successfully at revision b50fe0a81d843ba1199e1513623db36d044d3686.
- 2026-03-13: Destination namespace core-model auto-created and Active.
- 2026-03-13: Rollback drill passed: targetRevision switched to 1bf68fd60bf7e256c40a3122d9e4d182a670200e, then restored to main.
- 2026-03-13: Final post-rollback state confirmed as revision b50fe0a81d843ba1199e1513623db36d044d3686 with Sync=Synced and Health=Healthy.

## Phase 5 - Migrate Proxy Layer First
Status: IN_PROGRESS

Tasks:
- [ ] Deploy ollama-proxy in K3s namespace core-model.
- [ ] Port env settings from Compose to ConfigMap/Secret.
- [ ] Add readiness/liveness probes.
- [ ] Expose service via ingress/internal LB.

Validation:
- [ ] /api/models responds from K3s proxy.
- [ ] /metrics available and scraped.
- [ ] Latency not worse than Compose baseline by more than agreed threshold.

Execution notes:
- 2026-03-13: Phase 5 started.
- 2026-03-13: Added base manifests for ollama-proxy: ConfigMap, Secret template, Deployment, Service, and ServiceMonitor under k8s/base/proxy.
- 2026-03-13: Probes configured on /metrics; traffic cutover not performed.

## Phase 6 - GPU Runtime Enablement
Status: TODO

Tasks:
- [ ] Install NVIDIA container toolkit on GPU nodes.
- [ ] Install NVIDIA device plugin (or GPU operator).
- [ ] Verify GPU allocatable resources in k8s.

Validation:
- [ ] Test pod can see GPU.
- [ ] Scheduling to intended GPU node works.

## Phase 7 - Migrate Model Workloads Incrementally
Status: TODO

Order:
1. Non-critical model route.
2. One primary embedding route.
3. One primary chat route.
4. Remaining workloads.

Tasks:
- [ ] Deploy first model workload as StatefulSet/Deployment.
- [ ] Apply resource requests/limits and node selectors.
- [ ] Add PodDisruptionBudget and anti-affinity where useful.
- [ ] Benchmark against Compose baseline.

Validation:
- [ ] Model parity checks pass.
- [ ] No regression in error rate/timeout profile.

## Phase 8 - Controlled Cutover
Status: TODO

Tasks:
- [ ] Route partial traffic to K3s proxy (canary).
- [ ] Observe metrics/logs for agreed window.
- [ ] Increase traffic gradually to 100%.
- [ ] Keep Compose hot-standby until stability window closes.

Validation:
- [ ] Stable SLO in full traffic window.
- [ ] On-call rollback runbook tested.

## Phase 9 - Decommission Legacy Compose Paths
Status: TODO

Tasks:
- [ ] Disable legacy services not required.
- [ ] Archive compose configs and rollback notes.
- [ ] Keep minimal emergency fallback scripts.

Validation:
- [ ] No production dependency on old Compose path.

## Rollback Rules (Always Active)
- Any failed gate returns to previous stable phase.
- Do not continue forward with unresolved BLOCKED items.
- Keep last known-good Compose deployment ready until Phase 9 DONE.

## Agent Execution Queue
1. Phase 0 freeze checklist
2. Phase 1 bootstrap
3. Phase 2 workers + labels/taints
4. Phase 3 observability
5. Phase 5 proxy migration (before heavy model migration)
6. Phase 6 GPU enablement
7. Phase 7 model workloads
8. Phase 8 cutover
9. Phase 9 decommission

## Session Log Template
- Date:
- Phase:
- Commands run:
- Output summary:
- Decision:
- Next step:
- Status update:





K10afe5c5d9cc43f8aeccf022f88893b0a3fdeda809322055f602383b54410255e6::server:5c50c44324909fad527c8daa4b702229
