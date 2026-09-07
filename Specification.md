# Specification

# 1. Introduction

### 1.1. Purpose

(현재 기술 동향) 자율주행에서 센서 데이터를 바탕으로 데이터를 가공하여 ego 주변 상황을 파악하고 경로를 계획하는 연구가 계속되고 있다. 최근 learning-based 경로 계획을 통해 ego가 놓인 주변의 복합적이고 복잡한 상황을 처리하려는 노력이 계속되고 있으며 주로 Transformer 모델을 사용하여 주변 object 간의 관계, object와 map의 관계를 이해하여 ego가 놓인 주변 환경의 conext를 잘 이해할 수 있으며 예상 주행 경로를 추론하여 오브젝트들의 움직임을 예상하려고 하고 있다.

(문제) 반면 Transformer는 통계학적 모델을 기반으로 하기 때문에 물리적인 현상을 이해하기 위해 상대적으로 큰 모델 사이즈를 요구하여 가려진 물체의 경로를 예상하는 데에 불리하다. 뿐만 아니라 Transformer는 시간축으로의 장기 기억 보존에 불리하며 공간 및 연산 복잡도로 인하여 자율주행과 같은 엣지 컴퓨팅 환경에서 real-time 성능을 확보하는 데에 불리하다. 또한, 기존 Transformer 기반 방식은 프레임 간 센서 폐색(Occlusion)이 발생했을 때 이전 시점의 운동 관성을 연속성 있게 유지하지 못하고 예측 궤적이 급격히 불연속해지는 한계가 있다.

(해결 방안 제시) 이 연구에서는 수학 모델을 기반으로 하는 Mamba를 채용하여 물리 현상에 대해 inductive bias를 주어 효율적이고 효과적으로 이해할 수 있도록 하며, 뒤이어 하이브리드 구조로 Transformer Decoder와 직렬로 연결하여 각 오브젝트들의 운동을 주변 Map 환경과 융합하여 커브길에 인한 곡선 주행, 신호 대기로 인한 정차 예상과 같이 오브젝트의 예상되는 움직임을 한 차원 더 깊게 이해할 수 있도록 한다.

# 2. Overall Description

## 2.1. Scope

 이 연구는 ego 주변의 오브젝트 정보를 가지고 오브젝트의 이동에 대한 물리적인 현상을 효과적으로 이해하며, 이것을 Map 정보와 융합하여 예상되는 경로를 도출한다. 특히, 가려진 물체에 대해 이전 데이터를 기반으로 미래의 위치를 예측할 수 있도록 한다. 예를 들어 가려진 물체가 곡선 레인 위에 있었다면 언더스티어 현상 없이 해당 곡선을 잘 따라가도록 한다. 뿐만 아니라 Mamba를 통해 확보된 물리적 운동 벡터 f_t는 Transformer Decoder의 Cross-Attention 과정에서 신호등 상태, 정지선 정보 등 공간 맥락(Map Context)과 결합되어 '정지선 앞 감속' 또는 '주황불 통과/정지 판단'과 같은 고차원 주행 의도(Intention)로 진화한다.

 반면 이 연구는 영상에서부터 시작하는 영상 처리, 3D 공간 표현과 같은 앞단의 과정은 직접 수행하지 않으며, 단순히 annotation을 가지고 운동을 이해하고자 한다.

## 2.2. Assumption

 카메라 센서의 한계는 다양하게 있지만 이 연구에서는 오직 가려짐 현상에만 관심을 가지며, 노이즈와 같은 기타 한계는 없이 완전하다고 가정한다.

 또한 오브젝트의 탐지·추적(re-identification)은 이미 완료되어 시나리오 전체에서 안정적인 track id가 주어진다고 가정한다(Waymo Open Motion Dataset이 제공하는 오라클 id 사용). 즉 이 연구는 "누가 같은 오브젝트인가"를 다시 푸는 tracking 문제는 다루지 않고, id가 주어졌을 때 그 물리적 상태를 추정/예측하는 문제에 집중한다.

## 2.3 Environment
 runpod에서 학습할 것이기 때문에 dockerfile을 작성해야 한다. HW는 RTX 4090을 사용할 것으로 이것을 고려하여 OS 및 라이브러리 버전을 맞춘다.
 Waymo 데이터셋은 Network Volume에 올리고 배치 1이 학습되는 동안 배치 2의 데이터를 불러오는 방식으로 한다. 이 프로젝트를 본격적으로 시작할 때 Network Volume에 데이터셋을 다운로드하나 보관 비용이 한 달에 35$ 수준으로 예상되므로 프로젝트를 1-2주 이내에 집약적으로 수행하고 결과를 확인하였으면 바로 Network Volume에서 424GB의 그 데이터셋을 지우는 방식으로 한다.

# 3. Specification Requirements

## 3.1. IO Specification

#### 3.1.1. Selected Attributes

FOV

- 360 degree surround view, 100m patch

Object (Waymo Open Motion Dataset의 `Track` 기준으로 재정의)

- Track ID (Waymo `id`. 시나리오 내에서만 유효한 persistent id — [2.2 Assumption](#22-assumption) 참고)
- class (Waymo `object_type`: VEHICLE / PEDESTRIAN / CYCLIST. nuScenes의 세분류(23종)와 달리 3종뿐이므로 클래스 관련 서브 태스크의 스코프도 이에 맞춘다)
- x, y, z (ego-centric frame). Waymo global frame 좌표를 **t0 시점의 ego(SDC) pose 하나로 윈도우 전체를 1회 변환**한다. 매 스텝마다 그 시점 ego pose로 재중심(re-center)하지 않는다 — 재중심하면 준관성(quasi-inertial) 좌표계가 깨져 "등속 운동 ⇒ 일정 위치 변화"라는 관성 해석이 성립하지 않기 때문이다.
- theta (t0 ego frame 기준)
- vx, vy (t0 ego frame 기준 속도. Waymo가 주는 global velocity에서 ego 속도를 빼 상대속도로 변환)
- W, H, L (length, width, height)
- valid (해당 timestep에 실제 관측되었는지를 나타내는 binary flag. nuScenes의 4단계 visibility(v0-40 ~ v80-100)와 달리 Waymo는 이진값만 제공하므로, "가려짐 정도"가 아니라 "관측 유무"로 취급한다. **invalid 스텝의 x, y, z 등은 (0, 0, 0)으로 채워져 있어 위치 GT가 없다** — [3.3.2 Occlusion 프로토콜](#332-occlusion-학습평가-프로토콜) 참고)

Map (Waymo `map_features` / `dynamic_map_states` 기준으로 재정의)

- lane (entry_lanes / exit_lanes 토폴로지로 lane connector 역할까지 포함. lane_type: FREEWAY / SURFACE_STREET / BIKE_LANE)
- road_line (차선·도로 경계 구분선. type으로 lane divider·road divider 성격을 구분 — 예: 흰색 계열은 lane divider, 황색/이중선 계열은 road divider에 대응)
- road_edge (도로의 물리적 경계. Waymo에는 nuScenes의 drivable_area 같은 명시적 폴리곤이 없어 이것으로 주행 가능 영역을 근사)
- crosswalk (ped_crossing에 대응)
- stop_sign (표지판 정지 지점, point feature)
- dynamic_map_states → lane_state (신호등 상태: STOP / CAUTION / GO / ARROW_* 등을 **매 timestep마다** 제공하며, stop_point로 정지 위치도 함께 포함 — nuScenes의 정적 traffic_light 기하 + stop_line 폴리곤 역할을 이 하나가 대체하고, 오히려 nuScenes에는 없던 실시간 신호 상태까지 준다)

#### 3.1.2. Output Schema

- GT
    - class
    - x, y, z
    - theta
    - W, H, L
    - vx, vy
- Multi-modal Trajectory + Probability
    - 미래 경로를 단 1개가 아니라 K개(예: K=6)의 대표 모드(Mode) 경로로 출력하고, 각 모드가 일어날 확률(Probability) p_k을 함께 예측하는 방식
- (추후 업데이트 예정) Gaussian Mixture Model
    - 매 타임스텝 t마다 미래 위치를 단순 좌표 $(x,y)$가 아니라 2D 가우시안 분포의 파라미터 (평균 μ_x, μ_y, 표준편차 σx, σy, 상관계수 ρ)로 출력하는 방식

#### 3.1.3. Target

- **Δt = 0.1s (10Hz), 서브샘플링하지 않는다.** Mamba는 시퀀스 길이에 O(L)이라 91스텝은 부담이 아니며, occlusion 시 발생하는 궤적 불연속(jerk)은 고주파 현상이므로 다운샘플하면 정작 측정하려는 현상이 가려진다.
- WOMD 한 시나리오 = 91스텝 × 10Hz = 9.1초. 원본에는 **모든 트랙의 91스텝 전체가 GT로 들어있다**(x, y, z, heading, vx, vy, valid).
- "과거 관측 구간 / 미래 예측 구간"의 분할은 리더보드 채점 규칙일 뿐 학습 제약이 아니다. 본 연구의 관심사(가림 관성)는 tracking·filtering 쪽이므로, 리더보드의 1.1초/8초 대신 **과거 ~4초 / 미래 ~5초** 정도로 균형을 잡아 per-step 감독 신호를 늘린다.
  - 학습: sliding t0 (한 시나리오에서 t0를 옮겨가며 여러 학습 샘플 생성, per-step 신호 최대화)
  - 평가: t0 = step 10 고정 (`current_time_index`), WOMD 리더보드 수치와 비교 가능하게 정렬
- Target: t0+1, …, t0+N (N = 미래 구간 스텝 수, 기본 50 = 5초) 시점의 실제 3D Box 중심 좌표 $(x,\ y)$ 궤적. Baseline 대비 안정화되면 8초(80스텝)로 확장 검토.

## 3.2. Functional Requirements

#### 3.2.1. Mamba

- Goal: 과거의 운동을 가지고 가려진 물체도 마치 관측된 것처럼 현재 상태(t)를 복원/추정
- Input
    - $x_{t-1}^{corr}$ — **정의: $x_{t}^{corr} := f_{t}$** (3.2.3 MLP 출력). 직전 스텝에서 보정된 feature를 그대로 되먹인다. Transformer Decoder 출력 $f_t^{updated}$는 **되먹임하지 않는다**(map 추론이 재귀 상태를 누적 오염시키는 것 방지).
    - 초기화: $h_{-1}=0$ (또는 learned init). 첫 스텝(t=0)은 $f_{-1}$이 없으므로 $x_0^{raw}$ 임베딩을 직접 첫 입력으로 사용.
- Do
    - $\text{Mamba}(x_{t-1}^{corr},\ h_{t-1})$
    - 이 재귀(**루프 1**)는 **과거 관측 구간(t = 0 … t0)에서만** 돈다. 미래는 Head가 one-shot 디코딩한다(3.2.5).
    - 과거 운동 기억 $h_{t-1}$ 과 같은 시간에서의 correction된 object 데이터 $x_{t-1}^{corr}$와 결합하여 현재 운동 기억 $h_t$과 오브젝트들의 상태 추정값인 $y_t$를 생성
    - ~~중간에 건물이나 다른 차에 가려져 observation(GT bbox)이 끊기더라도 해당 Mamba의 State Space Model 특성을 이용해 가려지기 전 속도/방향(관성)을 유지하며 Hidden State를 업데이트함.~~
    - (학습 시)Mamba가 관성을 잘 학습하도록, 가려지지 않은 구간에서는 y_t 자체만으로도 $x_t^{raw}$를 예측하게 하는 Aux Loss($L_{aux}$)를 추가. 더불어 $y_t$가 $x_{t+1}^{raw}$를 예측하게 하는 next-step loss($L_{next}$)로 "현재 상태 → 다음 상태" 물리 전이를 매 스텝 감독한다. (전체 loss는 3.2.6 참고)
- Output
    - 과거 운동 기억 $h_t$
    - 상태 추정값 $y_t$ ~~(디코딩 시 object slot (서브 태스크로 헤드를 달아 오브젝트의 GT 학습 및 추론 가능)~~

#### 3.2.2. Embedding

- Goal: 센서로부터 인지된 데이터를 임베딩(이 연구에서는 인지 과정을 거쳤다고 가정)
- Input
    - Object GT Annotation
    - Map GT Annotation
- Do
    - $\text{Embedding(Object)}$: 3.1.1 Object 속성(x, y, z, theta, W, H, L, vx, vy, class, valid)을 슬롯별 토큰으로 임베딩.
    - $\text{Embedding(Map)}$ — **정적/동적 분리**:
        - **정적 지오메트리**(lane, road_line, road_edge, crosswalk, stop_sign): 9.1초 내내 불변. **시나리오당 1회만** 인코딩하고 캐시한다. polyline은 VectorNet 방식(polyline을 세그먼트로 쪼개 세그먼트별 MLP → max-pool → polyline당 토큰 1개)으로 벡터화.
        - **동적 신호 상태**(dynamic_map_states.lane_state): 매 timestep 바뀌므로 스텝별로 인코딩. 상태 enum은 {UNKNOWN, ARROW_STOP, ARROW_CAUTION, ARROW_GO, STOP, CAUTION, GO, FLASHING_STOP, FLASHING_CAUTION} — **UNKNOWN(신호 인지 실패) 케이스를 명시적으로 포함**. 매 스텝 이 작은 임베딩을 해당 lane의 정적 토큰에 더해 $x_t^{map}$을 구성.
        - 학습·추론 모두 동일 구조: 정적 map은 새 지역 진입 시 1회 인코딩 후 캐시, 매 프레임에는 동적 신호만 갱신. (안 바뀌는 map을 0.1초마다 재인코딩하는 것은 real-time 목표와 배치됨)
    - **Slot Assignment**: 시나리오마다 가변 개수인 오브젝트를 고정 300개 슬롯에 배치.
        - Waymo는 시나리오 전체에서 유지되는 persistent track id를 이미 제공하므로(오라클), re-identification 없이 **id → slot index를 시나리오 단위로 한 번만 결정**하고 전체 타임스텝에 고정한다.
        - 배정 우선순위: ① WOMD가 지정한 `tracks_to_predict`(예측 challenge용으로 큐레이션된 관심 오브젝트) 우선 배정 → ② 남은 슬롯은 ego 기준 100m 패치 내 오브젝트를 첫 등장 시각 순서로 배정.
        - 오브젝트가 300개보다 적으면 남는 슬롯은 학습되는 "empty slot" 임베딩으로 채우고 loss에서 마스킹, 300개보다 많으면(희귀 케이스) 우선순위 밖 오브젝트를 드롭.
        - 슬롯이 프레임마다 재정렬되지 않으므로 Mamba의 hidden state $h_t$가 슬롯 전환 없이 한 물리적 오브젝트에 안정적으로 귀속된다.
- Output
    - $x_{t}^{raw}$ (Object Token. 고정 길이 300의 슬롯 텐서로 배치됨)
    - $x_{t}^{map}$(Map Token)

#### 3.2.3. MLP

- Goal
    - 과거 기억 기반 예상 오브젝트 위치 $y_t$와 실제 센서로부터 들어온 $x_{t}^{raw}$를 결합하여, 관측 여부와 무관하게 항상 "현재 그 오브젝트의 상태"를 온전히 담은 하나의 Feature로 보정
- Input
    - $y_t$
    - $x_{t}^{raw}$
    - $valid_t$ (해당 오브젝트가 t 시점에 실제 관측되었는지를 나타내는 binary flag)
- Do
    - $\text{MLP}(y_t,\ x_{t}^{raw},\ valid_t)$
    - $valid_t$를 명시적으로 넣는 이유: Waymo는 관측되지 않은 timestep의 $x_t^{raw}$를 0으로 채워 제공하므로, $valid_t$ 없이는 "실제로 그 위치(원점)에 있다"와 "관측 자체가 없다"를 네트워크가 구분할 수 없다. $valid_t$가 이 둘을 구분하는 유일한 신호.
    - 학습된 게이트(Learned Gate)로 동작: $valid_t=1$이면 $x_t^{raw}$의 비중을 높게, $valid_t=0$이면 $y_t$(Mamba의 예측)만으로 상태를 구성하도록 MLP가 학습된다. 즉 하드 스위치가 아니라 **관측 신뢰도에 따라 두 입력을 블렌딩하는 소프트한 보정(Soft Correction)**에 가깝다 — 가려짐 직전/직후, 관측이 시작·종료되는 전환 구간의 궤적을 부드럽게 이어주는 효과를 기대.
- Output
    - Feature $f_{t}$ — 보정된 상태 추정값(Corrected State Estimate). 관측 여부와 무관하게 해당 오브젝트의 현재 위치/속도/관성 정보를 항상 온전히 담고 있는 Feature Vector. 두 곳으로 간다: ① Transformer Decoder의 입력(Q) ② **다음 스텝 Mamba의 입력 $x_t^{corr}$로 되먹임**(3.2.1).

#### 3.2.4. Transformer Decoder

- Goal
    - object와 map간 상호 연결을 통해 오브젝트가 가려진 경우 기존 운동량을 가지고 차로의 모양을 따라 주행시키거나 빨간불 && 정지선이 있는 경우 차량이 정지선 앞에 멈출 것이라는 것을 좀 더 잘 알 수 있도록 함. (공간 탐색)
- Input
    - Q: $f_t$
    - K, V: $f_t$, $x_{t}^{map}$
- Do
    - $\text{Cross-Attention(Q, K, V)}$
    - FFN
- Output
    - $f_{t}^{updated}$ — Head로만 전달되며, **다음 스텝 Mamba로 되먹이지 않는다**(3.2.1 참고).
    - "단순히 v의 속도로 직진 중인 차"에서 "앞에 정지선과 빨간불이 있어서 곧 감속/정지할 차"라는 고차원 공간 맥락(Spatial Context) 및 주행 의도(Intention)가 입혀진 Feature 완성. (서브 태스크로 헤드를 달아 오브젝트의 ‘더 정밀한’ GT 학습 및 추론 가능)

#### 3.2.5. Head

- Goal: Object 각각 현재 위치 및 경로 예측 결과를 직접 열어서 확인함.
- Input: $f_{t}^{updated}$
- Do
    - $\text{GTHead}$ — **과거 관측 구간의 매 스텝**에 적용, 그 시점 box 상태를 복원(occlusion 스텝 포함). map 융합 이전 $f_t$와 이후 $f_t^{updated}$ 양쪽에 각각 달아 비교(3.2.6 $L_{recon}$).
    - $\text{TrajectoryHead}(f_{t0}^{updated})$ — **마지막 관측 시점 t0의 feature 하나**로부터 미래 전 구간(t0+1 … t0+N) × K개 모드를 **one-shot 회귀**(자기회귀 rollout 아님).
- Output
    - GT 추론 (과거 구간, per-step)
        - class
        - $(x,\ y,\ z)$
        - $\theta$
        - $(W,\ H,\ L)$
        - $(v_x,\ v_y)$
    - K개의 대표 궤적 및 확률 (Multi-modality 해결)
        - K=6개의 대표 경로(앵커/모드)를 두어 직진, 좌회전, 우회전, 감속 등 분기되는 거시적인 주행 의도를 따로따로 커버함.
        - (학습 시) Winner-Take-All: K개 모드 중 GT와 가장 가까운 1개 궤적에만 회귀 Loss를 흘리고(WTA/Variety Loss), 별도의 K-way 분류 헤드가 "어느 모드가 winner인가"를 cross-entropy로 학습. 추론 시 이 확률로 6개 궤적을 랭킹. (전체 loss는 3.2.6)
    - (추후 업데이트 예정) 각 경로 타임스텝의 가우시안 분포 σ (위치 불확실성 해결)
        - k번째 경로 내에서 t0+1 … t0+N으로 미래로 갈수록 좌표 예측이 얼마나 불확실한지(위치 오차 범위)를 표준편차 σx,σy로 나타냄. 먼 미래 타임스텝일수록 σ 값이 자동으로 커지면서 위치 불확실성을 가우시안 타원(Uncertainty Ellipse) 형태로 나타내게 됨

#### 3.2.6. Loss

전체 손실은 아래의 가중합. $\lambda$는 실험으로 결정(시작값: 회귀 텀 1.0, 분류 텀 0.5, aux/next 0.5).

$$L_{total} = \lambda_1 L_{aux} + \lambda_2 L_{next} + \lambda_3 L_{recon}(f_t) + \lambda_4 L_{recon}(f_t^{updated}) + \lambda_5 L_{traj}^{WTA} + \lambda_6 L_{mode}^{CE}$$

| 텀 | 붙는 곳 | 구간 | 정답 | 역할 |
|---|---|---|---|---|
| $L_{aux}$ | $y_t$ (Mamba 직후) | 과거, valid=1 | $x_t^{raw}$ | Mamba 원 상태가 관측을 재현(관성) |
| $L_{next}$ | $y_t$ | 과거, valid=1 | $x_{t+1}^{raw}$ | "현재→다음" 물리 전이 감독 |
| $L_{recon}(f_t)$ | GTHead($f_t$) | 과거 전체 (**synthetic 가림 스텝 포함**) | GT box | Mamba+MLP 단독의 가림 복원력 격리 |
| $L_{recon}(f_t^{updated})$ | GTHead($f_t^{updated}$) | 과거 전체 (synthetic 가림 스텝 포함) | GT box | map 융합 후 복원력 (융합 기여도 = 두 텀 차이) |
| $L_{traj}^{WTA}$ | TrajectoryHead | 미래 | 미래 궤적 | winner 모드만 회귀 |
| $L_{mode}^{CE}$ | 분류 헤드 | 미래 | winner 인덱스 | K개 모드 확률 |

- **마스킹 규칙**
    - empty slot: 모든 텀에서 제외
    - $L_{aux}$, $L_{next}$: valid=1 스텝만 (자연 invalid 스텝은 GT가 0이라 제외)
    - $L_{recon}$: **인위적으로 마스킹한 스텝**에서만 계산 (자연 invalid 스텝은 위치 GT 없음). 원래 valid=1이던 스텝을 가렸으므로 정답은 그대로 존재.
    - $L_{traj}$, $L_{mode}$: focal agent(`tracks_to_predict`) 중심, 미래 valid=1 스텝만. 그 외 트랙도 미래가 valid하면 포함 가능.

## 3.3. Dataset & Training Protocol

#### 3.3.1. Dataset

- Waymo Open Motion Dataset (scenario protos, TFRecord). 한 시나리오 = 91스텝 × 10Hz = 9.1초.
- 학습셋 = 1000 shard ≈ 48만 시나리오, 압축 ~424GB. **1~2주·단일 4090** 제약상 전량 학습은 비현실적이므로 **100~200 shard 서브셋**으로 시작하고, sliding t0로 시나리오당 여러 샘플을 뽑아 유효 학습량을 늘린다.
- Split: 공식 training shard에서 자체적으로 train/val 분리(예: 마지막 10 shard를 val). 공식 validation/test는 최종 참고 지표용.
- 전처리 캐시: TFRecord 파싱 + t0 기준 좌표 변환 + 정적 map 벡터화 결과를 디스크에 캐시(매 epoch 재파싱 방지).

#### 3.3.2. Occlusion 학습·평가 프로토콜

- **문제**: Waymo는 invalid(가림) 스텝의 위치를 (0,0,0)으로 채우고 별도 GT를 주지 않는다. 따라서 자연 발생한 가림 구간 자체로는 복원을 직접 감독·채점할 수 없다.
- **Synthetic Occlusion (주력 학습 신호)**: valid=1로 이어진 깨끗한 트랙 구간에서 일부 스텝을 **입력에서만** 가린다(valid→0, 좌표→0). 정답은 원래 값 그대로 유지 → 그 구간에 $L_{recon}$을 직접 건다.
    - 가림 통계를 자연 분포에 맞춘다: 내부 gap 보유 트랙 ≈ 전체의 31.6%(focal agent 22.5%), gap 길이 median 0.3s / max 2.2s. 균등 랜덤이 아니라 이 분포에서 gap 길이·개수를 샘플링하고, 멀거나 작거나 가려지기 쉬운 오브젝트에 가중.
- **자연 Occlusion (실검증)**: 실제 gap을 가진 트랙에 대해 ① 재등장 스텝의 위치 오차, ② gap 구간 궤적 연속성/jerk, ③ 그 트랙의 미래 예측 minADE/FDE를 측정. → 3.4.3 Occlusion Slice의 구체 정의.

## 3.4. Performance Requirements

 소속 연구실/랩 없이 개인이 RunPod 단발성 환경에서 진행하는 프로젝트이므로, 외부 SOTA 논문의 공식 레포를 재구현해 동일 조건에서 맞대결하는 방식은 현실적이지 않다고 판단하고 다음과 같이 검증 전략을 잡는다.

#### 3.4.1. 절대 성능 참고 (Reference, 공정 비교 아님)

- WOMD Motion Prediction 공식 리더보드/논문(MTR, Wayformer 등)에 보고된 minADE / minFDE / Miss Rate 수치를 **참고선**으로만 인용한다.
- 전처리, split, 학습 규모(GPU-시간)가 다르므로 "동일 조건 비교"가 아니라 "동일 데이터셋에서 이 정도 난이도다"라는 맥락 제공용임을 리포트에 명시한다.

#### 3.4.2. 내부 Ablation 비교 (핵심 검증 — 직접 통제 가능)

 이 연구의 주장("Mamba 기반 관성 유지가 occlusion 상황에서 Transformer 단독보다 낫다")을 검증하는 가장 정확하고 실행 가능한 방법은 남의 레포가 아니라 **내가 만든 모델의 모듈을 갈아끼우며 비교**하는 것이다.

- Baseline A — Constant Velocity/Turn 외삽: 학습 없이 마지막 관측 속도로 직선/등회전 외삽. 모든 trajectory prediction 연구의 표준 sanity baseline.
- Baseline B — Transformer-only: Mamba를 제거하고 동일한 Embedding / MLP / Transformer Decoder / Head 구조에 일반 temporal encoder(positional encoding + self-attention, 혹은 GRU)로 대체. **이 연구가 반박하려는 대상 그 자체**이므로 가장 중요한 비교군.
- Baseline C (optional) — GRU/LSTM Hybrid: Mamba 자리에 GRU/LSTM을 넣어, "순환 구조 일반의 효과"와 "Mamba 특유의 효과"를 분리.
- Full Model: 제안 구조(Mamba + Transformer Decoder).

#### 3.4.3. 지표

- minADE_k, minFDE_k, Miss Rate@2m (K=6) — WOMD 공식 지표와 동일한 정의로 자체 val split에서 측정.
- **Occlusion Slice 지표(차별점, [3.3.2](#332-occlusion-학습평가-프로토콜) 참고)**:
    - Synthetic 가림 셋: 마스킹한 구간의 위치 복원 오차(ADE/FDE)를 직접 측정 (정답을 알고 있음).
    - 자연 가림 셋: `valid=False`가 내부에 있었던 트랙만 모아 ① 재등장 위치 오차 ② gap 구간 궤적 연속성/jerk ③ 미래 예측 minADE/FDE 재계산.
    - 공개 리더보드가 보통 보고하지 않는 지표이므로, 이 연구의 핵심 기여를 가장 직접적으로 보여주는 결과가 된다.

#### 3.4.4. 정리

 "SOTA를 이겼다"가 아니라 "동일 조건에서 Mamba 모듈이 있고 없고의 차이"를 보이는 것이 개인 프로젝트/포트폴리오 관점에서 현실적이며, 방법론적으로도 이 연구가 실제로 주장하는 바(메커니즘 단위 기여)에 더 정확히 부합한다.