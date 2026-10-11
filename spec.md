0. 한 줄 요약

Mel은 타건 하나당 로그멜 텐서 1개 + 그 타건의 onset 시각을 준다. AFE는 그걸 그대로 CNN에 넣는다. 규격은 학습·배포에서 완전히 동일해야 한다.

1. 전제 (확정됨)
세그멘테이션/온셋 검출 = Mel 담당 소유 (확정). Mel이 오디오 → 온셋검출 → 100ms 윈도잉 → 로그멜까지 한 번에 처리하고, AFE에는 이미 타건 단위로 잘린 로그멜 + onset_s를 준다.
참고 구현: 기존 src/segment.py(온셋검출 밴드 400~12000Hz, 최소간격 60ms, 앞5/뒤95ms)를 Mel이 참고.
학습 때는 키로거 onset_s가 정답으로 있으니 그 시각에서 자르면 됨. 실시간에선 소리로 온셋을 검출.
마이크 48kHz 지원 = 확정(지원 예정).
온디바이스(ESP32-S3)와 오프라인(학습, Windows)에서 같은 추출 파라미터·같은 구현을 써야 한다. 파라미터가 다르면 학습-배포 도메인 시프트로 정확도가 무너진다.
2. 추출 파라미터 (고정값 — 이대로 뽑아 줄 것)

AFE 기존 코드(src/features.py) 기준. 이 값들로 고정:

sample_rate = 48000 (48kHz). ← ESP32 I2S 마이크가 지원하는지 Mel이 확인. 불가하면 셋이 합의해 통일 후 재고정.
n_fft = 1024
win = 8 ms (= 384 samples @48k)
hop = 2 ms (= 96 samples @48k)
n_mels = 64
fmin = 200 Hz, fmax = 20000 Hz
클립 길이 = 100 ms (온셋 기준 앞 5ms + 뒤 95ms; press+release 포함)
시간축 프레임 = 48 로 고정 (길면 center-crop, 짧으면 edge-pad)
값 변환 = log(mel + 1e-6) 후 per-clip z-score 정규화 (클립별 평균0/표준편차1)
출력 dtype = float32, shape = (1, 64, 48) ← 이게 CNN 입력 shape

이 파라미터 묶음에 버전/해시를 붙여서 함께 넘길 것(mel_spec_version, param_hash). 학습·배포가 같은 해시인지 AFE가 검증한다.

3. 런타임 데이터 포맷 (실시간/추론 시)

타건이 검출될 때마다 아래 레코드 1개를 준다(스트림) 또는 세션 배열로 준다(배치).

json
{
  "idx": 0,                       // 타건 순번(0부터, 시간순, 누락 없이 연속)
  "onset_s": 1.234,               // 이 타건의 시작 시각(초). Mel/세그멘테이션이 부여
  "logmel": "<float32[64][48]>"   // per-clip z-score까지 끝난 로그멜. shape (n_mels=64, frames=48)
}

세션(배치) 파일이면 헤더 + 레코드 배열:

json
{
  "mel_spec_version": "1.0",
  "param_hash": "ab12cd34",
  "sample_rate": 48000,
  "n_mels": 64,
  "frames": 48,
  "session_id": "p1_near_s0",
  "records": [ { "idx": 0, "onset_s": 1.234, "logmel": "..." }, ... ]
}

규칙:

logmel은 정규화까지 끝난 최종 텐서. AFE는 그대로 CNN에 넣는다(추가 전처리 안 함).
idx는 시간순·연속. 특수키(Space/Backspace/Shift/Caps)도 타건이면 레코드로 포함한다 (Q2에 따라 AFE가 그 소리를 특수토큰으로 분류하므로, Mel은 자모/특수키 구분 없이 "타건이면 다" 준다).
onset_s는 AFE가 계산하지 않고 이 값을 그대로 LM으로 전달한다. 타이핑 간격(gap)은 AFE가 onset들로 계산.

전달 방식(스트림 vs 파일)과 직렬화(JSON에 배열 인라인 vs 바이너리 .npy/링버퍼)는 Mel 편한 쪽으로 정하되, shape·dtype·정규화 규격만 위와 동일하면 된다.

4. 학습용 데이터 (오프라인)

AFE 학습은 (로그멜, 라벨) 쌍이 필요하다. 라벨은 이미 data/metadata.csv(clip_id, jamo, keytype, ...)에 있다. Mel은 아래 둘 중 하나를 제공:

방식 A (권장): precomputed 로그멜 — 각 학습 클립의 로그멜을 clip_id 키로 뽑아서 준다 (예: mel/<clip_id>.npy, shape (64,48)). AFE가 metadata의 라벨과 join.
방식 B: 공유 추출기 — 2장 파라미터로 고정된 추출 함수/라이브러리를 준다. AFE가 오프라인에서 동일 적용.

어느 쪽이든 2장 파라미터·해시가 배포와 동일해야 한다. (A가 실수 여지가 적어 권장.)

5. 온디바이스 양자화 경계 (ESP32-S3, TFLite Micro int8)
CNN은 int8로 배포된다(내가 Q1에서 확정). 입력도 int8로 양자화된다.
그래서 온디바이스 Mel은 float32 로그멜을 만든 뒤, AFE가 정해 준 입력 quantization(scale, zero-point)으로 int8 변환해서 CNN에 넣어야 한다. 이 input_scale/input_zero_point는 AFE가 export 때 산출해서 Mel에게 준다.
즉 Mel은: 오디오 → 로그멜(float, 2장 규격) → (AFE가 준 scale/zp로) int8 → CNN. 학습 때는 float 그대로.
6. 그래서 Mel이 나한테 줄 것 (체크리스트)

Mel 책임 = 온셋검출 + 윈도잉 + 로그멜 (세그멘테이션 포함, 확정).

 타건 단위 로그멜 텐서 (float32, shape (64,48), per-clip z-score 완료)
 각 타건의 onset_s (Mel의 온셋검출 결과)
 idx (시간순 연속, 특수키 타건 포함)
 헤더 메타: mel_spec_version, param_hash, sample_rate, n_mels, frames, session_id
 학습용: clip_id 키로 된 precomputed 로그멜(방식 A 권장) 또는 고정 추출기(방식 B)
7. Mel이 나한테 안 줘도 되는 것 (경계)
조합된 음절/텍스트, 자모 판정, 확률 — 전부 AFE·LM 몫.
gap_ms/리듬 — AFE가 onset으로 계산.
라벨(jamo/keytype) — 라벨은 녹음·metadata 쪽에서 온다. Mel은 신호(로그멜)만.
8. 합의 항목

확정:

O1 ✅ 세그멘테이션 = Mel 소유(온셋검출 + 윈도잉 + 로그멜).
O2 ✅ 마이크 48kHz 지원.

남은 항목(비블로킹, 붙여가며 확정):

O3. 전달 방식: 실시간 스트림 vs 세션 파일, 직렬화 포맷(npy/바이너리/JSON).
O4. 학습 로그멜: 방식 A(precomputed, 권장) vs B(공유 추출기).
O5. 입력 quantization(scale/zp) 전달 — AFE export 후 Mel에 공유(5장).
