# Kế hoạch chuyển dataset → DKASC Alice Springs

**Dự án:** Tái lập bài báo *Temporal Knowledge Graph Reasoning for Real-Time Solar Power Forecasting Using Satellite Data* (Hoàng Đức Long, Nguyễn Thị Hồng Khánh — ĐH Điện lực)
**Mục tiêu task:** Reproduce **thứ hạng (ranking)** của bảng so sánh nhiều mô hình — đúng thứ tự là "đạt", không bắt buộc khớp số tuyệt đối.
**Ngày lập:** 14/06/2026

---

## 0. Bối cảnh — vì sao đổi

Pipeline cũ ghép **3 nguồn dữ liệu lệch địa lý (geographically mismatched)**: PV của OPSD (châu Âu) + khí tượng NSRDB + ảnh Himawari (châu Á). Vì không cùng vị trí (not co-located), đồ thị tri thức (temporal knowledge graph) không có ý nghĩa vật lý thật, và MAE của paper không thể tái lập.

**DKASC Alice Springs** giải quyết gốc rễ vấn đề: PV power + dữ liệu khí tượng **đo tại cùng một nơi**, mở hoàn toàn (open-access, tải CSV trực tiếp), 15+ năm dữ liệu, tọa độ −23.76 / 133.87.

> **Lưu ý quan trọng:** Bản paper mới (file `BaiBao2025_84`, ngày 13/06) **đã viết lại Mục 4 sang DKASC** rồi — nhưng còn dở: Mục 5 vẫn giữ bảng kết quả cũ (mâu thuẫn đơn vị & mô hình), và tên mô hình đề xuất bị lệch (Mục 3 = TKG, Mục 4.3 = "SatCNN-LSTM-Attn"). **Cần hỏi cô Khánh xác nhận file 84 là bản chuẩn** trước khi code.

---

## 1. Hai quyết định phải chốt trước

### Quyết định A — Nhánh vệ tinh (satellite branch)

| Phương án | Nội dung | Đánh đổi (trade-off) |
|---|---|---|
| **A1. Giữ 3 nhánh** | Tải Himawari-8 cho đúng vùng Alice Springs 2020–2022 (Himawari-8 phủ cả Úc) | Sửa được lệch địa lý + giữ kiến trúc gốc. Tốn công tải. |
| **A2. Rút 2 nhánh** | Chỉ dùng PV + khí tượng DKASC, bỏ nhánh vệ tinh | Đơn giản, nhanh. Mô hình đề xuất mất thành phần "satellite". |

### Quyết định B — Bộ mô hình cho bảng so sánh

| Phương án | Bộ mô hình | Việc phát sinh |
|---|---|---|
| **B1. Theo file 84** | Persistence, ARIMA, LSTM, Transformer, TFT + Đề xuất | Thêm **3 thuật toán mới**: Persistence, ARIMA, TFT |
| **B2. Giữ bộ cũ** | LSTM, GRU, Transformer, Temporal-GNN + Đề xuất | Không thêm thuật toán, chỉ đổi data |

> **Khuyến nghị:** **A1 + B1** — đúng tinh thần bản paper mới và mạnh nhất về mặt khoa học. Nhưng cần cô Khánh xác nhận trước khi đổ công.

---

## 2. Checklist công việc

### Phase 1 — Tải & chuẩn bị dữ liệu

- [ ] Tải PV + weather CSV của **một array** ở Alice Springs từ dkasc.solar (không cần API key). Chọn array có lịch sử dài, dữ liệu sạch.
- [ ] Cố định khung thời gian (time range) **01/2020 – 12/2022**, tần suất gốc 5 phút.
- [ ] Lấy đúng các cột file 84 dùng: `Pac` (target, kW), `GHI`, `Tamb`, `RH`, `WS`.
- [ ] *(Nếu A1)* Tải Himawari-8 ROI quanh Alice Springs (−23.76, 133.87) cho 2020–2022, crop nhỏ giống cách đã làm cho Việt Nam.
- [ ] Viết `scripts/download_dkasc.py` thay cho `download_opsd.py` + `download_nsrdb.py`.

### Phase 2 — Sửa data pipeline (phần nặng nhất)

- [ ] `src/data_pipeline/` đọc **một nguồn co-located** thay vì ghép 3 nguồn → **bỏ bước "intersection gate"** ghép cặp UTC giữa các châu lục.
- [ ] Đổi lưới căn thời gian 10 phút → **5 phút** (hoặc resample về horizon đã chọn).
- [ ] **Lọc ban đêm**: bỏ mẫu `GHI < 5 W/m²` (bước *mới*, chưa có trong pipeline cũ).
- [ ] Giữ kỷ luật chống rò rỉ (anti-leakage): split **trước**, rồi clip μ±5σ và Min-Max **chỉ fit trên train**. Code cũ đã đúng — chỉ trỏ vào data mới.
- [ ] Đổi `splits.py` sang **chia theo mốc thời gian cố định** của file 84: train 01/2020–09/2021 · val 10–12/2021 · test cả năm 2022.

### Phase 3 — Sửa "hợp đồng kích thước" (dimension contract)

- [ ] `src/common/shapes.py`: đổi `N_METEO_FEATURES` **7 → 4** (GHI, Tamb, RH, WS). Đây là chỗ **hardcoded** dễ quên nhất — sai là vỡ toàn bộ encoder.
- [ ] Chốt `N_HORIZONS`: file 84 báo cáo ∆t = 30 phút → 1 horizon, hay vẫn giữ 3 (10/30/60).
- [ ] *(Nếu A2)* Bỏ `sat_seq` khỏi batch và khỏi `TKGSolarModel.forward` → còn 2 nhánh (meteo + graph).

### Phase 4 — Thêm thuật toán mới (nếu B1)

- [ ] **Persistence**: `P̂(t+Δt) = P(t)`. Không tham số, **không train** — class trả thẳng giá trị cuối.
- [ ] **ARIMA(5,1,2)**: dùng `statsmodels`, fit theo từng chuỗi, **không qua Adam**.
- [ ] **TFT (Temporal Fusion Transformer)**: dùng `pytorch-forecasting` — baseline nặng nhất.
- [ ] *(Bỏ GRU và Temporal-GNN nếu theo B1.)*

### Phase 5 — Chạy benchmark & xuất bảng

- [ ] Train **tất cả mô hình trên cùng data / cùng split / cùng seed**.
- [ ] Tính MAE / RMSE / MAPE (+ sRMSE như file 84) sau khi **inverse-scale về kW**.
- [ ] Xuất bảng và kiểm tra **thứ hạng (ranking)** — mục tiêu Đề xuất sai số thấp nhất.

---

## 3. Kiểm tra file train — những chỗ phải sửa

> Tin vui: `train_loop.py` viết kiểu **model-agnostic** (`pred = model(batch); loss = loss_fn(...)`), nên vòng train gần như giữ nguyên. Chỉ có 3 chỗ phải đụng:

1. **`evaluate_mae(model, loader, splits.scalers, ...)`** — inverse-scale bằng `scalers`. Phải đảm bảo nhận **scaler mới fit trên DKASC** (đơn vị kW), nếu không số MAE vô nghĩa. (Tự động đúng nếu Phase 2 chuẩn.)

2. **Persistence & ARIMA không vừa vòng train hiện tại.** `fit()` giả định mọi mô hình là `nn.Module` + Adam + `loss.backward()`. Persistence (không tham số) và ARIMA (statsmodels) **không backward được** → cần **nhánh eval-only** bỏ qua `fit()`, chỉ gọi `evaluate`. Đây là điểm dễ vỡ nhất khi thêm B1.

3. **Hàm loss mô hình đề xuất** (`src/advanced_loss/`, có *physics-informed*). Nếu loss vật lý dùng hằng số gắn vị trí/đơn vị cũ (clear-sky theo tọa độ VN, hoặc PV chuẩn hóa 0–1), phải xem lại cho hợp DKASC (Alice Springs, kW). Nếu chưa chắc: tạm chạy mô hình đề xuất với **MSE thuần** để có baseline sạch trước, rồi mới bật physics loss.

**Giữ nguyên:** logic resume/checkpoint trên Drive và interface `forward(batch) -> [B, H]`.

---

## 4. Bảng tra nhanh — đổi gì ở file nào

| File / Module | Thay đổi |
|---|---|
| `scripts/download_*.py` | Thay bằng `download_dkasc.py` (+ Himawari-AliceSprings nếu A1) |
| `src/data_pipeline/loaders.py` | Đọc 1 nguồn co-located thay vì 3 nguồn |
| `src/data_pipeline/time_alignment.py` | Bỏ intersection gate; lưới 5 phút; lọc đêm `GHI<5` |
| `src/data_pipeline/splits.py` | Chia theo mốc thời gian cố định 2020–2022 |
| `src/common/shapes.py` | `N_METEO_FEATURES 7→4`; rà lại `N_HORIZONS`, `sat_channels` |
| `src/fusion_predictor/tkg_solar_model.py` | *(Nếu A2)* bỏ nhánh `sat_seq` |
| `src/lstm_baseline/` + baseline mới | Thêm Persistence, ARIMA, TFT *(nếu B1)* |
| `src/training/train_loop.py` | Thêm **eval-only path** cho Persistence/ARIMA |
| `src/advanced_loss/` | Rà hằng số physics-informed theo DKASC/kW |
| `configs/paper_config.yaml` | Cập nhật path, features, horizon, split, đơn vị |

---

## 5. Rủi ro & lưu ý

- **DKASC không có ảnh vệ tinh trong gói tải chuẩn** — chỉ PV + khí tượng. Nếu chọn A1 phải tự lấy Himawari-8 riêng cho Alice Springs.
- **Himawari-8** vận hành tới hết 2022 (sau đó là Himawari-9) → khung 2020–2022 vẫn nằm trong giai đoạn Himawari-8, hợp lệ.
- **Mâu thuẫn nội bộ trong file 84** (2 bảng kết quả lệch nhau, tên mô hình lệch) → xác nhận với cô Khánh đâu là bản chuẩn trước khi reproduce.
- Cập nhật `docs/assumptions.md` và `doi_chieu_paper_vs_code.md`: bỏ giả định lệch địa lý cũ, thêm các quyết định mới (A, B, khung thời gian, danh sách feature DKASC).

---

## 6. Câu hỏi cần hỏi cô Khánh (Discord)

1. File 84 (bản DKASC) có phải bản chuẩn để tái lập không?
2. Mô hình đề xuất là **TKG đầy đủ** (Mục 3) hay **SatCNN-LSTM-Attn** (Mục 4.3)?
3. Bộ baseline cuối cùng: theo file 84 (Persistence/ARIMA/LSTM/Transformer/TFT) hay giữ bộ cũ?
4. Báo cáo 1 horizon (30 phút) hay cả 3 (10/30/60)?

---

## Phụ lục — Glossary (học tiếng Anh)

| Thuật ngữ | Nghĩa |
|---|---|
| co-located | cùng vị trí địa lý |
| anti-leakage | chống rò rỉ dữ liệu (thống kê chỉ fit trên train) |
| dimension contract | quy ước kích thước tensor cố định giữa các module |
| model-agnostic | không phụ thuộc mô hình cụ thể |
| eval-only path | luồng chỉ đánh giá, không huấn luyện |
| trade-off | sự đánh đổi |
| baseline | mô hình nền để so sánh |
| time range | khoảng thời gian dữ liệu |
