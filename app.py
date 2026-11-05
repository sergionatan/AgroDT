from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import DATA_PATH, FEATURES, MODEL_DIR, TARGET, ensure_dataset
from data_gen import generate_mock_data, save_dataset
from stream_bridge import (
    buffered_count,
    clear_buffer,
    fetch_events_http,
    listening_port,
    start_bridge_server,
)


MODEL_PATH = MODEL_DIR / "olive_productivity_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DEFAULT_STREAM_PATH = Path("data/stream/olive_sensor_stream.jsonl")


def ensure_model_ready() -> None:
    if DATA_PATH.exists() and MODEL_PATH.exists():
        return

    df = generate_mock_data()
    save_dataset(df, DATA_PATH)
    from analysis import run_eda, train_model

    run_eda(df)
    train_model(df)


@st.cache_data
def load_data() -> pd.DataFrame:
    return ensure_dataset()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def olive_health_index(row: dict[str, float]) -> float:
    score = 100.0

    if not 18 <= row["temperature_c"] <= 30:
        score -= 12
    if row["temperature_c"] > 32:
        score -= 8
    if row["temperature_c"] > 35:
        score -= 10
    if row["temperature_c"] > 38:
        score -= 12
    if not 45 <= row["humidity_pct"] <= 75:
        score -= 10
    if not 18 <= row["soil_moisture_pct"] <= 40:
        score -= 18
    if not 6.2 <= row["soil_ph"] <= 7.4:
        score -= 15

    for nutrient in ("nitrogen_mgkg", "phosphorus_mgkg", "potassium_mgkg"):
        score -= max(0, abs(row[nutrient] - {"nitrogen_mgkg": 35, "phosphorus_mgkg": 20, "potassium_mgkg": 200}[nutrient]) * 0.06)

    score += min(row["lai"], 6.0) * 2.0
    score += min(row["fruit_maturation_rate"], 100) * 0.08
    return max(0.0, min(100.0, score))


def health_label(score: float) -> str:
    if score >= 80:
        return "Excelente"
    if score >= 60:
        return "Moderada"
    return "Crítica"


def read_jsonl_tail(path: Path, max_lines: int) -> list[dict]:
    if not path.exists() or max_lines <= 0:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = lines[-max_lines:]
    events: list[dict] = []
    for line in tail:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    rows = []
    for e in events:
        p = e.get("payload") or {}
        rows.append(
            {
                "event_id": e.get("event_id"),
                "event_ts": e.get("event_ts"),
                **p,
            }
        )
    return pd.DataFrame(rows)


def merge_event_streams(*lists: list[dict]) -> list[dict]:
    by_id: dict[int, dict] = {}
    for lst in lists:
        for e in lst:
            eid = e.get("event_id")
            if eid is None:
                continue
            try:
                by_id[int(eid)] = e
            except (TypeError, ValueError):
                continue
    return [by_id[k] for k in sorted(by_id.keys())]


def render_stream_panel(
    get_events: Callable[[], list[dict]],
    model,
    last_n: int,
    refresh_sec: int,
    planted_area_ha: float,
    *,
    empty_hint: str,
) -> None:
    fragment = getattr(st, "fragment", None)

    def body() -> None:
        events = get_events()
        if not events:
            st.info(empty_hint)
            return

        stream_df = events_to_dataframe(events)
        tail_df = stream_df.tail(last_n).copy()
        if tail_df.empty:
            st.warning("Não foi possível montar a tabela de eventos.")
            return

        latest = tail_df.iloc[-1]
        row = {k: float(latest[k]) for k in FEATURES if k in latest and pd.notna(latest[k])}
        if len(row) != len(FEATURES):
            st.error("Payload do stream incompleto para predição.")
            st.dataframe(tail_df.tail(5), use_container_width=True)
            return

        pred = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        obs_prod = float(latest[TARGET]) if TARGET in latest and pd.notna(latest[TARGET]) else None
        obs_rev = float(latest["revenue_eur"]) if "revenue_eur" in latest and pd.notna(latest["revenue_eur"]) else None
        oil_p = float(latest["oil_price_eur_l"]) if "oil_price_eur_l" in latest and pd.notna(latest["oil_price_eur_l"]) else 6.6

        h = olive_health_index(row)
        fut_harvest = pred * planted_area_ha
        fut_rev = fut_harvest * 210 * oil_p

        st.caption(f"Última leitura: `{latest.get('event_ts', '—')}` · event_id **{latest.get('event_id', '—')}**")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Temp. (stream)", f"{row['temperature_c']:.1f} °C")
        k2.metric("Saúde (heurística)", f"{h:.1f}/100", delta=health_label(h))
        k3.metric("Produtividade (modelo)", f"{pred:.2f} ton/ha")
        if obs_prod is not None:
            k4.metric("Produtividade (dataset)", f"{obs_prod:.2f} ton/ha", delta=f"{pred - obs_prod:+.2f} vs modelo")
        else:
            k4.metric("Produtividade (dataset)", "—")
        k5.metric("Faturamento (stream)", f"{obs_rev:,.0f} EUR" if obs_rev is not None else "—")

        st.metric(
            "Projeção What-if (área sidebar) — colheita / faturamento",
            f"{fut_harvest:,.2f} ton · {fut_rev:,.0f} EUR",
            help="Usa produtividade prevista pelo modelo e área plantada definida na barra lateral.",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Série do stream — clima")
            chart_clim = tail_df.set_index("event_id")[["temperature_c", "humidity_pct", "rainfall_mm"]]
            st.line_chart(chart_clim)
        with c2:
            st.subheader("Série do stream — solo e LAI")
            chart_soil = tail_df.set_index("event_id")[["soil_moisture_pct", "lai", "soil_ph"]]
            st.line_chart(chart_soil)

        st.subheader("Últimos eventos")
        show_cols = [
            "event_id",
            "event_ts",
            "temperature_c",
            "soil_moisture_pct",
            TARGET,
            "harvest_volume_ton",
            "revenue_eur",
        ]
        show_cols = [c for c in show_cols if c in tail_df.columns]
        st.dataframe(tail_df[show_cols].tail(25), use_container_width=True, hide_index=True)

    if fragment is not None:
        @fragment(run_every=timedelta(seconds=refresh_sec))
        def _auto() -> None:
            body()

        _auto()
    else:
        body()
        if st.button("Atualizar leitura do stream"):
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Digital Twin - Oliveiras", layout="wide")
    ensure_model_ready()

    df = load_data()
    model = load_model()

    st.title("Digital Twin para Plantação de Oliveiras")
    st.caption("Simulação integrada de clima, solo, crescimento e faturamento.")

    with st.sidebar:
        st.header("Parâmetros globais")
        planted_area_ha = st.number_input("Área Plantada (ha)", min_value=1.0, max_value=500.0, value=55.0, step=1.0)
        oil_price_eur_l = st.number_input("Preço do Azeite (EUR/L)", min_value=2.0, max_value=15.0, value=6.6, step=0.1)

        st.header("Simulador What-if")
        temperature_c = st.slider("Temperatura (°C)", 5.0, 45.0, 25.0, 0.1)
        humidity_pct = st.slider("Umidade Relativa (%)", 20.0, 100.0, 60.0, 0.1)
        rainfall_mm = st.slider("Pluviosidade (mm/dia)", 0.0, 30.0, 4.0, 0.1)

        soil_ph = st.slider("pH do Solo", 4.5, 8.5, 6.8, 0.01)
        nitrogen_mgkg = st.slider("Nitrogênio (mg/kg)", 5.0, 70.0, 35.0, 0.1)
        phosphorus_mgkg = st.slider("Fósforo (mg/kg)", 5.0, 50.0, 20.0, 0.1)
        potassium_mgkg = st.slider("Potássio (mg/kg)", 80.0, 350.0, 200.0, 0.5)
        soil_moisture_pct = st.slider("Umidade do Solo (%)", 5.0, 60.0, 28.0, 0.1)

        lai = st.slider("LAI", 0.8, 7.0, 3.8, 0.05)
        fruit_maturation_rate = st.slider("Taxa de Maturação (%)", 0.0, 100.0, 65.0, 0.5)

    row = {
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "rainfall_mm": rainfall_mm,
        "soil_ph": soil_ph,
        "nitrogen_mgkg": nitrogen_mgkg,
        "phosphorus_mgkg": phosphorus_mgkg,
        "potassium_mgkg": potassium_mgkg,
        "soil_moisture_pct": soil_moisture_pct,
        "lai": lai,
        "fruit_maturation_rate": fruit_maturation_rate,
    }

    tab_sim, tab_stream = st.tabs(["Simulador What-if", "Stream ao vivo"])

    with tab_sim:
        input_df = pd.DataFrame([row], columns=FEATURES)
        predicted_productivity = float(model.predict(input_df)[0])
        projected_harvest_ton = predicted_productivity * planted_area_ha
        projected_revenue_eur = projected_harvest_ton * 210 * oil_price_eur_l

        health_score = olive_health_index(row)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Produtividade Prevista (ton/ha)", f"{predicted_productivity:.2f}")
        col2.metric("Volume Estimado da Colheita (ton)", f"{projected_harvest_ton:,.2f}")
        col3.metric("Faturamento Futuro Estimado (EUR)", f"{projected_revenue_eur:,.2f}")
        col4.metric("Saúde da Oliveira", f"{health_score:.1f}/100", delta=health_label(health_score))

        st.subheader("Séries Temporais de Clima")
        ts_df = df[["date", "temperature_c", "humidity_pct", "rainfall_mm"]].copy()
        ts_df["date"] = pd.to_datetime(ts_df["date"])
        ts_df = ts_df.set_index("date")
        st.line_chart(ts_df.tail(365))

        st.subheader("Correlação entre Solo e Produtividade")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["soil_moisture_pct"], df[TARGET], alpha=0.6)
        ax.set_xlabel("Umidade do Solo (%)")
        ax.set_ylabel("Produtividade (ton/ha)")
        ax.set_title("Relação Solo x Produtividade")
        st.pyplot(fig)
        plt.close(fig)

    with tab_stream:
        st.markdown(
            "Os eventos podem chegar **direto ao Streamlit** via bridge HTTP (`POST /ingest`) iniciado nesta aba, "
            "e/ou pelo arquivo **JSONL**. O simulador `stream_sim.py` envia para o bridge por padrão."
        )
        bridge_port = st.number_input(
            "Porta do bridge HTTP (ingestão)",
            min_value=1024,
            max_value=65535,
            value=8765,
            step=1,
            key="stream_bridge_port",
        )
        started_here, bound_port = start_bridge_server(port=int(bridge_port))
        active = listening_port()
        if active is not None and int(active) != int(bridge_port):
            st.caption(
                f"O bridge já está escutando na porta **{active}**; altere o comando do simulador ou reinicie o Streamlit para usar **{bridge_port}**."
            )
        base_url = f"http://127.0.0.1:{bound_port}"
        if started_here:
            st.success(f"Bridge HTTP ativo em **{base_url}** — endpoint `POST /ingest`, leitura `GET /events`.")
        else:
            st.warning(
                f"Não foi possível abrir o servidor nesta porta (pode estar em uso). "
                f"Tente acessar **{base_url}** se outro processo já expõe o bridge, ou altere a porta e recarregue a página."
            )

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            source = st.radio(
                "Fonte dos dados",
                ("HTTP (bridge)", "Arquivo JSONL", "HTTP + JSONL"),
                horizontal=True,
                key="stream_source",
            )
        with sc2:
            stream_path_str = st.text_input(
                "Caminho do JSONL (se usar arquivo)",
                value=str(DEFAULT_STREAM_PATH),
                key="stream_jsonl_path",
            )
        with sc3:
            last_n = st.slider("Eventos no gráfico", min_value=10, max_value=400, value=80, step=10, key="stream_last_n")
        refresh_sec = st.slider("Auto-atualizar (s)", min_value=1, max_value=15, value=2, key="stream_refresh_sec")

        def get_stream_events() -> list[dict]:
            lim = max(500, last_n * 3)
            events_http: list[dict] = []
            events_file: list[dict] = []
            if source in ("HTTP (bridge)", "HTTP + JSONL"):
                events_http = fetch_events_http(base_url, limit=lim)
            if source in ("Arquivo JSONL", "HTTP + JSONL"):
                events_file = read_jsonl_tail(Path(stream_path_str), max_lines=lim)
            if source == "HTTP + JSONL":
                return merge_event_streams(events_http, events_file)
            if source == "HTTP (bridge)":
                return events_http
            return events_file

        with st.expander("Comandos do simulador"):
            st.code(
                f"python stream_sim.py --interval-sec 0.5 --max-events 200 --bridge-url {base_url}",
                language="bash",
            )
            st.caption("Use `--no-jsonl` para enviar somente ao bridge. Use `--bridge-url \"\"` para gravar só em arquivo.")

        if source in ("HTTP (bridge)", "HTTP + JSONL"):
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Limpar buffer HTTP desta sessão", key="clear_stream_buffer"):
                    clear_buffer()
                    st.rerun()
            with b2:
                st.caption(f"Eventos no buffer HTTP (servidor): **{buffered_count()}**.")

        empty_hint = (
            "Nenhum evento recebido ainda.\n\n"
            f"1) Confirme o bridge em **{base_url}** (mensagem verde acima).\n"
            f"2) Rode: `python stream_sim.py --bridge-url {base_url}`\n\n"
            "Ou escolha **Arquivo JSONL** e gere o arquivo com `stream_sim.py` sem bridge."
        )
        if source == "Arquivo JSONL":
            empty_hint = (
                "Nenhum evento no arquivo JSONL.\n\n"
                "`python stream_sim.py --interval-sec 0.5 --max-events 200 --bridge-url \"\" --truncate`"
            )

        render_stream_panel(get_stream_events, model, last_n, refresh_sec, planted_area_ha, empty_hint=empty_hint)

    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        st.subheader("Métricas do Modelo")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics.get('r2', 0):.3f}")
        m2.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        m3.metric("MAE", f"{metrics.get('mae', 0):.3f}")


if __name__ == "__main__":
    main()
