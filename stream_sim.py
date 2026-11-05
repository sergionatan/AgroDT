from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from analysis import DATA_PATH, ensure_dataset


def build_event(row: pd.Series, event_id: int) -> dict:
    return {
        "event_id": event_id,
        "event_ts": pd.Timestamp.utcnow().isoformat(),
        "source": "olive-orchard-sensor-gateway",
        "payload": {
            "date": str(row["date"]),
            "temperature_c": float(row["temperature_c"]),
            "humidity_pct": float(row["humidity_pct"]),
            "rainfall_mm": float(row["rainfall_mm"]),
            "soil_ph": float(row["soil_ph"]),
            "nitrogen_mgkg": float(row["nitrogen_mgkg"]),
            "phosphorus_mgkg": float(row["phosphorus_mgkg"]),
            "potassium_mgkg": float(row["potassium_mgkg"]),
            "soil_moisture_pct": float(row["soil_moisture_pct"]),
            "lai": float(row["lai"]),
            "fruit_maturation_rate": float(row["fruit_maturation_rate"]),
            "productivity_ton_ha": float(row["productivity_ton_ha"]),
            "harvest_volume_ton": float(row["harvest_volume_ton"]),
            "oil_price_eur_l": float(row["oil_price_eur_l"]),
            "revenue_eur": float(row["revenue_eur"]),
        },
    }


def post_event_to_bridge(bridge_base_url: str, event: dict) -> bool:
    url = f"{bridge_base_url.rstrip('/')}/ingest"
    body = json.dumps(event, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return 200 <= resp.status < 300 or resp.status == 204
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def stream_dataset(
    df: pd.DataFrame,
    output_path: Path | None,
    interval_sec: float,
    max_events: int | None = None,
    append: bool = True,
    bridge_base_url: str | None = None,
) -> int:
    fp = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        fp = output_path.open(mode, encoding="utf-8")

    count = 0
    bridge_fail_once = False
    try:
        for idx, row in df.iterrows():
            event = build_event(row, event_id=idx + 1)
            if fp is not None:
                fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                fp.flush()

            if bridge_base_url:
                ok = post_event_to_bridge(bridge_base_url, event)
                if not ok and not bridge_fail_once:
                    bridge_fail_once = True
                    print(
                        f"Aviso: não foi possível enviar para o bridge ({bridge_base_url}). "
                        "Confirme que o Streamlit está aberto (aba Stream) ou que o servidor escuta nesta URL."
                    )

            count += 1
            print(
                f"evento={event['event_id']} "
                f"temp={event['payload']['temperature_c']:.1f}C "
                f"umidade_solo={event['payload']['soil_moisture_pct']:.1f}% "
                f"faturamento={event['payload']['revenue_eur']:.2f} EUR"
            )

            if max_events is not None and count >= max_events:
                break
            time.sleep(interval_sec)
    finally:
        if fp is not None:
            fp.close()

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulador de streaming de dados do Digital Twin.")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Intervalo entre eventos.")
    parser.add_argument("--max-events", type=int, default=50, help="Quantidade máxima de eventos.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stream/olive_sensor_stream.jsonl"),
        help="Arquivo de saída JSONL para eventos simulados.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Se definido, sobrescreve o arquivo de saída em vez de anexar.",
    )
    parser.add_argument(
        "--bridge-url",
        type=str,
        default="http://127.0.0.1:8765",
        help="URL base do bridge HTTP (Streamlit). Ex.: http://127.0.0.1:8765 — use string vazia para desativar.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Não grava arquivo JSONL; apenas envia eventos ao bridge (se --bridge-url estiver definido).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = ensure_dataset(DATA_PATH).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    bridge = (args.bridge_url or "").strip() or None
    out = None if args.no_jsonl else args.output
    if out is None and bridge is None:
        raise SystemExit("Defina --bridge-url ou remova --no-jsonl para gravar JSONL.")

    total = stream_dataset(
        df=df,
        output_path=out,
        interval_sec=args.interval_sec,
        max_events=args.max_events,
        append=not args.truncate,
        bridge_base_url=bridge,
    )
    print(f"\nStreaming finalizado. Eventos emitidos: {total}")
    if out is not None:
        print(f"Arquivo: {out.resolve()}")
    if bridge:
        print(f"Bridge HTTP: {bridge}")


if __name__ == "__main__":
    main()
