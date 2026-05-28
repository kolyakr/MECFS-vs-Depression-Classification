"""Metrics section generation for monitoring reports."""
from pathlib import Path
import io
import base64
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.db.session import get_engine

logger = logging.getLogger(__name__)


def append_metrics_section(output_html: Path) -> None:
    """Append metrics trend section to HTML report if metrics table exists."""
    try:
        engine = get_engine()
        metrics_df = pd.read_sql_table("metrics", con=engine)
        
        if metrics_df.empty:
            logger.info("Metrics table is empty, skipping metrics section")
            return
        
        if "created_at" in metrics_df.columns:
            metrics_df = metrics_df.sort_values("created_at")
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True, facecolor='#1a1a1a')
        axes = ax.ravel()
        x = metrics_df["created_at"] if "created_at" in metrics_df.columns else range(len(metrics_df))
        
        for i, (col, title) in enumerate([
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score")
        ]):
            if col in metrics_df.columns:
                axes[i].plot(x, metrics_df[col], marker='o', color='#888888', linewidth=2, markersize=6)
                axes[i].set_title(title, color='#cccccc')
                axes[i].set_facecolor('#000000')
                axes[i].tick_params(colors='#888888')
                axes[i].grid(True, alpha=0.3, color='#444444')
                try:
                    vals = pd.to_numeric(metrics_df[col], errors='coerce').dropna().astype(float).values
                    if len(vals):
                        ymin = max(0.0, float(np.min(vals)) - 0.05)
                    else:
                        ymin = 0.0
                    axes[i].set_ylim(ymin, 1.0)
                except Exception:
                    axes[i].set_ylim(0, 1)
        
        buf = io.BytesIO()
        fig.autofmt_xdate(rotation=30)
        fig.savefig(buf, format='png')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        recent = metrics_df.tail(10)
        cols = [c for c in [
            "created_at", "run_type", "accuracy", "precision", "recall", "f1",
            "n_train", "n_inference"
        ] if c in recent.columns]
        
        table_rows = "".join(
            "<tr style=\"color: #cccccc;\">" + 
            "".join(f"<td style=\"color: #cccccc;padding:8px 10px;border-bottom:1px solid #333333;\">{recent.iloc[i][c]}</td>" for c in cols) + 
            "</tr>"
            for i in range(len(recent))
        )

        html_extra = f"""
<section style=\"font-family: Inter, Arial, sans-serif; margin: 24px; padding: 20px; border-radius: 12px; background: #000000; color: #cccccc; box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #333333;\">
  <h1 style=\"margin: 0 0 12px; font-size: 28px; font-weight: 700; letter-spacing: .3px; color: #ffffff;\">Monitoring summary</h1>
  <h2 style=\"margin: 8px 0 12px; font-size: 18px; font-weight: 600; color: #aaaaaa;\">Metrics Trend</h2>
  <div style=\"margin: 12px 0 20px; display: flex; justify-content: center;\">
    <img alt=\"metrics-trend\" src=\"data:image/png;base64,{img_b64}\" style=\"max-width: 100%; border-radius: 8px; background: #1a1a1a; padding: 8px; border: 1px solid #444444;\"/>
  </div>
  <div style=\"display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px;\">
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">Accuracy</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{float(metrics_df['accuracy'].iloc[-1]) if 'accuracy' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">Precision</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{float(metrics_df['precision'].iloc[-1]) if 'precision' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">Recall</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{float(metrics_df['recall'].iloc[-1]) if 'recall' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">F1-score</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{float(metrics_df['f1'].iloc[-1]) if 'f1' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">n_train</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{int(metrics_df['n_train'].iloc[-1]) if 'n_train' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
    <div style=\"background:#1a1a1a;border:1px solid #333333;border-radius:10px;padding:10px;\"><div style=\"color:#888888;font-size:12px;\">n_inference</div><div style=\"font-size:18px;font-weight:700;color:#cccccc;\">{int(metrics_df['n_inference'].iloc[-1]) if 'n_inference' in metrics_df.columns and len(metrics_df) else 'n/a'}</div></div>
  </div>
  <h3 style=\"margin: 8px 0; font-size: 16px; font-weight: 600; color: #aaaaaa;\">Recent runs</h3>
  <table style=\"width:100%; border-collapse:collapse; background:#1a1a1a; border:1px solid #333333; border-radius: 8px; overflow: hidden;\">
    <thead>
      <tr style=\"background:#2a2a2a; color:#cccccc; text-align:left;\">{''.join(f'<th style="padding:8px 10px;border-bottom:1px solid #444444;">{c}</th>' for c in cols)}</tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</section>
"""
        
        html_text = output_html.read_text(encoding="utf-8")
        inserted = False
        
        if "</body>" in html_text:
            html_text = html_text.replace("</body>", html_extra + "</body>", 1)
            inserted = True
        elif "</html>" in html_text:
            html_text = html_text.replace("</html>", html_extra + "</html>", 1)
            inserted = True
        else:
            html_text = html_text + html_extra
            inserted = True
        
        output_html.write_text(html_text, encoding="utf-8")
        logger.info(f"Inserted metrics trend section (rows={len(metrics_df)}) at bottom of body")
        
    except Exception as e:
        logger.warning(f"Could not append metrics trend section: {e}")

