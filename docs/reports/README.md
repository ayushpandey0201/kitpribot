# Document Folder — Samsung PRISM Worklet 25ST31BMS

Reports and decks in chronological order (dates verified from slide content):

| #   | File                                           | What it is                                                                   | Date        |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------- | ----------- |
| 00  | 00_Kickoff_Preliminary_Discussion_Mar2025.pptx | Preliminary discussion — team, professor, worklet pitch                      | 4 Mar 2025  |
| 00  | 00_Worklet_Details_25ST31BMS.pptx              | Worklet details deck shared at kick-off (problem statement, KPIs)            | kick-off    |
| 01  | 01_March_Monthly_Connect_PRISM.pptx            | Monthly connect — preprocessing pipeline, log-mel spectrograms               | 09 Mar 2026 |
| 02  | 02_April_Monthly_Connect_PRISM.pdf             | Monthly connect — 5,000-clip dataset, 70/15/15 split, mixing validation      | Apr 2026    |
| 03  | 03_June_Monthly_Connect_PRISM.pptx             | Monthly connect — v1 label-corruption discovery, v2 dataset rebuild          | 09 Jun 2026 |
| 04  | 04_July_Monthly_Connect_PRISM.pptx             | Monthly connect — kitpri_v4 dataset (7,200 clips), architecture benchmarking | 27 Jul 2026 |

To be added: end review PPT (with KPI details per the worklet details deck).

## Engineering report

- [`KitPri_v4_Engineering_Report.tex`](KitPri_v4_Engineering_Report.tex) / [`.pdf`](KitPri_v4_Engineering_Report.pdf) —
  publication-grade engineering report of the complete v4 pipeline: dataset
  synthesis → AST teacher → knowledge distillation → INT8 quantization →
  deployment (Docker, ARM guard, and the 24/7 Oracle-cloud Telegram bot),
  with all metrics traced to `results/` artifacts. The LaTeX source is fully
  self-contained (TikZ/pgfplots, data inlined — no external images): compile
  with `tectonic KitPri_v4_Engineering_Report.tex`, `pdflatex`, or Overleaf.
