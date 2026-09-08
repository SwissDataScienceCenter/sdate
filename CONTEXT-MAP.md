# Context Map

This repo hosts several largely-independent CT reconstruction/denoising research pipelines under `sdate/`. Each has its own vocabulary; only the ones below have been modeled so far.

## Contexts

- [Time-Resolved Diffusion Denoising](./sdate/tr_diffusion/CONTEXT.md) — self-supervised (Noise2Noise/Noise2Void) and diffusion-based denoising of time-resolved CT projections, plus the reconstruction pipelines built on top of it

## Not yet modeled

Other independent pipelines exist in this repo (e.g. `sdate/tr_naf`, `sdate/sino_hexplane`, `sdate/anomaly`, `isodiffusion/`) but have not had a domain-modeling pass — do not assume the `tr_diffusion` glossary applies to them.
