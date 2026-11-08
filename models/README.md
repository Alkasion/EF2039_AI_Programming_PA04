# Models

`python -m src.train --exp <name> --seed <seed>` saves the epoch with the lowest validation loss to
`models/<name>_seed<seed>.pt`
(a PyTorch `state_dict` of torchvision ResNet-18 with a 2-class head, ~45 MB).

The 20 trained checkpoints (4 experiments x seeds 42–46) are attached to the GitHub Release `v1.0` of this repository.
Download them into this folder to evaluate without retraining:

```bash
for e in real_mixed synth_mixed real_motion_only real_defocus_only; do
  for s in 42 43 44 45 46; do
    curl -L -o models/${e}_seed$s.pt \
      https://github.com/Alkasion/EF2039_AI_Programming_PA04/releases/download/v1.0/${e}_seed$s.pt
  done
done
```
