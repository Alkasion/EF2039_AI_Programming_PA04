# Models

`python -m src.train --exp <name>` saves the epoch with the lowest validation loss to `models/<name>.pt`
(a PyTorch `state_dict` of torchvision ResNet-18 with a 2-class head, ~45 MB).

The four trained checkpoints are attached to the GitHub Release `v1.0` of this repository.
Download them into this folder to evaluate without retraining:

```bash
for e in real_mixed synth_mixed real_motion_only real_defocus_only; do
  curl -L -o models/$e.pt https://github.com/Alkasion/EF2039_AI_Programming_PA04/releases/download/v1.0/$e.pt
done
```
