## Ablations
1. Qwen8b, LTD, 10k steps per phase, (co-optimized) --> Nish
2. Qwen8b, LTD, 10k steps per phase, (not co-optimized) --> Nish
3. Qwen8b, Supervised Learning, 10k timesteps of data, 1 epoch --> Nish
4. Qwen8b, Supervised Learning, 10k timesteps of data, 20 epochs --> Nish
---
1. Qwen14b, LTD, 10k steps per phase, (co-optimized) --> Alistair
2. Qwen14b, LTD, 10k steps per phase, (not co-optimized) --> Alistair
3. Qwen14b, Supervised Learning, 10k timesteps of data, 1 epoch --> Alistair
4. Qwen14b, Supervised Learning, 10k timesteps of data, 20 epochs --> Alistair
---
1. Qwen8b, LTD, 20k steps per phase, (co-optimized) --> Neil
2. Qwen8b, LTD, 20k steps per phase, (not co-optimized) --> Neil
3. Qwen8b, Supervised Learning, 20k timesteps of data, 1 epochs --> Neil
4. Qwen8b, Supervised Learning, 20k timesteps of data, 20 epochs --> Neil
---
1. Qwen14b, LTD, 20k steps per phase, (co-optimized) --> Neil
2. Qwen14b, LTD, 20k steps per phase, (not co-optimized) --> Neil
3. Qwen14b, Supervised Learning, 20k timesteps of data, 1 epoch --> Neil
4. Qwen14b, Supervised Learning, 20k timesteps of data, 20 epochs --> Neil
---
1. Qwen8b, LTD, 50k steps per phase, (co-optimized) --> Neil
2. Qwen8b, LTD, 50k steps per phase, (not co-optimized) --> Neil
3. Qwen8b, Supervised Learning, 50k timesteps of data, 1 epochs --> Neil
4. Qwen8b, Supervised Learning, 50k timesteps of data, 20 epochs --> Neil
---
1. Qwen14b, LTD, 50k steps per phase, (co-optimized) --> Neil
2. Qwen14b, LTD, 50k steps per phase, (not co-optimized) --> Neil
3. Qwen14b, Supervised Learning, 50k timesteps of data, 1 epoch --> Neil
4. Qwen14b, Supervised Learning, 50k timesteps of data, 20 epochs --> Neil
---
1. Qwen8b, LTD, 100k steps per phase, (co-optimized) --> Alistair
2. Qwen8b, LTD, 100k steps per phase, (not co-optimized) --> Alistair
3. Qwen8b, Supervised Learning, 100k timesteps of data, 1 epoch --> Alistair
4. Qwen8b, Supervised Learning, 100k timesteps of data, 20 epochs --> Alistair
---
1. Qwen14b, LTD, 100k steps per phase, (co-optimized) --> Alistair
2. Qwen14b, LTD, 100k steps per phase, (not co-optimized) --> Alistair
3. Qwen14b, Supervised Learning, 100k timesteps of data, 1 epoch --> Alistair
4. Qwen14b, Supervised Learning, 100k timesteps of data, 20 epochs --> Alistair

## Notes
- The usage instructions aren't perfect, but they provide a pretty clear guide of how to run the code on Modal
- Save all data in the same format that I do. You will have to delete unnecessary files (ex: zip files of policies) while preserving the key structure
- Name result folders in the same format that I do
- When you finish one ablation, make sure you `rm -rf` the old data on modal! For supervised learning, the dataset can be reused, but everything else should be cleared out. For LTD, everything should be cleared out.
- If you're unsure, start with a quick 10k run for practice.
- For 1 epoch, set checkpoint epochs to be 1
- For 20 epochs, set checkpoint epochs to be 20 (this checkpoints at the first, best training, and last epochs)
- For 50 epochs, set checkpoint epochs to be 50 (this checkpoints at the first, best training, and last epochs)
- Nish Note: I've spent 20+ hrs coding everything up. Thousands of lines of code have been produced. I'm not doing more work.