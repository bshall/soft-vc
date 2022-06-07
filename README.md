# Soft Speech Units for Improved Voice Conversion

Official repository for [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://ieeexplore.ieee.org/abstract/document/9746484).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/hubert/main/diagram.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

For modularity, each component of the system is housed in a separate repository. Please visit the following links for more details:

- [HuBERT content encoders](https://github.com/bshall/hubert)
- [Acoustic Models](https://github.com/bshall/acoustic-model)
- [HiFiGAN vocoder](https://github.com/bshall/hifigan)

## Example Usage

### Programmatic Usage

```python
import torch, torchaudio

# Load the content encoder (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

# Load the acoustic model (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load the vocoder (either hifigan_hubert_soft or hifigan_hubert_discrete)
hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()

# Load the source audio
source, sr = torchaudio.load("path/to/wav")
assert sr == 16000
source = source.unsqueeze(0).cuda()

# Convert to the target speaker
with torch.inference_mode():
    # Extract speech units
    units = hubert.units(source)
    # Generate target spectrogram
    mel = acoustic.generate(units).transpose(1, 2)
    # Generate audio waveform
    target = hifigan(mel)
```

## Citation

If you found this work helpful please consider citing our paper:

```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```