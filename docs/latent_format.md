# Latent format

The release expects latents to be generated or exported separately. Each latent
file must preserve the relative path of its source image and use the `.pt`
extension.

```text
images/
  train/<class>/<sample>.jpg
  val/<class>/<sample>.jpg
pc_latents/
  train/<class>/<sample>.pt
  val/<class>/<sample>.pt
audio_latents/
  train/<class>/<sample>.pt
  val/<class>/<sample>.pt
```

The loaders match files by relative path. A complete sample therefore has the
same `<class>/<sample>` path in all three roots.

## Tensor shapes

- 3D latents: floating point tensor `[tokens, channels]` or `[channels, tokens]`;
  the default configuration uses 64 channels and pads/truncates to 256 tokens.
- Audio latents: integer tensor `[codebooks, time]`; the default configuration
  uses four codebooks and pads/truncates to 250 time steps.

The audio vocabulary is 2048 by default. Change the CLI values only when the
latents were produced with a different codec configuration.
