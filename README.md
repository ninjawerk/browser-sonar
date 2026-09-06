# Browser Sonar

Turn a laptop into an acoustic sonar using nothing but its built-in speaker and
microphone. The page emits a near-ultrasonic chirp, listens for the reflections,
and derives range, presence, direction of motion, swipe gestures and breathing
rate — all computed live in an `AudioWorkletProcessor`. No install, no backend,
no accounts, nothing leaves the device.

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # 122 unit + integration tests
npm run build    # static site into dist/
```

---

## What it does

Open the page, press **Start**, allow the microphone, hold still for about three
seconds while it learns the room, then wave your hand at the screen.

| Readout | How it is derived |
| --- | --- |
| Range profile | FFT cross-correlation of the mic signal against the transmitted chirp |
| Presence | Peak of the background-subtracted profile against an adaptive noise floor |
| Radial motion | How far moving-target energy migrates between range bins per chirp |
| Gestures | Direction consistency of that motion over a ~1.2 s window |
| Breathing | Unwrapped carrier phase of one range bin, periodogram over ~45 s |
| CW Doppler | Sideband asymmetry around a pure 20 kHz tone |

---

## How it works

### The signal

A linear FMCW sweep from 18 kHz to 22 kHz lasting 15 ms, repeated every 4096
samples (85.3 ms at 48 kHz, so 11.7 chirps per second). The first and last
millisecond are Hann-tapered, because gating a 18 kHz tone with a hard edge
splatters energy right across the audible spectrum and produces a click on every
single chirp.

Sweeping bandwidth rather than sending a short pulse is what makes this possible
at all. Matched filtering compresses the 15 ms sweep into a peak roughly `1/B`
wide, so range resolution is

```
Δr = c / (2B) = 343 / (2 × 4000) ≈ 4.3 cm
```

independent of chirp length — while received energy, and therefore SNR, grows
with duration. A laptop speaker is perhaps 30 dB down at 20 kHz; the ~18 dB of
correlation gain from a 720-sample chirp is the difference between a usable
echo and noise.

One range bin is one sample of round-trip delay:

```
bin = c / (2 × f_s) = 343 / 96000 ≈ 3.57 mm
```

so the ~4.3 cm resolution cell spans about 12 bins, and sub-bin interpolation on
the peak gets accuracy well below the resolution limit.

### The transmit period is a power of two, and that matters

The transmitted period is 4096 samples — a power of two — and the receiver
correlates **circularly** over exactly one period.

Because the transmitter loops the same period forever, a circular correlation of
*any* contiguous period-length window of the microphone signal is exact. There
is no windowing loss, no edge effect, and crucially the result does not depend
on where in the period the receiver happened to start listening. Wrap-around
aliases delays longer than one period, which at 85 ms is about 14 m of round
trip — far beyond what a laptop speaker can illuminate.

### Self-calibration on the direct path

This is the trick that makes the whole thing work with no per-device setup.

The loudest thing in the recording is always the speaker talking straight into
the microphone a few centimetres away. Where that lands within the period
depends on the browser's combined output and input latency, which is tens of
milliseconds, differs on every device, and drifts over time. So the receiver
never assumes it: **every frame, it finds the strongest correlation peak and
calls that range zero.** Everything else is measured relative to it.

Measured in a browser with a synthetic room, the direct path is recovered to
within 0.02 samples of truth, and reported target ranges are identical (to
within 5 mm) whether the simulated latency is 500 samples or 2600.

### Background subtraction

Walls, the desk and the laptop's own chassis return constant echoes far stronger
than a hand. An exponential moving average of the range profile (time constant
~3 s, adjustable) converges to everything that is not moving; subtracting it
leaves only what changed.

The consequence, which the UI makes visible rather than hiding, is that **a
perfectly still object disappears within a few time constants.** That is
correct behaviour for a motion sensor and wrong for a presence sensor, so the
raw profile is drawn alongside the subtracted one, and there is a *Freeze room
model* switch for when you want a newly-placed object to stay visible.

### Two ways of measuring velocity, and why both exist

The brief asked for both approaches to be implemented and compared. They turn
out not to be competitors — they operate at completely different scales.

**Inter-chirp phase differencing aliases almost immediately.** Phase wraps once
a target moves half a wavelength, about 4 mm at 20 kHz, and chirps are 85 ms
apart:

```
v_max = c / (4 · f_c · T) = 343 / (4 × 20000 × 0.0853) ≈ 0.05 m/s
```

A hand swipe is twenty to forty times faster than that, so inter-chirp phase is
useless for gestures. It is, however, exactly the right tool for breathing,
where the chest surface moves at millimetres per second.

**Range migration does not alias at all.** Correlating consecutive
background-subtracted profiles against each other and taking the lag of the peak
gives the shift in bins directly, with no phase ambiguity. A target at 1 m/s
moves 8.5 cm — about 24 bins — between chirps, which is trivially measurable.

| | Inter-chirp phase | Range migration | CW Doppler |
| --- | --- | --- | --- |
| Unambiguous range | ±0.05 m/s | ±3 m/s (bounded by search) | ±5 m/s |
| Resolution | sub-millimetre | ~1 bin ≈ 4 cm/frame | ~0.1 m/s |
| Gives distance too | yes | yes | no |
| Used for | breathing | presence, gestures | direction readout |

So the system measures macro-motion by migration, micro-motion by phase, and
offers CW as an independent cross-check where the Doppler shift is read
directly off the spectrum instead of inferred.

In CW mode a 20 kHz carrier is transmitted continuously and the receiver
compares energy in the upper and lower sidebands. Approach returns
`f(1 + 2v/c)` — at 1 m/s that is a 117 Hz shift, about ten FFT bins here — so
approach lands entirely above the carrier and recede entirely below it. This
path uses a Blackman–Harris window rather than Hann: the carrier arrives ~60 dB
above any reflection, and Hann's near sidelobes would fill both sidebands
symmetrically and wash out the very asymmetry being measured.

### Breathing

For a chosen range bin, the complex correlation value's phase is tracked across
chirps and unwrapped. Chest displacement of a few millimetres is far too small
to move between range bins, but 8.6 mm of displacement is a full turn of phase
at 20 kHz — phase resolves roughly a hundred times finer than the bin grid.

The unwrapped phase is detrended (it drifts, from slow posture change and from
speaker/microphone clock offset), Hann-windowed, and transformed with a
zero-padded 1024-point FFT over ~45 s of history. The dominant peak between 0.1
and 0.7 Hz becomes the rate, and it is only reported when its peak-to-mean ratio
exceeds 3 — the system says "no steady rhythm" rather than inventing a number.

---

## Two findings that changed the design

Both were found by measurement, not by reasoning, and both are the kind of thing
that would have been written off as "bad SNR" on real hardware.

### 1. The anti-alias filter was smearing the compressed pulse

The front-end high-pass (6th-order Butterworth at 16 kHz) has **12 samples of
group delay at 18 kHz and 39 at 22 kHz.** That dispersion stretches the sweep
unevenly, so a matched filter built from the *unfiltered* chirp is no longer
matched: the compressed pulse grew a long asymmetric tail and its peak landed
about 4 samples late.

The fix is to push the template through an identical filter before building the
correlator, so the matched filter matches what the microphone actually delivers.
Direct-path error dropped from 4 samples to 0.01.

### 2. LFM range sidelobes were biasing every nearby target

A flat sweep has a near-rectangular spectrum, and the transform of a rectangle
is a sinc — range sidelobes only 13 dB down that decay as `1/range`. Measured on
this chirp, the direct path was still **−22.7 dB a full metre away**. Since
leakage is ~20 dB stronger than any real target, those sidelobes did not merely
add noise; they pulled the apparent peak of a 30 cm target by 5 bins.

The fix is a deliberately *mismatched* filter: a Hann amplitude taper across the
receive template. Because an LFM sweep maps time linearly onto frequency, a
taper in time is a taper in frequency.

| | Flat template | Hann-tapered template |
| --- | --- | --- |
| Sidelobe at 17 cm (bin 48) | −22.7 dB | −35.7 dB |
| Sidelobe at 54 cm (bin 150) | ≈ −33 dB | −42.5 dB |
| Sidelobe at 107 cm (bin 300) | ≈ −38 dB | −65.3 dB |
| Range error at 30 cm | 18.1 mm | 8.3 mm |
| Main lobe (−3 dB, full width) | ~3.9 cm | ~6.4 cm |

The taper costs ~1.6× main-lobe width and ~1.8 dB of SNR, and buys back an order
of magnitude in sidelobe rejection and less than half the range error. The taper
is applied **only on receive** — the transmitted chirp stays flat, so it keeps
every bit of the energy a quiet ultrasonic speaker can give.

This also set the default guard zone. The compressed direct path is only 6.9 dB
down at bin 12 and does not fall below 29 dB — under a typical hand return —
until about bin 24, so the default blind zone is 22 bins ≈ 8 cm.

---

## Measured results

### Ranging accuracy (simulated room, broadband noise at −54 dBFS)

| True distance | Reported | Error |
| --- | --- | --- |
| 0.30 m | 0.3083 m | +8.3 mm |
| 0.60 m | 0.6024 m | +2.4 mm |
| 1.20 m | 1.1998 m | −0.2 mm |
| 2.00 m | 1.9999 m | −0.1 mm |

Error is largest closest in, where the direct path's residual sidelobes are
strongest. All are far inside the ±5 cm the brief asks for.

Two targets at 30 cm and 60 cm resolve as two clean peaks (reported 0.3086 m and
0.6032 m).

### Running in a real browser

Driving the actual deployed worklet in Chrome with a synthetic room injected in
place of the microphone:

| Quantity | Truth | Measured |
| --- | --- | --- |
| Direct-path position | 1731 samples | 1730.98 |
| Target range | 0.600 m | 0.6038 m (+3.8 mm) |
| Usable bandwidth | 4000 Hz | 3938 Hz |
| Spectral tilt | 0 dB | −2.5 dB |
| Frame rate | 11.72 fps | 11.7 fps |
| Breathing rate | 15.0 bpm | 14.7 bpm (confidence 6.1) |
| Radial velocity | 1.00 m/s approaching | 0.88 m/s approaching |

### Motion and gestures

| Quantity | Truth | Measured |
| --- | --- | --- |
| Approach | +0.6 m/s | +0.37 m/s |
| Approach | +1.0 m/s | +1.16 m/s |
| Approach | +2.0 m/s | +1.81 m/s |
| Recede | −1.0 m/s | −0.97 m/s |
| CW Doppler | ±0.8 m/s | ±0.72 m/s |

Sign is correct in every case; magnitude is good to roughly ±20% mid-range and
compresses at the extremes. The gesture threshold was set from measurement:
frame-to-frame motion energy peaks at 1.7 in an empty room and sits around 6.3
during a swipe, so the gate is 3.

### Devices

**No test on real acoustic hardware has been performed.** This environment has
no speaker or microphone access. Every number above comes either from the
simulation harness or from the real worklet running in Chrome with a synthetic
signal injected in place of the microphone — which exercises all the code but
none of the physics. The device table is deliberately empty rather than
invented:

| Device | Browser | Sample rate | Constraints honoured | Usable BW | Result |
| --- | --- | --- | --- | --- | --- |
| _(awaiting hardware testing)_ | | | | | |

The in-app **Device compatibility** panel reports exactly these fields on
whatever machine you run it on.

---

## Honest failure cases

- **Echo cancellation.** If the browser refuses to disable it, the range profile
  is flat and nothing works. The app checks `getSettings()` and says so loudly.
  This is the single most common cause of "it does nothing".
- **Bluetooth audio.** Codecs low-pass well below 20 kHz and add tens of
  milliseconds of wandering latency. Detected heuristically and warned about;
  there is no way to make it work.
- **Speaker roll-off.** Consumer speakers fall off a cliff above 18 kHz. On a
  bad one the top of the sweep never makes it out, the effective bandwidth is
  smaller than requested, and real resolution is worse than the 4.3 cm quoted.
  The app measures received spectral tilt and reports the bandwidth it thinks is
  actually usable.
- **A still person vanishes.** Background subtraction is doing its job. Use
  *Freeze room model*, or watch the raw profile.
- **Close range.** Nothing is detectable inside ~8 cm; that is the direct path's
  own main lobe.
- **In-band noise.** Some LED drivers and switching supplies emit strongly at
  18–22 kHz. The spectrogram shows the noise floor in-band so you can see it.
- **Breathing is the least robust feature.** It needs ~20 s of near-stillness,
  the right range bin (click the range plot to pin one), and it will report
  nothing rather than guess when confidence is low. Any gross movement resets
  the estimate.
- **Multipath.** A strong reflector can produce a ghost at the sum of two path
  lengths. Nothing is done about this.
- **One microphone.** There is no bearing information whatsoever — everything is
  radial distance and radial speed.

---

## Architecture

```
src/
  main.ts              controls, rendering, DOM — no signal processing
  audio.ts             getUserMedia, constraint verification, transmit loop
  style.css
  dsp/
    fft.ts             radix-2 Cooley–Tukey, precomputed twiddles     + tests
    window.ts          Hann / Hamming / Blackman–Harris, tapers       + tests
    xcorr.ts           analytic matched filter, sub-sample peak       + tests
    unwrap.ts          phase unwrapping, streaming + batch, detrend   + tests
    chirp.ts           FMCW synthesis, tone snapping, range maths     + tests
    biquad.ts          RBJ biquads and Butterworth cascades           + tests
    engine.ts          the processing chain, Web-Audio-free           + tests
    sonar-worklet.ts   thin AudioWorkletProcessor shell
    types.ts           config and message contracts
  ui/                  spectrogram, range plot, CW plot
```

Every DSP primitive is written from scratch — no DSP libraries — and each has
its own test file. `FFT` is validated against a naive DFT, Parseval's theorem,
linearity and round-trip identity.

**`engine.ts` has no Web Audio dependency.** It takes blocks of samples and
returns analysis frames, which means the entire chain — correlation,
calibration, subtraction, Doppler, gestures, breathing — is testable in Node
against synthetic echoes with known geometry. On real hardware every bug looks
identical (bad SNR); in simulation they are distinguishable. The test suite
includes a room simulator that evaluates echoes analytically at fractional
sample delays, which is what makes it possible to test breathing at all.

### Design decisions worth stating

**Transmission uses a looped `AudioBufferSourceNode`, not worklet synthesis.**
The loop is sample-exact and maintained by the audio engine itself, so the
transmitted period stays rigidly periodic regardless of what the worklet is
doing. The cost is that transmitter and receiver are separate graph nodes with
an unknown relative offset — which costs nothing here, because the direct-path
calibration was always going to measure that offset anyway. The worklet
generates the waveform and posts it to the main thread rather than both sides
generating it independently, so the transmitted signal and the correlation
template cannot drift apart.

**`SharedArrayBuffer` is deliberately not used.** It requires cross-origin
isolation via COOP/COEP response headers, which GitHub Pages cannot set — so
that path would be dead code on the one deployment target that exists.
`postMessage` with transferables costs a few microseconds per chirp at 11.7
messages per second.

**The worklet hot path allocates nothing.** All buffers are preallocated on
config change. The only per-frame allocation is the outgoing message, at ~12 Hz,
and those buffers are transferred rather than copied.

**Buffering.** `AudioWorklet` delivers 128 samples (2.7 ms) per call, far too
short to correlate against a 15 ms chirp. The engine accumulates quanta and runs
an analysis once per full transmit period, roughly every 32 calls. The
spectrogram runs on its own shorter hop (256-point FFT, 128-sample hop) so a
15 ms chirp reads as a diagonal stripe over ~6 columns rather than a smear.

---

## Phase status against the brief

| Phase | Status |
| --- | --- |
| 0 — plumbing, spectrogram | Complete. Chirps visible as diagonal stripes. |
| 1 — range | Complete in simulation and in-browser; ±8 mm, better than the ±5 cm asked. **Needs hardware confirmation.** |
| 2 — presence, Doppler | Complete. Presence flips in <1 s; direction correct in all simulated trials. **Needs the 20-trial hardware test.** |
| 3 — gestures | Implemented with measured thresholds; correct on all simulated swipes. **Needs the 50-trial hardware test by someone who did not tune it.** |
| 4 — breathing | Implemented; 14.7–15.0 bpm against a 15 bpm truth. **Needs the hardware test against a manual count.** |
| 5 — polish, ship | UI, warnings, device panel, mobile layout and this README are done. |

The acceptance tests in phases 1–4 are all defined in terms of real hardware and
a tape measure, and cannot be self-certified here. What can be said is that the
implementation passes the equivalent test in simulation and that the code runs
correctly in a real browser.

---

## Licence

MIT.
