/**
 * Scrolling spectrogram (waterfall).
 *
 * This is the Phase 0 instrument: if the chirp is not visible here as a
 * diagonal stripe, nothing downstream can possibly work, and the cause is
 * almost always that the browser is still applying echo cancellation.
 *
 * Drawing uses a wrap-around write pointer into an offscreen canvas and two
 * blits per repaint, rather than scrolling pixels every frame. Columns arrive
 * in batches of ~32 per chirp, so a per-column scroll would be 375 full-canvas
 * copies a second for no visual benefit.
 */

export class SpectrogramView {
  private readonly ctx: CanvasRenderingContext2D;
  private buffer: HTMLCanvasElement;
  private bufferCtx: CanvasRenderingContext2D;
  private column = 0;
  private bins = 0;
  private hzPerBin = 0;
  private columnImage: ImageData | null = null;
  private readonly palette: Uint8ClampedArray;
  /** Frequency range displayed, Hz. */
  private minHz = 0;
  private maxHz = 24000;

  constructor(
    canvas: HTMLCanvasElement,
    private readonly width = 900,
    private readonly height = 260,
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.ctx.imageSmoothingEnabled = false;

    this.buffer = document.createElement('canvas');
    this.buffer.width = width;
    this.buffer.height = height;
    const bctx = this.buffer.getContext('2d');
    if (!bctx) throw new Error('2D canvas context unavailable');
    this.bufferCtx = bctx;
    this.bufferCtx.fillStyle = '#05070d';
    this.bufferCtx.fillRect(0, 0, width, height);

    this.palette = buildPalette();
  }

  setBand(minHz: number, maxHz: number): void {
    this.minHz = minHz;
    this.maxHz = maxHz;
  }

  clear(): void {
    this.bufferCtx.fillStyle = '#05070d';
    this.bufferCtx.fillRect(0, 0, this.width, this.height);
    this.column = 0;
  }

  /**
   * @param data  columns * bins bytes, each a dB value mapped to 0..255
   */
  push(data: Uint8Array, bins: number, hzPerBin: number): void {
    if (bins <= 0) return;
    if (bins !== this.bins || !this.columnImage) {
      this.bins = bins;
      this.columnImage = this.bufferCtx.createImageData(1, this.height);
    }
    this.hzPerBin = hzPerBin;
    const columns = Math.floor(data.length / bins);
    const img = this.columnImage;
    const px = img.data;

    for (let c = 0; c < columns; c++) {
      const base = c * bins;
      for (let y = 0; y < this.height; y++) {
        // Bottom of the canvas is minHz, top is maxHz.
        const frac = 1 - y / (this.height - 1);
        const hz = this.minHz + frac * (this.maxHz - this.minHz);
        const bin = Math.round(hz / this.hzPerBin);
        const v = bin >= 0 && bin < bins ? data[base + bin] : 0;
        const p = v * 4;
        const o = y * 4;
        px[o] = this.palette[p];
        px[o + 1] = this.palette[p + 1];
        px[o + 2] = this.palette[p + 2];
        px[o + 3] = 255;
      }
      this.bufferCtx.putImageData(img, this.column, 0);
      this.column = (this.column + 1) % this.width;
    }
  }

  render(): void {
    const { ctx, height } = this;
    // Two blits stitch the ring buffer into a left-to-right timeline with the
    // newest column at the right edge.
    const tail = this.width - this.column;
    ctx.drawImage(this.buffer, this.column, 0, tail, height, 0, 0, tail, height);
    ctx.drawImage(this.buffer, 0, 0, this.column, height, tail, 0, this.column, height);
    this.drawAxis();
  }

  private drawAxis(): void {
    const { ctx, width, height } = this;
    ctx.save();
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.fillStyle = 'rgba(220,230,245,0.75)';
    ctx.lineWidth = 1;

    const step = this.maxHz - this.minHz > 12000 ? 4000 : 2000;
    const first = Math.ceil(this.minHz / step) * step;
    for (let hz = first; hz <= this.maxHz; hz += step) {
      const y = Math.round(
        (1 - (hz - this.minHz) / (this.maxHz - this.minHz)) * (height - 1),
      ) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.fillText(`${(hz / 1000).toFixed(0)}k`, 6, y - 8);
    }
    ctx.restore();
  }

  /** Draw a band highlight so the user can see where the chirp should appear. */
  markBand(f0: number, f1: number): void {
    const { ctx, width, height } = this;
    const toY = (hz: number) =>
      (1 - (hz - this.minHz) / (this.maxHz - this.minHz)) * (height - 1);
    const yTop = toY(Math.max(f0, f1));
    const yBottom = toY(Math.min(f0, f1));
    ctx.save();
    ctx.strokeStyle = 'rgba(120, 220, 255, 0.55)';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, yTop + 0.5);
    ctx.lineTo(width, yTop + 0.5);
    ctx.moveTo(0, yBottom + 0.5);
    ctx.lineTo(width, yBottom + 0.5);
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * An "inferno"-like ramp: black through purple and red to yellow-white.
 * Perceptually monotonic in lightness, so faint returns stay visible.
 */
function buildPalette(): Uint8ClampedArray {
  const stops: [number, number, number, number][] = [
    [0.0, 4, 6, 14],
    [0.2, 34, 12, 64],
    [0.4, 108, 22, 92],
    [0.6, 186, 54, 62],
    [0.8, 240, 126, 32],
    [1.0, 252, 246, 190],
  ];
  const out = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = stops[0];
    let b = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) {
        a = stops[s];
        b = stops[s + 1];
        break;
      }
    }
    const span = b[0] - a[0] || 1;
    const k = (t - a[0]) / span;
    out[i * 4] = a[1] + (b[1] - a[1]) * k;
    out[i * 4 + 1] = a[2] + (b[2] - a[2]) * k;
    out[i * 4 + 2] = a[3] + (b[3] - a[3]) * k;
    out[i * 4 + 3] = 255;
  }
  return out;
}
