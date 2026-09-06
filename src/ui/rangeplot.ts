/**
 * Range profile plot: correlation magnitude against one-way distance.
 *
 * Both curves are drawn because they answer different questions. The raw
 * profile shows everything the room returns — useful for Phase 1, where you
 * hold a hand at a measured distance and look for a peak. The
 * background-subtracted profile shows only what changed, which is what
 * presence and gesture detection actually run on.
 */

import type { DetectedTarget } from '../dsp/types';

export interface RangePlotFrame {
  raw: Float32Array;
  subtracted: Float32Array;
  metersPerBin: number;
  targets: DetectedTarget[];
  guardBins: number;
  trackedBin: number | null;
}

export class RangePlotView {
  private readonly ctx: CanvasRenderingContext2D;
  /** Peak-hold on the vertical scale, so the plot does not jump about. */
  private scale = 0.05;
  private maxMeters = 3;

  constructor(
    canvas: HTMLCanvasElement,
    private readonly width = 900,
    private readonly height = 260,
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
  }

  setMaxRange(meters: number): void {
    this.maxMeters = meters;
  }

  render(frame: RangePlotFrame): void {
    const { ctx, width, height } = this;
    const padL = 46;
    const padR = 12;
    const padT = 12;
    const padB = 26;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#080b14';
    ctx.fillRect(0, 0, width, height);

    const binsShown = Math.min(
      frame.raw.length,
      Math.ceil(this.maxMeters / frame.metersPerBin),
    );
    if (binsShown <= 1) return;

    // Track the peak with a slow decay: an instantly-rescaling y axis makes
    // small changes look enormous and large ones look like nothing.
    let peak = 0;
    for (let i = frame.guardBins; i < binsShown; i++) {
      if (frame.raw[i] > peak) peak = frame.raw[i];
    }
    this.scale = Math.max(peak * 1.15, this.scale * 0.97, 0.005);

    const x = (bin: number) => padL + (bin / (binsShown - 1)) * plotW;
    const y = (v: number) => padT + plotH - Math.min(1, v / this.scale) * plotH;

    this.drawGrid(padL, padT, plotW, plotH, binsShown, frame.metersPerBin);

    // Guard region: the direct path and its immediate skirt, where nothing can
    // be measured.
    ctx.fillStyle = 'rgba(255, 90, 90, 0.10)';
    ctx.fillRect(padL, padT, Math.max(1, x(frame.guardBins) - padL), plotH);

    this.drawCurve(frame.raw, binsShown, x, y, 'rgba(120, 190, 255, 0.55)', 1.25);
    this.drawCurve(
      frame.subtracted,
      binsShown,
      x,
      y,
      'rgba(120, 255, 170, 0.95)',
      1.6,
      true,
    );

    // Detected targets. Targets arrive strongest-first; every one gets a
    // marker line, but a label is only drawn when there is room for it, so
    // that a cluster of nearby peaks does not print illegible overlapping text.
    ctx.save();
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const labelled: number[] = [];
    for (const t of frame.targets) {
      const bin = t.rangeMeters / frame.metersPerBin;
      if (bin > binsShown) continue;
      const px = x(bin);
      ctx.strokeStyle = 'rgba(255, 214, 102, 0.9)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      if (labelled.length < 3 && labelled.every((p) => Math.abs(p - px) > 46)) {
        ctx.fillStyle = 'rgba(255, 214, 102, 0.95)';
        ctx.fillText(`${(t.rangeMeters * 100).toFixed(0)} cm`, px + 4, padT + 12);
        labelled.push(px);
      }
    }
    ctx.restore();

    if (frame.trackedBin !== null && frame.trackedBin < binsShown) {
      const px = x(frame.trackedBin);
      ctx.strokeStyle = 'rgba(190, 140, 255, 0.9)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
    }

    // Legend.
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = 'rgba(120, 190, 255, 0.9)';
    ctx.fillText('raw', width - padR - 96, padT + 12);
    ctx.fillStyle = 'rgba(120, 255, 170, 0.95)';
    ctx.fillText('moving only', width - padR - 96, padT + 26);
  }

  private drawCurve(
    data: Float32Array,
    count: number,
    x: (b: number) => number,
    y: (v: number) => number,
    colour: string,
    lineWidth: number,
    clampNegative = false,
  ): void {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      const v = clampNegative ? Math.max(0, data[i]) : data[i];
      const px = x(i);
      const py = y(v);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  private drawGrid(
    padL: number,
    padT: number,
    plotW: number,
    plotH: number,
    binsShown: number,
    metersPerBin: number,
  ): void {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.fillStyle = 'rgba(210,222,240,0.7)';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.lineWidth = 1;

    const maxM = binsShown * metersPerBin;
    const stepCm = maxM > 2 ? 50 : 20;
    for (let cm = 0; cm <= maxM * 100; cm += stepCm) {
      const bin = cm / 100 / metersPerBin;
      const px = Math.round(padL + (bin / (binsShown - 1)) * plotW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      const label = `${cm}`;
      // Flip the last label inside the plot so it does not run off the canvas.
      const w = ctx.measureText(label).width;
      const overflows = px + 3 + w > padL + plotW;
      ctx.fillText(label, overflows ? px - 3 - w : px + 3, padT + plotH + 14);
    }
    // Unit marker lives in the left margin, clear of the tick labels.
    ctx.fillText('cm', 8, padT + plotH + 14);
    ctx.restore();
  }
}
