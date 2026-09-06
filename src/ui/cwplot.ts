/**
 * CW Doppler spectrum: the transmitted tone with its reflection sidebands.
 *
 * This is the plot where approach and recede are directly visible rather than
 * inferred. Everything to the right of the carrier is energy that came back at
 * a higher frequency — a reflector closing on the microphone — and everything
 * to the left came back lower. The carrier itself is clipped off the top of
 * the scale on purpose; it is 40–60 dB above anything interesting.
 */

export class CwPlotView {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    canvas: HTMLCanvasElement,
    private readonly width = 900,
    private readonly height = 260,
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
  }

  render(spectrum: Float32Array | null, hzPerBin: number, velocity: number): void {
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#080b14';
    ctx.fillRect(0, 0, width, height);

    if (!spectrum || spectrum.length === 0) {
      ctx.fillStyle = 'rgba(220,230,245,0.5)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.fillText('CW spectrum appears in continuous-tone mode', 24, height / 2);
      return;
    }

    const padL = 46;
    const padR = 12;
    const padT = 12;
    const padB = 26;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const centre = (spectrum.length - 1) >> 1;

    // Fixed dB window: relative levels matter more than absolute here, and a
    // sliding scale would make the sidebands appear to breathe.
    const topDb = -20;
    const bottomDb = -110;
    const x = (i: number) => padL + (i / (spectrum.length - 1)) * plotW;
    const y = (db: number) =>
      padT + plotH - ((Math.min(topDb, Math.max(bottomDb, db)) - bottomDb) / (topDb - bottomDb)) * plotH;

    // Sideband shading.
    ctx.fillStyle = 'rgba(120, 255, 170, 0.07)';
    ctx.fillRect(x(centre), padT, plotW / 2, plotH);
    ctx.fillStyle = 'rgba(255, 140, 140, 0.07)';
    ctx.fillRect(padL, padT, x(centre) - padL, plotH);

    // Grid.
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.fillStyle = 'rgba(210,222,240,0.7)';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let db = topDb; db >= bottomDb; db -= 20) {
      const py = Math.round(y(db)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.fillText(`${db}`, 8, py + 4);
    }
    // Frequency offset ticks, in Hz relative to the carrier.
    const spanHz = ((spectrum.length - 1) / 2) * hzPerBin;
    const stepHz = spanHz > 500 ? 250 : 100;
    for (let f = -Math.floor(spanHz / stepHz) * stepHz; f <= spanHz; f += stepHz) {
      const i = centre + f / hzPerBin;
      const px = Math.round(x(i)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.fillText(`${f > 0 ? '+' : ''}${f}`, px + 3, padT + plotH + 14);
    }
    ctx.restore();

    // Spectrum.
    ctx.save();
    ctx.strokeStyle = 'rgba(150, 210, 255, 0.95)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < spectrum.length; i++) {
      const px = x(i);
      const py = y(spectrum[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();

    // Carrier marker and readout.
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x(centre), padT);
    ctx.lineTo(x(centre), padT + plotH);
    ctx.stroke();
    ctx.restore();

    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = 'rgba(255,140,140,0.9)';
    ctx.fillText('receding', padL + 8, padT + 14);
    ctx.fillStyle = 'rgba(120,255,170,0.95)';
    const approachLabel = 'approaching';
    ctx.fillText(approachLabel, padL + plotW - 8 - ctx.measureText(approachLabel).width, padT + 14);
    ctx.fillStyle = 'rgba(220,230,245,0.85)';
    ctx.fillText(`${velocity >= 0 ? '+' : ''}${velocity.toFixed(2)} m/s`, x(centre) + 6, padT + plotH - 8);
    ctx.fillText('Hz from carrier', padL, padT + plotH + 14);
  }
}
