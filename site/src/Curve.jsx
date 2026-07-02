// Inline SVG chart: truth reference (dashed ink) vs model mean (series blue)
// with the K-sample band. No chart lib; nulls render as gaps.
const W = 520, H = 250, M = { l: 52, r: 14, t: 12, b: 34 }

function scales(curves) {
  const xs = curves.x
  const xLog = (curves.x_label || '').includes('(N)')
  const fin = (a) => a.filter((v) => v != null && isFinite(v))
  const ys = [...fin(curves.y_truth), ...fin(curves.y_model),
              ...fin(curves.ci?.lo || []), ...fin(curves.ci?.hi || [])]
  const ymin = Math.min(0, ...ys), ymax = Math.max(...ys, 1e-9)
  const tx = (v) => {
    const [a, b] = xLog ? [Math.log(xs[0]), Math.log(xs[xs.length - 1])] : [xs[0], xs[xs.length - 1]]
    const u = xLog ? Math.log(v) : v
    return M.l + ((u - a) / (b - a || 1)) * (W - M.l - M.r)
  }
  const ty = (v) => H - M.b - ((v - ymin) / (ymax - ymin || 1)) * (H - M.t - M.b)
  return { tx, ty, ymin, ymax, xLog }
}

function path(xs, ys, tx, ty) {
  let d = '', pen = false
  xs.forEach((x, i) => {
    const y = ys[i]
    if (y == null || !isFinite(y)) { pen = false; return }
    d += `${pen ? 'L' : 'M'}${tx(x).toFixed(1)},${ty(y).toFixed(1)}`
    pen = true
  })
  return d
}

function fmt(v) {
  if (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)) return v.toExponential(0)
  return +v.toFixed(2)
}

export default function Curve({ curves, marker }) {
  const { tx, ty, ymin, ymax } = scales(curves)
  const xs = curves.x
  const lo = curves.ci?.lo || [], hi = curves.ci?.hi || []
  let band = ''
  const ok = xs.map((_, i) => lo[i] != null && hi[i] != null && isFinite(lo[i]) && isFinite(hi[i]))
  if (ok.some(Boolean)) {
    const up = xs.map((x, i) => ok[i] ? `${tx(x)},${ty(hi[i])}` : null).filter(Boolean)
    const dn = xs.map((x, i) => ok[i] ? `${tx(x)},${ty(lo[i])}` : null).filter(Boolean).reverse()
    band = `M${up.join('L')}L${dn.join('L')}Z`
  }
  const yticks = [ymin, (ymin + ymax) / 2, ymax]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="sub-test curve"
         style={{ width: '100%', height: 'auto' }}>
      {yticks.map((v, i) => (
        <g key={i}>
          <line x1={M.l} x2={W - M.r} y1={ty(v)} y2={ty(v)} stroke="var(--grid)" strokeWidth="1" />
          <text x={M.l - 6} y={ty(v) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{fmt(v)}</text>
        </g>
      ))}
      <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke="var(--baseline)" strokeWidth="1" />
      {xs.map((x, i) => (
        <text key={i} x={tx(x)} y={H - M.b + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">
          {fmt(x)}
        </text>
      ))}
      <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--muted)">
        {curves.x_label}
      </text>
      {band && <path d={band} fill="var(--series-1)" opacity="0.18" />}
      <path d={path(xs, curves.y_model, tx, ty)} fill="none" stroke="var(--series-1)" strokeWidth="2" />
      <path d={path(xs, curves.y_truth, tx, ty)} fill="none" stroke="var(--ink)"
            strokeWidth="2" strokeDasharray="5 3" />
      {marker != null && xs[marker] != null && (
        <line x1={tx(xs[marker])} x2={tx(xs[marker])} y1={M.t} y2={H - M.b}
              stroke="var(--muted)" strokeWidth="1" strokeDasharray="2 3" />
      )}
    </svg>
  )
}
