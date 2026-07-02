import { useEffect, useMemo, useState } from 'react'
import Curve from './Curve.jsx'
import copy from './copy.js'

const BUDGET_ORDER = { '100k': 0, '5k': 1 }
const CAP_ORDER = { '2x64': 0, '3x256': 1, '4x512': 2 }

function sortModels(models) {
  return [...models].sort((a, b) =>
    (BUDGET_ORDER[a.budget] ?? 9) - (BUDGET_ORDER[b.budget] ?? 9) ||
    (CAP_ORDER[a.capacity] ?? 9) - (CAP_ORDER[b.capacity] ?? 9))
}

function Chip({ subtest, selected, onClick }) {
  return (
    <span className={`chip ${selected ? 'selected' : ''}`} onClick={onClick}
          role="button" tabIndex={0}>
      <span className={`dot ${subtest.light}`} />
      {subtest.score.toFixed(2)}
    </span>
  )
}

function LightLegend({ config }) {
  return (
    <div className="light-legend" aria-label="traffic light legend">
      <span><span className="dot green" /> ≤ {config.light_green} {copy['legend-green']}</span>
      <span><span className="dot yellow" /> ≤ {config.light_yellow} {copy['legend-yellow']}</span>
      <span><span className="dot red" /> &gt; {config.light_yellow} {copy['legend-red']}</span>
    </div>
  )
}

function Detail({ scene, model, subtest, config, namebrand }) {
  const [mi, setMi] = useState(4)
  const entries = subtest.entries
  const entry = entries[Math.min(mi, entries.length - 1)]
  return (
    <div className="card">
      <h3>
        {model.model_id} — {subtest.name}
        {subtest.diverged && <span className="badge-diverged">diverged</span>}
      </h3>
      <div className="sub">
        <span className={`dot ${subtest.light}`} style={{ display: 'inline-block', marginRight: 6 }} />
        score {subtest.score.toFixed(3)} · poke site: {entry.poke_id}
      </div>
      <div className="detail">
        <div>
          <Curve curves={entry.curves}
                 marker={entry.curves.x_label.includes('(N)') ? mi : null} />
          <div className="legend">
            <span><span className="swatch" style={{ background: 'var(--ink)' }} />simulated ground truth</span>
            <span><span className="swatch" style={{ background: 'var(--series-1)' }} />
              {subtest.details?.K === 1 ? 'model (deterministic)' : 'model mean (band: K-sample 10–90%)'}</span>
          </div>
          {subtest.name === 'momentum' && <div className="note">{copy['momentum-note']}</div>}
          {subtest.diverged && <div className="note">{copy['divergence-note']}</div>}
        </div>
        <div>
          {entry.clip_url ? (
            <>
              <video key={entry.clip_url} autoPlay loop muted playsInline controls>
                <source src={entry.clip_url} type="video/webm" />
                <source src={entry.clip_url.replace('.webm', '.mp4')} type="video/mp4" />
              </video>
              <div className="slider-row">
                <input type="range" min="0" max={entries.length - 1} value={mi}
                       aria-label="poke magnitude"
                       onChange={(e) => setMi(+e.target.value)} />
                <span className="mag">
                  {entry.magnitude.toFixed(2)}{entry.curves.x_label.includes('(N)') ? ' N' : ''}
                </span>
              </div>
              <div className="note">
                {namebrand ? copy['namebrand-clip-caption'] : copy['clip-caption']}
              </div>
            </>
          ) : (
            <div className="note">No clip for this row; the curve is the audit
              output. Poke channel: {entry.poke_id}.</div>
          )}
        </div>
      </div>
    </div>
  )
}

function NamebrandCard({ nb }) {
  const r2 = Object.entries(nb.r2_per_coord)
  return (
    <div className="card">
      <p className="intro">{copy['namebrand-caveat']}</p>
      {nb.killed && <p className="intro" style={{ color: 'var(--crit)' }}>
        {copy['namebrand-killed']}</p>}
      <div className="sub" style={{ marginBottom: 8 }}>
        agent return (mean): {nb.agent_return_mean} · decoder: {nb.decoder_type} ·
        min observable-position R²: {nb.min_observable_pos_r2} ·
        excluded: {nb.excluded_coords.join(', ')}
      </div>
      <table className="report" style={{ maxWidth: 560 }}>
        <thead><tr><th>coordinate</th><th>holdout R²</th></tr></thead>
        <tbody>
          {r2.map(([k, v]) => (
            <tr key={k}>
              <td className="model-id">{k}</td>
              <td style={{ fontVariantNumeric: 'tabular-nums',
                           color: v < 0.9 ? 'var(--crit)' : 'var(--ink)' }}>
                {v.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ScenePage({ scene, config }) {
  const models = useMemo(() => sortModels(scene.models), [scene])
  const subtestNames = models[0]?.subtests.map((s) => s.name) ?? []
  const [sel, setSel] = useState({ model: 0, subtest: 0 })
  const model = models[sel.model]
  const subtest = model?.subtests[sel.subtest]
  const nb = models[0]?.namebrand
  return (
    <>
      {nb && <NamebrandCard nb={nb} />}
      <div className="card">
        <p className="intro">{copy['report-intro']}</p>
        <LightLegend config={config} />
        <table className="report">
          <thead>
            <tr>
              <th>model</th><th>capacity</th><th>data</th><th>holdout NLL</th>
              {subtestNames.map((n) => <th key={n}>{n}</th>)}
            </tr>
          </thead>
          <tbody>
            {models.map((m, i) => (
              <tr key={m.model_id}>
                <td className="model-id">{m.model_id}</td>
                <td>{m.capacity}</td>
                <td>{m.budget}</td>
                <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {m.holdout_nll == null ? '—'
                    : Math.abs(m.holdout_nll) >= 1e4 ? m.holdout_nll.toExponential(1)
                    : m.holdout_nll.toFixed(2)}
                </td>
                {m.subtests.map((s, j) => (
                  <td key={s.name}>
                    <Chip subtest={s}
                          selected={sel.model === i && sel.subtest === j}
                          onClick={() => setSel({ model: i, subtest: j })} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {model && subtest &&
        <Detail scene={scene} model={model} subtest={subtest} config={config}
                namebrand={!!nb} />}
    </>
  )
}

function MethodPage({ config }) {
  const thresholds = copy['method-thresholds']
    .replace('{green}', config.light_green)
    .replace('{yellow}', config.light_yellow)
    .replace('{k}', config.k_samples)
  return (
    <div className="card method">
      <h2>{copy['method-title']}</h2>
      <p>{copy['method-body']}</p>
      <p>{thresholds}</p>
      <p>{copy['momentum-note']}</p>
      <p>{copy['divergence-note']}</p>
    </div>
  )
}

export default function App() {
  const [manifest, setManifest] = useState(null)
  const [tab, setTab] = useState(0)
  useEffect(() => {
    fetch('manifest.json').then((r) => r.json()).then(setManifest)
  }, [])
  if (!manifest) return <div className="loading">loading manifest…</div>
  const tabs = [...manifest.scenes.map((s) => s.name), 'method']
  return (
    <div className="wrap">
      <h1>{copy.title}</h1>
      <p className="tagline">{copy.tagline}</p>
      <div className="tabs">
        {tabs.map((t, i) => (
          <button key={t} className={`tab ${tab === i ? 'active' : ''}`}
                  onClick={() => setTab(i)}>{t}</button>
        ))}
      </div>
      {tab < manifest.scenes.length
        ? <ScenePage scene={manifest.scenes[tab]} config={manifest.config} />
        : <MethodPage config={manifest.config} />}
      <footer>{copy.footer}</footer>
    </div>
  )
}
