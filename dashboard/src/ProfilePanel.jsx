import React, { useEffect, useMemo, useState } from 'react';
import { ComposedChart, XAxis, YAxis, CartesianGrid, Line, Area, ReferenceLine, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || '';
const FETCH_HEADERS = { 'ngrok-skip-browser-warning': 'true' };
const REF_TEMP = 26.0;
const MAX_DEPTH_PLOT = 280;

/**
 * Leipper-style T(z) panel: Argo observation and RTOFS model on the same axes
 * with the 26°C reference line and the integrated heat-content region shaded.
 * X-axis: temperature (°C). Y-axis: depth (m), inverted (0 at top).
 */
export default function ProfilePanel({ castId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!castId) return;
    setData(null);
    setError(null);
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/profile?cast_id=${encodeURIComponent(castId)}`,
                              { headers: FETCH_HEADERS });
        if (!r.ok) throw new Error(await r.text());
        setData(await r.json());
      } catch (e) { setError(e.message); }
    })();
  }, [castId]);

  // Build a merged depth-indexed series: at each depth row, the Argo temperature
  // and the RTOFS temperature are aligned for plotting. We plot as two Lines.
  const merged = useMemo(() => {
    if (!data) return [];
    const a = data.argo;
    const m = data.model;
    const c = data.corrected;
    const rows = [];
    // Argo on its native depth grid
    for (let i = 0; i < a.depth_m.length; i++) {
      const d = a.depth_m[i];
      if (d <= MAX_DEPTH_PLOT) rows.push({ depth: d, argo_t: a.temperature_c[i], mod_t: null, corrected_t: null });
    }
    // Model on its grid
    for (let i = 0; i < m.depth_m.length; i++) {
      const d = m.depth_m[i];
      if (d <= MAX_DEPTH_PLOT) rows.push({ depth: d, argo_t: null, mod_t: m.temperature_c[i], corrected_t: null });
    }
    if (c.depth_m && c.temperature_c) {
      for (let i = 0; i < c.depth_m.length; i++) {
        const d = c.depth_m[i];
        if (d <= MAX_DEPTH_PLOT) rows.push({ depth: d, argo_t: null, mod_t: null, corrected_t: c.temperature_c[i] });
      }
    }
    rows.sort((a, b) => a.depth - b.depth);
    return rows;
  }, [data]);

  // Separate Area datasets for shading the two integrals.
  // Each bar spans from 26°C to T(z) for depths < D26, so we use (baseline=26, top=T).
  const argoShade = useMemo(() => {
    if (!data?.argo?.d26_m) return [];
    const d26 = data.argo.d26_m;
    const out = [];
    for (let i = 0; i < data.argo.depth_m.length; i++) {
      const d = data.argo.depth_m[i];
      if (d > d26) break;
      const t = data.argo.temperature_c[i];
      if (t >= REF_TEMP) out.push({ depth: d, excess: t });
    }
    out.push({ depth: d26, excess: REF_TEMP });
    return out;
  }, [data]);

  const modelShade = useMemo(() => {
    if (!data?.model?.d26_m) return [];
    const d26 = data.model.d26_m;
    const out = [];
    for (let i = 0; i < data.model.depth_m.length; i++) {
      const d = data.model.depth_m[i];
      if (d > d26) break;
      const t = data.model.temperature_c[i];
      if (t >= REF_TEMP) out.push({ depth: d, excess: t });
    }
    out.push({ depth: d26, excess: REF_TEMP });
    return out;
  }, [data]);

  const correctedShade = useMemo(() => {
    if (!data?.corrected?.d26_m || !data?.corrected?.depth_m || !data?.corrected?.temperature_c) return [];
    const d26 = data.corrected.d26_m;
    const out = [];
    for (let i = 0; i < data.corrected.depth_m.length; i++) {
      const d = data.corrected.depth_m[i];
      if (d > d26) break;
      const t = data.corrected.temperature_c[i];
      if (t >= REF_TEMP) out.push({ depth: d, excess: t });
    }
    out.push({ depth: d26, excess: REF_TEMP });
    return out;
  }, [data]);

  if (error) return <div style={{ color: 'var(--danger)' }}>Profile load failed: {error}</div>;
  if (!data) return <div style={{ color: 'var(--muted)' }}>Loading profile…</div>;

  const a = data.argo;
  const m = data.model;
  const c = data.corrected;
  const delta = data.delta_tchp_kj_cm2;

  return (
    <>
      <h4>T(z) — Argo vs RTOFS vs Final</h4>
      <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.4rem' }}>
        Cast {data.platform} · {data.obs_time} · ({data.lat.toFixed(2)}, {data.lon.toFixed(2)})
      </div>
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <ComposedChart layout="vertical" data={merged} margin={{ top: 8, right: 18, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#1f2b4a" strokeDasharray="3 4" />
            <XAxis type="number" domain={[23, 32]} tick={{ fontSize: 11, fill: '#94a3b8' }}
                   label={{ value: 'Temperature (°C)', position: 'insideBottom', offset: -4, fill: '#94a3b8', fontSize: 11 }} />
            <YAxis type="number" dataKey="depth" reversed domain={[0, MAX_DEPTH_PLOT]}
                   tick={{ fontSize: 11, fill: '#94a3b8' }}
                   label={{ value: 'Depth (m)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#0b1220', border: '1px solid #1f2b4a', fontSize: '0.8rem' }}
              labelFormatter={(v) => `Depth ${v} m`}
            />
            {/* Shaded integration regions */}
            <Area data={argoShade} dataKey="excess" stroke="none"
                  fill="#0ea5e9" fillOpacity={0.18} isAnimationActive={false} connectNulls />
            <Area data={modelShade} dataKey="excess" stroke="none"
                  fill="#f97316" fillOpacity={0.18} isAnimationActive={false} connectNulls />
            <Area data={correctedShade} dataKey="excess" stroke="none"
                  fill="#22c55e" fillOpacity={0.16} isAnimationActive={false} connectNulls />
            {/* Curves */}
            <Line type="monotone" dataKey="argo_t" stroke="#0ea5e9" strokeWidth={2.2}
                  dot={false} connectNulls name="Argo observation" isAnimationActive={false} />
            <Line type="monotone" dataKey="mod_t" stroke="#f97316" strokeWidth={2.2}
                  dot={false} connectNulls name="RTOFS model" isAnimationActive={false} />
            <Line type="monotone" dataKey="corrected_t" stroke="#f8fafc" strokeWidth={4.8}
                  strokeOpacity={0.75} strokeDasharray="4 3" dot={false} connectNulls
                  legendType="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="corrected_t" stroke="#22c55e" strokeWidth={2.8}
                  strokeDasharray="4 3"
                  dot={{ r: 1.8, fill: '#22c55e', stroke: '#dcfce7', strokeWidth: 0.6 }}
                  activeDot={{ r: 3.5 }}
                  connectNulls name="Final corrected proxy" isAnimationActive={false} />
            <ReferenceLine x={REF_TEMP} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '26 °C', fill: '#ef4444', fontSize: 11, position: 'top' }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem',
                    fontSize: '0.78rem', marginTop: '0.35rem' }}>
        <div style={{ color: '#0ea5e9' }}>
          <b>Argo</b> · TCHP {a.tchp_kj_cm2?.toFixed(0)} kJ/cm² · D26 {a.d26_m?.toFixed(0)} m
        </div>
        <div style={{ color: '#f97316' }}>
          <b>RTOFS</b> · TCHP {m.tchp_kj_cm2?.toFixed(0)} kJ/cm² · D26 {m.d26_m?.toFixed(0)} m
        </div>
      </div>
      {delta != null && (
        <div style={{ fontSize: '0.78rem', marginTop: '0.2rem', color: delta > 0 ? 'var(--good)' : 'var(--danger)' }}>
          Model gap: obs − model = {delta > 0 ? '+' : ''}{delta.toFixed(0)} kJ/cm²
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem', fontSize: '0.78rem', marginTop: '0.35rem' }}>
        <div style={{ color: '#22c55e' }}>
          <b>Final corrected</b> · TCHP {c.tchp_kj_cm2?.toFixed(0)} kJ/cm²
        </div>
        <div style={{ color: 'var(--muted)' }}>
          Correction {c.correction_delta_kj_cm2 > 0 ? '+' : ''}{c.correction_delta_kj_cm2?.toFixed(1) ?? '0.0'} via {c.correction_source}
        </div>
      </div>
      {c.curve_note && (
        <div style={{ fontSize: '0.72rem', marginTop: '0.35rem', color: 'var(--muted)' }}>
          {c.curve_note}
        </div>
      )}
    </>
  );
}
