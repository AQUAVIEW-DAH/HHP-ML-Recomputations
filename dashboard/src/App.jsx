import React, { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { Activity, Waves, Search, MapPin, Layers, Wind, TrendingUp, TrendingDown } from 'lucide-react';
import ProfilePanel from './ProfilePanel.jsx';

// Leaflet default icons (CDN)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const API_BASE = import.meta.env.VITE_API_BASE || '';
const FETCH_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

const TCHP_BINS = [
  { label: '< 40 kJ/cm²',  max: 40, color: '#38bdf8' },
  { label: '40–60',        max: 60, color: '#34d399' },
  { label: '60–80',        max: 80, color: '#facc15' },
  { label: '80–120',       max: 120, color: '#fb923c' },
  { label: '> 120',        max: Infinity, color: '#ef4444' },
];

const CORRECTION_BINS = [
  { label: '< -20', max: -20, color: '#7f1d1d' },
  { label: '-20 to -10', max: -10, color: '#dc2626' },
  { label: '-10 to -3', max: -3, color: '#fb7185' },
  { label: '-3 to +3', max: 3, color: '#94a3b8' },
  { label: '+3 to +10', max: 10, color: '#4ade80' },
  { label: '+10 to +20', max: 20, color: '#22c55e' },
  { label: '> +20', max: Infinity, color: '#166534' },
];

function tchpColor(v) {
  if (v == null || Number.isNaN(v)) return '#64748b';
  return TCHP_BINS.find((b) => v < b.max)?.color || TCHP_BINS.at(-1).color;
}

function correctionColor(v) {
  if (v == null || Number.isNaN(v)) return '#64748b';
  return CORRECTION_BINS.find((b) => v < b.max)?.color || CORRECTION_BINS.at(-1).color;
}

function windColor(kt) {
  if (kt == null) return '#64748b';
  if (kt >= 137) return '#7c2d12';  // Cat 5
  if (kt >= 113) return '#dc2626';  // Cat 4
  if (kt >= 96)  return '#f97316';  // Cat 3
  if (kt >= 83)  return '#fbbf24';  // Cat 2
  if (kt >= 64)  return '#eab308';  // Cat 1
  if (kt >= 34)  return '#0ea5e9';  // TS
  return '#64748b';                   // TD
}

function MapClick({ onClick, enabled }) {
  useMapEvents({
    click(e) { if (enabled) onClick(e.latlng); },
  });
  return null;
}

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [track, setTrack] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState(null);
  const [target, setTarget] = useState(null);
  const [observationsLayer, setObservationsLayer] = useState(null);
  const [rawTchpField, setRawTchpField] = useState(null);
  const [correctedTchpField, setCorrectedTchpField] = useState(null);
  const [correctionField, setCorrectionField] = useState(null);
  const [fieldMode, setFieldMode] = useState('corrected_tchp');
  const [showTrack, setShowTrack] = useState(true);
  const [selectedCastId, setSelectedCastId] = useState(null);

  // Load metadata + track once
  useEffect(() => {
    (async () => {
      try {
        const [m, t] = await Promise.all([
          fetch(`${API_BASE}/metadata`, { headers: FETCH_HEADERS }).then((r) => r.json()),
          fetch(`${API_BASE}/track/milton`, { headers: FETCH_HEADERS }).then((r) => r.json()),
        ]);
        setMetadata(m);
        setTrack(t);
        if (m.default_query_time) setSelectedDate(m.default_query_time.slice(0, 10));
        else if (m.available_dates?.length) setSelectedDate(formatDate(m.available_dates.at(-1)));
      } catch (err) {
        setQueryError(`Failed to load metadata: ${err.message}`);
      }
    })();
  }, []);

  // Load per-date layers when date changes
  useEffect(() => {
    if (!selectedDate) return;
    const t = `${selectedDate}T12:00:00Z`;
    (async () => {
      try {
        const [obs, tchp, corrected, correction] = await Promise.all([
          fetch(`${API_BASE}/map_layer?time=${encodeURIComponent(t)}&layer=observations`,
                { headers: FETCH_HEADERS }).then((r) => r.json()),
          fetch(`${API_BASE}/map_layer?time=${encodeURIComponent(t)}&layer=tchp&stride=10`,
                { headers: FETCH_HEADERS }).then((r) => r.json()),
          fetch(`${API_BASE}/map_layer?time=${encodeURIComponent(t)}&layer=corrected_tchp&stride=10`,
                { headers: FETCH_HEADERS }).then((r) => r.json()),
          fetch(`${API_BASE}/map_layer?time=${encodeURIComponent(t)}&layer=correction&stride=10`,
                { headers: FETCH_HEADERS }).then((r) => r.json()),
        ]);
        setObservationsLayer(obs);
        setRawTchpField(tchp);
        setCorrectedTchpField(corrected);
        setCorrectionField(correction);
      } catch (err) {
        console.error('layer load error', err);
      }
    })();
  }, [selectedDate]);

  const handleMapClick = async ({ lat, lng }) => {
    if (!selectedDate) return;
    setTarget({ lat, lng });
    setQueryLoading(true);
    setQueryError(null);
    setQueryResult(null);
    setSelectedCastId(null);
    try {
      const res = await fetch(`${API_BASE}/tchp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...FETCH_HEADERS },
        body: JSON.stringify({ lat, lon: lng, time: `${selectedDate}T12:00:00Z` }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const data = await res.json();
      setQueryResult(data);
      if (data.nearby_observations?.[0]) setSelectedCastId(data.nearby_observations[0].cast_id);
    } catch (err) {
      setQueryError(err.message);
    } finally {
      setQueryLoading(false);
    }
  };

  const ready = Boolean(metadata && selectedDate && track);
  const fieldData = useMemo(() => {
    if (fieldMode === 'tchp') return rawTchpField;
    if (fieldMode === 'corrected_tchp') return correctedTchpField;
    if (fieldMode === 'correction') return correctionField;
    return null;
  }, [fieldMode, rawTchpField, correctedTchpField, correctionField]);
  const fieldPoints = useMemo(() => fieldData?.points || [], [fieldData]);
  const obsPoints = useMemo(() => observationsLayer?.points || [], [observationsLayer]);
  const trackPoints = useMemo(() => track?.points || [], [track]);
  const fieldLegend = fieldMode === 'correction' ? CORRECTION_BINS : TCHP_BINS;
  const fieldLegendTitle = fieldMode === 'correction' ? 'ML correction legend (kJ/cm²)' : 'TCHP legend (kJ/cm²)';
  const fieldLabel = fieldMode === 'tchp'
    ? 'Raw model field'
    : fieldMode === 'corrected_tchp'
      ? 'Final corrected field'
      : fieldMode === 'correction'
        ? 'Correction field'
        : 'Field hidden';

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="header">
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <Waves size={26} color="var(--accent)" />
            <h1>HHP — Milton 2024</h1>
          </div>
          <p className="subtitle">
            Hurricane Heat Potential replay sandbox for Hurricane Milton (Oct 4–11, 2024). The
            default ML artifact is trained on Argo GDAC matchups recomputed with TEOS-10-backed
            HHP values from the 2024 storm set. Click any point in the Gulf of Mexico to query the
            corrected TCHP estimate, the raw RTOFS baseline, and the nearest Argo observation pair.
          </p>
        </div>

        {metadata && (
          <div className="card">
            <h3>Replay controls</h3>
            <div className="stat-row"><span className="k">Mode</span><span className="v">{metadata.mode}</span></div>
            <div className="stat-row"><span className="k">Storm</span><span className="v">{metadata.storm.name} {metadata.storm.season}</span></div>
            <div className="stat-row"><span className="k">Peak</span><span className="v">Cat {metadata.storm.peak_category} · {metadata.storm.peak_wind_kt} kt</span></div>
            <div className="stat-row"><span className="k">Pairs</span><span className="v">{metadata.replay_pair_count} / {metadata.replay_platforms} floats</span></div>
            <div className="stat-row"><span className="k">ML model</span><span className="v">{metadata.ml_model_name || 'raw_rtofs'}</span></div>
            <div className="stat-row"><span className="k">Train storms</span><span className="v">{metadata.ml_model_train_events?.join(', ') || 'n/a'}</span></div>
            <label style={{ display: 'block', marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--muted)' }}>Replay date</label>
            <select value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)}>
              {(metadata.available_dates || []).map((d) => (
                <option key={d} value={formatDate(d)}>{formatDate(d)}</option>
              ))}
            </select>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem', marginTop: '0.5rem' }}>
              <button className="glass" style={{ marginTop: 0 }} onClick={() => setFieldMode('tchp')}>
                <Layers size={14} /> Raw field
              </button>
              <button className="glass" style={{ marginTop: 0 }} onClick={() => setFieldMode('corrected_tchp')}>
                <Layers size={14} /> Final field
              </button>
              <button className="glass" style={{ marginTop: 0 }} onClick={() => setFieldMode('correction')}>
                <Layers size={14} /> Correction
              </button>
              <button className="glass" style={{ marginTop: 0 }} onClick={() => setFieldMode('none')}>
                <Layers size={14} /> Hide field
              </button>
            </div>
            <button className="glass" onClick={() => setShowTrack((v) => !v)}>
              <Wind size={14} /> {showTrack ? 'Hide' : 'Show'} Milton track
            </button>
            <div style={{ marginTop: '0.55rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
              Active map layer: <b style={{ color: 'var(--text)' }}>{fieldLabel}</b>
            </div>
          </div>
        )}

        {queryLoading && <div className="card">Querying…</div>}
        {queryError && (
          <div className="card" style={{ borderColor: 'var(--danger)' }}>
            <h3 style={{ color: 'var(--danger)' }}>Query failed</h3>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>{queryError}</div>
          </div>
        )}

        {queryResult && !queryLoading && (
          <>
            <div className="card">
              <h3>
                <Activity size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                TCHP estimate
              </h3>
              <div>
                <span className="big-number">{queryResult.best_tchp_kj_cm2?.toFixed(0)}</span>
                <span className="unit">kJ/cm²</span>
              </div>
              <div className="stat-row" style={{ marginTop: '0.5rem' }}>
                <span className="k">Confidence</span>
                <span className={`badge badge-${queryResult.confidence.toLowerCase()}`}>{queryResult.confidence}</span>
              </div>
              <div className="stat-row"><span className="k">Raw RTOFS</span><span className="v">{queryResult.model_tchp_kj_cm2?.toFixed(0)} kJ/cm²</span></div>
              <div className="stat-row"><span className="k">ML correction</span><span className="v">{queryResult.correction_delta_kj_cm2?.toFixed(1) ?? '0.0'} kJ/cm²</span></div>
              <div className="stat-row"><span className="k">Correction source</span><span className="v">{queryResult.correction_source}</span></div>
              <div className="stat-row"><span className="k">D26</span><span className="v">{queryResult.best_d26_m?.toFixed(0)} m</span></div>
              <div className="stat-row"><span className="k">Surface T</span><span className="v">{queryResult.model_surface_t_c?.toFixed(1)} °C</span></div>
              <div className="stat-row"><span className="k">Grid cell</span><span className="v">{queryResult.model_grid_lat.toFixed(2)}, {queryResult.model_grid_lon.toFixed(2)}</span></div>
            </div>

            <div className="card">
              <h3><Search size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                Nearby Argo obs ({queryResult.nearby_observations.length})</h3>
              {queryResult.nearby_observations.length === 0 && (
                <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                  No Argo profiles within {queryResult.support_radius_km} km / {queryResult.support_window_hr} h.
                </div>
              )}
              {queryResult.nearby_observations.slice(0, 5).map((o) => {
                const delta = o.tchp_delta_kj_cm2;
                const DeltaIcon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : null;
                const deltaColor = delta > 0 ? 'var(--good)' : delta < 0 ? 'var(--danger)' : 'var(--muted)';
                const isSelected = selectedCastId === o.cast_id;
                return (
                  <button
                    key={o.cast_id}
                    onClick={() => setSelectedCastId(o.cast_id)}
                    style={{
                      width: '100%', textAlign: 'left', padding: '0.55rem 0.6rem',
                      background: isSelected ? 'rgba(14,165,233,0.12)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${isSelected ? 'var(--argo)' : 'var(--border)'}`,
                      borderRadius: 8, marginTop: 6, cursor: 'pointer', color: 'var(--text)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span>Argo {o.platform}</span>
                      <span style={{ color: 'var(--argo)', fontWeight: 600 }}>{o.obs_tchp_kj_cm2?.toFixed(0)} kJ/cm²</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--muted)' }}>
                      <span>{o.distance_km.toFixed(1)} km · Δt {o.time_delta_hr.toFixed(0)} h</span>
                      {DeltaIcon && (
                        <span style={{ color: deltaColor, display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                          <DeltaIcon size={12} /> {delta > 0 ? '+' : ''}{delta?.toFixed(0)}
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </>
        )}

        {!queryResult && !queryLoading && (
          <div className="card" style={{ textAlign: 'center', padding: '1.2rem' }}>
            <MapPin size={28} color="var(--muted)" />
            <p style={{ color: 'var(--muted)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              {ready ? 'Click a point in the Gulf of Mexico to query TCHP.' : 'Loading replay metadata…'}
            </p>
          </div>
        )}

        <div className="card">
          <h3>{fieldLegendTitle}</h3>
          {fieldLegend.map((b) => (
            <div key={b.label} className="stat-row" style={{ margin: 0 }}>
              <span className="legend-chip">
                <span className="dot" style={{ background: b.color }} />
                {b.label}
              </span>
            </div>
          ))}
          <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: '0.5rem' }}>
            Shay et al. 2000 rapid-intensification threshold: <b>50 kJ/cm²</b>
          </div>
        </div>
      </aside>

      <div className="map-pane">
        <MapContainer center={[24.0, -88.0]} zoom={6} className="map-container" zoomControl={true}>
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; OpenStreetMap contributors &copy; CARTO'
          />
          <MapClick onClick={handleMapClick} enabled={ready} />

          {fieldMode !== 'none' && fieldPoints.map((p, i) => (
            <CircleMarker
              key={`f-${i}`}
              center={[p.lat, p.lon]}
              radius={6}
              pathOptions={{
                color: fieldMode === 'correction' ? correctionColor(p.value) : tchpColor(p.value),
                weight: 0.2,
                fillColor: fieldMode === 'correction' ? correctionColor(p.value) : tchpColor(p.value),
                fillOpacity: 0.45,
              }}
            >
              <Popup>
                <b>{fieldMode === 'tchp' ? 'RTOFS TCHP' : fieldMode === 'corrected_tchp' ? 'Corrected TCHP' : 'ML correction'}</b><br />
                {p.value.toFixed(1)} kJ/cm²<br />
                ({p.lat.toFixed(2)}, {p.lon.toFixed(2)})
              </Popup>
            </CircleMarker>
          ))}

          {showTrack && trackPoints.length > 0 && (
            <>
              <Polyline
                positions={trackPoints.map((p) => [p.lat, p.lon])}
                pathOptions={{ color: '#e2e8f0', weight: 1.5, opacity: 0.55, dashArray: '3 4' }}
              />
              {trackPoints.map((p, i) => (
                <CircleMarker
                  key={`t-${i}`}
                  center={[p.lat, p.lon]}
                  radius={4}
                  pathOptions={{ color: '#0f172a', weight: 0.6, fillColor: windColor(p.wind_kt), fillOpacity: 0.95 }}
                >
                  <Popup>
                    <b>Milton</b><br />
                    {p.iso_time}<br />
                    {p.wind_kt} kt · Cat {p.sshs}<br />
                    {p.pres_mb ? `${p.pres_mb} mb · ` : ''}{p.nature || ''}
                  </Popup>
                </CircleMarker>
              ))}
            </>
          )}

          {obsPoints.map((o, i) => (
            <CircleMarker
              key={`o-${i}`}
              center={[o.lat, o.lon]}
              radius={7}
              pathOptions={{
                color: '#f8fafc', weight: 1.2,
                fillColor: tchpColor(o.value), fillOpacity: 0.92,
              }}
              eventHandlers={{ click: () => setSelectedCastId(o.cast_id) }}
            >
              <Popup>
                <b>Argo {o.platform}</b><br />
                Obs TCHP: {o.value?.toFixed(1) ?? '—'} kJ/cm²<br />
                Model TCHP: {o.model_value?.toFixed(1) ?? '—'}<br />
                {o.obs_time}
              </Popup>
            </CircleMarker>
          ))}

          {target && (
            <Marker position={[target.lat, target.lng]}>
              <Popup>Query point<br />({target.lat.toFixed(3)}, {target.lng.toFixed(3)})</Popup>
            </Marker>
          )}
        </MapContainer>

        <div className="map-overlay top-left">
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Date</div>
          <div><b>{selectedDate || '…'}</b></div>
        </div>

        {selectedCastId && (
          <div className="profile-panel">
            <button className="close" onClick={() => setSelectedCastId(null)}>✕</button>
            <ProfilePanel castId={selectedCastId} />
          </div>
        )}
      </div>
    </div>
  );
}

function formatDate(yyyymmdd) {
  return `${yyyymmdd.slice(0,4)}-${yyyymmdd.slice(4,6)}-${yyyymmdd.slice(6)}`;
}
