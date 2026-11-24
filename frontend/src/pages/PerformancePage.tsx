import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { BarChart3, Activity, AlertCircle, CheckCircle2, ChevronRight, Sliders } from 'lucide-react';
import { getModelDetail, getAllMetrics, getSpeedLimitAnalysis } from '../services/api';
import { MetricDetail, ModelMetricRow } from '../types';

export const PerformancePage: React.FC = () => {
  const [availableModels, setAvailableModels] = useState<ModelMetricRow[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('svm');
  const [detail, setDetail] = useState<MetricDetail | null>(null);
  const [speedLimitStudy, setSpeedLimitStudy] = useState<any>(null);
  const [selectedCell, setSelectedCell] = useState<{ trueId: number; predId: number; count: number } | null>(null);

  useEffect(() => {
    getAllMetrics().then((res) => {
      setAvailableModels(res.models);
      if (res.models.length > 0) {
        // Prefer custom_cnn or svm
        const preferred = res.models.find(m => m.model_id === 'custom_cnn') || res.models[0];
        setSelectedModelId(preferred.model_id);
      }
    });

    getSpeedLimitAnalysis().then((res) => setSpeedLimitStudy(res));
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      getModelDetail(selectedModelId)
        .then((data) => setDetail(data))
        .catch((err) => console.error('Failed to load metric detail:', err));
    }
  }, [selectedModelId]);

  if (!detail) {
    return (
      <div className="py-20 text-center font-mono text-xs text-neutral">
        Loading evaluation matrix and convergence histories...
      </div>
    );
  }

  const historyData = detail.history
    ? detail.history.epoch.map((ep, idx) => ({
        epoch: ep,
        trainAcc: detail.history!.train_acc[idx] * 100.0,
        valAcc: detail.history!.val_acc[idx] * 100.0,
        trainLoss: detail.history!.train_loss[idx],
        valLoss: detail.history!.val_loss[idx],
      }))
    : [];

  const cm = detail.confusion_matrix || [];

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="border-b border-ink/15 pb-6">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-4">
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-signal font-semibold">
              Module 05 • Empirical Error Analysis & Convergence
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-bold text-ink tracking-tight mt-1">
              Performance Diagnostics
            </h1>
          </div>

          {/* Model Selector Pill */}
          <div className="flex items-center space-x-2 font-mono text-xs">
            <span className="text-neutral">Active Model:</span>
            <div className="flex flex-wrap gap-1">
              {availableModels.map((m) => (
                <button
                  key={m.model_id}
                  onClick={() => setSelectedModelId(m.model_id)}
                  className={`px-2.5 py-1 border transition-colors ${
                    selectedModelId === m.model_id
                      ? 'border-signal bg-signal text-paper font-semibold'
                      : 'border-ink/20 hover:border-ink/50 bg-paper-light text-ink/80'
                  }`}
                >
                  {m.model_id.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* High-Level Scorecard */}
      <section className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Top-1 Accuracy</span>
          <span className="font-serif text-3xl font-bold text-signal mt-1 block">
            {(detail.accuracy * 100.0).toFixed(2)}%
          </span>
          <span className="font-mono text-[10px] text-neutral">Holdout Validation</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Macro F1-Score</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {detail.macro_f1.toFixed(4)}
          </span>
          <span className="font-mono text-[10px] text-neutral">Unweighted Class Average</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Weighted F1</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {detail.weighted_f1.toFixed(4)}
          </span>
          <span className="font-mono text-[10px] text-neutral">Frequency Weighted</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Precision / Recall</span>
          <span className="font-serif text-2xl font-bold text-ink mt-1 block">
            {detail.precision.toFixed(3)} / {detail.recall.toFixed(3)}
          </span>
          <span className="font-mono text-[10px] text-neutral">Class Balance Ratio</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Inference Speed</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {detail.inference_latency_ms.toFixed(2)} <span className="text-sm font-mono font-normal">ms</span>
          </span>
          <span className="font-mono text-[10px] text-neutral">Per Photographic Query</span>
        </div>
      </section>

      {/* Dedicated Speed Limit Disambiguation Card */}
      <section className="border border-ink/20 bg-paper-light p-6">
        <div className="border-b border-ink/15 pb-4 mb-4">
          <span className="font-mono text-xs uppercase tracking-widest text-signal font-bold">
            Critical Safety Focus
          </span>
          <h2 className="font-serif text-2xl font-bold text-ink mt-1">
            Speed Limit 30 km/h vs Speed Limit 50 km/h Disambiguation Study
          </h2>
          <p className="font-mono text-xs text-neutral mt-0.5">
            Analysis of visual glyph confusion between German speed limits #1 (30 km/h) and #2 (50 km/h).
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono text-xs">
          <div className="border border-ink/15 p-4 bg-paper">
            <span className="font-semibold text-ink block mb-2">Class #1: Speed Limit 30 km/h</span>
            <div className="space-y-1 text-neutral">
              {speedLimitStudy && speedLimitStudy[selectedModelId] ? (
                <>
                  <p>Validation Instances: {speedLimitStudy[selectedModelId].class_1_speed_30.total}</p>
                  <p className="text-ink font-semibold">Correct Predictions: {speedLimitStudy[selectedModelId].class_1_speed_30.correct}</p>
                  <p className="text-signal">Misclassified as 50 km/h: {speedLimitStudy[selectedModelId].class_1_speed_30.misclassified_as_50}</p>
                  <p>Recall: {(speedLimitStudy[selectedModelId].class_1_speed_30.recall * 100).toFixed(2)}%</p>
                </>
              ) : (
                <p>Analyzing confusion matrix records...</p>
              )}
            </div>
          </div>

          <div className="border border-ink/15 p-4 bg-paper">
            <span className="font-semibold text-ink block mb-2">Class #2: Speed Limit 50 km/h</span>
            <div className="space-y-1 text-neutral">
              {speedLimitStudy && speedLimitStudy[selectedModelId] ? (
                <>
                  <p>Validation Instances: {speedLimitStudy[selectedModelId].class_2_speed_50.total}</p>
                  <p className="text-ink font-semibold">Correct Predictions: {speedLimitStudy[selectedModelId].class_2_speed_50.correct}</p>
                  <p className="text-signal">Misclassified as 30 km/h: {speedLimitStudy[selectedModelId].class_2_speed_50.misclassified_as_30}</p>
                  <p>Recall: {(speedLimitStudy[selectedModelId].class_2_speed_50.recall * 100).toFixed(2)}%</p>
                </>
              ) : (
                <p>Analyzing confusion matrix records...</p>
              )}
            </div>
          </div>

          <div className="border border-ink/15 p-4 bg-paper-dark/30 font-sans text-xs text-ink/80 leading-relaxed">
            <span className="font-mono font-bold uppercase tracking-wider text-signal block mb-1">
              Topological Root Cause
            </span>
            Both signs share identical red circular rims (diameter 80% of window) and white internal disks. Differentiating "3" from "5" relies entirely on whether the top segment forms a closed loop ("3") or a horizontal crossbar with vertical stem ("5"). Classical HOG handles this via cell gradient angles, while deep CNNs extract distinct stroke kernels.
          </div>
        </div>
      </section>

      {/* Convergence Curves (if Deep Learning) */}
      {historyData.length > 0 && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="border border-ink/20 bg-paper-light p-6">
            <h3 className="font-serif text-xl font-bold text-ink mb-1">Accuracy Trajectory</h3>
            <p className="font-mono text-xs text-neutral mb-4">Training vs. Validation Accuracy over Epochs</p>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="epoch" stroke="#85877F" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                  <YAxis domain={[50, 100]} stroke="#85877F" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#202A28', color: '#EEECE5', fontFamily: 'IBM Plex Mono', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="trainAcc" stroke="#85877F" name="Train Acc (%)" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="valAcc" stroke="#B84D38" name="Val Acc (%)" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="border border-ink/20 bg-paper-light p-6">
            <h3 className="font-serif text-xl font-bold text-ink mb-1">Loss Trajectory</h3>
            <p className="font-mono text-xs text-neutral mb-4">Cross-Entropy Loss (Generalization Gap Monitor)</p>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="epoch" stroke="#85877F" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                  <YAxis stroke="#85877F" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#202A28', color: '#EEECE5', fontFamily: 'IBM Plex Mono', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="trainLoss" stroke="#85877F" name="Train Loss" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="valLoss" stroke="#202A28" name="Val Loss" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      )}

      {/* Interactive 43-Class Confusion Matrix Heatmap */}
      <section className="border border-ink/20 bg-paper-light p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-ink/15 pb-4 mb-6">
          <div>
            <h2 className="font-serif text-2xl font-bold text-ink">43-Class Interactive Confusion Matrix</h2>
            <p className="font-mono text-xs text-neutral mt-0.5">
              Rows: Ground Truth Class (0–42) • Columns: Model Prediction (0–42). Hover or click cells for exact counts.
            </p>
          </div>
          {selectedCell && (
            <div className="bg-ink text-paper px-4 py-2 font-mono text-xs border border-ink mt-2 md:mt-0">
              <span>True: #{selectedCell.trueId}</span>
              <span className="mx-2">→</span>
              <span>Pred: #{selectedCell.predId}</span>
              <span className="mx-2">|</span>
              <span className="text-signal font-bold">{selectedCell.count} instances</span>
            </div>
          )}
        </div>

        {/* 43x43 Matrix Grid */}
        <div className="overflow-x-auto pb-4">
          <div className="inline-block min-w-[750px]">
            <div className="grid grid-cols-[40px_repeat(43,_minmax(14px,_1fr))] gap-[1px] bg-ink/10 p-[1px]">
              {/* Header Row */}
              <div className="bg-paper-dark/60 font-mono text-[9px] text-neutral text-center p-0.5">T\P</div>
              {Array.from({ length: 43 }).map((_, colIdx) => (
                <div key={colIdx} className="bg-paper-dark/60 font-mono text-[8px] text-neutral text-center p-0.5">
                  {colIdx}
                </div>
              ))}

              {/* Rows */}
              {cm.map((row, rIdx) => (
                <React.Fragment key={rIdx}>
                  {/* Row Label */}
                  <div className="bg-paper-dark/60 font-mono text-[8px] text-neutral text-center p-0.5 flex items-center justify-center font-bold">
                    {rIdx}
                  </div>
                  {/* Row Cells */}
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const isError = !isDiagonal && val > 0;
                    
                    let bg = '#EEECE5';
                    if (isDiagonal) {
                      bg = val > 0 ? '#A6AD91' : '#EEECE5';
                    } else if (isError) {
                      bg = val > 3 ? '#B84D38' : '#CE5E47';
                    }

                    return (
                      <div
                        key={cIdx}
                        onClick={() => setSelectedCell({ trueId: rIdx, predId: cIdx, count: val })}
                        onMouseEnter={() => setSelectedCell({ trueId: rIdx, predId: cIdx, count: val })}
                        style={{ backgroundColor: bg }}
                        className={`aspect-square flex items-center justify-center font-mono text-[8px] cursor-pointer transition-transform hover:scale-125 hover:z-10 ${
                          isError ? 'text-paper font-bold' : isDiagonal && val > 0 ? 'text-ink font-semibold' : 'text-neutral/20'
                        }`}
                        title={`True #${rIdx} -> Pred #${cIdx}: ${val}`}
                      >
                        {val > 0 ? (val > 99 ? '•' : val) : ''}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Matrix Legend */}
        <div className="mt-4 flex items-center space-x-6 text-xs font-mono text-neutral border-t border-ink/10 pt-4">
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-sage border border-ink/20 inline-block"></span>
            <span>Correct Classification (Diagonal)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-signal inline-block"></span>
            <span>Misclassification (Off-Diagonal Error)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-paper border border-ink/20 inline-block"></span>
            <span>Zero Count</span>
          </div>
        </div>
      </section>
    </div>
  );
};
