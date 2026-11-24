import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { Database, Filter, Eye, AlertCircle, Info, Sparkles } from 'lucide-react';
import { getFullEDA, getSampleImageUrl } from '../services/api';
import { EDASummary } from '../types';

export const DataExplorerPage: React.FC = () => {
  const [edaData, setEdaData] = useState<EDASummary | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedClassDetail, setSelectedClassDetail] = useState<number | null>(null);

  useEffect(() => {
    getFullEDA()
      .then((data) => setEdaData(data))
      .catch((err) => console.error('Failed to load EDA data:', err));
  }, []);

  if (!edaData) {
    return (
      <div className="py-20 text-center font-mono text-xs text-neutral">
        Loading empirical dataset statistics...
      </div>
    );
  }

  // Prepare chart data
  const chartData = Object.keys(edaData.class_counts).map((cid) => {
    const idNum = parseInt(cid);
    const detail = edaData.class_details[idNum];
    return {
      classId: idNum,
      name: detail?.class_name || `Class ${idNum}`,
      category: detail?.category || 'Other',
      count: edaData.class_counts[cid],
    };
  });

  const categories = ['All', ...Object.keys(edaData.categories_breakdown)];

  const filteredGallery = Object.values(edaData.class_details).filter((item) => {
    if (selectedCategory === 'All') return true;
    return item.category === selectedCategory;
  });

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="border-b border-ink/15 pb-6">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-4">
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-signal font-semibold">
              Module 03 • Exploratory Data Analysis
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-bold text-ink tracking-tight mt-1">
              Dataset Profiler & 43-Class Gallery
            </h1>
          </div>
          <p className="font-mono text-xs text-neutral max-w-md">
            Full empirical inspection of the German Traffic Sign Recognition Benchmark (GTSRB) across dimensional, frequency, and photometric domains.
          </p>
        </div>
      </section>

      {/* Metric Cards Row */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Total Training Images</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {edaData.total_train_images.toLocaleString()}
          </span>
          <span className="font-mono text-[10px] text-neutral">Across 43 Class Subfolders</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Official Test Images</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {edaData.total_test_images.toLocaleString()}
          </span>
          <span className="font-mono text-[10px] text-neutral">Ground Truth Untouched</span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Class Imbalance Ratio</span>
          <span className="font-serif text-3xl font-bold text-signal mt-1 block">
            {edaData.imbalance_metrics.imbalance_ratio}×
          </span>
          <span className="font-mono text-[10px] text-neutral">
            Min: {edaData.imbalance_metrics.min_class_instances} / Max: {edaData.imbalance_metrics.max_class_instances}
          </span>
        </div>

        <div className="border border-ink/15 p-5 bg-paper-light">
          <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block">Gini Coefficient</span>
          <span className="font-serif text-3xl font-bold text-ink mt-1 block">
            {edaData.imbalance_metrics.gini_coefficient}
          </span>
          <span className="font-mono text-[10px] text-neutral">Moderate Category Inequality</span>
        </div>
      </section>

      {/* Class Frequency Bar Chart */}
      <section className="border border-ink/20 bg-paper-light p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-ink/15 pb-4 mb-6">
          <div>
            <h2 className="font-serif text-2xl font-bold text-ink">Class Frequency Distribution</h2>
            <p className="font-mono text-xs text-neutral mt-0.5">
              Empirical sample count for all 43 GTSRB categories (N = 39,209)
            </p>
          </div>
          <span className="font-mono text-xs text-neutral mt-2 sm:mt-0">
            Dashed Line: Mean Count ({edaData.imbalance_metrics.mean_instances_per_class})
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis
                dataKey="classId"
                stroke="#85877F"
                tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }}
                interval={1}
              />
              <YAxis
                stroke="#85877F"
                tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-ink text-paper p-3 font-mono text-xs border border-ink shadow-lg">
                        <p className="font-bold">Class #{data.classId}: {data.name}</p>
                        <p className="text-sage mt-1">Category: {data.category}</p>
                        <p className="text-signal font-semibold">Instances: {data.count}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine
                y={edaData.imbalance_metrics.mean_instances_per_class}
                stroke="#202A28"
                strokeDasharray="4 4"
                label={{ value: 'Mean', position: 'top', fill: '#85877F', fontSize: 10 }}
              />
              <Bar dataKey="count" fill="#B84D38" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Dimensional & Photometric Analysis Table */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border border-ink/15 p-6 bg-paper-light">
          <h3 className="font-serif text-xl font-bold text-ink mb-4">Geometric Variance</h3>
          <p className="font-mono text-xs text-neutral mb-4">
            Analysis of raw captured image boundaries prior to resizing.
          </p>
          <dl className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Resolution Range (Width):</dt>
              <dd className="font-semibold text-ink">{edaData.dimension_stats.min_width}px – {edaData.dimension_stats.max_width}px</dd>
            </div>
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Resolution Range (Height):</dt>
              <dd className="font-semibold text-ink">{edaData.dimension_stats.min_height}px – {edaData.dimension_stats.max_height}px</dd>
            </div>
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Median Sign Dimension:</dt>
              <dd className="font-semibold text-ink">{edaData.dimension_stats.median_width} × {edaData.dimension_stats.median_height} px</dd>
            </div>
            <div className="flex justify-between py-1">
              <dt className="text-neutral">Mean Aspect Ratio (W/H):</dt>
              <dd className="font-semibold text-signal">{edaData.dimension_stats.mean_aspect_ratio} (Near Square)</dd>
            </div>
          </dl>
        </div>

        <div className="border border-ink/15 p-6 bg-paper-light">
          <h3 className="font-serif text-xl font-bold text-ink mb-4">Photometric Profile</h3>
          <p className="font-mono text-xs text-neutral mb-4">
            Color channel averages across road condition variations.
          </p>
          <dl className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Mean Red Channel Intensity:</dt>
              <dd className="font-semibold text-ink">{edaData.photometric_stats.mean_red} / 255.0 ({edaData.photometric_stats.normalized_mean_rgb[0]})</dd>
            </div>
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Mean Green Channel Intensity:</dt>
              <dd className="font-semibold text-ink">{edaData.photometric_stats.mean_green} / 255.0 ({edaData.photometric_stats.normalized_mean_rgb[1]})</dd>
            </div>
            <div className="flex justify-between py-1 border-b border-ink/10">
              <dt className="text-neutral">Mean Blue Channel Intensity:</dt>
              <dd className="font-semibold text-ink">{edaData.photometric_stats.mean_blue} / 255.0 ({edaData.photometric_stats.normalized_mean_rgb[2]})</dd>
            </div>
            <div className="flex justify-between py-1">
              <dt className="text-neutral">Normalization Pipeline:</dt>
              <dd className="font-semibold text-signal">ImageNet Standard (μ, σ)</dd>
            </div>
          </dl>
        </div>
      </section>

      {/* 43-Class Interactive Canonical Sample Gallery */}
      <section className="border-t border-ink/15 pt-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="font-serif text-3xl font-bold text-ink">43-Class Canonical Benchmark Gallery</h2>
            <p className="font-mono text-xs text-neutral mt-1">
              Representative examples sampled from the official GTSRB training corpus. Click any card to inspect.
            </p>
          </div>

          {/* Filter Pills */}
          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1 border transition-colors ${
                  selectedCategory === cat
                    ? 'border-signal bg-signal text-paper'
                    : 'border-ink/20 hover:border-ink/50 bg-paper-light text-ink/80'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
          {filteredGallery.map((sign) => {
            const isSelected = selectedClassDetail === sign.class_id;
            return (
              <div
                key={sign.class_id}
                onClick={() => setSelectedClassDetail(isSelected ? null : sign.class_id)}
                className={`border p-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-signal bg-signal/10 ring-1 ring-signal'
                    : 'border-ink/15 hover:border-ink/40 bg-paper-light'
                }`}
              >
                <div className="aspect-square bg-paper-dark/30 flex items-center justify-center p-1 overflow-hidden">
                  <img
                    src={getSampleImageUrl(sign.sample_file)}
                    alt={sign.class_name}
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="mt-2 font-mono text-[10px]">
                  <span className="font-bold text-signal">#{sign.class_id}</span>
                  <p className="font-sans font-medium text-ink truncate" title={sign.class_name}>
                    {sign.class_name}
                  </p>
                  <div className="flex justify-between text-neutral mt-0.5">
                    <span>{sign.count} imgs</span>
                    <span className="text-[9px] uppercase">{sign.category.slice(0, 5)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
