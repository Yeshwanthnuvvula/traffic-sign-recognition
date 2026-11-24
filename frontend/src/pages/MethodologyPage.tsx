import React from 'react';
import { Layers, ShieldCheck, Cpu, Database, Sliders, CheckCircle2 } from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="border-b border-ink/15 pb-6">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-4">
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-signal font-semibold">
              Module 06 • Scientific Research Protocol
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-bold text-ink tracking-tight mt-1">
              Methodology & Experimental Pipeline
            </h1>
          </div>
          <p className="font-mono text-xs text-neutral max-w-md">
            Mathematical formulations, architecture specifications, and safety-critical data engineering standards.
          </p>
        </div>
      </section>

      {/* 6-Stage Workflow Diagram */}
      <section className="border border-ink/20 bg-paper-light p-6">
        <h2 className="font-serif text-2xl font-bold text-ink mb-6">End-to-End Pipeline Workflow</h2>
        <div className="grid grid-cols-1 md:grid-cols-6 gap-3 font-mono text-xs">
          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 01</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">Ingestion</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                GTSRB official training (39,209) & test sets (12,630) verified across 43 classes.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-neutral">Data Verification</span>
          </div>

          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 02</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">Profiling</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                Class imbalance (10.71×), Gini coefficient (0.4124), dimensional & RGB analysis.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-neutral">Empirical EDA</span>
          </div>

          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 03</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">Preprocessing</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                48×48 RGB resize, ImageNet normalization, controlled affine jitter. Zero flips.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-signal">Safety Pipeline</span>
          </div>

          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 04</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">Feature Eng.</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                9-bin HOG extraction (900-dim) + StandardScaler for SVM, KNN, Random Forest.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-neutral">Spatial Gradients</span>
          </div>

          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 05</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">Deep Learning</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                Custom 3-block CNN & Pretrained ResNet-18 with Apple Silicon MPS acceleration.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-neutral">Representation Learning</span>
          </div>

          <div className="border border-ink/20 p-4 bg-paper flex flex-col justify-between">
            <div>
              <span className="text-signal font-bold">STAGE 06</span>
              <h3 className="font-serif text-lg font-bold text-ink mt-1">OOD Rejection</h3>
              <p className="font-sans text-[11px] text-neutral mt-2">
                Calibrated MSP & Shannon entropy thresholds prevent false acceptance of noise.
              </p>
            </div>
            <span className="mt-4 text-[10px] uppercase text-signal">Safety Guardrail</span>
          </div>
        </div>
      </section>

      {/* Mathematical Formulations & Safety Invariant Detail */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border border-ink/20 bg-paper-light p-6 space-y-4">
          <h2 className="font-serif text-2xl font-bold text-ink">Mathematical Foundations</h2>
          
          <div className="border-b border-ink/10 pb-3">
            <h3 className="font-mono text-xs uppercase font-bold text-signal">1. Predictive Shannon Entropy</h3>
            <p className="font-mono text-xs text-neutral mt-1">
              H(p) = - ∑ (p_i · log p_i) / log(K)
            </p>
            <p className="font-sans text-xs text-ink/80 mt-1">
              Measures uncertainty across the 43 classes. Values approaching 1.0 indicate severe ambiguity or non-road imagery, triggering automatic out-of-distribution rejection.
            </p>
          </div>

          <div className="border-b border-ink/10 pb-3">
            <h3 className="font-mono text-xs uppercase font-bold text-signal">2. Histogram of Oriented Gradients (HOG)</h3>
            <p className="font-mono text-xs text-neutral mt-1">
              L2-Hys Normalization: v_norm = v / sqrt(||v||_2^2 + ε)
            </p>
            <p className="font-sans text-xs text-ink/80 mt-1">
              Constructed with 48×48 window, 8×8 pixel cells, 2×2 cell blocks with 50% overlap, yielding exactly 900 gradient orientation histogram features robust to illumination shifts.
            </p>
          </div>

          <div>
            <h3 className="font-mono text-xs uppercase font-bold text-signal">3. Stratified K-Fold Guarantee</h3>
            <p className="font-sans text-xs text-ink/80 mt-1">
              Guarantees identical class proportions (e.g., 5.7% for Speed Limit 50 km/h, 0.5% for Speed Limit 20 km/h) in both training and holdout validation sets.
            </p>
          </div>
        </div>

        <div className="border border-ink/20 bg-paper-light p-6 space-y-4">
          <h2 className="font-serif text-2xl font-bold text-ink">Safety Invariants & Protocol</h2>

          <div className="p-4 bg-signal/10 border border-signal/30 text-xs font-mono space-y-2">
            <span className="font-bold uppercase text-signal flex items-center">
              <ShieldCheck className="w-4 h-4 mr-2" />
              Invariant I: Directional Semantic Preservation
            </span>
            <p className="font-sans text-ink/80 leading-relaxed">
              Horizontal or vertical flips are strictly prohibited in the augmentation pipeline. Signs such as "Turn Right Ahead" (#33) versus "Turn Left Ahead" (#34), and "Road narrows on right" (#24) rely strictly on horizontal chirality.
            </p>
          </div>

          <div className="p-4 bg-paper border border-ink/15 text-xs font-mono space-y-2">
            <span className="font-bold uppercase text-ink flex items-center">
              <CheckCircle2 className="w-4 h-4 mr-2 text-sage" />
              Invariant II: Pristine Test Integrity
            </span>
            <p className="font-sans text-ink/80 leading-relaxed">
              The official 12,630 test images remain completely isolated until final evaluation benchmarking. No test sample participates in feature scaling, hyperparameter tuning, or validation checkpoints.
            </p>
          </div>

          <div className="p-4 bg-paper border border-ink/15 text-xs font-mono space-y-2">
            <span className="font-bold uppercase text-ink flex items-center">
              <Sliders className="w-4 h-4 mr-2 text-neutral" />
              Invariant III: Controlled Photometric Perturbation
            </span>
            <p className="font-sans text-ink/80 leading-relaxed">
              Simulated road disturbances include polygonal shadow mask variation, mild camera defocus blur (σ &lt; 1.2), and contrast scaling (±25%) to mirror authentic German autobahn environmental shifts.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
