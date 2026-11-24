import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Cpu, Layers, ShieldCheck, Database, Award, BookOpen } from 'lucide-react';

export const CoverPage: React.FC = () => {
  return (
    <div className="space-y-16">
      {/* Editorial Journal Paper Masthead */}
      <section className="border-b border-ink/20 pb-12 pt-4">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-8">
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-signal font-semibold bg-signal/10 px-2.5 py-1 border border-signal/20">
              Scientific Monograph • Computer Vision & Machine Learning
            </span>
            <div className="mt-4 font-mono text-xs text-neutral space-x-4">
              <span>Date: Academic Session 2024–2025</span>
              <span>•</span>
              <span>Dataset: GTSRB Official Benchmark</span>
              <span>•</span>
              <span>Scope: 43 Standardized Sign Classes</span>
            </div>
          </div>
          <div className="text-right">
            <span className="font-mono text-xs text-neutral">Document ID: SGV-2024-GTSRB-43</span>
          </div>
        </div>

        {/* Editorial Heading */}
        <h1 className="font-serif text-5xl md:text-7xl font-bold tracking-tight text-ink leading-[1.08] max-w-5xl">
          Understanding Traffic Signs Through Machine Learning.
        </h1>
        <p className="mt-6 text-xl md:text-2xl text-ink/80 font-serif italic max-w-3xl leading-relaxed">
          An experimental study of classical gradient descriptors and deep residual neural networks for automated traffic sign recognition and out-of-distribution rejection.
        </p>

        {/* Action CTAs */}
        <div className="mt-8 flex flex-wrap items-center gap-4 pt-4 border-t border-ink/10">
          <Link
            to="/studio"
            className="inline-flex items-center px-6 py-3 bg-ink text-paper font-mono text-xs uppercase tracking-wider hover:bg-signal transition-colors group"
          >
            <span>Launch Recognition Studio</span>
            <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link
            to="/performance"
            className="inline-flex items-center px-6 py-3 border border-ink/30 text-ink font-mono text-xs uppercase tracking-wider hover:bg-paper-dark transition-colors"
          >
            <span>Review Performance Matrix</span>
          </Link>
          <Link
            to="/report"
            className="inline-flex items-center px-6 py-3 border border-ink/30 text-ink font-mono text-xs uppercase tracking-wider hover:bg-paper-dark transition-colors"
          >
            <span>Read Academic Thesis</span>
          </Link>
        </div>
      </section>

      {/* Abstract & Summary Grid */}
      <section className="grid grid-cols-1 md:grid-cols-12 gap-8">
        <div className="md:col-span-8 space-y-6">
          <div className="border-l-2 border-signal pl-6">
            <h2 className="font-serif text-2xl font-bold text-ink">Abstract</h2>
            <p className="mt-3 text-sm md:text-base text-ink/80 leading-relaxed font-sans">
              Automated Traffic Sign Recognition (TSR) constitutes a safety-critical perception component of autonomous driving systems and advanced driver assistance systems (ADAS). In this investigation, we systematically benchmark five foundational machine learning and deep learning paradigms on the standardized German Traffic Sign Recognition Benchmark (GTSRB), spanning 43 distinct traffic sign classes. We examine Support Vector Machines (Linear and RBF kernels) with Histogram of Oriented Gradients (HOG), K-Nearest Neighbors (KNN), Random Forest ensembles, a customized 3-block Convolutional Neural Network with spatial dropout and batch normalization, and an ImageNet-pretrained ResNet-18 residual network. We address subtle inter-class visual confusions—specifically disambiguating Speed Limit 30 km/h from Speed Limit 50 km/h—and introduce predictive entropy calibration and out-of-distribution (OOD) rejection thresholds to prevent catastrophic false acceptances of non-road imagery.
            </p>
          </div>

          {/* Key Empirical Findings Table */}
          <div className="mt-8 border border-ink/15 bg-paper-light p-6">
            <h3 className="font-mono text-xs uppercase tracking-widest text-ink font-bold mb-4 flex items-center">
              <Award className="w-4 h-4 mr-2 text-signal" />
              Empirical Performance Summary (Validation Set)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono text-left">
                <thead>
                  <tr className="border-b border-ink/20 text-neutral uppercase">
                    <th className="py-2 pr-4">Model Architecture</th>
                    <th className="py-2 px-3">Type</th>
                    <th className="py-2 px-3">Feature Input</th>
                    <th className="py-2 px-3">Top-1 Accuracy</th>
                    <th className="py-2 px-3">Macro F1</th>
                    <th className="py-2 pl-3">Inference Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink/10">
                  <tr>
                    <td className="py-2.5 pr-4 font-semibold text-ink">Support Vector Machine (RBF)</td>
                    <td className="py-2.5 px-3 text-neutral">Classical ML</td>
                    <td className="py-2.5 px-3">HOG (900-dim)</td>
                    <td className="py-2.5 px-3 font-semibold text-signal">97.13%</td>
                    <td className="py-2.5 px-3 font-semibold">0.9752</td>
                    <td className="py-2.5 pl-3">7.22 ms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-semibold text-ink">K-Nearest Neighbors (k=5)</td>
                    <td className="py-2.5 px-3 text-neutral">Instance-based</td>
                    <td className="py-2.5 px-3">HOG (900-dim)</td>
                    <td className="py-2.5 px-3">92.73%</td>
                    <td className="py-2.5 px-3">0.9357</td>
                    <td className="py-2.5 pl-3">0.29 ms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-semibold text-ink">Random Forest (100 Trees)</td>
                    <td className="py-2.5 px-3 text-neutral">Ensemble</td>
                    <td className="py-2.5 px-3">HOG (900-dim)</td>
                    <td className="py-2.5 px-3">91.97%</td>
                    <td className="py-2.5 px-3">0.9207</td>
                    <td className="py-2.5 pl-3">0.01 ms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-semibold text-ink">Custom TrafficSignCNN</td>
                    <td className="py-2.5 px-3 text-neutral">Deep Learning</td>
                    <td className="py-2.5 px-3">RGB 48×48 px</td>
                    <td className="py-2.5 px-3 font-semibold text-signal">In Training</td>
                    <td className="py-2.5 px-3 font-semibold">—</td>
                    <td className="py-2.5 pl-3">&lt; 3.0 ms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-semibold text-ink">ResNet-18 (Transfer Learning)</td>
                    <td className="py-2.5 px-3 text-neutral">Residual Net</td>
                    <td className="py-2.5 px-3">RGB 48×48 px</td>
                    <td className="py-2.5 px-3 font-semibold text-signal">Scheduled</td>
                    <td className="py-2.5 px-3 font-semibold">—</td>
                    <td className="py-2.5 pl-3">&lt; 4.5 ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Sidebar Metadata & Research Scope */}
        <div className="md:col-span-4 space-y-6">
          <div className="border border-ink/20 p-6 bg-paper-dark/30">
            <h3 className="font-mono text-xs uppercase tracking-widest text-ink font-bold border-b border-ink/15 pb-2">
              Research Specifications
            </h3>
            <dl className="mt-4 space-y-3 text-xs font-mono">
              <div className="flex justify-between">
                <dt className="text-neutral">Benchmark:</dt>
                <dd className="font-semibold text-ink">GTSRB Official</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Total Classes:</dt>
                <dd className="font-semibold text-ink">43 Standard Signs</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Training Set:</dt>
                <dd className="font-semibold text-ink">39,209 Images</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Test Set:</dt>
                <dd className="font-semibold text-ink">12,630 Images</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Color Space:</dt>
                <dd className="font-semibold text-ink">Normalized RGB</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Input Dimension:</dt>
                <dd className="font-semibold text-ink">48 × 48 px</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral">Safety Protocol:</dt>
                <dd className="font-semibold text-signal">Zero Horizontal Flips</dd>
              </div>
            </dl>
          </div>

          {/* Safety Notice Callout */}
          <div className="border border-signal/30 bg-signal/5 p-6">
            <div className="flex items-start space-x-3">
              <ShieldCheck className="w-5 h-5 text-signal flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-mono text-xs uppercase tracking-wider text-signal font-bold">
                  Safety Invariant Rule
                </h4>
                <p className="mt-1 text-xs text-ink/80 leading-relaxed font-sans">
                  Unlike generic vision benchmarks, traffic signs possess strict directional semantics. Horizontal mirroring transforms "Turn Left" into "Turn Right" and alters priority vectors. Data augmentation strictly prohibits horizontal or vertical flips.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Six Pillars of the Study */}
      <section className="border-t border-ink/15 pt-12">
        <h2 className="font-serif text-3xl font-bold text-ink">Investigation Pillars</h2>
        <p className="mt-2 text-sm font-mono text-neutral">Modular experimental architecture of the SIGNAL / VISION laboratory.</p>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="border border-ink/15 p-6 bg-paper-light hover:border-signal transition-colors">
            <span className="font-mono text-xs text-signal font-bold">PILLAR 01</span>
            <h3 className="font-serif text-xl font-bold text-ink mt-2">Empirical Profiling</h3>
            <p className="mt-2 text-xs text-ink/75 leading-relaxed font-sans">
              Comprehensive statistical profiling covering class frequency imbalance (10.71× ratio, Gini = 0.4124), resolution variance, and photometric channel distributions.
            </p>
          </div>

          <div className="border border-ink/15 p-6 bg-paper-light hover:border-signal transition-colors">
            <span className="font-mono text-xs text-signal font-bold">PILLAR 02</span>
            <h3 className="font-serif text-xl font-bold text-ink mt-2">Gradient vs Deep Learning</h3>
            <p className="mt-2 text-xs text-ink/75 leading-relaxed font-sans">
              Direct mathematical comparison between hand-crafted spatial descriptors (Dalal-Triggs HOG) and hierarchical representation learning (CNN and ResNet-18).
            </p>
          </div>

          <div className="border border-ink/15 p-6 bg-paper-light hover:border-signal transition-colors">
            <span className="font-mono text-xs text-signal font-bold">PILLAR 03</span>
            <h3 className="font-serif text-xl font-bold text-ink mt-2">Uncertainty & OOD Rejection</h3>
            <p className="mt-2 text-xs text-ink/75 leading-relaxed font-sans">
              Rejection of unrelated non-traffic imagery using predictive Shannon entropy and calibrated Maximum Softmax Probability (MSP) baselines.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
