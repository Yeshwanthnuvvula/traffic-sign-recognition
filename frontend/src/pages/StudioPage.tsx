import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Activity,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  RotateCcw,
  Sparkles,
  Sliders,
  Image as ImageIcon
} from 'lucide-react';
import {
  predictTrafficSign,
  getAvailableModels,
  getFullEDA,
  getSampleImageUrl
} from '../services/api';
import { PredictionResult, ModelSummary, EDASummary } from '../types';

export const StudioPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('svm');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [edaData, setEdaData] = useState<EDASummary | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load available models and sample images
  useEffect(() => {
    const fetchMeta = async () => {
      try {
        const [modelsRes, edaRes] = await Promise.all([
          getAvailableModels().catch(() => ({ models: [], default: 'svm' })),
          getFullEDA().catch(() => null)
        ]);

        if (modelsRes.models && modelsRes.models.length > 0) {
          setModels(modelsRes.models);
          setSelectedModel(modelsRes.models[0].id);
        } else {
          // Fallback defaults
          const defaultList: ModelSummary[] = [
            { id: 'svm', name: 'Support Vector Machine (RBF)', type: 'Classical ML' },
            { id: 'knn', name: 'K-Nearest Neighbors', type: 'Classical ML' },
            { id: 'rf', name: 'Random Forest', type: 'Ensemble ML' },
            { id: 'custom_cnn', name: 'Custom TrafficSignCNN', type: 'Deep Learning' },
            { id: 'resnet18', name: 'ResNet-18 (Deep Residual)', type: 'Deep Learning' }
          ];
          setModels(defaultList);
          setSelectedModel('svm');
        }

        if (edaRes) {
          setEdaData(edaRes);
        }
      } catch (err) {
        console.error('Error fetching studio metadata:', err);
      }
    };
    fetchMeta();
  }, []);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, PPM).');
      return;
    }
    setError(null);
    setResult(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleQuickPreset = async (classId: number) => {
    if (!edaData?.class_details[classId]) return;
    try {
      setLoading(true);
      setError(null);
      const sampleFile = edaData.class_details[classId].sample_file;
      const imgUrl = getSampleImageUrl(sampleFile);
      
      const response = await fetch(imgUrl);
      const blob = await response.blob();
      const file = new File([blob], `sample_class_${classId}.png`, { type: 'image/png' });
      
      setSelectedFile(file);
      setPreviewUrl(imgUrl);
      
      // Auto analyze
      const pred = await predictTrafficSign(file, selectedModel);
      setResult(pred);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Failed to analyze sample image.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const pred = await predictTrafficSign(selectedFile, selectedModel);
      setResult(pred);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Inference execution failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-12">
      {/* Studio Header */}
      <section className="border-b border-ink/15 pb-6">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-4">
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-signal font-semibold">
              Module 02 • Real-Time Computer Vision Workspace
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-bold text-ink tracking-tight mt-1">
              Traffic Sign Recognition Studio
            </h1>
          </div>
          <p className="font-mono text-xs text-neutral max-w-md">
            Interactive visual laboratory executing deterministic 48×48 preprocessing, model inference, and uncertainty evaluation.
          </p>
        </div>
      </section>

      {/* Main Workspace 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Input & Inspection Station */}
        <div className="lg:col-span-5 space-y-6">
          <div className="border border-ink/20 bg-paper-light p-6">
            <h2 className="font-mono text-xs uppercase tracking-widest text-ink font-bold mb-4 flex items-center justify-between">
              <span>01. Image Ingestion Station</span>
              {selectedFile && (
                <button
                  onClick={handleReset}
                  className="text-neutral hover:text-signal flex items-center text-[11px] normal-case"
                >
                  <RotateCcw className="w-3 h-3 mr-1" /> Reset
                </button>
              )}
            </h2>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-none p-8 text-center cursor-pointer transition-all ${
                previewUrl
                  ? 'border-ink/30 bg-paper-dark/20'
                  : 'border-ink/30 hover:border-signal bg-paper hover:bg-signal/5'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.ppm"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
              />

              {previewUrl ? (
                <div className="space-y-4">
                  <div className="relative inline-block border border-ink/20 shadow-subtle bg-black/5 p-2">
                    <img
                      src={previewUrl}
                      alt="Input Preview"
                      className="max-h-56 max-w-full mx-auto object-contain"
                    />
                  </div>
                  <p className="font-mono text-xs text-neutral">
                    {selectedFile?.name} ({(selectedFile ? selectedFile.size / 1024 : 0).toFixed(1)} KB)
                  </p>
                  <p className="font-mono text-[11px] text-ink/60">Click or drop to replace image</p>
                </div>
              ) : (
                <div className="space-y-3 py-6">
                  <Upload className="w-8 h-8 text-neutral mx-auto stroke-[1.5]" />
                  <div className="font-serif text-lg text-ink font-bold">
                    Drop Traffic Sign Photograph Here
                  </div>
                  <p className="font-mono text-xs text-neutral max-w-xs mx-auto">
                    Accepts PNG, JPEG, or GTSRB PPM files. Dimensions will be normalized to 48×48 px.
                  </p>
                </div>
              )}
            </div>

            {/* Quick Test Presets from Dataset */}
            {edaData && (
              <div className="mt-6 pt-4 border-t border-ink/10">
                <span className="font-mono text-[11px] uppercase tracking-wider text-neutral block mb-2.5">
                  Or Test Canonical Benchmark Examples:
                </span>
                <div className="grid grid-cols-6 gap-2">
                  {[1, 2, 14, 13, 17, 33].map((cid) => {
                    const sample = edaData.class_details[cid];
                    if (!sample) return null;
                    return (
                      <button
                        key={cid}
                        onClick={() => handleQuickPreset(cid)}
                        className="group relative border border-ink/20 hover:border-signal p-1 bg-paper hover:bg-signal/5 transition-all text-center"
                        title={sample.class_name}
                      >
                        <img
                          src={getSampleImageUrl(sample.sample_file)}
                          alt={sample.class_name}
                          className="w-full aspect-square object-cover"
                        />
                        <span className="block font-mono text-[9px] text-neutral group-hover:text-signal truncate mt-1">
                          #{cid}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Model Selection Card */}
          <div className="border border-ink/20 bg-paper-light p-6">
            <h2 className="font-mono text-xs uppercase tracking-widest text-ink font-bold mb-4 flex items-center">
              <Sliders className="w-3.5 h-3.5 mr-2 text-signal" />
              02. Classifier Architecture Selector
            </h2>
            <div className="space-y-2 font-mono text-xs">
              {models.map((m) => (
                <label
                  key={m.id}
                  className={`flex items-center justify-between p-3 border cursor-pointer transition-colors ${
                    selectedModel === m.id
                      ? 'border-signal bg-signal/10 text-ink font-semibold'
                      : 'border-ink/15 hover:border-ink/40 bg-paper text-ink/80'
                  }`}
                >
                  <div className="flex items-center">
                    <input
                      type="radio"
                      name="modelSelector"
                      value={m.id}
                      checked={selectedModel === m.id}
                      onChange={() => setSelectedModel(m.id)}
                      className="accent-signal mr-3"
                    />
                    <span>{m.name}</span>
                  </div>
                  <span className="text-[10px] uppercase text-neutral tracking-wider border border-ink/15 px-2 py-0.5">
                    {m.type}
                  </span>
                </label>
              ))}
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              className={`w-full mt-6 py-3 px-4 font-mono text-xs uppercase tracking-wider text-paper transition-all flex items-center justify-center ${
                !selectedFile || loading
                  ? 'bg-neutral cursor-not-allowed'
                  : 'bg-signal hover:bg-signal-dark active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-spin" />
                  <span>Computing Inference Pass...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  <span>Execute Classification</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Prediction Diagnostics & Visualization */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="border border-signal bg-signal/10 p-4 font-mono text-xs text-signal flex items-start space-x-3">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-bold uppercase tracking-wider">Inference Notification</p>
                <p className="mt-1 font-sans">{error}</p>
              </div>
            </div>
          )}

          {result ? (
            <div className="space-y-6">
              {/* Primary Prediction Banner */}
              <div
                className={`border p-6 ${
                  result.uncertainty.is_rejected
                    ? 'border-signal bg-signal/5'
                    : result.uncertainty.is_uncertain
                    ? 'border-neutral bg-neutral/5'
                    : 'border-ink bg-paper-light'
                }`}
              >
                <div className="flex items-start justify-between border-b border-ink/15 pb-4">
                  <div>
                    <span className="font-mono text-xs uppercase tracking-widest text-neutral">
                      Class #{result.predicted_class_id} • {result.category}
                    </span>
                    <h3 className="font-serif text-3xl font-bold text-ink mt-1">
                      {result.predicted_class_name}
                    </h3>
                  </div>

                  {/* Confidence Badge */}
                  <div className="text-right">
                    <span className="font-mono text-2xl font-bold text-signal">
                      {result.confidence_percentage.toFixed(1)}%
                    </span>
                    <span className="block font-mono text-[10px] uppercase text-neutral">
                      Confidence
                    </span>
                  </div>
                </div>

                {/* Status Notice */}
                {result.uncertainty.is_rejected && (
                  <div className="mt-4 p-3 bg-signal/15 border border-signal/30 text-xs font-mono text-signal flex items-center">
                    <XCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                    <span>Rejection Warning: Potential Out-Of-Distribution (Non-Traffic Sign) Input</span>
                  </div>
                )}
                {!result.uncertainty.is_rejected && result.uncertainty.is_uncertain && (
                  <div className="mt-4 p-3 bg-neutral/15 border border-neutral/30 text-xs font-mono text-ink flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-2 flex-shrink-0 text-signal" />
                    <span>Ambiguity Warning: Low decision margin between top candidate classes.</span>
                  </div>
                )}

                {/* Preprocessing & Performance Metrics Ribbon */}
                <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono border-t border-ink/10 pt-4">
                  <div>
                    <span className="text-neutral block">Inference Latency:</span>
                    <span className="font-semibold text-ink">{result.inference_time_ms} ms</span>
                  </div>
                  <div>
                    <span className="text-neutral block">Input Geometry:</span>
                    <span className="font-semibold text-ink">
                      {result.image_metadata.width}×{result.image_metadata.height} px
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral block">Normalized Entropy:</span>
                    <span className="font-semibold text-ink">
                      {result.uncertainty.normalized_entropy.toFixed(3)}
                    </span>
                  </div>
                  <div>
                    <span className="text-neutral block">Active Model:</span>
                    <span className="font-semibold text-ink uppercase">{result.model_id}</span>
                  </div>
                </div>
              </div>

              {/* Top 3 Posterior Probabilities */}
              <div className="border border-ink/20 bg-paper-light p-6">
                <h4 className="font-mono text-xs uppercase tracking-widest text-ink font-bold mb-4">
                  Top 3 Posterior Distribution Candidates
                </h4>
                <div className="space-y-4">
                  {result.top_3_predictions.map((c, idx) => (
                    <div key={c.class_id} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="font-semibold text-ink">
                          {idx + 1}. [#{c.class_id}] {c.class_name}
                        </span>
                        <span className="text-signal font-bold">{c.percentage.toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-paper-dark h-2 rounded-none overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            idx === 0 ? 'bg-signal' : 'bg-neutral'
                          }`}
                          style={{ width: `${Math.max(2, c.percentage)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Speed Limit Disambiguation Specific Note */}
              {(result.predicted_class_id === 1 || result.predicted_class_id === 2) && (
                <div className="border border-ink/20 bg-paper-dark/30 p-4 text-xs font-mono">
                  <span className="font-bold uppercase tracking-wider text-signal block mb-1">
                    Speed Limit Disambiguation Check (30 km/h vs 50 km/h)
                  </span>
                  <p className="font-sans text-ink/80 leading-relaxed">
                    Categories 1 (Speed 30) and 2 (Speed 50) share identical circular red outer boundaries and white internal discs, differentiating solely in the central glyph's upper loop ("3" vs "5"). Current decision margin: {(result.uncertainty.decision_margin * 100).toFixed(1)}%.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="border border-ink/20 bg-paper-light p-12 text-center h-full flex flex-col items-center justify-center min-h-[400px]">
              <ImageIcon className="w-12 h-12 text-neutral/40 mb-4 stroke-1" />
              <h3 className="font-serif text-2xl font-bold text-ink">Awaiting Image Ingestion</h3>
              <p className="mt-2 text-xs font-mono text-neutral max-w-sm">
                Upload a traffic sign image or click one of the benchmark presets on the left to observe real-time classification.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
