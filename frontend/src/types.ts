export interface PredictionCandidate {
  class_id: number;
  class_name: string;
  category: string;
  probability: number;
  percentage: number;
}

export interface UncertaintyInfo {
  confidence: number;
  decision_margin: number;
  entropy: number;
  normalized_entropy: number;
  is_uncertain: boolean;
  is_rejected: boolean;
  status_message: string;
}

export interface ImageMetadata {
  width: number;
  height: number;
  aspect_ratio: number;
  channels: number;
  is_valid_quality: boolean;
  quality_note: string;
}

export interface PredictionResult {
  predicted_class_id: number;
  predicted_class_name: string;
  category: string;
  confidence: number;
  confidence_percentage: number;
  top_3_predictions: PredictionCandidate[];
  inference_time_ms: number;
  model_id: string;
  uncertainty: UncertaintyInfo;
  image_metadata: ImageMetadata;
}

export interface ModelSummary {
  id: string;
  name: string;
  type: string;
}

export interface ModelMetricRow {
  model_id: string;
  model_name: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  precision: number;
  recall: number;
  train_time_seconds: number;
  inference_latency_ms: number;
}

export interface MetricDetail extends ModelMetricRow {
  hyperparameters: Record<string, any>;
  train_samples: number;
  val_samples: number;
  confusion_matrix: number[][];
  classification_report: Record<string, any>;
  history?: {
    epoch: number[];
    train_loss: number[];
    train_acc: number[];
    val_loss: number[];
    val_acc: number[];
    lr: number[];
  };
}

export interface EDASummary {
  dataset_name: string;
  total_classes: number;
  total_train_images: number;
  total_test_images: number;
  class_counts: Record<string, number>;
  class_details: Record<string, {
    class_id: number;
    class_name: string;
    category: string;
    count: number;
    sample_file: string;
  }>;
  imbalance_metrics: {
    min_class_instances: number;
    min_class_id: number;
    min_class_name: string;
    max_class_instances: number;
    max_class_id: number;
    max_class_name: string;
    imbalance_ratio: number;
    gini_coefficient: number;
    mean_instances_per_class: number;
    std_instances_per_class: number;
  };
  dimension_stats: {
    min_width: number;
    max_width: number;
    median_width: number;
    min_height: number;
    max_height: number;
    median_height: number;
    mean_aspect_ratio: number;
  };
  photometric_stats: {
    mean_red: number;
    mean_green: number;
    mean_blue: number;
    normalized_mean_rgb: number[];
  };
  categories_breakdown: Record<string, {
    classes: number[];
    total_instances: number;
  }>;
}

export interface ExperimentTrial {
  experiment_id: string;
  model_type: string;
  hyperparameters: Record<string, any>;
  training_duration_s: number;
  val_accuracy: number;
  val_macro_f1: number;
  status: string;
}
