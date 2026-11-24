import axios from 'axios';
import {
  PredictionResult,
  ModelSummary,
  ModelMetricRow,
  MetricDetail,
  EDASummary,
  ExperimentTrial
} from '../types';

const API_BASE = '/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const getHealth = async () => {
  const resp = await apiClient.get('/health');
  return resp.data;
};

export const getDatasetSummary = async () => {
  const resp = await apiClient.get('/dataset/summary');
  return resp.data;
};

export const getFullEDA = async (): Promise<EDASummary> => {
  const resp = await apiClient.get<EDASummary>('/eda');
  return resp.data;
};

export const getAvailableModels = async (): Promise<{ models: ModelSummary[]; default: string }> => {
  const resp = await apiClient.get('/models');
  return resp.data;
};

export const getAllMetrics = async (): Promise<{ models: ModelMetricRow[]; selected_best_model: string }> => {
  const resp = await apiClient.get('/metrics');
  return resp.data;
};

export const getModelDetail = async (modelId: string): Promise<MetricDetail> => {
  const resp = await apiClient.get<MetricDetail>(`/metrics/${modelId}`);
  return resp.data;
};

export const getExperiments = async (): Promise<{ total_trials: number; experiments: ExperimentTrial[] }> => {
  const resp = await apiClient.get('/experiments');
  return resp.data;
};

export const getSpeedLimitAnalysis = async () => {
  const resp = await apiClient.get('/speed-limit-analysis');
  return resp.data;
};

export const predictTrafficSign = async (
  imageFile: File | Blob,
  modelId: string = 'resnet18'
): Promise<PredictionResult> => {
  const formData = new FormData();
  formData.append('file', imageFile);

  const resp = await apiClient.post<PredictionResult>(`/predict?model_id=${encodeURIComponent(modelId)}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return resp.data;
};

export const getSampleImageUrl = (sampleFile: string) => {
  // sampleFile looks like 'samples/class_00.png'
  const filename = sampleFile.split('/').pop();
  return `/api/static/samples/${filename}`;
};
