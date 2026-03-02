export type JobStatus = {
  id: string;
  mode: string;
  quality: string;
  status: string;
  stage: string;
  stage_timings?: Record<string, Record<string, string>>;
  error_message: string | null;
  output_url: string | null;
  runpod_job_id: string | null;
  request_id: string | null;
  input_config: Record<string, unknown>;
};

export const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
