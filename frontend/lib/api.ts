export type JobStatus = {
  id: string;
  status: string;
  stage: string;
  error_message: string | null;
  output_url: string | null;
  runpod_job_id: string | null;
  request_id: string | null;
};

export const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
