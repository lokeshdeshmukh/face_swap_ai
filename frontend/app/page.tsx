"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { apiBase, type JobStatus } from "../lib/api";

type Mode = "video_swap" | "photo_sing";
type Quality = "fast" | "balanced" | "max";
type AspectRatio = "9:16" | "1:1" | "4:5";

export default function HomePage() {
  const [mode, setMode] = useState<Mode>("video_swap");
  const [quality, setQuality] = useState<Quality>("balanced");
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>("9:16");
  const [enable4k, setEnable4k] = useState(false);

  const [referenceVideo, setReferenceVideo] = useState<File | null>(null);
  const [sourceImage, setSourceImage] = useState<File | null>(null);
  const [drivingAudio, setDrivingAudio] = useState<File | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const pollEnabled = useMemo(
    () => Boolean(jobId && job?.status !== "done" && job?.status !== "failed"),
    [jobId, job?.status],
  );

  useEffect(() => {
    if (!jobId) return;

    let alive = true;

    const fetchJob = async () => {
      const response = await fetch(`${apiBase}/v1/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job: ${response.status}`);
      }
      const data = (await response.json()) as JobStatus;
      if (alive) setJob(data);
    };

    fetchJob().catch((e: unknown) => {
      if (alive) setError(e instanceof Error ? e.message : "Unexpected polling error");
    });

    if (!pollEnabled) return;

    const timer = setInterval(() => {
      fetchJob().catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : "Unexpected polling error");
      });
    }, 3000);

    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [jobId, pollEnabled]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (!referenceVideo || !sourceImage) {
      setError("Reference video and source image are required.");
      return;
    }

    const form = new FormData();
    form.append("mode", mode);
    form.append("quality", quality);
    form.append("enable_4k", String(enable4k));
    form.append("aspect_ratio", aspectRatio);
    form.append("reference_video", referenceVideo);
    form.append("source_image", sourceImage);
    if (drivingAudio) form.append("driving_audio", drivingAudio);

    setIsSubmitting(true);
    try {
      const response = await fetch(`${apiBase}/v1/jobs`, {
        method: "POST",
        body: form,
      });
      const data = (await response.json()) as { id?: string; detail?: string };
      if (!response.ok || !data.id) {
        throw new Error(data.detail ?? `Create job failed: ${response.status}`);
      }
      setJobId(data.id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unexpected create job error");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main>
      <h1>TrueFaceSwapVideo POC</h1>
      <p className="muted">
        Local-first orchestration with Runpod Serverless. Upload Instagram-style inputs and track job stages.
      </p>

      <form className="card" onSubmit={onSubmit}>
        <div className="grid">
          <div>
            <label htmlFor="mode">Mode</label>
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="video_swap">Exact Movement Face Swap</option>
              <option value="photo_sing">Photo to Singing</option>
            </select>
          </div>

          <div>
            <label htmlFor="quality">Quality</label>
            <select id="quality" value={quality} onChange={(e) => setQuality(e.target.value as Quality)}>
              <option value="fast">Fast</option>
              <option value="balanced">Balanced</option>
              <option value="max">Max</option>
            </select>
          </div>

          <div>
            <label htmlFor="aspect">Aspect Ratio</label>
            <select
              id="aspect"
              value={aspectRatio}
              onChange={(e) => setAspectRatio(e.target.value as AspectRatio)}
            >
              <option value="9:16">9:16 (Instagram Reels)</option>
              <option value="1:1">1:1</option>
              <option value="4:5">4:5</option>
            </select>
          </div>

          <div>
            <label htmlFor="enable4k">4K Enhance</label>
            <select id="enable4k" value={String(enable4k)} onChange={(e) => setEnable4k(e.target.value === "true")}>
              <option value="false">Off</option>
              <option value="true">On</option>
            </select>
          </div>
        </div>

        <div className="grid" style={{ marginTop: 12 }}>
          <div>
            <label htmlFor="reference">Reference Video</label>
            <input
              id="reference"
              type="file"
              accept="video/*"
              onChange={(e) => setReferenceVideo(e.target.files?.[0] ?? null)}
            />
          </div>

          <div>
            <label htmlFor="source">Source Image</label>
            <input
              id="source"
              type="file"
              accept="image/*"
              onChange={(e) => setSourceImage(e.target.files?.[0] ?? null)}
            />
          </div>

          <div>
            <label htmlFor="audio">Driving Audio (optional; falls back to reference video audio)</label>
            <input
              id="audio"
              type="file"
              accept="audio/*"
              onChange={(e) => setDrivingAudio(e.target.files?.[0] ?? null)}
            />
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <button disabled={isSubmitting} type="submit">
            {isSubmitting ? "Submitting..." : "Create Job"}
          </button>
        </div>
      </form>

      {error ? (
        <section className="card">
          <strong className="error">Error</strong>
          <p className="error">{error}</p>
        </section>
      ) : null}

      {jobId ? (
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Job Status</h2>
          <p>
            <span className="status-pill">ID: {jobId}</span>
            <span className="status-pill">Status: {job?.status ?? "loading"}</span>
            <span className="status-pill">Stage: {job?.stage ?? "loading"}</span>
          </p>

          {job?.error_message ? <p className="error">{job.error_message}</p> : null}

          {job?.output_url ? (
            <div>
              <video controls src={job.output_url} />
              <p>
                <a href={job.output_url} target="_blank" rel="noreferrer" style={{ color: "#7de5c6" }}>
                  Download Result
                </a>
              </p>
            </div>
          ) : (
            <p className="muted">Waiting for Runpod completion callback...</p>
          )}
        </section>
      ) : null}
    </main>
  );
}
