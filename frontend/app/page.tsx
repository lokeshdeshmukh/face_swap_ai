"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { apiBase, type JobStatus } from "../lib/api";

type Mode =
  | "portrait_reenactment"
  | "full_body_reenactment"
  | "ai_video_generate"
  | "photo_to_video"
  | "video_swap"
  | "photo_sing";
type Quality = "fast" | "balanced" | "max";
type AspectRatio = "9:16" | "1:1" | "4:5";

const GENERATION_MODES: Mode[] = ["portrait_reenactment", "full_body_reenactment", "ai_video_generate", "photo_to_video"];
const REENACTMENT_MODES: Mode[] = ["portrait_reenactment", "full_body_reenactment"];

export default function HomePage() {
  const [mode, setMode] = useState<Mode>("full_body_reenactment");
  const [quality, setQuality] = useState<Quality>("balanced");
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>("9:16");
  const [enable4k, setEnable4k] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [motionPreset, setMotionPreset] = useState("cinematic_dolly");
  const [stylePreset, setStylePreset] = useState("studio_realism");
  const [durationSeconds, setDurationSeconds] = useState("5");
  const [seed, setSeed] = useState("");

  const [referenceVideo, setReferenceVideo] = useState<File | null>(null);
  const [sourceImages, setSourceImages] = useState<File[]>([]);
  const [sourceVideo, setSourceVideo] = useState<File | null>(null);
  const [drivingAudio, setDrivingAudio] = useState<File | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isGenerationMode = GENERATION_MODES.includes(mode);
  const isReenactmentMode = REENACTMENT_MODES.includes(mode);
  const isFullBodyReenactmentMode = mode === "full_body_reenactment";
  const usesPromptDrivenGeneration = isGenerationMode && !isReenactmentMode;
  const needsReferenceVideo = !isGenerationMode || isReenactmentMode;

  const pollEnabled = useMemo(
    () => Boolean(jobId && job?.status !== "done" && job?.status !== "failed"),
    [jobId, job?.status],
  );

  const orderedStages = useMemo(() => {
    const timings = job?.stage_timings ?? {};
    return Object.keys(timings);
  }, [job?.stage_timings]);

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

    if (sourceImages.length === 0 && !sourceVideo) {
      setError("At least one identity image or identity video is required.");
      return;
    }
    if (needsReferenceVideo && !referenceVideo) {
      setError(
        isReenactmentMode ? "Driving video is required for reenactment modes." : "Reference video is required for legacy modes."
      );
      return;
    }
    if (mode === "ai_video_generate" && !prompt.trim()) {
      setError("Prompt is required for generation modes.");
      return;
    }

    const form = new FormData();
    form.append("mode", mode);
    form.append("quality", quality);
    form.append("enable_4k", String(enable4k));
    form.append("aspect_ratio", aspectRatio);
    if (usesPromptDrivenGeneration) {
      form.append("prompt", prompt);
      form.append("negative_prompt", negativePrompt);
      form.append("motion_preset", motionPreset);
      form.append("style_preset", stylePreset);
      form.append("duration_seconds", durationSeconds);
      if (seed.trim()) form.append("seed", seed.trim());
    }
    if (referenceVideo) form.append("reference_video", referenceVideo);
    sourceImages.forEach((file) => form.append("source_images", file));
    if (sourceVideo) form.append("source_video", sourceVideo);
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
      setJob(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unexpected create job error");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main>
      <h1>TrueFaceSwapVideo</h1>
      <p className="muted">
        Reenactment-first workflow for in-house identity video pipelines. Legacy face-swap modes remain available for
        compatibility.
      </p>

      <form className="card" onSubmit={onSubmit}>
        {isFullBodyReenactmentMode ? (
          <p className="muted" style={{ marginTop: 0 }}>
            Full Body Reenactment is for driver-video performance transfer. Use identity images with clear face plus
            upper-body context, and upload the performer clip as the driving video.
          </p>
        ) : null}

        <div className="grid">
          <div>
            <label htmlFor="mode">Mode</label>
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="portrait_reenactment">Portrait Reenactment</option>
              <option value="full_body_reenactment">Full Body Reenactment</option>
              <option value="ai_video_generate">AI Video Generation (Experimental)</option>
              <option value="photo_to_video">Photo to Video (Experimental)</option>
              <option value="video_swap">Legacy Face Swap</option>
              <option value="photo_sing">Legacy Photo Sing</option>
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
              <option value="9:16">9:16</option>
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

        {usesPromptDrivenGeneration ? (
          <div className="grid" style={{ marginTop: 12 }}>
            <div>
              <label htmlFor="prompt">Prompt</label>
              <textarea
                id="prompt"
                rows={4}
                value={prompt}
                placeholder=""
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="negativePrompt">Negative Prompt</label>
              <textarea
                id="negativePrompt"
                rows={4}
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="motionPreset">Motion Preset</label>
              <input id="motionPreset" value={motionPreset} onChange={(e) => setMotionPreset(e.target.value)} />
            </div>

            <div>
              <label htmlFor="stylePreset">Style Preset</label>
              <input id="stylePreset" value={stylePreset} onChange={(e) => setStylePreset(e.target.value)} />
            </div>

            <div>
              <label htmlFor="durationSeconds">Duration (seconds)</label>
              <input
                id="durationSeconds"
                type="number"
                min={2}
                max={20}
                value={durationSeconds}
                onChange={(e) => setDurationSeconds(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="seed">Seed (optional)</label>
              <input id="seed" value={seed} onChange={(e) => setSeed(e.target.value)} />
            </div>
          </div>
        ) : null}

        <div className="grid" style={{ marginTop: 12 }}>
          <div>
            <label htmlFor="sourceImages">Identity Images</label>
            <input
              id="sourceImages"
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setSourceImages(Array.from(e.target.files ?? []))}
            />
            <p className="muted" style={{ marginTop: 6 }}>
              {sourceImages.length > 0 ? `${sourceImages.length} image(s) selected` : "No identity images selected"}
            </p>
          </div>

          <div>
            <label htmlFor="sourceVideo">Identity Video (optional)</label>
            <input
              id="sourceVideo"
              type="file"
              accept="video/*"
              onChange={(e) => setSourceVideo(e.target.files?.[0] ?? null)}
            />
            <p className="muted" style={{ marginTop: 6 }}>
              {sourceVideo ? sourceVideo.name : "No identity video selected"}
            </p>
          </div>

            <div>
              <label htmlFor="reference">{isReenactmentMode ? "Driving Video" : isGenerationMode ? "Motion Reference Video (optional)" : "Reference Video"}</label>
              <input
                id="reference"
                type="file"
                accept="video/*"
                onChange={(e) => setReferenceVideo(e.target.files?.[0] ?? null)}
              />
              <p className="muted" style={{ marginTop: 6 }}>
                {referenceVideo
                  ? referenceVideo.name
                  : isReenactmentMode
                    ? isFullBodyReenactmentMode
                      ? "Upload the performer video whose full-body motion and expressions should drive the identity."
                      : "Upload the performer video whose motion and expressions should drive the identity."
                    : "No reference video selected"}
              </p>
            </div>

          <div>
            <label htmlFor="audio">
              {mode === "photo_sing" ? "Driving Audio (optional)" : "Audio / Narration (optional)"}
            </label>
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
            <span className="status-pill">Mode: {job?.mode ?? mode}</span>
            <span className="status-pill">Status: {job?.status ?? "loading"}</span>
            <span className="status-pill">Stage: {job?.stage ?? "loading"}</span>
          </p>

          {orderedStages.length > 0 ? (
            <p className="muted">
              Pipeline: {orderedStages.map((stage) => (stage === (job?.stage ?? "") ? `[${stage}]` : stage)).join(" -> ")}
            </p>
          ) : null}

          {job?.input_config?.prompt ? (
            <p className="muted">Prompt: {String(job.input_config.prompt)}</p>
          ) : null}

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
            <p className="muted">Waiting for worker completion callback...</p>
          )}
        </section>
      ) : null}
    </main>
  );
}
