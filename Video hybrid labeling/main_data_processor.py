import asyncio
import os
import logging
from typing import List, Dict, Any
import cv2
import numpy as np
from collections import deque

from Val_vod_downloader import AutomatedVODDownloader
from Rule_event_detector import RuleBasedEventDetector
from Val_data_collector import AutomatedValorantDataCollector

logging.basicConfig(level=logging.INFO)


def ensure_dirs(paths: List[str]):
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not create directory {p}: {e}")


def build_event_dicts(detection_result: Dict[str, Any], min_threshold: float) -> List[Dict[str, Any]]:
    """Convert detector output into list of event dicts expected by the clip extractor/DB.

    Each event dict contains: event_type, timestamp, importance_score
    """
    events = []
    importance = detection_result.get("importance_score", 0.0)
    timestamp = detection_result.get("timestamp", 0.0)
    for evt in detection_result.get("events", []):
        if importance >= min_threshold:
            events.append({
                "event_type": evt,
                "timestamp": timestamp,
                "importance_score": importance,
            })
    return events

def extract_video_data(video_path: str, sample_rate: int = 30):
    """Extract frames from video. Returns (frames, audio, hud_data, video_fps, sample_interval).

    For now, just extracts frames. Audio and HUD extraction can be added later.
    sample_rate is interpreted as "take every sample_rate-th frame".
    This function no longer imposes an artificial frame limit — it will sample across the whole video.
    """
    logging.info(f"Extracting frames from {video_path} (sampling every {sample_rate} frames)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return np.array([]), np.array([]), [], 30.0, sample_rate

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames (take every Nth frame to reduce memory usage)
        if frame_count % sample_rate == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    logging.info(f"Extracted {len(frames)} sampled frames from {frame_count} total frames (video_fps={video_fps})")

    # Return frames and metadata needed to map sampled frames back to time
    return np.array(frames) if frames else np.array([]), np.array([]), [], float(video_fps), sample_rate


def main():
    # Initialize collector (loads config and prepares DB)
    # Load config.json from the script directory so relative runs find the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    logging.info(f"Using config at: {config_path}")
    collector = AutomatedValorantDataCollector(config_path)
    config = collector.config

    # Some config keys may be missing; provide safe defaults
    storage = config.get("storage", {})
    raw_dir = storage.get("raw_clips_dir", "./raw_clips")
    processed_dir = storage.get("processed_data_dir", "./processed_data")
    db_path = storage.get("database_path", "./valorant_data.db")

    processing_cfg = config.get("processing", {})
    min_importance = processing_cfg.get("min_importance_threshold", 0.5)

    ensure_dirs([raw_dir, processed_dir])

    # Initialize downloader and detector
    downloader = AutomatedVODDownloader(config)
    detector = RuleBasedEventDetector()

    async def download_and_process():
        # Run scraping/monitoring in background (best-effort)
        # try:
        #     await asyncio.gather(
        #         downloader.scrape_youtube_highlights(),
        #         downloader.moniter_twitch_streams(),
        #         return_exceptions=True,
        #     )
        # except Exception as e:
        #     logging.warning(f"Background fetchers raised: {e}")
        logging.info("Skipping background scraping for this test run")

        # Example: single manual URL to demonstrate pipeline
        example_url = config.get("processing", {}).get("example_vod_url", "https://www.youtube.com/watch?v=EXAMPLE")

        # For testing: enqueue exactly one example video and then process a single queue item
        logging.info(f"Queueing example VOD for processing: {example_url}")
        await downloader.queue_for_processing(example_url, priority=2)

        # Dequeue one item using public consumer API
        item = await downloader.pop_next()
        if item is None:
            logging.error("No queued item received (timeout or empty). Exiting.")
            return
        priority, queued_url = item
        logging.info(f"Dequeued {queued_url} (priority={priority}) for processing")

        vod_path = downloader.download_vod(queued_url)
        if not vod_path:
            logging.error("Download failed or returned no path. Exiting example run.")
            return

        logging.info(f"Downloaded VOD to {vod_path}")

        # Stream-process the video: sample frames, form segments (with optional overlap), detect, and immediately ingest clips
        sample_rate = processing_cfg.get('sample_rate', 30)  # sample every Nth frame
        segment_duration = processing_cfg.get('segment_duration', 5.0)  # seconds per segment
        segment_overlap = processing_cfg.get('segment_overlap', 0.0)  # 0.0..0.9 fraction of overlap

        def process_video_stream(video_file: str):
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                logging.error(f"Could not open video for streaming: {video_file}")
                return 0

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            logging.info(f"Streaming video (fps={fps}, total_frames={total_frames})")

            # number of sampled frames per segment
            frames_per_segment = max(1, int(segment_duration * fps / float(sample_rate)))
            # compute step size to allow overlap (step = frames_per_segment * (1 - overlap))
            overlap = float(segment_overlap)
            if not 0.0 <= overlap < 1.0:
                overlap = 0.0
            step = max(1, int(frames_per_segment * (1.0 - overlap)))

            sampled = deque()
            frame_idx = 0
            processed_events = 0
            last_logged_percent = -1.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps
                    sampled.append((frame.copy(), timestamp))

                    if len(sampled) >= frames_per_segment:
                        segment_frames = [f for f, t in sampled]
                        start_time = sampled[0][1]

                        results = detector.analyze_clip_segment(np.array(segment_frames), np.array([]), [], start_time=start_time)

                        logging.info(f"Segment at {start_time:.1f}s: events={results.get('events')}, importance={results.get('importance_score'):.2f}")

                        events = build_event_dicts(results, min_importance)
                        if events:
                            try:
                                collector.ingest_clips(input_video=video_file, event_times=events)
                                processed_events += len(events)
                                logging.info(f"Saved {len(events)} clip(s) at ~{start_time:.1f}s")
                            except Exception as e:
                                logging.error(f"Failed to ingest clips for segment at {start_time}: {e}")

                        # advance the sliding window by `step` sampled frames (supports overlap)
                        for _ in range(step):
                            if sampled:
                                sampled.popleft()
                            else:
                                break

                frame_idx += 1

                # progress logging every ~5%
                if total_frames > 0:
                    percent = (frame_idx / total_frames) * 100.0
                    if percent - last_logged_percent >= 5.0:
                        last_logged_percent = percent
                        logging.info(f"Processing progress: {percent:.1f}% (frame {frame_idx}/{total_frames})")

            cap.release()
            logging.info(f"Streaming processing complete, processed events: {processed_events}")
            return processed_events

        processed_count = await asyncio.to_thread(process_video_stream, vod_path)
        logging.info(f"Total clips processed and saved: {processed_count}")

        try:
            downloader.mark_task_done()
        except Exception:
            pass

    # Run the async pipeline
    asyncio.run(download_and_process())


if __name__ == "__main__":
    main()
