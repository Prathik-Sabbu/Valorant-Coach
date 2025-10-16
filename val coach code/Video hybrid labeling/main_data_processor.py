import asyncio
import os
from Val_vod_downloader import AutomatedVODDownloader
from Rule_event_detector import RuleBasedEventDetector
from Clip_shortener import extract_event_clips
from Val_data_collector import AutomatedValorantDataCollector

def main():
    collector = AutomatedVODDownloader("config.json")
    config = collector.config
    downloader = AutomatedVODDownloader(config)
    detector = RuleBasedEventDetector()

    raw_dir = config["storage"]["raw_clips_dir"]
    processed_dir = config["storage"]["processed_data_dir"]
    db_path = config["storage"]["database_path"]

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    async def download_and_process():

        await downloader.scrape_youtube_highlights()
        await downloader.moniter_twitch_streams()

        # Example: manually feed one URL for testing
        url = "https://www.youtube.com/watch?v=EXAMPLE"

        vod_path = downloader.download_vod(url)
        if not vod_path:
            print("⚠️ Download failed")
            return
        
        # TODO: Run HUD analyzer on vod_path (frames, audio, HUD data)
        # Example placeholders:
        frames = []    # numpy arrays from cv2
        audio = []     # audio samples
        hud_data = []  # extracted HUD timeline

        results = detector.analyze_clip_segment(frames, audio, hud_data, start_time=0.0)

        if results["importance_score"] >= config["processing"]["min_importance_threshold"]:
            events = [
                {
                    "event_type": evt,
                    "timestamp": results["timestamp"]
                    "importance_score": results["importance_score"]
                }
                for evt in results["events"]
            ]
            extract_event_clips(
                input_video=vod_path,
                event_times=events,
                output_dir=processed_dir,
                pre_event=2,
                post_event=2,
                db_path=db_path
            )

    asyncio.run(download_and_process())

if __name__ == "__main__":
    main()
