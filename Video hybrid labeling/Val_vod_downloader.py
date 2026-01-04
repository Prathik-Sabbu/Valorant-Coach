from typing import Dict, Optional, List, Any
import asyncio
import aiohttp
import logging
import yt_dlp
import os

class AutomatedVODDownloader:
    "Downloads VODs from various sources automatically"

    def __init__(self, config: Dict):
        self.config = config
        self.ydl_opts = {
            'format': 'best[height<=1080]',
            'outtmpl': f"{config['storage']['raw_clips_dir']}/%(title)s.%(ext)s",
            'writeinfojson': True,
            'writesubtitles': True
        }
        # simple in-memory priority queue to hold URLs for downstream processing
        # lower numbers == higher priority
        self._processing_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    async def moniter_twitch_streams(self):
        "moniter twich streams to get vod clips"

        async with aiohttp.ClientSession() as session:

            for channel in self.config['data_sources']['twitch_channels']:

                try:
                    is_live, stream_info = await self.check_twitch_stream(session, channel)

                    # ensure we check for the string properly
                    if is_live and 'valorant' in stream_info.get('game', '').lower():
                        await self.queue_for_processing(stream_info.get('url'), priority=1)

                except Exception as e:
                    logging.error(f'Error monitering {channel}: {e}')

    async def scrape_youtube_highlights(self):
        'scrape youtube for valorant highlights and pro games'

        for channel in self.config['data_sources']['youtube_channels']:
            search_queries = [
                f"{channel} valorant ace",
                f"{channel} valorant clutch",
                f"{channel} valorant highlights",
                f"{channel} VALORANT Champions",
                f"{channel} VALORANT Masters",               
            ]
            
            for query in search_queries:
                try:
                    videos = await self.search_youtube_videos(query, max_result=10)

                    for video in videos:
                        if self.is_relevent_video(video):
                            await self.queue_for_processing(video.get('url'), priority=2)
                except Exception as e:
                    logging.error(f"Error scraping YouTube: {e}")

    # --- Helper methods to make the downloader runnable without external services ---
    async def check_twitch_stream(self, session: aiohttp.ClientSession, channel: str) -> tuple[bool, Dict[str, Any]]:
        """Check whether the given Twitch channel is live and return basic stream info.

        This is a minimal, best-effort implementation. If Twitch credentials are provided
        in config (client id / oauth), the method could be expanded to query the Helix API.
        Without credentials this returns (False, {}).
        """
        # If user supplied credentials, you could implement a proper Helix request here.
        twitch_cfg = self.config.get('data_sources', {}).get('twitch', {})
        if not twitch_cfg.get('client_id') or not twitch_cfg.get('oauth_token'):
            # no credentials: return not-live
            return False, {}

        # Placeholder: real implementation would call Twitch Helix endpoints
        try:
            # Example structure returned if live: {'url': 'https://www.twitch.tv/..', 'game': 'Valorant'}
            # Here we simply return not live to avoid external dependencies in tests.
            return False, {}
        except Exception as e:
            logging.debug(f"check_twitch_stream failed for {channel}: {e}")
            return False, {}

    def download_vod(self, url: str)-> Optional[str]:
        'Download a single VOD'

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Use yt-dlp's prepare_filename to get the full path
                filepath = ydl.prepare_filename(info)
                
                if os.path.exists(filepath):
                    return filepath
                
                logging.error(f"Downloaded file not found at {filepath}")
                return None
            
        except Exception as e:
            logging.error(f"Download failed for {url}: {e}")
            return None

    async def queue_for_processing(self, url: str, priority: int = 10):
        """Enqueue a URL for downstream processing. Uses an in-memory PriorityQueue.

        This is intentionally simple so the pipeline is runnable for testing. In a
        production system you'd push to a durable queue or task system.
        """
        if url is None:
            logging.debug("queue_for_processing called with None url; skipping")
            return
        # ensure queue exists
        await self._processing_queue.put((priority, url))
        logging.info(f"Queued {url} with priority {priority}")

    async def pop_next(self, timeout: Optional[float] = None) -> Optional[tuple]:
        """Pop the next (priority, url) item from the internal queue.

        If timeout is provided, waits up to timeout seconds and returns None on timeout.
        This provides a public, async-safe consumer interface so external code doesn't
        touch the private queue directly.
        """
        try:
            if timeout is None:
                item = await self._processing_queue.get()
                return item
            else:
                return await asyncio.wait_for(self._processing_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def mark_task_done(self):
        """Mark a previously retrieved queue item as done (wrapper for task_done())."""
        try:
            self._processing_queue.task_done()
        except Exception:
            # ignore if queue state is inconsistent
            logging.debug("mark_task_done: queue task_done failed or queue empty")

    async def search_youtube_videos(self, query: str, max_result: int = 5) -> List[Dict[str, Any]]:
        """Search YouTube using yt_dlp's "ytsearch" extractor in a thread to avoid blocking.

        Returns a list of video dicts with at least 'url' and 'title'.
        """
        def blocking_search(q, n):
            opts = {**self.ydl_opts, 'quiet': True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                # 'ytsearchN:' is supported by yt_dlp to search YouTube
                info = ydl.extract_info(f"ytsearch{n}:{q}", download=False)
                return info.get('entries', []) if info else []

        try:
            entries = await asyncio.to_thread(blocking_search, query, max_result)
            results: List[Dict[str, Any]] = []
            for e in entries:
                # normalize to have a url field
                vid_url = e.get('webpage_url') or e.get('url') or None
                results.append({'url': vid_url, 'title': e.get('title', ''), 'info': e})
            return results
        except Exception as e:
            logging.error(f"YouTube search failed for '{query}': {e}")
            return []

    def is_relevent_video(self, video: Dict[str, Any]) -> bool:
        """Basic relevance check: title or description contains 'valorant' and video length reasonable.

        The `video` dict is expected to contain at least 'title' or 'info' keys from yt_dlp.
        """
        if not video:
            return False
        title = (video.get('title') or '').lower()
        info = video.get('info') or {}
        description = (info.get('description') or '').lower()

        if 'valorant' in title or 'valorant' in description:
            # optional: filter out very short clips if duration available
            duration = info.get('duration')
            if duration is None or duration >= 5:
                return True
        return False

