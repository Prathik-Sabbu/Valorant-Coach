from typing import Dict, Optional
import aiohttp
import logging
import yt_dlp


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

    async def moniter_twitch_streams(self):
        "moniter twich streams to get vod clips"

        async with aiohttp.ClientSession() as session:

            for channel in self.config['data_sources']['twitch_channels']:

                try:
                    is_live, stream_info = await self.check_twitch_stream(session, channel)

                    if is_live and ('valorant' or 'Valorant') in stream_info.get('game', '').lower():
                        await self.queue_for_processing(stream_info['url'], priority = 1)

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
                    videos = await self.search_youtube_videos(query, max_result = 10)

                    for video in videos:
                        if self.is_relevent_video(video):
                            await self.queue_for_processing(video['url'], priority=2)
                except Exception as e:
                    logging.error(f"Error scraping YouTube: {e}")

    def download_vod(self, url: str)-> Optional[str]:
        'Download a single VOD'

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info.get('filepath') or f"{info['title']}.{info['ext']}"
            
        except Exception as e:
            logging.error(f"Download failed for {url}: {e}")
            return None

