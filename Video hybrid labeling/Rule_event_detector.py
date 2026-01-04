from typing import List, Dict
import logging
import numpy as np
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class RuleBasedEventDetector:
    "Detects important events using heuristic rules"

    def __init__(self):
        # Event detection thresholds
        self.thresholds = {
            'ace': {'min_kills': 5, 'max_time_span': 30},
            'clutch': {'max_team_alive': 2, 'min_enemies': 3},
            'multi_kill': {'min_kills': 2, 'max_time_span': 10},
            'round_win': {'detect_round_end': True},
            'smart_ability_usage': {
                'ultimate_combo_kill': True,      # Ultimate ability + kill within 3 seconds
                'ability_multikill': 2,           # 2+ kills with single ability use
                'clutch_ability_usage': True,     # Ability usage in 1vX situations
                'game_winning_ability': True,     # Ability that secures round win
                'creative_ability_usage': {       # Unusual/skillful ability combinations
                    'min_ability_combo': 2,       # Using 2+ abilities in sequence
                    'max_combo_time': 5,          # Within 5 seconds
                    'includes_movement': True     # Involves movement abilities
                },
                'ability_save': {                 # Smart ability economy
                    'last_round_save': True,      # Saving ultimate for final round
                    'eco_round_save': True        # Not using expensive abilities on save rounds
                }
            }
        }

        self.audio_signatures = {
            'gunshot': ([1000, 3000], 0.8),  # freq range, min confidence
            'ability_cast': ([500, 1500], 0.6),
            'elimination_sound': ([800, 2400], 0.9)
        }
        # OCR availability flag — will be set to False if the tesseract binary is not found
        self._ocr_available = True

    def analyze_clip_segment(self, frames: np.ndarray, audio: np.ndarray, hud_data: List[Dict], start_time: float) -> Dict:
        "analize clip segmant and detect events"

        events_detected = []
        time = []
        importance_score = 0.0

        kill_events = self.extract_kill_events(hud_data)
        if len(kill_events) >= 5:
            events_detected.append('ace')
            time.append(start_time)
            importance_score = 0.95
        elif len(kill_events) >=2:
            events_detected.append('multi_kill')
            time.append(start_time)
            importance_score = max(importance_score, 0.7)
        
        clutch_info = self.detect_clutch_situation(hud_data)
        if clutch_info['is_clutch']:
            events_detected.append('clutch')
            time.append(start_time)
            importance_score = max(importance_score, 0.8)

        ability_analysis = self.detect_smart_ability_usage(hud_data, kill_events, audio)
        if ability_analysis['smart_usage_detected']:
            events_detected.append('smart_ability_usage')
            time.append(start_time)
            importance_score = max(importance_score, ability_analysis['importance_boost'])
        
        audio_events = self.analyze_audio_events(audio)
        if audio_events['high_activity']:
            importance_score = max(importance_score, 0.6)
            time.append(start_time)
        
        motion_intensity = self.analyze_motion(frames)
        if motion_intensity > 0.7:
            importance_score = max(importance_score, 0.5)
            time.append(start_time)

        text_events = self.analyze_screen_text(frames)
        if text_events['ace_text'] or text_events['clutch_text']:
            importance_score = max(importance_score, 0.8)
            time.append(start_time)
        
        return {
            'events': events_detected,
            'importance_score': importance_score,
            'kill_count': len(kill_events),
            'clutch_info': clutch_info,
            'ability_analysis': ability_analysis,
            'audio_activity': audio_events,
            'motion_intensity': motion_intensity,
            'detected_text': text_events,
            'timestamp' : start_time
        } 

    def extract_kill_events(self, hud_data: List[Dict]) -> List[Dict]:
        "Extract kill events from HUD analysis over time"
        kills = []

        for data in hud_data:
            if 'kill_feed' in data:
                for kill in data['kill_feed']:
                    is_new = not any(k['victim'] == kill['victim'] and abs(k['timestamp'] - kill['timestamp']) < 1.0 for k in kills)

                    if is_new:
                        kills.append(kill)
        return kills

    def detect_clutch_situation(self, hud_data: List[Dict]) -> Dict:
        "Detect if this is a clutch situation"
        for data in hud_data:
            if 'team_status' in data:
                team_alive = data['team_status'].get('allies_alive', 5)
                enemies_alive = data['team_status'].get('enemies_alive', 5)

                if team_alive <= 2 and enemies_alive >= 3:
                    return {
                        'is_clutch': True,
                        'team_alive': team_alive,
                        'enemies_alive': enemies_alive,
                        'clutch_ratio': f"{team_alive}v{enemies_alive}"
                    }
        return {'is_clutch': False}

    def detect_smart_ability_usage(self, hud_data: List[Dict], kill_events: List[Dict], audio: np.ndarray) -> Dict:
        "Detect intelligent/skillful ability usage patterns"
        smart_usage = {
            'smart_usage_detected': False,
            'usage_types': [],
            'importance_boost': 0.0,
            'details': {}
        }
        
        ability_events = []
        ultimate_usage = []

        for i, frame_data in enumerate(hud_data):
            if 'ability_usage' in frame_data and frame_data['ability_usage']:
                ability_events.append({
                    'timestamp': i * 0.5,  # Assuming 2 FPS HUD analysis
                    'type': 'ability',
                    'ultimate': frame_data.get('ultimate_used', False)
                })
                if frame_data.get('ultimate_used', False):
                    ultimate_usage.append({
                        'timestamp': i * 0.5,
                        'round_context': frame_data.get('round_info', {})
                    })
        ultimate_combos = self.detect_ultimate_combos(ultimate_usage, kill_events)
        if ultimate_combos:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('ultimate_combo')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.8)
            smart_usage['details']['ultimate_combos'] = ultimate_combos
        
        ability_multikills = self.detect_ability_multikills(ability_events, kill_events)
        if ability_multikills:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('ability_multikill')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.75)
            smart_usage['details']['ability_multikills'] = ability_multikills
        
        clutch_ability = self.detect_clutch_ability_usage(ability_events, hud_data)
        if clutch_ability:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('clutch_ability')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.7)
            smart_usage['details']['clutch_abilities'] = clutch_ability
        
        creative_combos = self.detect_creative_ability_combos(ability_events)
        if creative_combos:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('creative_combo')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.65)
            smart_usage['details']['creative_combos'] = creative_combos

        game_winning = self.detect_game_winning_abilities(ability_events, hud_data)
        if game_winning:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('game_winning')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.85)
            smart_usage['details']['game_winning'] = game_winning
        
        smart_economy = self.detect_smart_ability_economy(ultimate_usage, hud_data)
        if smart_economy:
            smart_usage['smart_usage_detected'] = True
            smart_usage['usage_types'].append('smart_economy')
            smart_usage['importance_boost'] = max(smart_usage['importance_boost'], 0.6)
            smart_usage['details']['smart_economy'] = smart_economy
        
        return smart_usage

    def detect_ultimate_combos(self, ultimate_usage: List[Dict], kill_events: List[Dict]) -> List[Dict]:
        "Detect ultimate ability + kill combinations"
        combos = []

        for ult in ultimate_usage:
            nearby_kills = [
                kill for kill in kill_events
                if abs(kill.get('timestamp', 0) - ult['timestamp']) <= 3.0
            ]

            if len(nearby_kills) >= 1:
                combos.append({
                    'ultimate_time': ult['timestamp'],
                    'kills': nearby_kills,
                    'kill_count': len(nearby_kills),
                    'time_span': max([kill.get('timestamp', 0) for kill in nearby_kills]) - ult['timestamp']
                })
        
        return combos

    def detect_ability_multikills(self, ability_events: List[Dict], kill_events: List[Dict]) -> List[Dict]:
        "Detect multiple kills from single ability usage"
        multikills = []

        for ability in ability_events:
            nearby_kills = [
                kill for kill in kill_events
                if 0 <= kill.get('timestamp', 0) - ability['timestamp'] <= 5.0
            ]
            
            if len(nearby_kills) >= self.thresholds['smart_ability_usage']['ability_multikill']:
                multikills.append({
                    'ability_time': ability['timestamp'],
                    'kills': nearby_kills,
                    'kill_count': len(nearby_kills)
                })
        
        return multikills

    def detect_clutch_ability_usage(self, ability_events: List[Dict], hud_data: List[Dict]) -> List[Dict]:
        "Detect ability usage during clutch situations (1vX)"
        clutch_abilities = []
        for ability in ability_events:
            # Check HUD around ability timestamp for clutch situation
            frame_idx = int(ability['timestamp'] * 2)  # assuming 2 FPS HUD
            if frame_idx < len(hud_data):
                team_alive = hud_data[frame_idx]['team_status'].get('allies_alive', 5)
                enemies_alive = hud_data[frame_idx]['team_status'].get('enemies_alive', 5)
                if team_alive <= 2 and enemies_alive >= 2:
                    clutch_abilities.append(ability)
        return clutch_abilities
    
    def detect_creative_ability_combos(self, ability_events: List[Dict]) -> List[Dict]:
        "Detect unusual or skillful combinations of multiple abilities"
        combos = []
        ability_events_sorted = sorted(ability_events, key=lambda x: x['timestamp'])
        for i in range(len(ability_events_sorted) - 1):
            combo = [ability_events_sorted[i]]
            for j in range(i+1, len(ability_events_sorted)):
                if ability_events_sorted[j]['timestamp'] - ability_events_sorted[i]['timestamp'] <= 5:
                    combo.append(ability_events_sorted[j])
                else:
                    break
            if len(combo) >= 2:
                combos.append({'combo': combo, 'start_time': combo[0]['timestamp'], 'end_time': combo[-1]['timestamp']})
        return combos

    def detect_game_winning_abilities(self, ability_events: List[Dict], hud_data: List[Dict]) -> List[Dict]:
        "Detect abilities that secure round wins"
        winning_abilities = []
        for ability in ability_events:
            frame_idx = int(ability['timestamp'] * 2)
            if frame_idx < len(hud_data):
                round_end = hud_data[frame_idx].get('round_end', False)
                if round_end:
                    winning_abilities.append(ability)
        return winning_abilities
    
    def detect_smart_ability_economy(self, ultimate_usage: List[Dict], hud_data: List[Dict]) -> List[Dict]:
        "Detect saving ultimates or expensive abilities for key rounds"
        smart_saves = []
        for ult in ultimate_usage:
            frame_idx = int(ult['timestamp'] * 2)
            if frame_idx < len(hud_data):
                round_info = hud_data[frame_idx].get('round_info', {})
                if round_info.get('eco_round', False) or round_info.get('last_round', False):
                    smart_saves.append(ult)
        return smart_saves
    
    def analyze_audio_events(self, audio: np.ndarray, sr: int = 22050) -> Dict:
        "Simple heuristic: detect periods with high amplitude or gunshot/ability frequency peaks"
        high_activity = False
        if len(audio) == 0:
            return {'high_activity': False}
        # Compute simple RMS over audio
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0.05:  # threshold for "loud activity"
            high_activity = True
        return {'high_activity': high_activity}

    def analyze_motion(self, frames: np.ndarray) -> float:
        "Compute optical flow magnitude to estimate motion intensity"
        if len(frames) < 2:
            return 0.0
        motion_magnitudes = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion_magnitudes.append(np.mean(mag))
            prev_gray = curr_gray
        return float(np.mean(motion_magnitudes))

    def analyze_screen_text(self, frames: np.ndarray) -> Dict:
        "Use OCR to detect on-screen text indicating key events (e.g., ACE, CLUTCH)"
        ace_text_detected = False
        clutch_text_detected = False
        # If OCR has previously been found to be unavailable, skip heavy OCR work
        if not getattr(self, '_ocr_available', True):
            logging.debug("OCR disabled: skipping analyze_screen_text")
            return {'ace_text': False, 'clutch_text': False}
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                text = pytesseract.image_to_string(gray)
            except pytesseract.pytesseract.TesseractNotFoundError:
                # Disable OCR for future calls and warn once
                logging.warning("Tesseract binary not found. OCR will be disabled — install Tesseract and add to PATH to enable OCR.")
                self._ocr_available = False
                return {'ace_text': False, 'clutch_text': False}
            except Exception as e:
                logging.debug(f"OCR failed for one frame: {e}")
                continue

            if 'ACE' in text.upper():
                ace_text_detected = True
            if 'CLUTCH' in text.upper():
                clutch_text_detected = True
        return {'ace_text': ace_text_detected, 'clutch_text': clutch_text_detected}