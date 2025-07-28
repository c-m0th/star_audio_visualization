import pygame
import numpy as np
import librosa
from moviepy import VideoClip
import random
import math
import pygame.surfarray

# 配置参数
WIDTH, HEIGHT = 1920, 1080  # 16:9 分辨率
FPS = 30
BPM = 85
BEAT_DURATION = 60 / BPM  # 节拍持续时间（秒）
BACKGROUND_COLOR = (0, 0, 0)
STAR_COLOR = (255, 255, 255)
NUM_STAR_GROUPS = 80  # 大幅增加星组数量
STARS_PER_GROUP = 7
MAX_STAR_RADIUS = 5
MIN_STAR_RADIUS = 1
BASE_SPEED = 3.0  # 基础移动速度
MAX_SPEED = 8.0  # 最大移动速度
SCREEN_DIAGONAL = math.sqrt(WIDTH**2 + HEIGHT**2)  # 屏幕对角线长度

def load_audio(audio_path):
    """加载音频文件并分析特征"""
    y, sr = librosa.load(audio_path, sr=None)
    
    # 计算人声频段能量（300-3000Hz）
    vocal_energy = []
    hop_length = 512
    frame_count = 0
    for i in range(0, len(y), hop_length):
        frame = y[i:i+hop_length]
        if len(frame) < hop_length:
            continue
        stft = librosa.stft(frame)
        freq_bins = librosa.fft_frequencies(sr=sr)
        vocal_band = np.logical_and(freq_bins >= 300, freq_bins <= 3000)
        energy = np.sum(np.abs(stft[vocal_band])**2)
        vocal_energy.append(energy)
        frame_count += 1
    
    # 归一化能量
    max_energy = max(vocal_energy) if max(vocal_energy) > 0 else 1
    vocal_energy = [e / max_energy for e in vocal_energy]
    
    # 检测节拍
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return y, sr, vocal_energy, beat_frames, frame_count

class StarGroup:
    def __init__(self, group_id):
        self.id = group_id
        self.stars = []
        self.connections = []
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = BASE_SPEED  # 初始速度为基础速度
        self.distance = random.uniform(0, 200)  # 初始距离
        self.reset()
        
    def reset(self):
        """重置星组位置和连接"""
        self.stars = []
        self.connections = []
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = BASE_SPEED
        self.distance = random.uniform(0, 200)  # 靠近中心位置开始
        
        # 创建随机分布的星星
        for _ in range(STARS_PER_GROUP):
            angle_offset = random.uniform(-1, 1)  # 增加角度偏移范围
            dist_offset = random.uniform(1, 2)  # 增加距离偏移范围
            self.stars.append({
                'angle_offset': angle_offset,
                'dist_offset': dist_offset,
                'pulse_phase': random.uniform(0, 2 * math.pi)
            })
        
        # 随机创建连接
        for i in range(len(self.stars)):
            for j in range(i + 1, len(self.stars)):
                if random.random() < 0.35:  # 增加连接概率
                    self.connections.append({
                        'stars': (i, j),
                        'alpha': 0,
                        'active': False
                    })
    
    def update(self, time, vocal_energy):
        """更新星组位置和状态"""
        # 根据距离中心的比例计算速度：离中心越远速度越快
        distance_ratio = self.distance / SCREEN_DIAGONAL
        self.speed = BASE_SPEED + (MAX_SPEED - BASE_SPEED) * distance_ratio**3
        
        # 更新距离
        self.distance += self.speed
        
        # 计算当前中心位置
        center_x = WIDTH/2 + math.cos(self.angle) * self.distance
        center_y = HEIGHT/2 + math.sin(self.angle) * self.distance
        
        # 如果星组移出屏幕边界，重置
        if (center_x < -100 or center_x > WIDTH + 100 or 
            center_y < -100 or center_y > HEIGHT + 100):
            self.reset()
            return
        
        # 更新连接状态
        beat_index = int(time / BEAT_DURATION)
        
        # 计算每帧透明度变化量
        alpha_change = 255 * BPM / (60 * FPS)
        
        for conn in self.connections:
            # 在奇数拍开始时激活新连接
            if beat_index % 2 == 0 and not conn['active']:
                conn['active'] = True
                conn['alpha'] = 0
            
            # 在偶数拍开始时关闭连接
            elif beat_index % 2 == 1 and conn['active']:
                conn['active'] = False
            
            # 更新透明度
            if conn['active']:
                conn['alpha'] = min(255, conn['alpha'] + alpha_change)
            else:
                conn['alpha'] = max(0, conn['alpha'] - alpha_change)

def create_frame(t, stars, vocal_energy, total_frames, duration):
    """创建视频帧"""
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill(BACKGROUND_COLOR)
    
    # 获取当前时间对应的能量值
    if total_frames > 0:
        energy_idx = min(total_frames - 1, int(t * total_frames / duration))
        current_energy = vocal_energy[energy_idx] if vocal_energy else 0
    else:
        current_energy = 0
    
    # 更新和绘制所有星组
    for group in stars:
        group.update(t, current_energy)
        
        # 计算星组中心位置
        center_x = WIDTH/2 + math.cos(group.angle) * group.distance
        center_y = HEIGHT/2 + math.sin(group.angle) * group.distance
        
        # 绘制星星
        for star in group.stars:
            # 计算星星位置（添加随机偏移）
            angle = group.angle + star['angle_offset']
            distance = group.distance * star['dist_offset']
            x = center_x + math.cos(angle) * distance
            y = center_y + math.sin(angle) * distance
            
            # 根据人声能量调整星星大小
            pulse = 0.5 + 0.5 * math.sin(star['pulse_phase'] + current_energy * 15)
            radius = MIN_STAR_RADIUS + (MAX_STAR_RADIUS - MIN_STAR_RADIUS) * pulse
            
            pygame.draw.circle(surface, STAR_COLOR, (int(x), int(y)), int(radius))
        
        # 绘制连接线
        for conn in group.connections:
            if conn['alpha'] > 0:
                i, j = conn['stars']
                star1 = group.stars[i]
                star2 = group.stars[j]
                
                # 计算端点位置
                angle1 = group.angle + star1['angle_offset']
                distance1 = group.distance * star1['dist_offset']
                x1 = center_x + math.cos(angle1) * distance1
                y1 = center_y + math.sin(angle1) * distance1
                
                angle2 = group.angle + star2['angle_offset']
                distance2 = group.distance * star2['dist_offset']
                x2 = center_x + math.cos(angle2) * distance2
                y2 = center_y + math.sin(angle2) * distance2
                
                # 根据透明度绘制线
                alpha = min(255, max(0, int(conn['alpha'])))
                color = (255, 255, 255, alpha)
                
                # 创建临时surface绘制透明线
                line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surface, color, (x1, y1), (x2, y2), 1)
                surface.blit(line_surface, (0, 0))
    
    # 将Surface转换为NumPy数组（MoviePy兼容格式）
    frame_array = pygame.surfarray.array3d(surface)
    frame_array = np.transpose(frame_array, (1, 0, 2))
    return frame_array

def main(audio_file, output_file):
    """主函数：处理音频并生成视频"""
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.HIDDEN)
    
    # 加载和分析音频
    print("正在分析音频...")
    y, sr, vocal_energy, beat_frames, total_frames = load_audio(audio_file)
    duration = len(y) / sr
    
    # 创建星组
    stars = [StarGroup(i) for i in range(NUM_STAR_GROUPS)]
    
    # 创建视频剪辑
    print("正在生成视频...")
    def make_frame(t):
        return create_frame(t, stars, vocal_energy, total_frames, duration)
    
    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(
        output_file,
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        bitrate='8000k',
        audio=audio_file
    )
    print(f"视频已保存至: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python star_visualizer.py input.wav output.mp4")
    else:
        main(sys.argv[1], sys.argv[2])