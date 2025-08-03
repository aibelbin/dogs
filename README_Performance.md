# Raspberry Pi Performance Optimization

## Key Changes Made:

### 1. **Lightweight Model**
- Changed from YOLOv5s to YOLOv5n (nano version)
- 75% smaller model size
- 3x faster inference

### 2. **Frame Skipping**
- Only processes every 2nd frame by default
- Reuses last detection results for skipped frames
- Doubles effective FPS

### 3. **Resolution Optimization**
- Camera: 480x360 (down from 640x480)
- Detection: 320x240 (then scaled up)
- 60% fewer pixels to process

### 4. **CPU Optimizations**
- Model set to eval mode
- Gradients disabled for inference
- Memory usage reduced

### 5. **Lower Confidence**
- Reduced from 0.5 to 0.4
- Catches more dogs with less certainty
- Faster processing

## Expected Performance:
- **Before**: 2-5 FPS on Raspberry Pi 4
- **After**: 8-15 FPS on Raspberry Pi 4

## Installation on Raspberry Pi:

```bash
# Install CPU-only PyTorch (faster on Pi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements_rpi.txt

# Run the optimized version
python main.py
```

## Additional Pi Optimizations:

### GPU Memory Split (optional):
```bash
sudo raspi-config
# Advanced Options > Memory Split > Set to 128 or 256
```

### CPU Governor:
```bash
# Set to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Disable Desktop (for headless):
```bash
sudo systemctl set-default multi-user.target
```

## Adjustable Parameters:

- `skip_frames`: Higher = faster but less responsive
- `confidence_threshold`: Lower = more detections
- Camera resolution: Lower = faster
- Detection resolution: Lower = faster but less accurate
