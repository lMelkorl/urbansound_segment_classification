#!/bin/bash
set -e

# Create checkpoint folder if not exists
mkdir -p urbansound_segment_task/goals/goal2_esresnext/checkpoints

echo "Downloading ESResNeXt pre-trained checkpoint..."
wget -O urbansound_segment_task/goals/goal2_esresnext/checkpoints/ESResNeXtFBSP_AudioSet.pt \
  "https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt"

echo "✅ Done. Checkpoints saved under urbansound_segment_task/goals/goal2_esresnext/checkpoints/"
