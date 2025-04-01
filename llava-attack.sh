GPU_ID=$1
OUTPUT_PATH=$2
MODEL_PATH=$3
EPS=$4

# MODEL_PATH='ys-zong/llava-v1.5-7b-Posthoc-lora'
# OUTPUT_PATH='outputs/rebuttal/attack/llava_vlguard/'

if [ ! -d "$OUTPUT_PATH/attack" ]; then
  mkdir -p "$OUTPUT_PATH/attack"
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python llava_v1_5_visual_attack.py \
--model-path $MODEL_PATH \
--eps $EPS \
--alpha 1 \
--n_iters 5000 \
--constrained \
--save_dir $OUTPUT_PATH/attack-$EPS 2>&1 | tee -a "$OUTPUT_PATH/attack-$EPS/training_log.txt"

wait

save_image_path="$OUTPUT_PATH/attack/LLaVA-v15-7B-eps$EPS-alpha1-iters5000-constrained.bmp"

bash omi_eval_rtp.sh $OUTPUT_PATH/attack/LLaVA-v15-7B-eps$EPS-alpha1-iters5000-constrained/attack/ $save_image_path $MODEL_PATH &
bash omi_eval_rtp_diffpure.sh $OUTPUT_PATH/attack/LLaVA-v15-7B-eps$EPS-alpha1-iters5000-constrained/diffpure_50_50/ $save_image_path $MODEL_PATH 50 &

wait
bash omi_eval_rtp_diffpure.sh $OUTPUT_PATH/attack/LLaVA-v15-7B-eps$EPS-alpha1-iters5000-constrained/diffpure_100_100/ $save_image_path $MODEL_PATH 100 &
bash omi_eval_rtp_diffpure.sh $OUTPUT_PATH/attack/LLaVA-v15-7B-eps$EPS-alpha1-iters5000-constrained/diffpure_150_150/ $save_image_path $MODEL_PATH 150 &

wait
