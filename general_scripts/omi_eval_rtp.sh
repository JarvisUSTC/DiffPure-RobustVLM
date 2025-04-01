output_path=$1
image_prompt_path=$2
model_path=$3

# if output path does not exist, create it
if [ ! -d $output_path ]; then
    mkdir -p $output_path
fi

# for i in {0..6}
# do
#   # 限制可见GPU
#   nohup bash -c "CUDA_VISIBLE_DEVICES=$i python qwen2vl_general_inference.py --model_path $model_path --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_$i.jsonl --output_file $output_path/rtp_prompts_challenges_$i.jsonl" > $i.log 2>&1 &
# done

# CUDA_VISIBLE_DEVICES=7 python qwen2vl_general_inference.py --model_path $model_path  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_7.jsonl --output_file $output_path/rtp_prompts_challenges_7.jsonl
# 限制可见GPU
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python omi_general_inference.py --model_path $model_path --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts.jsonl --output_file $output_path/rtp_prompts_challenges.jsonl" > $output_path/output.log 2>&1 &

echo "Started all processes"

# wait for all processes to finish
wait

echo "All processes finished"

# # merge all the jsonl outputs
# # loop
# for i in {0..7}
# do
#     cat $output_path/rtp_prompts_challenges_$i.jsonl >> $output_path/rtp_prompts_challenges.jsonl
# done

# echo "Merged all jsonl outputs"

# # remove the individual jsonl outputs
# for i in {0..7}
# do
#     rm $output_path/rtp_prompts_challenges_$i.jsonl
# done

# echo "Removed individual jsonl outputs"

# # move the logs to output path
# for i in {0..7}
# do
#     mv $i.log $output_path
# done

# echo "Moved logs"

# evaluate the generated prompts
python get_metric.py --input $output_path/rtp_prompts_challenges.jsonl --output $output_path/rtp_prompts_challenges_metrics.jsonl

echo "Evaluated the generated prompts"

# cal metric
python cal_metrics_markdown.py --input $output_path/rtp_prompts_challenges_metrics.jsonl

echo "Calculated metrics"