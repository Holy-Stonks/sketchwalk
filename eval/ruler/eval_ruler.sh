# model="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
model="meta-llama/Llama-3.1-8B-Instruct"
# change model to the path of your model if needed
basedir=./results/llama
threshold=5e-4
export CUDA_VISIBLE_DEVICES=0
export PROFILE_FILE=${basedir}/${threshold}.txt # Comment this line to disable profiling
bash run_seer.sh \
    $model \
    SketchWalk \
    $basedir \
    $threshold

## Get profiled sparsity
python averaged_sparsity.py --file $PROFILE_FILE