export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

maindir=$1
datadir=${maindir}data
codedir=${maindir}code

test_data=${datadir}/mtbenchplus/mtbenchplus.json

settings=("1k", "10k")
models=("t5-3b" "vicuna-7b" "vicuna-13b" "vicuna-33b")

for model in "${models[@]}"
    do
    for setting in "${settings[@]}"
        do
        finetuned_model_path=${maindir}model/${model}_${setting}/
        case ${model} in 
            "vicuna-33b")
                RAYGPUS=2
                ;;
            "t5-3b"|"vicuna-7b"|"vicuna-13b")
                RAYGPUS=1
                ;;
        esac
        python3 ${codedir}/codes/eval/get_model_infer_memochat.py \
            --model-path ${finetuned_model_path} \
            --question-file ${test_data} \
            --answer-file ${datadir}/mtbenchplus/mtbenchplus_testing/mtbenchplus_testing_${model}_${setting}.json \
            --num-gpus $GPU_NUM_PER_NODE \
            --ray-num-gpus ${RAYGPUS} \
            --prompt-path ${datadir}/prompts.json
        done
    done
