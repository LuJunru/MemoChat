export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

maindir=$1
datadir=${maindir}data
codedir=${maindir}code

settings=("10k")
models=("t5-3b" "vicuna-7b" "vicuna-13b" "vicuna-33b")

for model in "${models[@]}"
    do
    for setting in "${settings[@]}"
        do
        python3 ${codedir}/codes/api/llm_judge.py \
            ${datadir}/mtbenchplus/mtbenchplus_testing/mtbenchplus_testing_${model}_${setting}.json \
            gpt-4 \
            YourOpenAIKey \
            ${datadir}/llm_judge/llm_judge_gpt-4_${model}_${setting}.json \
            ${datadir}/prompts.json
        done
    done

gpt_settings=("2k" "memochat")

for gpt_setting in "${gpt_settings[@]}"
    do
    python3 ${codedir}/codes/api/llm_judge.py \
        ${datadir}/mtbenchplus/mtbenchplus_testing/mtbenchplus_testing_gpt-3.5-turbo-${gpt_setting}.json \
        gpt-4 \
        YourOpenAIKey \
        ${datadir}/llm_judge/llm_judge_gpt-4_gpt-3.5-turbo-${gpt_setting}.json \
        ${datadir}/prompts.json
    done

