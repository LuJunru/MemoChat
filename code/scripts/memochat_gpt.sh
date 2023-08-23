export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

maindir=$1
datadir=${maindir}data
codedir=${maindir}code

gpt_settings=("2k" "memochat")

for gpt_setting in "${gpt_settings[@]}"
    do
    python3 ${codedir}/codes/api/gpt_${gpt_setting}.py \
        ${datadir}/mtbenchplus/mtbenchplus.json \
        gpt-3.5-turbo \
        YourOpenAIKey \
        ${datadir}/mtbenchplus/mtbenchplus_testing/mtbenchplus_testing_gpt-3.5-turbo-${gpt_setting}.json \
        ${datadir}/prompts.json
    done
