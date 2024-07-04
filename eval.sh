#!/bin/zsh
source ~/.zshrc

# set up python environment
conda activate vstream

# set important configurations
ngpus=1
gputype=4090

# auto calculate configurations
gpus_list=$(seq -s, 0 $((ngpus - 1)))
date_device="$(date +%m%d)_${ngpus}${gputype}"
#your_model_checkpoint_path = "/ckpt"
echo start eval
# define your openai info here
OPENAIKEY="sk-proj-QF2mlTYRd2kOgAWLBuxNT3BlbkFJZCish6BWeNWOszMXNhaU"
#OPENAIBASE = None
#OPENAITYPE = None
#OPENAIVERSION = None



for dataset in msvd msrvtt
do
    echo start eval ${dataset}
    python -m flash_vstream.eval_video.eval_any_dataset_features \
        --model-path "./ckpt/Flash-VStream-7b"\
        --dataset ${dataset} \
        --num_chunks $ngpus \
        --api_key $OPENAIKEY \
#        --api_base $OPENAIBASE \
#        --api_type $OPENAITYPE \
#        --api_version $OPENAIVERSION \
        --test \
        >> ${date_device}_vstream-7b-eval-${dataset}.log 2>&1 
done

