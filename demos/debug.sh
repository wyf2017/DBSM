#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "\n************ start of ${0} ${*} ************\n\n\n"
#set -x


## arch of model
arch=${1-DispNetC}
loss_name=${2-SV-SL1}
bn=${3-2}
freq_optim=${4-1}

mode_down_disp="max" # avg/max
mode_down_img="Simple" # Simple/Gaussion/DoG

freq_print=20
maxdisp=64
mode="Train"
loadmodel="None"


## dataset--kitti
crop_width=384
crop_height=128
dir_root="/media/qjc/D/data/kitti/"
datas_train="k15-two,k12-two"
datas_val="k15-two,k12-two"
dir_datas_train=${dir_root}
dir_datas_val=${dir_root}


## log_filepath and dir_save
flag="debug--${mode}_${arch}/${loss_name}_${mode:0:1}(${datas_train})"
dir_save="./results/${flag}"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"


## epochs
epochs=2
nloop=2
epochs_warmup=0
freq_save=1
echo


## optimer
lr=0.001
lr_epoch0=1
lr_stride=10
lr_decay=0.5
weight_decay=0.0001


## kargs_all
kargs0="
        --mode ${mode}  --arch ${arch}  --maxdisp ${maxdisp}  --loadmodel ${loadmodel}" # model
kargs1="
        --loss_name ${loss_name}  --flag_FC
        --mode_down_disp ${mode_down_disp}  --mode_down_img ${mode_down_img}"           # lossfun  --flag_FCTF 
kargs2="
        --datas_train ${datas_train}  --dir_datas_train ${dir_datas_train} 
        --datas_val ${datas_val}  --dir_datas_val ${dir_datas_val} 
        --bn ${bn}  --crop_width ${crop_width}  --crop_height ${crop_height}"          # dataset
kargs3="
        --epochs ${epochs}  --nloop ${nloop}  --epochs_warmup ${epochs_warmup}
        --freq_save ${freq_save}"                                                      # epochs
kargs4="
        --lr ${lr}  --freq_optim ${freq_optim}  --weight_decay ${weight_decay} 
        --lr_epoch0 ${lr_epoch0}  --lr_stride ${lr_stride}  --lr_decay ${lr_decay}"    # optimer
kargs5="
        --freq_print ${freq_print}  --dir_save ${dir_save}" # others
#echo -e "karga as follow:\n
#        kargs_model:   ${kargs0}\n
#        kargs_lossfun: ${kargs1}\n
#        kargs_dataset: ${kargs2}\n
#        kargs_epochs:  ${kargs3}\n
#        kargs_optimer: ${kargs4}\n
#        kargs_others:  ${kargs5}\n"
echo

## main entry (train model)
path_weight="${dir_save}/weight_final.pkl"
if test -e ${path_weight} ; then
    echo -e "\nTraining finished! \n\n path_weight: ${path_weight}\n\n"
else
    ./main.py ${kargs0} ${kargs1} ${kargs2} ${kargs3} ${kargs4} ${kargs5} 2>&1 | tee -a "$LOG"
fi


echo -e "\n************ end of ${0} ${*} ************\n\n\n"


