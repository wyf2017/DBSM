#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
#set -x


## args parse
arch=${1-DispNetC}
loss_name=${2-SL1}
maxdisp=${3-192}
bn=${4-1}
flag_train=${5-"T(sf-tr)_F(k15-tr,k12-tr)"} # "T(sf-tr)"} # 
datas_val=${6-"k15-te"} # "k12-te"} # 
dir_datas_val=${7-"/media/qjc/D/data/kitti"}
echo


## loadmodel
mode="Submission"
loadmodel="./results/Train_${arch}_${loss_name}_${flag_train}/weight_final.pkl"


# log_filepath and dir_save
flag="${mode}_${datas_val}/${arch}_${loss_name}_${flag_train}"
dir_save="./results/${flag}"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"
echo


# Submission
freq_print=20
if test -e ${loadmodel} ; then
    ./main.py --mode ${mode} --arch $arch --maxdisp $maxdisp \
                   --loadmodel $loadmodel \
                   --datas_val $datas_val --dir_datas_val $dir_datas_val  \
                   --bn $bn\
                   --freq_print $freq_print \
                   --dir_save $dir_save \
                   2>&1 | tee -a "$LOG"
else
    echo -e "Weight file[$loadmodel] is not exist, please check !!! "
fi

echo


echo -e "************ end of ${0}\n\n\n"




