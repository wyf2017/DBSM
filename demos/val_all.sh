#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
#set -x

#archs="DispNetC WSMCnet MBFnet MTLnet"
#losses="SV-SL1 SV-CE SV-SL1+CE"
#losses="LUSV-A LUSV-AS1 DUSV-AC1 DUSV-AS1C1 DUSV-AS1C1 DUSV-AS1C1-M"
#losses="DUSV-AS1C1-M DUSV-AS2C1-M DUSV-AS3C1-M DUSV-AS4C1-M SV-SL1"

entry="./demos/val.sh"
archs="MTLnet"
losses="SV-SL1"
maxdisp=256
bn=1

flag_train="T(sf-tr)_F(kme-tr)" # "T(sf-tr)" # "T(sf-tr)_F(k15-tr,k12-tr)" # "F(k15-tr,k12-tr)" # 
dir_datas="/media/qjc/D/data"
datas_val="middeval-val" # "eth3d-val" # "k15-val,k12-val" # "sf-val" # "k15-val" # 
dir_datas_val="${dir_datas}/MiddEval3" # "${dir_datas}/eth3d" # "${dir_datas}/kitti" # "${dir_datas}/sceneflow" # 


# train
for arch in ${archs}; do
    for loss_name in ${losses}; do
        kargs="${arch} ${loss_name} ${maxdisp} ${bn} ${flag_train} ${datas_val} ${dir_datas_val}"
        ${entry} ${kargs}
    done
done


echo -e "************ end of ${0} ${*} ************\n\n\n"


