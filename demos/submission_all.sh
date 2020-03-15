#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
set -x


#archs="DispNetC WSMCnet MBFnet MTLnet"
#losses="SV-SL1 SV-CE SV-SL1+CE"
#losses="LUSV-A LUSV-AS1 DUSV-AC1 DUSV-AS1C1 DUSV-AS1C1 DUSV-AS1C1-M"
#losses="DUSV-AS1C1-M DUSV-AS2C1-M DUSV-AS3C1-M DUSV-AS4C1-M SV-SL1"


entry="./demos/submission.sh"
archs="MTLnet"
losses="SV-SL1"
maxdisp=192
bn=1

flag_train="T(sf-tr)_F(k15-tr,k12-tr)_F(k15,k12)" # "T(sf-tr)" # 
datas_val="k12-te" # "k15-te" # 
dir_datas_val="/media/qjc/D/data/kitti"


# train
for arch in ${archs}; do
    for loss_name in ${losses}; do
        kargs="${arch} ${loss_name} ${maxdisp} ${bn} ${flag_train} ${datas_val} ${dir_datas_val}"
        ${entry} ${kargs}
    done
done


echo -e "************ end of ${0} ${*} ************\n\n\n"


