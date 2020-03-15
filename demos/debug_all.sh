#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
#set -x

entry="./demos/debug.sh"
bn=2
freq_optim=1

archs="MTLnet"
losses="SV-SL1"

#archs="DispNetC WSMCnet MBFnet MTLnet"
#losses="SV-SL1 SV-CE SV-SL1+CE"
#losses="${losses} DUSV-AS1C1-M DUSV-AS2C1-M DUSV-AS3C1-M DUSV-AS2C2-M"
#losses="${losses} LUSV-AS1 LUSV-AS2 LUSV-AS3 LUSV-AS2-EC"

# test
for arch in ${archs}; do
    for loss_name in ${losses}; do
        kargs="${arch} ${loss_name} ${bn} ${freq_optim}"
        ${entry} ${kargs}
    done
done


# collect
dir_save="./results"
for arch in ${archs}; do
    loss_passed=""
    loss_error=""
    for loss_name in ${losses}; do
        path_weight="${dir_save}/debug--Train_${arch}/${loss_name}_T(k15-two,k12-two)/weight_final.pkl"
        if test -e ${path_weight} ; then
            loss_passed="${loss_passed} ${loss_name},"
        else
            loss_error="${loss_error} ${loss_name},"
        fi
    done
    echo -e "\n arch: ${arch} \n loss_passed: ${loss_passed} \n loss_error: ${loss_error} \n\n "
done


echo -e "************ end of ${0} ${*} ************\n\n\n"


